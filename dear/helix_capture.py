"""One continuous, opt-in Nsight capture. Python 3.8; no per-step GPU sync."""
import atexit
import os
from functools import wraps
import helix_nvtx


def add_arguments(parser):
    parser.add_argument('--nvtx-capture-start', type=int,
                        default=int(os.environ.get('HELIX_CAPTURE_START', '40')),
                        help='zero-based benchmark_step invocation, including warmup')
    parser.add_argument('--nvtx-capture-steps', type=int,
                        default=int(os.environ.get('HELIX_CAPTURE_STEPS', '0')),
                        help='core source steps; zero disables capture control')
    parser.add_argument('--nvtx-capture-tail', type=int,
                        default=int(os.environ.get('HELIX_CAPTURE_TAIL', '1')),
                        help='extra calls to consume deferred updates, at least one')
    parser.add_argument('--nvtx-autograd', type=int, choices=[0, 1],
                        default=int(os.environ.get('HELIX_NVTX_AUTOGRAD', '0')),
                        help='emit automatic autograd/ATen NVTX operators once per window')


class Capture:
    def __init__(self, args, torch_module=None):
        self.start = args.nvtx_capture_start
        self.steps = args.nvtx_capture_steps
        self.tail = args.nvtx_capture_tail
        self.autograd = bool(args.nvtx_autograd)
        self.enabled = self.steps > 0
        self.active = False
        self.completed = False
        self._torch = torch_module
        self._ops = None
        self._span = None
        if self.start < 0 or self.steps < 0 or self.tail < 1:
            raise ValueError('capture start/steps must be nonnegative; tail must be >= 1')
        if self.enabled:
            if not helix_nvtx.ENABLED or not args.cuda:
                raise ValueError('capture requires HELIX_NVTX=1 before import and CUDA')
            if args.overlap_summary or args.overlap_timeline:
                raise ValueError('disable legacy overlap profiling for this NVTX capture')
            if args.loss_log_every:
                raise ValueError('use --loss-log-every 0; loss.item() alters the trace')
            total = 1 + args.num_warmup_batches + args.num_iters * args.num_batches_per_iter
            if self.start + self.steps + self.tail > total:
                raise ValueError('capture window exceeds total benchmark_step calls (%d)' % total)
            if args.compressor == 'halfrankk' and self.start <= args.compress_warmup:
                raise ValueError('capture must start after compression warmup')
            atexit.register(self.close)

    def annotate_step(self, fn):
        if not self.enabled and not helix_nvtx.ENABLED:
            return fn
        @wraps(fn)
        def wrapped(*args, **kwargs):
            step = fn.__globals__['step']
            self.before_step(step)
            try:
                with helix_nvtx.context(step=step):
                    with helix_nvtx.range('bert_step', step=step):
                        result = fn(*args, **kwargs)
            except BaseException:
                self.close()
                raise
            self.after_step(step)
            return result
        return wrapped

    def before_step(self, step):
        if not self.enabled or step != self.start:
            return
        if self._torch is None:
            import torch
            self._torch = torch
        # emit_nvtx enters once and may synchronize here. Never enter it per step.
        if self.autograd:
            self._ops = self._torch.autograd.profiler.emit_nvtx(record_shapes=False)
            self._ops.__enter__()
        try:
            self._torch.cuda.profiler.start()
        except BaseException:
            if self._ops is not None:
                self._ops.__exit__(None, None, None)
                self._ops = None
            raise
        self.active = True
        self._span = helix_nvtx.range('capture', source_begin=self.start,
                                      source_end=self.start + self.steps,
                                      tail_steps=self.tail)
        self._span.__enter__()

    def after_step(self, step):
        if self.active and step + 1 == self.start + self.steps + self.tail:
            self.close()
            self.completed = True

    def close(self):
        if not self.active:
            return
        self.active = False
        try:
            # One window-end drain of submitted GPU work; no optimizer call or
            # extra forward is inserted. Tail calls are ordinary training calls.
            with helix_nvtx.range('capture_end_sync'):
                if self._ops is not None:
                    self._ops.__exit__(None, None, None)
                    self._ops = None
                else:
                    self._torch.cuda.synchronize()
        finally:
            try:
                self._span.__exit__(None, None, None)
            finally:
                self._torch.cuda.profiler.stop()

    def finish(self):
        self.close()
        if self.enabled and not self.completed:
            raise RuntimeError('capture did not complete; do not use this report as a complete window')
