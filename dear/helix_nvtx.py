"""Optional host-side stage annotations; no CUDA synchronization is added.

Set HELIX_NVTX=1 before Python starts. Python 3.8 compatible.
"""
import inspect
import os
import threading
from contextlib import contextmanager, nullcontext
from functools import wraps

ENABLED = os.environ.get('HELIX_NVTX', '0') == '1'
_api = None
_local = threading.local()


@contextmanager
def _active_context(fields):
    previous = getattr(_local, 'fields', {})
    _local.fields = dict(previous, **fields)
    try:
        yield
    finally:
        _local.fields = previous


def context(**fields):
    """Propagate CPU metadata on the current thread, without reading tensors."""
    return _active_context(fields) if ENABLED else nullcontext()


def _label(stage, fields):
    parts = ['helix/' + stage]
    worker = os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('RANK'))
    if worker is not None:
        parts.append('worker=' + worker)
    combined = dict(getattr(_local, 'fields', {}))
    combined.update(fields)
    for key, value in combined.items():
        # Never format a tensor: that can implicitly synchronize the device.
        if value is not None and isinstance(value, (str, int, float, bool)):
            parts.append('%s=%s' % (key, value))
    return ' | '.join(parts)


@contextmanager
def _active_range(stage, fields):
    global _api
    if _api is None:
        import torch
        _api = torch.cuda.nvtx
    _api.range_push(_label(stage, fields))
    try:
        yield
    finally:
        _api.range_pop()


def range(stage, **fields):
    return _active_range(stage, fields) if ENABLED else nullcontext()


def annotate(stage, arg_names=()):
    """Decorate synchronous host calls; push/pop remain on the calling thread."""
    def decorate(fn):
        if not ENABLED:
            return fn
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            values = signature.bind_partial(*args, **kwargs).arguments
            fields = {key: values.get(key) for key in arg_names}
            owner = values.get('self')
            if owner is not None:
                fields.setdefault('runtime_step', getattr(owner, '_num_steps', None))
                fields.setdefault('active_factor', getattr(owner, '_active_compression_factor', None))
                inherited = getattr(_local, 'fields', {})
                runtime_step = getattr(owner, '_num_steps', None)
                if runtime_step is not None:
                    deferred = stage in ('forward_gate', 'module_update', 'optimizer_update')
                    fields.setdefault('source_step', inherited.get(
                        'source_step', runtime_step - 1 if deferred else runtime_step))
                    if deferred:
                        fields.setdefault('consume_step', runtime_step)
                name = values.get('name')
                if name is None and 'p' in values:
                    name = getattr(owner, '_param_names', {}).get(values['p'])
                    fields.setdefault('name', name)
                if name is not None:
                    indices = getattr(owner, '_param_group_idx', {}).get(name)
                    if indices is not None:
                        fields.setdefault('group_idx', indices[0])
                if 'module' in values:
                    module_name = getattr(owner, '_module_names', {}).get(values['module'])
                    fields.setdefault('module_name', module_name)
                    indices = getattr(owner, '_module_group_idx', {}).get(module_name)
                    if indices is not None:
                        fields.setdefault('group_idx', indices[0])
            with context(**fields):
                with range(stage, **fields):
                    return fn(*args, **kwargs)
        return wrapped
    return decorate


def hook(owner, parameter):
    """Place a range on the actual autograd hook thread, not just the caller."""
    def decorate(fn):
        if not ENABLED:
            return fn

        @wraps(fn)
        def wrapped(*args, **kwargs):
            name = owner._param_names.get(parameter)
            indices = owner._param_group_idx.get(name)
            fields = dict(name=name, source_step=owner._num_steps,
                          group_idx=indices[0] if indices else None)
            with context(**fields):
                with range('bp_hook', **fields):
                    return fn(*args, **kwargs)
        return wrapped
    return decorate
