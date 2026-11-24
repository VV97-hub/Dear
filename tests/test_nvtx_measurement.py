"""Meaningful CPU regressions: import order, unchanged math, capture, attribution."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeNVTX:
    def __init__(self): self.calls = []
    def range_push(self, text): self.calls.append(('push', text))
    def range_pop(self): self.calls.append(('pop',))


def args(**kwargs):
    values = dict(nvtx_capture_start=2, nvtx_capture_steps=2, nvtx_capture_tail=1,
                  nvtx_autograd=1, cuda=True, overlap_summary=False,
                  overlap_timeline=False, loss_log_every=0, num_warmup_batches=1,
                  num_iters=10, num_batches_per_iter=1, compressor='halfrankk',
                  compress_warmup=0)
    values.update(kwargs)
    return SimpleNamespace(**values)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {'HELIX_NVTX':'1'}):
            self.nvtx = load('nvtx_for_capture','dear/helix_nvtx.py')
        self.nvtx._api = FakeNVTX()
        with patch.dict(sys.modules, {'helix_nvtx': self.nvtx}):
            self.capture = load('capture_test','dear/helix_capture.py')
        self.events = []
        events = self.events
        class Ops:
            def __enter__(self): events.append('ops_enter')
            def __exit__(self,*unused): events.append('ops_exit')
        self.torch = SimpleNamespace(
            autograd=SimpleNamespace(profiler=SimpleNamespace(emit_nvtx=lambda **k: Ops())),
            cuda=SimpleNamespace(profiler=SimpleNamespace(start=lambda:events.append('start'),
                                                        stop=lambda:events.append('stop')),
                                 synchronize=lambda:events.append('sync')))

    def test_one_capture_with_tail_no_per_step_sync(self):
        c = self.capture.Capture(args(),self.torch)
        for i in range(8):
            c.before_step(i)
            self.events.append(('step',i))
            c.after_step(i)
        c.finish()
        self.assertEqual(self.events.count('start'),1)
        self.assertEqual(self.events.count('ops_enter'),1)
        self.assertEqual(self.events.count('ops_exit'),1)
        self.assertNotIn('sync',self.events)  # automatic profiler owns boundary sync
        self.assertLess(self.events.index(('step',4)),self.events.index('stop'))
        self.assertLess(self.events.index('stop'),self.events.index(('step',5)))

    def test_manual_mode_syncs_once_at_end_and_incomplete_rejected(self):
        c = self.capture.Capture(args(nvtx_autograd=0),self.torch)
        c.before_step(2)
        c.after_step(2)
        with self.assertRaisesRegex(RuntimeError,'did not complete'): c.finish()
        c.close()
        self.assertEqual(self.events,['start','sync','stop'])

    def test_rejects_empty_window_strict_probes_and_loss_readback(self):
        for override in [dict(num_iters=0),dict(overlap_timeline=True),dict(loss_log_every=1),
                         dict(compress_warmup=5),dict(nvtx_capture_tail=0)]:
            with self.subTest(override=override),self.assertRaises(ValueError):
                self.capture.Capture(args(**override),self.torch)

    def test_deferred_metadata_and_hook_thread_context(self):
        nvtx = self.nvtx
        parameter = object()
        class Runtime:
            _num_steps=7
            _active_compression_factor='q'
            _param_names={parameter:'weight'}
            _param_group_idx={'weight':(3,0,0,1)}
            @nvtx.annotate('module_update',('group_idx',))
            def update(self,group_idx):
                with nvtx.range('reconstruct_matmul',rank=8): pass
        Runtime().update(3)
        text = nvtx._api.calls[1][1]
        self.assertIn('source_step=6',text)
        self.assertIn('consume_step=7',text)
        self.assertIn('group_idx=3',text)
        self.assertIn('rank=8',text)
        self.assertEqual(getattr(nvtx._local,'fields',{}),{})

    def test_compression_module_executes_decorators_after_import(self):
        # Execute the actual module definition with dependency stubs. This catches
        # the NameError that AST syntax/annotation-stripping checks alone missed.
        torch = SimpleNamespace(jit=SimpleNamespace(script=lambda fn:fn),tensor=lambda value:value)
        stubs={'torch':torch,'numpy':SimpleNamespace(),'utils':SimpleNamespace(),
               'scipy':SimpleNamespace(stats=SimpleNamespace()),'bit2byte':SimpleNamespace(),
               'helix_nvtx':self.nvtx}
        with patch.dict(sys.modules,stubs):
            module=load('compression_import_test','dear/compression.py')
        self.assertIn('halfrankk',module.compressors)

    def test_metadata_is_thread_local(self):
        barrier=threading.Barrier(2)
        def worker(group):
            with self.nvtx.context(group_idx=group):
                barrier.wait()
                with self.nvtx.range('thread_probe'): pass
        threads=[threading.Thread(target=worker,args=(g,)) for g in (3,9)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        labels=[c[1] for c in self.nvtx._api.calls if c[0]=='push']
        self.assertEqual(len(labels),2)
        self.assertEqual(sum('group_idx=3' in x for x in labels),1)
        self.assertEqual(sum('group_idx=9' in x for x in labels),1)

    def test_step_exception_stops_capture_and_preserves_error(self):
        c=self.capture.Capture(args(),self.torch)
        scope={'step':2}
        exec('def training_call():\n    raise ValueError("training failed")',scope)
        wrapped=c.annotate_step(scope['training_call'])
        with self.assertRaisesRegex(ValueError,'training failed'): wrapped()
        self.assertFalse(c.active)
        self.assertEqual(self.events,['ops_enter','start','ops_exit','stop'])


def dotted(node):
    if isinstance(node,ast.Name): return node.id
    if isinstance(node,ast.Attribute): return dotted(node.value)+'.'+node.attr
    if isinstance(node,ast.Call): return dotted(node.func)
    return ''


class StripAnnotations(ast.NodeTransformer):
    def visit_Import(self,node):
        return None if any(a.name in ('helix_nvtx','helix_capture') for a in node.names) else node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str): return None
        if isinstance(node.value,ast.Call) and dotted(node.value.func) in (
            'helix_capture.add_arguments','nvtx_capture.finish'): return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(n,ast.Name) and n.id=='nvtx_capture' for n in node.targets): return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        node.decorator_list=[d for d in node.decorator_list
                             if not dotted(d).startswith(('helix_nvtx.','nvtx_capture.'))]
        return self.generic_visit(node)
    def visit_With(self,node):
        if len(node.items)==1 and dotted(node.items[0].context_expr).startswith('helix_nvtx.'):
            body=[]
            for statement in node.body:
                value=self.visit(statement)
                body.extend(value if isinstance(value,list) else [value] if value else [])
            return body
        return self.generic_visit(node)


class RuntimeSemanticsTests(unittest.TestCase):
    def test_runtime_math_and_order_match_mature_baseline(self):
        for relative in ['dear/bert_benchmark.py','dear/compression.py',
                         'dear/dopt_rsag.py','dear/tensorfusion.py']:
            with self.subTest(file=relative):
                # Some Windows launchers forward GIT_CONFIG_COUNT without its
                # empty VALUE entries. Use explicit read-only safe.directory.
                env={k:v for k,v in os.environ.items() if not k.startswith('GIT_CONFIG_')}
                original=subprocess.check_output(
                    ['git','-c','safe.directory='+str(ROOT),'show','5f02802:'+relative],
                    cwd=ROOT,env=env).decode('utf-8')
                current=(ROOT/relative).read_text(encoding='utf-8')
                before=StripAnnotations().visit(ast.parse(original))
                after=StripAnnotations().visit(ast.parse(current))
                self.assertEqual(ast.dump(before),ast.dump(after))
                ast.parse(current,feature_version=8)


class AttributionTests(unittest.TestCase):
    def setUp(self): self.analysis=load('nvtx_analysis','dear/measurement_nvtx/analyze_nvtx.py')

    def fixture(self,path):
        c=sqlite3.connect(path)
        c.executescript('''
            create table StringIds(id integer,value text);
            create table NVTX_EVENTS(start integer,end integer,text text,globalTid integer,endGlobalTid integer,textId integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,deviceId integer,streamId integer,correlationId integer,globalPid integer,demangledName integer,shortName integer,graphId integer);
        ''')
        pid=1<<24;tid=pid+1;worker=pid+2;other=2<<24
        c.executemany('insert into StringIds values (?,?)',[(1,'gemm'),(2,'ncclReduceScatter'),(3,'backward_gemm')])
        ranges=[(0,100,'helix/capture | source_begin=2 | source_end=3',tid,None,None),
                (1,90,'helix/backward | step=2',tid,None,None),
                (8,30,'helix/compress_param | source_step=2 | group_idx=1',worker,None,None),
                (10,25,'helix/project_p | source_step=2 | rank=8',worker,None,None),
                (11,24,'aten::mm',worker,None,None),
                (35,45,'helix/rs_submit | source_step=2 | group_idx=1',worker,None,None),
                (50,60,'MmBackward0',worker,None,None),
                (0,80,'helix/codec_decode | source_step=1',other+1,None,None)]
        c.executemany('insert into NVTX_EVENTS values (?,?,?,?,?,?)',ranges)
        c.executemany('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?)',
                      [(12,14,worker,1),(36,38,worker,2),(52,54,worker,3),(12,14,other+1,1)])
        # Actual GPU executions are OUTSIDE the CPU launch/NVTX scopes.
        c.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?)',
                      [(100,140,0,7,1,pid,1,1,0),(110,150,0,17,2,pid,2,2,0),
                       (140,160,0,7,3,pid,3,3,0),(100,130,0,7,1,other,1,1,0)])
        c.commit();c.close()

    def test_launch_correlation_nested_ownership_other_thread_and_pid(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.fixture(p/'trace.sqlite')
            result=self.analysis.analyze(p/'trace.sqlite',p/'out')
            self.assertEqual(result['selected_kernel_count'],3)
            d=result['devices'][0]
            self.assertEqual(d['category_union_ns']['powersgd_encode'],40)
            self.assertEqual(d['nccl_codec_overlap_ns'],30)
            self.assertEqual(d['nccl_bp_overlap_ns'],10)
            self.assertEqual(d['nccl_without_bp_or_codec_ns'],0)
            self.assertEqual(result['selected_unclassified_count'],0)

    def test_union_counts_real_overlap_once(self):
        self.assertEqual(self.analysis.duration([(0,10),(5,15)]),15)
        self.assertEqual(self.analysis.intersection([(0,10),(5,15)],[(7,12)]),5)


if __name__=='__main__': unittest.main()
