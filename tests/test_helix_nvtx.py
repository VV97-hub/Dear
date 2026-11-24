"""CPU-only checks for annotations; these do not validate GPU execution."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'dear' / 'helix_nvtx.py'


def load(enabled):
    with patch.dict(os.environ, {'HELIX_NVTX': '1' if enabled else '0'}):
        spec = importlib.util.spec_from_file_location('helix_nvtx_test', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeNVTX:
    def __init__(self):
        self.calls = []

    def range_push(self, text):
        self.calls.append(('push', text))

    def range_pop(self):
        self.calls.append(('pop',))


class AnnotationTests(unittest.TestCase):
    def test_disabled_decorator_returns_original(self):
        nvtx = load(False)
        def fn(value):
            return value
        self.assertIs(nvtx.annotate('test')(fn), fn)
        with nvtx.range('disabled'):
            pass
        self.assertIsNone(nvtx._api)

    def test_nested_ranges_and_exception_are_balanced(self):
        nvtx = load(True)
        api = FakeNVTX()
        nvtx._api = api
        @nvtx.annotate('outer', ('group_idx',))
        def fn(group_idx):
            with nvtx.range('inner', rank=16):
                raise ValueError('sentinel')
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            fn(6)
        self.assertEqual([c[0] for c in api.calls], ['push','push','pop','pop'])
        self.assertIn('group_idx=6', api.calls[0][1])

    def test_tensor_like_metadata_is_never_formatted(self):
        nvtx = load(True)
        nvtx._api = FakeNVTX()
        class TensorLike:
            def __str__(self):
                raise AssertionError('Must not format tensor')
        with nvtx.range('safe', value=TensorLike()):
            pass
        self.assertNotIn('value=', nvtx._api.calls[0][1])

    def test_result_and_runtime_metadata(self):
        nvtx = load(True)
        nvtx._api = FakeNVTX()
        class Runtime:
            _num_steps = 7
            _active_compression_factor = 'p'
            @nvtx.annotate('update', ('group_idx',))
            def fn(self, group_idx, value):
                return value + 1
        self.assertEqual(Runtime().fn(group_idx=3, value=9), 10)
        self.assertIn('runtime_step=7', nvtx._api.calls[0][1])
        self.assertIn('active_factor=p', nvtx._api.calls[0][1])


if __name__ == '__main__':
    unittest.main()
