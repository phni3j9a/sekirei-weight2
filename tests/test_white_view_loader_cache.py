"""Tiny public source copies/mock workers; no actual runtime artifacts or runs."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'preparations/white-view-fit-operational-v2'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(1,str(HERE))
import fit_white_view_paired_linear as driver
import white_view_fit_operational_common as common

CONTRACT = ROOT/'scripts/white_view_build_contract.py'
WORKER = b'''from pathlib import Path
import white_view_build_contract as c
assert Path(c.__file__).resolve() == Path(__file__).resolve().with_name('white_view_build_contract.py')
def verify_runtime(*args): return {'manifest':{},'identity':{}}
'''


def fullref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(sys.modules)
        self.tmp = tempfile.TemporaryDirectory(dir=HERE, prefix='synthetic-loader-')
        self.directory = Path(self.tmp.name)
        self.adjacent = self.directory / 'white_view_build_contract.py'
        self.adjacent.write_bytes(CONTRACT.read_bytes())
        self.worker = self.directory / 'synthetic-build-worker.py'
        self.worker.write_bytes(WORKER)
        self.source_map = {str(self.adjacent): {k: fullref(self.adjacent)[k] for k in ('bytes', 'sha256')}}
        self.public = types.ModuleType('white_view_build_contract')
        self.public.__file__ = '/synthetic/public-R/white_view_build_contract.py'
        sys.modules['white_view_build_contract'] = self.public

    def tearDown(self):
        for name in set(sys.modules) - set(self.saved):
            if name.startswith(('_white_view_actual_build_', '_white_operational_')):
                del sys.modules[name]
        if 'white_view_build_contract' in self.saved:
            sys.modules['white_view_build_contract'] = self.saved['white_view_build_contract']
        else:
            sys.modules.pop('white_view_build_contract', None)
        self.tmp.cleanup()

    def load(self, which):
        if which == 'driver':
            return driver._load_runtime_validator(fullref(self.worker), self.source_map)
        reader = common.Reader()
        reader.pin(self.adjacent, self.source_map[str(self.adjacent)])
        return common.load_build_validator(reader, fullref(self.worker))

    def test_public_cache_restored_and_real_adjacent_retained(self):
        for which in ('driver', 'ops'):
            obj = self.load(which)
            self.assertIs(sys.modules['white_view_build_contract'], self.public)
            self.assertEqual(Path(obj.c.__file__), self.adjacent)
            self.assertIsNot(obj.c, self.public)

    def test_missing_public_cache_stays_missing(self):
        sys.modules.pop('white_view_build_contract')
        for which in ('driver', 'ops'):
            self.load(which)
            self.assertNotIn('white_view_build_contract', sys.modules)

    def test_exec_failure_restores_public_cache(self):
        self.worker.write_bytes(WORKER + b'raise RuntimeError("synthetic worker exec failed")\n')
        for which in ('driver', 'ops'):
            with self.assertRaisesRegex(RuntimeError, 'synthetic worker exec failed'):
                self.load(which)
            self.assertIs(sys.modules['white_view_build_contract'], self.public)
        self.assertNotIn('_white_view_actual_build_validator', sys.modules)
        self.assertNotIn('_white_operational_build_validator', sys.modules)

    def test_missing_or_bad_adjacent_pin_rejected(self):
        with self.assertRaises(ValueError):
            driver._load_runtime_validator(fullref(self.worker), {})
        with self.assertRaisesRegex(ValueError, 'externally pinned'):
            common.load_build_validator(common.Reader(), fullref(self.worker))
        self.source_map[str(self.adjacent)]['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'fixed adjacent'):
            driver._load_runtime_validator(fullref(self.worker), self.source_map)

    def test_same_bytes_public_object_never_substitutes_real_adjacent(self):
        self.public.__file__ = str(CONTRACT)
        for which in ('driver', 'ops'):
            obj = self.load(which)
            self.assertIsNot(obj.c, self.public)
            self.assertEqual(Path(obj.c.__file__), self.adjacent)
            self.assertEqual(self.adjacent.read_bytes(), CONTRACT.read_bytes())
            self.assertIs(sys.modules['white_view_build_contract'], self.public)

    def test_adjacent_mutation_rejected_before_worker_exec(self):
        self.adjacent.write_bytes(self.adjacent.read_bytes() + b'\n# synthetic mutation\n')
        for which in ('driver', 'ops'):
            with self.assertRaises(ValueError): self.load(which)
            self.assertIs(sys.modules['white_view_build_contract'], self.public)

    def test_ops_reuses_only_same_frozen_worker_and_adjacent(self):
        obj = self.load('ops')
        self.assertIs(self.load('ops'), obj)
        other = self.directory / 'different-worker.py'
        other.write_bytes(WORKER)
        reader = common.Reader(); reader.pin(self.adjacent, self.source_map[str(self.adjacent)])
        with self.assertRaisesRegex(ValueError, 'source import binding changed'):
            common.load_build_validator(reader, fullref(other))
        self.assertIs(sys.modules['white_view_build_contract'], self.public)



if __name__ == '__main__': unittest.main()
