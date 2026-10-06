"""Small public source snapshot checks; no compiler, engine, models or data."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent

class PublicSnapshotTests(unittest.TestCase):
    def test_file_set_sizes_and_hashes(self):
        meta=json.loads((ROOT/'public-source-manifest.json').read_text())
        expected=set(meta['files'])|{'public-source-manifest.json'}
        actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
        self.assertEqual(actual,expected)
        for name,ref in meta['files'].items():
            self.assertEqual(set(ref),{'bytes','sha256'})
            self.assertIs(type(ref['bytes']),int)
            path=ROOT/name;self.assertFalse(path.is_symlink())
            raw=path.read_bytes();self.assertEqual(len(raw),ref['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
    def test_four_postimages_and_weekly_profile_bound(self):
        meta=json.loads((ROOT/'public-source-manifest.json').read_text())
        self.assertEqual(len(meta['diagnostic_postimages']),4)
        for name,ref in meta['diagnostic_postimages'].items():
            self.assertEqual(meta['files']['source/'+name],ref)
        profile='source/crates/sekirei-train/src/weekly_nonlinear_profile.rs'
        self.assertEqual(meta['files'][profile],meta['baseline_alignment']['profile'])
        self.assertTrue(meta['Rust_source_v5_exact_v2'])
        self.assertTrue(meta['prototype_closed_before_io'])
    def test_activation_is_only_queued_boolean_change(self):
        source=(ROOT/'source/crates/sekirei-train/src/paired_nonlinear_raw_master_diag.rs').read_text()
        self.assertEqual(source.count('const PROTOTYPE_ONLY:bool=true;'),1)
        patch=(ROOT/'queued-activation.patch').read_text()
        removed=[line[1:] for line in patch.splitlines() if line.startswith('-') and not line.startswith('---')]
        added=[line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
        self.assertEqual(removed,['const PROTOTYPE_ONLY:bool=true;'])
        self.assertEqual(added,['const PROTOTYPE_ONLY:bool=false;'])

if __name__=='__main__':
    unittest.main()
