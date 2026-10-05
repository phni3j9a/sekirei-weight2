"""Run append-only pure suites in isolated processes; no actual artifacts."""
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')

class PortablePreparationTests(unittest.TestCase):
    def suite(self,name,count):
        directory=REPO/'preparations'/name
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'tests'))
        env.update({key:'1' for key in THREADS})
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover',
            '-s',str(directory),'-p','test_*.py'],cwd=REPO,env=env,
            capture_output=True,text=True,timeout=45)
        report=result.stdout+result.stderr
        self.assertEqual(result.returncode,0,report)
        self.assertEqual(re.findall(r'Ran (\d+) tests? in ',report),[str(count)],report)
        self.assertRegex(report,r'\nOK\s*$')
    def test_nas_18_pure(self):self.suite('white-view-nas-launcher-v1',18)
    def test_postformal_14_pure(self):self.suite('white-view-postformal-v1',14)
    def test_public_projection_14_pure(self):self.suite('white-view-public-projection-v1',14)
    def test_numeric_v2_23_pure(self):self.suite('white-view-numeric-audit-v2',23)
    def test_model_proof_v2_13_pure(self):self.suite('white-view-model-proof-v2',13)
    def test_evaluation_v4_43_pure(self):self.suite('white-view-evaluation-runtime-v4',43)

if __name__=='__main__':unittest.main()
