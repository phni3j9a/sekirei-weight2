import ast
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import struct
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import paired_linear_solver as old
import paired_linear_solver_verified_design as solver
import verified_design_gram as gram

OLD_SOURCE=Path(old.__file__)
OLD_SHA='ba9c86b37e3ed442db930332f16f3390ff5cd84a98431fcceea84cc4dd9b066b'


def hashes():
    return {'expected_producer_source_sha256':hashlib.sha256(Path(gram.__file__).read_bytes()).hexdigest(),
            'expected_solver_source_sha256':hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest()}


def produced(Z=None,d=None):
    design=gram.verified_design(Z or [[1,2],[-3,1],[0,4]],d or [5,-2,7])
    artifact=gram.produce_gram(design,expected_producer_source_sha256=hashes()['expected_producer_source_sha256'],
                                deadline_check=lambda:None)
    return design,artifact


def consume(artifact,**kwargs):
    args={'expected_design_sha256':artifact.design.design_sha256,'expected_gram_sha256':artifact.gram_sha256,
          'expected_producer_source_sha256':hashes()['expected_producer_source_sha256']}
    args.update(kwargs)
    return gram.consume_gram(artifact,**args)


try:
    import numpy as np
    PINNED_NUMPY = np.__version__ == '1.26.4'
except ImportError:
    PINNED_NUMPY = False


@unittest.skipUnless(PINNED_NUMPY, 'local pinned NumPy1.26.4 numerical fixtures')
class FullGramTests(unittest.TestCase):
    def test_old_source_bytes_and_generic_functions_unchanged(self):
        self.assertEqual(hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest(),OLD_SHA)
        a=ast.parse(OLD_SOURCE.read_text());b=ast.parse(Path(solver.__file__).read_text())
        before={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        after={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,node in before.items():
            if name!='solve_synthetic':self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(after[name],include_attributes=False),name)
    def test_same_FISTA_AST_after_only_origin_deadline_checks(self):
        a=ast.parse(OLD_SOURCE.read_text());b=ast.parse(Path(solver.__file__).read_text())
        body=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='solve_synthetic').body
        body=[n for n in body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='problem' for t in n.targets))]
        other=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='_solve_prepared').body
        class StripNewChecks(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in {'_check_deadline','_validate_problem_origin'}:return None
                return self.generic_visit(node)
            def visit_Call(self,node):return self.generic_visit(node)
            def visit_ExprGuard(self,node):return node
        other=[StripNewChecks().visit(n) for n in other]
        other=[n for n in other if n is not None]
        # Sole added _require is the internal N=problem.sample_count binding.
        other=[n for n in other if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Name) and n.value.func.id=='_require'
                 and any(isinstance(arg,ast.Constant) and arg.value=='internal sample count differs' for arg in n.value.args))]
        self.assertEqual([ast.dump(n,include_attributes=False) for n in body],
                         [ast.dump(n,include_attributes=False) for n in other])
    def test_Gram_transpose_rhs_and_cert_match_generic(self):
        design,artifact=produced();problem=consume(artifact)
        self.assertEqual(problem.gram,((10,-1),(-1,21)));self.assertEqual(problem.hz,(11,36))
        reference=old.prepare_problem(problem.gram,problem.hz,design.samples)
        for field in ('gram','K','hz','sample_count','dimension','lipschitz_exact','lipschitz_float'):
            self.assertEqual(getattr(problem,field),getattr(reference,field))
        self.assertEqual(old._certificate(problem,(0.25,-0.125),old.RHO),
                         old.exact_certificate(problem.gram,problem.hz,design.samples,(0.25,-0.125)))
    def test_singular_zero_and_entry_extremes(self):
        for Z,d in [([[1,1],[2,2]],[3,4]),([[0,0],[0,0]],[3,4]),
                    ([[40,-40],[-40,40]],[55779,-55779])]:
            design,a=produced(Z,d);p=consume(a)
            self.assertEqual(p.gram,old.prepare_problem(p.gram,p.hz,design.samples).gram)
    def test_wrong_domains_counts_and_mutable_bytes(self):
        for Z,d in [([[True]],[1]),([[1.0]],[1]),([[41]],[1]),([[1]],[False]),
                    ([[1]],[float('nan')]),([[1]],[55780]),([[1],[2,3]],[1,2]),([[1]],[])]:
            with self.subTest(Z=Z,d=d),self.assertRaises(gram.InvalidOrigin):gram.verified_design(Z,d)
        for zb,db,n,p in [(bytearray(b'\0'),struct.pack('<i',0),1,1),(b'\0',struct.pack('<i',0),True,1),
                          (b'\x29',struct.pack('<i',0),1,1),(b'\0',struct.pack('<i',55780),1,1)]:
            with self.assertRaises(gram.InvalidOrigin):gram.verified_design_bytes(zb,db,n,p)
    def test_bytes_and_numpy_strict_copy_bridge(self):
        np=gram._numpy();Z=np.array([[1,2],[-3,1],[0,4]],dtype=np.int8);d=np.array([5,-2,7],dtype='<i4')
        design=gram.verified_numpy_design(Z,d);Z[:]=40;d[:]=55779
        a=gram.produce_gram(design,expected_producer_source_sha256=hashes()['expected_producer_source_sha256'],deadline_check=lambda:None)
        self.assertEqual(consume(a).gram,((10,-1),(-1,21)))
        for badZ,badd in [(Z.astype(np.float64),d),(Z,d.astype(np.float64)),(Z[:,::-1],d),(Z,d[::-1])]:
            with self.assertRaises(gram.InvalidOrigin):gram.verified_numpy_design(badZ,badd)
    def test_origin_copies_and_changes_rejected(self):
        design,a=produced()
        for fake in [copy.copy(a),copy.deepcopy(a),replace(a)]:
            with self.assertRaises(gram.InvalidOrigin):consume(fake)
        with self.assertRaises(gram.InvalidOrigin):gram.produce_gram(replace(design),expected_producer_source_sha256=hashes()['expected_producer_source_sha256'],deadline_check=lambda:None)
        object.__setattr__(a,'gram',((1,2),(2,1)))
        digest=gram._gram_hash(design,a.gram,a.hz,a.producer_source_sha256,a.backend_metadata)
        object.__setattr__(a,'gram_sha256',digest)
        with self.assertRaises(gram.InvalidOrigin):consume(a,expected_gram_sha256=digest)
    def test_design_and_source_hash_changes_rejected(self):
        design,a=produced()
        for kwargs in [{'expected_design_sha256':'0'*64},{'expected_gram_sha256':'0'*64},
                       {'expected_producer_source_sha256':'0'*64}]:
            with self.assertRaises(gram.InvalidOrigin):consume(a,**kwargs)
        object.__setattr__(design,'z_bytes',b'\0'*6)
        with self.assertRaises(gram.InvalidOrigin):consume(a)
    def test_SFEN_driver_bytes_entry_no_duplicate_decode(self):
        zb=struct.pack('bbb',16,-16,0);db=struct.pack('<iii',2,-2,0)
        result=solver.solve_verified_design_bytes(zb,db,3,1,**hashes())
        self.assertTrue(result['certificate_f64']['solver_threshold_met'])
        self.assertFalse(result['verified_design_gram']['actual_dataset_provenance_claimed'])
    def test_same_generic_and_dedicated_fit_results(self):
        for Z,d in [([[16]],[2]),([[16]],[100]),([[1,2],[-3,1],[0,4]],[5,-2,7]),
                    ([[0,0],[0,0]],[5,-2])]:
            design,a=produced(Z,d);reference=solver.solve_synthetic(a.gram,a.hz,design.samples)
            result=solver.solve_verified_design(Z,d,**hashes())
            for key,value in reference.items():self.assertEqual(result[key],value,key)
            self.assertTrue(result['certificate_f64']['solver_threshold_met'])
            self.assertLessEqual(result['certificate_f32_saved_radius']['objective_delta_vs_zero'],0)
    def test_generic_indefinite_remains_rejected_and_no_free_problem(self):
        for fn in (old.prepare_problem,solver.prepare_problem):
            with self.assertRaises(old.InvalidProblem if fn is old.prepare_problem else solver.InvalidProblem):fn(((1,2),(2,1)),(0,0),1)
        p=solver.prepare_problem(((1,),),(0,),1)
        with self.assertRaises(solver.InvalidProblem):solver._solve_prepared(p,1,max_iterations=1)
    def test_wrong_external_solver_binding_and_deadline(self):
        h=hashes();h['expected_solver_source_sha256']='0'*64
        with self.assertRaises(solver.InvalidProblem):solver.solve_verified_design([[1]],[1],**h)
        for value in [True,float('inf'),0.0,1200.1]:
            with self.assertRaises(solver.InvalidProblem):solver._Deadline(value)
        deadline=solver._Deadline(1.0);deadline.started-=2.0
        with self.assertRaises(solver.DeadlineExceeded):deadline.check()
    def test_actual_entry_stays_blocked_and_no_fallback(self):
        with self.assertRaises(solver.PrototypeOnly):solver.runtime_entry()
        with self.assertRaises(RuntimeError):gram.runtime_entry()
        with self.assertRaises(solver.ConvergenceFailure):solver.solve_verified_design([[1,2],[-3,1],[0,4]],[5,-2,7],max_iterations=1,**hashes())

if __name__=='__main__':unittest.main()
