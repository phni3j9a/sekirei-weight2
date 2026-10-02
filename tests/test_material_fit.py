"""Only synthetic algebra and public fixtures; never fit real data or encode weights."""
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import unittest
REPO = Path(os.environ.get('SEKIREI_MATERIAL_FIT_REPO', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(REPO / 'scripts'))
import material_fit as fit


def load_public_helper():
    spec = importlib.util.spec_from_file_location('public_material_initializer', REPO / 'scripts/material_init.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve_linear(matrix, rhs):
    # Independent small Gaussian elimination for test reference only.
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    n = len(a)
    for i in range(n):
        pivot = max(range(i, n), key=lambda row: abs(a[row][i]))
        a[i], a[pivot] = a[pivot], a[i]
        divisor = a[i][i]
        if abs(divisor) < 1e-12:
            raise AssertionError('test reference is singular')
        a[i] = [v/divisor for v in a[i]]
        for j in range(n):
            if j != i:
                scale = a[j][i]
                a[j] = [x-scale*y for x,y in zip(a[j], a[i])]
    return [row[-1] for row in a]


class MaterialFitTests(unittest.TestCase):
    def test_prior_and_features_on_public_fixtures(self):
        import json
        helper = load_public_helper()
        fixtures = json.loads((REPO / 'tests/fixtures/material_init.json').read_text())
        self.assertEqual(fit.prices(fit.PRIOR), fit.NATIVE_VALUES)
        self.assertEqual(fit.group_totals(fit.PRIOR), (10800., 14980.))
        self.assertEqual(len(fixtures), 15)
        for fixture in fixtures:
            position = helper.parse_sfen(fixture['sfen'])
            features = fit.features(position)
            self.assertEqual(fit.dot(features, fit.PRIOR), fixture['expected_cp_stm'])
            flipped = dict(position, stm=1-position['stm'])
            self.assertEqual(fit.features(flipped), tuple(-v for v in features))
            self.assertEqual(fit.dot(features, fit.PRIOR), helper.material(position))

    def test_every_promoted_kind_maps_to_base_plus_increment(self):
        for kind in range(14):
            # Synthetic algebraic representation, not claimed to be a legal board.
            position = {'stm':0, 'pieces':[(0, kind, 0)], 'hands':[[0]*7, [0]*7]}
            row = fit.features(position)
            self.assertEqual(fit.dot(row, fit.PRIOR), fit.NATIVE_VALUES[kind])
            expected_active = 0 if kind == 7 else 2 if kind >= 8 else 1
            self.assertEqual(sum(row), expected_active)

    def test_ft_channel_arithmetic_without_encoding_models(self):
        import json
        helper = load_public_helper()
        fixtures = json.loads((REPO / 'tests/fixtures/material_init.json').read_text())
        for params in (tuple(int(v) for v in fit.PRIOR), tuple(int(v)+4 for v in fit.PRIOR)):
            fit.require_feasible(params,cap=fit.ENCODED_CAP)
            values=tuple(int(v) for v in fit.prices(params))
            for fixture in fixtures:
                position=helper.parse_sfen(fixture['sfen'])
                sums=[]
                for perspective in (0,1):
                    sums.append(sum(sum(fit.ft_material_channels(feature,values))
                                    for feature in helper.active_features(position,perspective)))
                actual=2*(sums[position['stm']]-sums[1-position['stm']])
                self.assertEqual(actual,fit.exact_predictions([fit.features(position)],params)[0])
            for kind in range(14):
                own=fit.ft_material_channels(kind*2,values)
                other=fit.ft_material_channels(kind*2+1,values)
                self.assertEqual(sum(own)*2,values[kind])
                self.assertEqual(other,(0,0))
            for bank in range(4):
                for kind in range(7):
                    for number in range(fit.HAND_MAX[kind]):
                        feature=fit.BOARD_INPUT+bank*fit.HAND_THRESHOLDS+fit.HAND_OFFSETS[kind]+number
                        expected=values[kind] if bank//2==bank%2 else 0
                        self.assertEqual(sum(fit.ft_material_channels(feature,values))*2,expected)

    def test_weighted_projection_kkt_and_idempotency(self):
        rng = random.Random(73)
        for _ in range(80):
            vector = [rng.uniform(-1000, 9000) for _ in range(13)]
            projected = fit.project(vector)
            fit.require_feasible(projected)
            for a,b in zip(projected, fit.project(projected)):
                self.assertAlmostEqual(a,b,places=8)
            for group in fit.GROUPS:
                total = math.fsum(projected[i]*weight for i,weight in group)
                active = [(vector[i]-projected[i])/weight for i,weight in group if projected[i]>1e-8]
                if total < fit.FIT_CAP-1e-6:
                    self.assertTrue(all(abs(v)<1e-8 for v in active))
                elif active:
                    self.assertLess(max(active)-min(active), 1e-7)
                    multiplier = active[0]
                    self.assertGreaterEqual(multiplier, -1e-8)
                    for i,weight in group:
                        if projected[i] == 0:
                            self.assertLessEqual(vector[i]-multiplier*weight,1e-7)

    def test_diagonal_fit_matches_closed_form_including_binding_caps(self):
        rows = [tuple(int(i==j) for i in range(13)) for j in range(13)]
        for target in (list(fit.PRIOR), [20000.]*13, [-20000.]*13):
            ridge = 0.1
            result = fit.fit(rows, target, ridge)
            expected = fit.project([(target[i]/13+ridge*fit.PRIOR[i])/(1/13+ridge) for i in range(13)])
            self.assertLess(math.dist(result['parameters'],expected),1e-7)
            self.assertEqual(result['status'],'converged')
            self.assertLessEqual(result['distance_bound_cp'],result['distance_tolerance_cp'])

    def test_correlated_fit_matches_independent_linear_system(self):
        rng = random.Random(149)
        rows = [tuple(rng.randrange(-1,2) for _ in range(13)) for _ in range(100)]
        wanted = [v+20 for v in fit.PRIOR]
        targets = [fit.dot(row,wanted) for row in rows]
        stats = fit.sufficient_statistics(rows,targets)
        for ridge in (0.01,0.1,10):
            result = fit.fit_statistics(stats,ridge)
            a = [[stats['gram'][i][j]+(ridge if i==j else 0) for j in range(13)] for i in range(13)]
            b = [stats['rhs'][i]+ridge*fit.PRIOR[i] for i in range(13)]
            expected = solve_linear(a,b)
            fit.require_feasible(expected)
            self.assertLessEqual(math.dist(expected,result['parameters']),0.00011)
            actual_mse = math.fsum((fit.dot(row,result['parameters'])-y)**2 for row,y in zip(rows,targets))/len(rows)
            self.assertAlmostEqual(actual_mse,result['train_mse_cp2'],places=6)

    def test_correlated_fit_with_active_budget_matches_reduced_exact_solution(self):
        rows = [[1 if i == 0 else 0 for i in range(13)],
                [1 if i in (0, 7) else 0 for i in range(13)]]
        ridge = 0.01
        result = fit.fit(rows, [400, 2000], ridge)
        total = fit.FIT_CAP / 18
        expected_pawn = (400 + 2*ridge*(total-400)) / (1+4*ridge)
        self.assertAlmostEqual(result['parameters'][0], expected_pawn, places=4)
        self.assertAlmostEqual(result['parameters'][7], total-expected_pawn, places=4)
        self.assertAlmostEqual(fit.group_totals(result['parameters'])[0], fit.FIT_CAP, places=7)
        for i in range(13):
            if i not in (0, 7):
                self.assertAlmostEqual(result['parameters'][i], fit.PRIOR[i], places=7)

    def test_rank_deficient_data_returns_prior_and_nonconvergence_fails(self):
        result = fit.fit([[0]*13]*2,[0,0],0.01)
        self.assertEqual(result['parameters'],list(fit.PRIOR))
        rows = [[1 if i in (0,7) else 0 for i in range(13)], [1 if i==0 else 0 for i in range(13)]]
        with self.assertRaisesRegex(fit.FitError,'did not converge'):
            fit.fit(rows,[800,100],0.01,max_iterations=1)
        with self.assertRaises(fit.FitError):
            fit.fit(rows,[float('nan'),0],0.1)
        with self.assertRaises(fit.FitError):
            fit.fit(rows,[0,0],0.)

    def test_rounding_margin_nonnegative_promotion_and_integer_cp(self):
        rng = random.Random(61)
        for _ in range(100):
            raw = fit.project([rng.uniform(-1,9000) for _ in range(13)])
            rounded = fit.round_parameters({'status':'converged','parameters':raw,
                                             'distance_bound_cp':0.,'distance_tolerance_cp':0.0001})
            self.assertTrue(all(type(v) is int and v>=0 and v%2==0 for v in rounded))
            for before,after in zip(fit.group_totals(raw),fit.group_totals(rounded)):
                self.assertLessEqual(after-before,36.0000001)
                self.assertLessEqual(after,fit.ENCODED_CAP)
            values=fit.prices(rounded)
            for kind in fit.PROMOTION_PARAMETER:
                self.assertGreaterEqual(values[kind],values[fit.BASE[kind]])
        for value,wanted in ((1.,0),(3.,4),(5.,4)):
            params=list(fit.PRIOR); params[0]=value
            rounded=fit.round_parameters({'status':'converged','parameters':params,
                                          'distance_bound_cp':0.,'distance_tolerance_cp':0.0001})
            self.assertEqual(rounded[0],wanted)
        with self.assertRaises(fit.FitError):
            fit.round_parameters({'status':'not_converged'})

    def test_core_validation_and_tie_selection_refuse_unverified(self):
        parameters=tuple(int(v) for v in fit.PRIOR)
        features=[(1,)+(0,)*12,(-1,)+(0,)*12]
        rows=[]
        for i,expected in enumerate((100,-100)):
            rows.append({'index':i,'native_core_cp':expected,'nearest_core_cp':expected,
                         'native_quantized_float_cp':float(expected),'nearest_quantized_float_cp':float(expected),
                         'material_cp':expected})
        self.assertEqual(fit.validate_probe_rows(rows,features,parameters)['count'],2)
        rows[1]['nearest_core_cp']+=1
        with self.assertRaises(fit.FitError): fit.validate_probe_rows(rows,features,parameters)
        items=[{'ridge':ridge,'split':'holdout','core_verified':False,
                'holdout_binding_sha256':'0'*64,'holdout':{'count':2,'absolute_error_sum_cp':100}} for ridge in fit.LAMBDAS]
        with self.assertRaises(fit.FitError): fit.rank_static_candidates(items)
        for item in items: item['core_verified']=True
        self.assertEqual(fit.rank_static_candidates(items)[0]['ridge'],100.)
        items[1]['holdout']['absolute_error_sum_cp']=98
        self.assertEqual(fit.rank_static_candidates(items)[0]['ridge'],0.1)


if __name__=='__main__':
    unittest.main()
