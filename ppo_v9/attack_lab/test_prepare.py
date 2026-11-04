import copy
import unittest
from unittest.mock import patch
import prepare


class ScaffoldChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [{'date': '2020-01-01', 'x': '1.0', 'flag': '0'},
                     {'date': '2020-01-02', 'x': '2.0', 'flag': '1'}]
        self.config = {'columns': ['x'], 'representation': 'standardized_observation_features',
                       'spec_reference': 'synthetic test', 'sigma': 0.1, 'noise_seed': 123}

    def test_identity_and_source_preservation(self):
        before = copy.deepcopy(self.rows)
        output, stats = prepare.transform(self.rows, self.config, 'identity')
        self.assertEqual(output, before)
        self.assertEqual(self.rows, before)
        self.assertEqual(stats['x']['max_abs_delta'], 0)

    def test_bad_shape_is_rejected(self):
        with patch.object(prepare, 'perturb_values', return_value=[]):
            with self.assertRaises(ValueError):
                prepare.transform(self.rows, self.config, 'attack')

    def test_nonfinite_is_rejected(self):
        with patch.object(prepare, 'perturb_values', return_value=[float('nan')]):
            with self.assertRaises(ValueError):
                prepare.transform(self.rows, self.config, 'attack')

    def test_zero_sigma_must_be_identity(self):
        self.config['sigma'] = 0
        with patch.object(prepare, 'perturb_values', return_value=[99]):
            with self.assertRaises(AssertionError):
                prepare.transform(self.rows, self.config, 'attack')

    def test_same_seed_reproduces_noise(self):
        first, _ = prepare.transform(self.rows, self.config, 'attack')
        second, _ = prepare.transform(self.rows, self.config, 'attack')
        self.assertEqual(first, second)
        self.assertNotEqual(first, self.rows)


if __name__ == '__main__':
    unittest.main()
