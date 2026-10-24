import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trifusesite.cli.repeat import build_plan


ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_validation_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-B', '-X', 'utf8', str(ROOT / 'main.py'), 'validate-datasets'],
                cwd=directory, check=True, capture_output=True, text=True, encoding='utf-8')
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(len(report['partitions']), 14)
        self.assertEqual(report['checked_files'], 2873)

    def test_preparation_check_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-B', '-X', 'utf8', str(ROOT / 'main.py'),
                 'prepare-datasets', '--tasks', 'smpi', '--splits', 'test', '--check-only'],
                cwd=directory, check=True, capture_output=True, text=True, encoding='utf-8')
        report = json.loads(result.stdout)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['proteins'], 348)
        self.assertEqual(report[0]['task'], 'smpi')

    def test_repeat_subprocesses_use_working_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = build_plan(ROOT / 'config/ppi.yaml', Path(directory) / 'repeat', seeds=[2025])
            self.assertEqual(len(plan['commands']), 4)
            for command in plan['commands']:
                result = subprocess.run(command['argv'] + ['--help'], cwd=directory,
                                        check=True, capture_output=True, text=True, encoding='utf-8')
                self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
