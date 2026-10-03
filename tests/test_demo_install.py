import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv


class InstalledDemoTest(unittest.TestCase):
    def test_installed_mode_refuses_missing_package_even_beside_a_checkout(self):
        demo = Path(__file__).resolve().parents[1] / "demos" / "demo.py"
        with tempfile.TemporaryDirectory(prefix="sourcemark-demo-install-") as raw:
            directory = Path(raw)
            environment = directory / "empty-venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["GIT_CONFIG_GLOBAL"] = os.devnull
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            result = subprocess.run(
                [str(python), str(demo), "--installed"], cwd=directory,
                env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("ModuleNotFoundError: No module named 'sourcemark'", result.stderr)
