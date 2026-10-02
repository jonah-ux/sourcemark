import contextlib
import io
import unittest

from sourcemark import __version__
from sourcemark.cli import main


class CliSmokeTest(unittest.TestCase):
    def test_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)

    def test_version(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(__version__, out.getvalue())


if __name__ == "__main__":
    unittest.main()
