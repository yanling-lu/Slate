"""Parquet reading must work, or fail with a plain message, on old pyarrow."""
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from portfolio_agent import parquet_io
from portfolio_agent.parquet_io import read_parquet

OLD_ERROR = OSError("Repetition level histogram size mismatch")


def sample(d):
    path = Path(d) / "x.parquet"
    pd.DataFrame({"permno": [1, 2, 3, 4], "filing_date": [date(2024, 5, 31), date(2024, 6, 3), date(2024, 6, 30), date(2024, 7, 1)],
                  "v": [1.0, 2.0, 3.0, 4.0]}).to_parquet(path)
    return path


class ParquetIoTest(unittest.TestCase):
    def test_plain_read_and_exact_inclusive_date_window(self):
        with tempfile.TemporaryDirectory() as d:
            path = sample(d)
            self.assertEqual(len(read_parquet(path)), 4)
            self.assertEqual(list(read_parquet(path, columns=["permno"]).columns), ["permno"])
            win = read_parquet(path, columns=["permno", "v"], between=("filing_date", "2024-06-01", "2024-06-30"))
            self.assertEqual(win["permno"].tolist(), [2, 3])                    # both ends inclusive, others excluded
            self.assertNotIn("filing_date", win.columns)                        # filter column was not requested

    def test_unrelated_errors_are_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            path = sample(d)
            with mock.patch.object(pd, "read_parquet", side_effect=OSError("disk on fire")):
                with self.assertRaises(OSError):
                    read_parquet(path)

    @unittest.skipUnless(parquet_io._have("fastparquet"), "fastparquet not installed")
    def test_old_pyarrow_error_falls_back_to_fastparquet(self):
        with tempfile.TemporaryDirectory() as d:
            path = sample(d)
            real = pd.read_parquet

            def fake(p, columns=None, engine=None, **kw):
                if engine == "pyarrow":
                    raise OLD_ERROR
                return real(p, columns=columns, engine=engine, **kw)
            with mock.patch.object(pd, "read_parquet", side_effect=fake):
                self.assertEqual(read_parquet(path)["v"].tolist(), [1.0, 2.0, 3.0, 4.0])

    @unittest.skipUnless(parquet_io._have("fastparquet"), "fastparquet not installed")
    def test_old_pyarrow_date_filter_goes_straight_to_fastparquet_and_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            path = sample(d)
            engines = []
            real = pd.read_parquet

            def spy(p, columns=None, engine=None, **kw):
                engines.append(engine)
                return real(p, columns=columns, engine=engine, **kw)
            with mock.patch.object(parquet_io, "_pyarrow_major", return_value=19), mock.patch.object(pd, "read_parquet", side_effect=spy):
                out = read_parquet(path, between=("filing_date", "2024-06-01", "2024-06-30"))
            self.assertEqual(engines, ["fastparquet"])                           # never touches the pyarrow filter path
            self.assertEqual(out["permno"].tolist(), [2, 3])

    def test_old_pyarrow_without_fastparquet_explains_the_fix_and_still_reads_good_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = sample(d)
            window = ("filing_date", "2024-06-01", "2024-06-30")
            with mock.patch.object(parquet_io, "_have", return_value=False), mock.patch.object(parquet_io, "_pyarrow_major", return_value=19):
                ok = read_parquet(path, between=window)                           # a readable file still works, filtered in pandas
                self.assertEqual(ok["permno"].tolist(), [2, 3])
                for kwargs in ({}, {"between": window}):
                    with mock.patch.object(pd, "read_parquet", side_effect=OLD_ERROR):
                        with self.assertRaises(RuntimeError) as ctx:
                            read_parquet(path, **kwargs)
                    msg = str(ctx.exception)
                    self.assertIn("pip install -U 'pyarrow>=25'", msg)
                    self.assertIn("fastparquet", msg)
                    self.assertIn("Repetition level histogram", msg)


class MissingInputMessageTest(unittest.TestCase):
    def run_main(self, *args):
        from portfolio_agent import run
        with self.assertRaises(SystemExit) as ctx:
            run.main(list(args))
        return str(ctx.exception)

    def test_missing_signals_file_says_how_to_create_it(self):
        with tempfile.TemporaryDirectory() as d:
            panel = Path(d) / "panel.parquet"
            panel.write_text("x")
            msg = self.run_main("--panel", str(panel), "--signals", "out_composite/signals.csv")
            self.assertIn("Signals file not found: out_composite/signals.csv", msg)
            self.assertIn("python -m base_signal.composites", msg)
            self.assertIn("generated, not stored in git", msg)
            msg2 = self.run_main("--panel", str(panel), "--signals", "out_8k/signals_8k.csv")
            self.assertIn("agents.eightk_signal", msg2)

    def test_missing_panel_says_where_the_data_goes(self):
        msg = self.run_main("--panel", "Provided Documents/nope.parquet", "--signals", "x.parquet")
        self.assertIn("Data file not found", msg)
        self.assertIn("Provided Documents", msg)


if __name__ == "__main__":
    unittest.main()
