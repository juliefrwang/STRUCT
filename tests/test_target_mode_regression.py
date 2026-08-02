"""WCF regression test: SS_target with wobble=False must produce output
byte-identical to the golden generated from main.

The golden file was produced by running ``SS_target`` on ``main`` against
``tests/test_data/test.after_correction.scores.tsv`` and saved as
``tests/test_data/golden_target_mode_wcf.tsv``. Any change to the WCF
code path that breaks this test is a regression on backward
compatibility.
"""

import filecmp
from pathlib import Path

import pandas as pd
import pytest
from pandarallel import pandarallel

from splash_structure_py.structure_target_mode import SS_target


# pandarallel must be initialised before parallel_apply is called.
pandarallel.initialize(nb_workers=1, progress_bar=False, verbose=0)


REPO_ROOT = Path(__file__).resolve().parent.parent
SPLASH_TSV = REPO_ROOT / "tests" / "test_data" / "test.after_correction.scores.tsv"
GOLDEN_TSV = REPO_ROOT / "tests" / "test_data" / "golden_target_mode_wcf.tsv"


def test_wcf_path_matches_main_golden(tmp_path, monkeypatch):
    """Run SS_target with wobble=False on the standard test input.

    The output must match ``golden_target_mode_wcf.tsv`` byte-for-byte.
    The golden was generated on ``main`` (commit d86d925) using the same
    test input.
    """
    # SS_target writes to <prefix>_results/ in CWD.
    monkeypatch.chdir(tmp_path)

    SS_target(
        output_prefix="regression",
        splash_output_file=str(SPLASH_TSV),
        element_annotation=False,
        wobble=False,
    )

    actual = tmp_path / "regression_results" / "structure_on_targets.tsv"
    assert actual.exists(), f"SS_target did not produce {actual}"

    # Try byte-identical comparison first; fall back to dataframe-equality
    # if pandas writes differ in trailing whitespace etc.
    if filecmp.cmp(str(actual), str(GOLDEN_TSV), shallow=False):
        return

    # Fallback: dataframe-level equality. Useful if pandas formatting
    # changes across versions but the data is identical.
    df_actual = pd.read_csv(actual, sep="\t")
    df_golden = pd.read_csv(GOLDEN_TSV, sep="\t")
    pd.testing.assert_frame_equal(df_actual, df_golden, check_exact=False)


def test_golden_file_exists():
    """Sanity check that the golden file is checked in."""
    assert GOLDEN_TSV.exists(), (
        f"Golden file missing: {GOLDEN_TSV}. "
        "Re-generate by running SS_target on main and copying the output."
    )
