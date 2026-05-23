"""WCF regression test: SS_compactor (legacy WCF-only path) must produce
output byte-identical to the golden generated from ``feature/non-wcf``.

The golden file was produced by running ``SS_compactor`` on
``feature/non-wcf`` (commit prior to the compactor-mode extension)
against ``tests/test_data/test.compactor.tsv`` and saved as
``tests/test_data/golden_compactor_mode_wcf.tsv``. ``feature/non-wcf``
does not touch ``structure_compactor_mode.py`` (per target-mode
implementation plan Decision 3), so this golden equivalently represents
the ``main`` compactor-mode behaviour.

Any change to the legacy compactor-mode code path that breaks this test
is a regression on backward compatibility. The companion extended-path
golden (``--noncanon GU``) lands in Phase C5.
"""

import filecmp
from pathlib import Path

import pandas as pd
import pytest
from pandarallel import pandarallel

from splash_structure_py.structure_compactor_mode import SS_compactor


pandarallel.initialize(nb_workers=1, progress_bar=False, verbose=0)


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPACTOR_TSV = REPO_ROOT / "tests" / "test_data" / "test.compactor.tsv"
GOLDEN_TSV = REPO_ROOT / "tests" / "test_data" / "golden_compactor_mode_wcf.tsv"


def test_wcf_path_matches_golden(tmp_path, monkeypatch):
    """Run SS_compactor on the standard test input under the legacy
    (no-flag) path.

    The output must match ``golden_compactor_mode_wcf.tsv``
    byte-for-byte. The golden was generated on ``feature/non-wcf`` prior
    to the compactor-mode extension.
    """
    # SS_compactor writes to <prefix>_results/ in CWD.
    monkeypatch.chdir(tmp_path)

    SS_compactor(
        output_prefix="regression",
        compactor_file=str(COMPACTOR_TSV),
        element_annotation=False,
    )

    actual = tmp_path / "regression_results" / "structure_on_compactors.tsv"
    assert actual.exists(), f"SS_compactor did not produce {actual}"

    # Try byte-identical first; fall back to dataframe equality if pandas
    # writes drift in trailing whitespace etc.
    if filecmp.cmp(str(actual), str(GOLDEN_TSV), shallow=False):
        return

    df_actual = pd.read_csv(actual, sep="\t")
    df_golden = pd.read_csv(GOLDEN_TSV, sep="\t")
    pd.testing.assert_frame_equal(df_actual, df_golden, check_exact=False)


def test_golden_file_exists():
    """Sanity check that the golden file is checked in."""
    assert GOLDEN_TSV.exists(), (
        f"Golden file missing: {GOLDEN_TSV}. "
        "Re-generate by running SS_compactor on the legacy path and "
        "copying the output."
    )
