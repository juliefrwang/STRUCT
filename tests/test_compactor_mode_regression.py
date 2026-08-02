"""WCF regression test: SS_compactor (legacy WCF-only path) must produce
output byte-identical to the golden saved in
``tests/test_data/golden_compactor_mode_wcf.tsv``.

The golden was first captured on ``feature/non-wcf`` prior to the
compactor-mode extension (Phase C0). Phase C1 refreshed it to absorb
the ``k = 80 → len(base_S1) + len(base_S2)`` fix in
``anchor_p_compactor_subdf`` (Decision C2 of
``IMPLEMENTATION_PLAN_nonwcf_compactor.md``) — this is a latent-bug
correction, not a behavioural change to the extended path, and shifts
only the ``anchor_p`` / ``anchor_p_BH`` columns. All upstream columns
(stem detection, mutation counts, ``compactor_p``) are unchanged from
the pre-fix golden.

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
GOLDEN_WCF_TSV = REPO_ROOT / "tests" / "test_data" / "golden_compactor_mode_wcf.tsv"
GOLDEN_GU_TSV = REPO_ROOT / "tests" / "test_data" / "golden_compactor_mode_GU.tsv"


def _assert_golden(actual_path, golden_path):
    """Byte-identical first, fall back to dataframe equality."""
    if filecmp.cmp(str(actual_path), str(golden_path), shallow=False):
        return
    df_actual = pd.read_csv(actual_path, sep="\t")
    df_golden = pd.read_csv(golden_path, sep="\t")
    pd.testing.assert_frame_equal(df_actual, df_golden, check_exact=False)


def test_wcf_path_matches_golden(tmp_path, monkeypatch):
    """Run SS_compactor on the standard test input under the legacy
    (no-flag) path. Must match ``golden_compactor_mode_wcf.tsv``."""
    monkeypatch.chdir(tmp_path)

    SS_compactor(
        output_prefix="regression",
        compactor_file=str(COMPACTOR_TSV),
        element_annotation=False,
    )

    actual = tmp_path / "regression_results" / "structure_on_compactors.tsv"
    assert actual.exists(), f"SS_compactor did not produce {actual}"
    _assert_golden(actual, GOLDEN_WCF_TSV)


def test_noncanon_GU_path_matches_golden(tmp_path, monkeypatch):
    """Run SS_compactor under the extended (G·U) path. Must match
    ``golden_compactor_mode_GU.tsv``.

    This locks the Phase C2 + C3 wiring: wobble-aware stem detection,
    find_mutation_ext per half, b_vector concatenation, target_p_ext
    dispatch, and wrap_anchor_p_compactor_ext. titv = 0.5 (uniform
    null) is the test default.
    """
    monkeypatch.chdir(tmp_path)

    SS_compactor(
        output_prefix="regression_GU",
        compactor_file=str(COMPACTOR_TSV),
        element_annotation=False,
        noncanon="GU",
        titv=0.5,
    )

    actual = tmp_path / "regression_GU_results" / "structure_on_compactors.tsv"
    assert actual.exists(), f"SS_compactor did not produce {actual}"
    _assert_golden(actual, GOLDEN_GU_TSV)


def test_golden_files_exist():
    """Sanity check that both goldens are checked in."""
    assert GOLDEN_WCF_TSV.exists(), (
        f"WCF golden missing: {GOLDEN_WCF_TSV}. Re-generate by running "
        "SS_compactor on the legacy path."
    )
    assert GOLDEN_GU_TSV.exists(), (
        f"G·U golden missing: {GOLDEN_GU_TSV}. Re-generate by running "
        "SS_compactor with noncanon='GU', titv=0.5."
    )
