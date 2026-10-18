"""Phase 1 레슨 노트북(.ipynb)을 생성한다.  사용법: python tools/build_notebooks.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lesson_01  # noqa: E402
import lesson_02  # noqa: E402
import lesson_03  # noqa: E402
import lesson_04  # noqa: E402
import lesson_05  # noqa: E402
import lesson_06  # noqa: E402
import lesson_07  # noqa: E402
import lesson_08  # noqa: E402
import lesson_09  # noqa: E402
from nbkit import write_notebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LESSONS = {
    "01_what_is_credit_model.ipynb": lesson_01,
    "02_rows_columns_target.ipynb": lesson_02,
    "03_missing_and_quality.ipynb": lesson_03,
    "04_variable_diagnostics.ipynb": lesson_04,
    "05_binning_woe_iv.ipynb": lesson_05,
    "06_logistic_regression.ipynb": lesson_06,
    "07_scorecard.ipynb": lesson_07,
    "08_evaluation_metrics.ipynb": lesson_08,
    "09_psi_stability.ipynb": lesson_09,
}

if __name__ == "__main__":
    for filename, module in LESSONS.items():
        out = ROOT / "notebooks" / filename
        write_notebook(out, module.cells(), module.TITLE)
        print("wrote", out.relative_to(ROOT))
