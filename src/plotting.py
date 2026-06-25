"""Shared figure-saving helper so EVERY plot lands in one place: report/figures/.

Usage in any notebook/script:
    from src.plotting import save_fig
    ...plot...
    save_fig("class_balance")          # -> report/figures/class_balance.png

The report references these as `figures/NAME.png` (paths relative to report/report.md).
Figures ARE committed (they go into the report), so report/figures/ is NOT gitignored.
"""
from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"


def save_fig(name: str, dpi: int = 150, tight: bool = True):
    """Save the current matplotlib figure as report/figures/<name>.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=dpi,
                bbox_inches="tight" if tight else None)
    return FIG_DIR / f"{name}.png"
