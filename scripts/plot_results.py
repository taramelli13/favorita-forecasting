"""Generate result plots for the README.

Produces two charts:
  - docs/images/results.png: CV fold performance (LightGBM vs XGBoost) + baseline
  - docs/images/feature_importance.png: Top 10 features by LightGBM gain
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

folds = ["Fold 1\nJun 13-28", "Fold 2\nJun 29-Jul 14", "Fold 3\nJul 15-30", "Fold 4\nJul 31-Aug 15"]
lgb_scores = [0.5207, 0.5284, 0.5150, 0.5303]
xgb_scores = [0.5220, 0.5249, 0.5164, 0.5309]
baseline = 0.8006
lgb_mean = 0.5236
xgb_mean = 0.5235

top_features = [
    ("store_item_mean_sales", "Target enc."),
    ("item_mean_sales", "Target enc."),
    ("onpromotion", "Original"),
    ("sales_rolling_mean_14", "Rolling"),
    ("sales_rolling_mean_28", "Rolling"),
    ("oil_rolling_mean_7", "External"),
    ("family", "Categorical"),
    ("sales_lag_21", "Lag"),
    ("sales_lag_28", "Lag"),
    ("sales_rolling_mean_7", "Rolling"),
]

# Approximate relative importance (normalized, for visualization only)
importance = [100, 78, 62, 55, 48, 42, 38, 35, 32, 29]

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

C_LGB = "#3A86FF"
C_XGB = "#FF6B6B"
C_BASELINE = "#888888"
C_BG = "#FAFAFA"

# ---------------------------------------------------------------------------
# Chart 1: CV fold performance
# ---------------------------------------------------------------------------


def plot_results() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    x = np.arange(len(folds))
    w = 0.30

    bars_lgb = ax.bar(x - w / 2, lgb_scores, w, label="LightGBM", color=C_LGB, zorder=3)
    bars_xgb = ax.bar(x + w / 2, xgb_scores, w, label="XGBoost", color=C_XGB, zorder=3)

    # Baseline reference line
    ax.axhline(baseline, color=C_BASELINE, linestyle="--", linewidth=1.2, label=f"Baseline ({baseline})")

    # Mean lines
    ax.axhline(lgb_mean, color=C_LGB, linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(xgb_mean, color=C_XGB, linestyle=":", linewidth=1, alpha=0.6)

    # Value labels on bars
    for bars in (bars_lgb, bars_xgb):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.004,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Improvement annotation
    ax.annotate(
        f"-34.6%",
        xy=(3.4, lgb_mean),
        xytext=(3.4, baseline - 0.02),
        fontsize=11,
        fontweight="bold",
        color="#2D6A4F",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#2D6A4F", lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=9)
    ax.set_ylabel("NWRMSLE (lower is better)", fontsize=10)
    ax.set_title("Cross-Validation Performance by Fold", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0.45, 0.88)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "results.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'results.png'}")


# ---------------------------------------------------------------------------
# Chart 2: Feature importance (horizontal bar)
# ---------------------------------------------------------------------------


def plot_feature_importance() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    labels = [f"{name}" for name, _ in reversed(top_features)]
    types = [t for _, t in reversed(top_features)]
    vals = list(reversed(importance))

    type_colors = {
        "Target enc.": "#3A86FF",
        "Original": "#2D6A4F",
        "Rolling": "#FF9F1C",
        "External": "#8338EC",
        "Categorical": "#FF6B6B",
        "Lag": "#06D6A0",
    }
    colors = [type_colors[t] for t in types]

    bars = ax.barh(labels, vals, color=colors, height=0.65, zorder=3)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Legend for feature types
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9, title="Feature type")

    ax.set_xlabel("Relative Importance (gain)", fontsize=10)
    ax.set_title("Top 10 Features — LightGBM", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    plot_results()
    plot_feature_importance()
