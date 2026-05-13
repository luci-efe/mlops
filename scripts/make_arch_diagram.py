"""Generate the architecture diagram embedded in the cartel + PDF.

Outputs evidence/m3/architecture.png (~ 2400x900 px, white background).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MLOPS_ROOT = Path(__file__).resolve().parents[1]
OUT = MLOPS_ROOT / "evidence" / "m3" / "architecture.png"


def box(ax, x, y, w, h, label, color):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor="#2b2b2b", facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=11, weight="bold", color="#1a1a1a", wrap=True)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#2b2b2b"),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=9, color="#444")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Training side (left)
    box(ax, 0.3, 4.2, 2.4, 1.0, "Dataset ULB\n(OpenML 1597)", "#dbe9ff")
    box(ax, 0.3, 2.8, 2.4, 1.0, "DVC\n(versionado de datos)", "#fff4d6")
    box(ax, 0.3, 1.4, 2.4, 1.0, "MLflow\n(7 corridas, registro)", "#d6f5e0")

    box(ax, 3.4, 2.8, 2.4, 1.0, "XGBoost\nentrenamiento", "#fde2e4")
    box(ax, 3.4, 1.4, 2.4, 1.0, "Modelo registrado\nfraud-detector@staging", "#d6e8ff")

    # Export
    box(ax, 6.5, 2.1, 2.4, 1.0, "Export ONNX\n(zipmap=False)", "#efe5ff")

    # Cloud storage
    box(ax, 6.5, 4.2, 2.4, 1.0, "Cloudflare R2\n(artefactos DVC + MLflow)", "#fff0e6")

    # Container & deploy
    box(ax, 9.6, 2.1, 2.6, 1.0, "Contenedor Docker\nFastAPI + onnxruntime", "#e7f4d3")
    box(ax, 12.9, 2.1, 2.8, 1.0, "Cloudflare Worker\n+ Durable Object", "#d3f0f0")
    box(ax, 12.9, 0.5, 2.8, 1.0, "Cliente / curl\nPOST /predict", "#f4f4f4")

    # Arrows
    arrow(ax, 2.7, 4.7, 3.4, 3.6)
    arrow(ax, 2.7, 3.3, 3.4, 3.1)
    arrow(ax, 2.7, 1.9, 3.4, 2.0)
    arrow(ax, 5.8, 3.3, 6.5, 2.9)
    arrow(ax, 5.8, 1.9, 6.5, 2.4)
    arrow(ax, 8.9, 2.6, 9.6, 2.6, "ONNX")
    arrow(ax, 12.2, 2.6, 12.9, 2.6, "HTTP")
    arrow(ax, 14.3, 2.1, 14.3, 1.5, "JSON")
    arrow(ax, 14.3, 1.5, 14.3, 2.1)

    # Title
    ax.text(8, 5.6, "Pipeline MLOps — entrenamiento reproducible → inferencia en el edge",
            ha="center", va="center", fontsize=14, weight="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
