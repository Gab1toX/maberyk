"""Genera el PNG de curvas de perdida a partir del log de entrenamiento.

Uso:
    python analisis/plot_loss_curves.py --log analisis/retrain_production_big.log --out docs/loss_curves.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"epoch\s+(\d+)/\d+\s+.*?train loss\s+([\d.]+),\s+val loss\s+([\d.]+)"
)


def parse(log_path: Path):
    epochs, train, val = [], [], []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train.append(float(match.group(2)))
            val.append(float(match.group(3)))
    return epochs, train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="docs/loss_curves.png")
    parser.add_argument("--title", default="Maberyk - Transformer condicional (256d / 6 capas)")
    args = parser.parse_args()

    epochs, train, val = parse(Path(args.log))
    if not epochs:
        raise SystemExit("No se encontraron lineas de epoch en el log.")

    best_idx = min(range(len(val)), key=lambda i: val[i])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="train loss", linewidth=2)
    ax.plot(epochs, val, label="val loss", linewidth=2)
    ax.scatter([epochs[best_idx]], [val[best_idx]], zorder=5, s=60)
    ax.annotate(
        f"mejor val = {val[best_idx]:.4f}",
        xy=(epochs[best_idx], val[best_idx]),
        xytext=(8, 12),
        textcoords="offset points",
    )

    gap = val[best_idx] - train[best_idx]
    ax.set_title(f"{args.title}\nbrecha train/val en el mejor punto: {gap:.2f} - firma de memorizacion")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[plot] guardado en {out}")
    print(f"[plot] epochs={len(epochs)} mejor val={val[best_idx]:.4f} en epoch {epochs[best_idx]}")


if __name__ == "__main__":
    main()
