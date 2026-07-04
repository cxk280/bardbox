"""
Render loss_log.csv to a loss-curve PNG for report.md.

Run (from inside module1_from_scratch/):
    python plot_loss.py

Reads loss_log.csv (written by train.py) and writes report_assets/loss_curve.png.
Deliberately plain matplotlib — the point is the curve going down, not chart art.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless: render straight to a file, no display needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "report_assets")


def main():
    steps, train, val = [], [], []
    with open(os.path.join(HERE, "loss_log.csv")) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(steps, train, label="train loss", color="#2563eb", linewidth=2)
    ax.plot(steps, val, label="val loss", color="#dc2626", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats/char)")
    ax.set_title("From-scratch GPT on tiny-shakespeare")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Annotate the final val loss so the reader sees where it landed.
    ax.annotate(
        f"final val: {val[-1]:.3f}",
        xy=(steps[-1], val[-1]),
        xytext=(-90, 20),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#dc2626"),
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "loss_curve.png")
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
