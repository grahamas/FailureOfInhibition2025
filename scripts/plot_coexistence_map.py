"""Plot discovered equilibrium counts; requires the optional matplotlib package."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_csv", type=Path)
    parser.add_argument("output", type=Path, help="new PNG or PDF output path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    with args.map_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    anchors = list(dict.fromkeys(row["anchor"] for row in rows))
    if not anchors:
        parser.error("map has no rows")
    xs = sorted({float(row["e_to_i"]) for row in rows})
    ys = sorted({float(row["failure_threshold"]) for row in rows})
    if min(len(xs), len(ys)) < 2:
        parser.error("plot requires at least two values on each axis")
    maximum = max(1, max(int(row["attracting_equilibria"]) for row in rows))
    cmap = plt.get_cmap("viridis", maximum + 1).copy()
    cmap.set_bad("#d9d9d9")
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
    fig, axes = plt.subplots(len(anchors), 2, figsize=(10, 4 * len(anchors)),
                             squeeze=False, constrained_layout=True)
    for row_index, anchor in enumerate(anchors):
        for column, condition in enumerate(("control", "failure_of_inhibition")):
            ax = axes[row_index, column]
            values = np.full((len(ys), len(xs)), np.nan)
            seen = set()
            for row in rows:
                if row["anchor"] != anchor or row["condition"] != condition:
                    continue
                key = (float(row["e_to_i"]), float(row["failure_threshold"]))
                if key in seen:
                    raise ValueError(f"duplicate map coordinate: {anchor} {condition} {key}")
                seen.add(key)
                if row["status"] == "completed":
                    values[ys.index(key[1]), xs.index(key[0])] = int(row["attracting_equilibria"])
            if len(seen) != len(xs) * len(ys):
                raise ValueError(f"incomplete rectangular map: {anchor} {condition}")
            mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(values),
                                 cmap=cmap, norm=norm, shading="nearest", rasterized=True)
            if condition == "failure_of_inhibition":
                ax.plot(2 * np.asarray(ys), ys, "--", color="white", linewidth=1.2,
                        label=r"$J_{I\leftarrow E}=2\theta_{off}$ heuristic")
                ax.legend(loc="upper left", fontsize=8, facecolor="#333333", labelcolor="white")
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
            ax.set_title(anchor.replace("_", " ") + " · " + ("monotone control" if column == 0 else "FoI"))
            ax.set_xlabel(r"E-to-I coupling $J_{I\leftarrow E}$")
            ax.set_ylabel(r"Failure threshold $\theta_{off}$")
    fig.colorbar(mesh, ax=axes, ticks=range(maximum + 1),
                 label="Discovered locally attracting equilibria")
    fig.suptitle("Equilibrium coexistence exploration\nIncomplete discovery; no biological regime labels", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    evidence = {"input": str(args.map_csv.resolve()),
                "input_sha256": hashlib.sha256(args.map_csv.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
                "meaning": "discovered locally attracting equilibrium counts, not exhaustive attractor counts"}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
