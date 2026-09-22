"""Render the three figures used by the minimal manuscript-claim report.

The input must be a completed coexistence artifact whose consumed files still
match its checksum manifest. Outputs are written only to a fresh directory.
No exhaustive-attractor, biological-state, or treatment inference is made.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = {
    "control": ("Monotone control", "#2166ac"),
    "failure_of_inhibition": ("Failure of inhibition", "#b2182b"),
}
FIGURE3_E_TO_I = 19.0
FIGURE3_FAILURE_THRESHOLD = 8.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inputs(directory, required):
    manifest_path = directory / "checksums.toml"
    manifest = tomllib.loads(manifest_path.read_text())["files"]
    for relative in required:
        path = directory / relative
        if relative not in manifest:
            raise ValueError(f"consumed input is not checksummed: {relative}")
        if not path.is_file() or sha256(path) != manifest[relative]:
            raise ValueError(f"consumed input is missing or changed: {relative}")
    return [directory / relative for relative in required]


def validate_coexistence_artifact(metadata_path, map_path):
    metadata = tomllib.loads(metadata_path.read_text())
    if metadata.get("execution_success") is not True:
        raise ValueError("coexistence artifact execution did not complete successfully")
    if metadata.get("failed_search_contexts"):
        raise ValueError("coexistence artifact records failed search contexts")

    evidence = rows(map_path)
    if not evidence:
        raise ValueError("coexistence map is empty")
    incomplete = [row for row in evidence if row["status"] != "completed"]
    if incomplete:
        raise ValueError(
            f"coexistence map contains {len(incomplete)} incomplete cells"
        )
    figure3 = [row for row in evidence if row["anchor"] == "figure3"]
    for condition in CONDITIONS:
        if not any(row["condition"] == condition for row in figure3):
            raise ValueError(f"Figure 3 map is missing condition: {condition}")
        if not any(
            row["condition"] == condition
            and float(row["e_to_i"]) == FIGURE3_E_TO_I
            and float(row["failure_threshold"]) == FIGURE3_FAILURE_THRESHOLD
            for row in figure3
        ):
            raise ValueError(
                "Figure 3 map is missing the phase-portrait cell "
                f"for condition: {condition}"
            )
    return figure3


def validate_phase_evidence(equilibrium_path, figure3_map):
    evidence = [
        row
        for row in rows(equilibrium_path)
        if row["anchor"] == "figure3"
        and float(row["e_to_i"]) == FIGURE3_E_TO_I
        and float(row["failure_threshold"]) == FIGURE3_FAILURE_THRESHOLD
    ]
    for condition in CONDITIONS:
        cell = next(
            row
            for row in figure3_map
            if row["condition"] == condition
            and float(row["e_to_i"]) == FIGURE3_E_TO_I
            and float(row["failure_threshold"]) == FIGURE3_FAILURE_THRESHOLD
        )
        selected = [row for row in evidence if row["condition"] == condition]
        discovered = int(cell["discovered_equilibria"])
        attracting = int(cell["attracting_equilibria"])
        if len(selected) != discovered:
            raise ValueError(
                f"Figure 3 {condition} equilibrium rows do not match map count"
            )
        if sum(row["stability"] == "Attracting" for row in selected) != attracting:
            raise ValueError(
                f"Figure 3 {condition} attracting rows do not match map count"
            )
    return evidence


def sigmoid(argument):
    return np.exp(-np.logaddexp(0.0, -argument))


def inhibitory_response(input_value, slope, onset_threshold, failure_threshold):
    return sigmoid(slope * (input_value - onset_threshold)) - sigmoid(
        slope * (input_value - failure_threshold)
    )


def balance(e, i, parameters, condition):
    excitatory_input = parameters["e_to_e"] * e - parameters["i_to_e"] * i
    inhibitory_input = parameters["e_to_i"] * e - parameters["i_to_i"] * i
    excitatory_rate = sigmoid(
        parameters["a_e"] * (excitatory_input - parameters["theta_e"])
    )
    inhibitory_rate = sigmoid(
        parameters["a_i"] * (inhibitory_input - parameters["theta_on"])
    )
    if condition == "failure_of_inhibition":
        inhibitory_rate -= sigmoid(
            parameters["a_i"] * (inhibitory_input - parameters["theta_off"])
        )
    return (
        -e + (1.0 - e) * excitatory_rate,
        -i + (1.0 - i) * inhibitory_rate,
    )


class FigurePackage:
    def __init__(self, output):
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.figures = []
        self.environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        }

    def save(self, figure, name, sources, claim, limitation):
        target = self.output / f"{name}.png"
        figure.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(figure)
        provenance = {
            "figure": name,
            "claim": claim,
            "limitation": limitation,
            "environment": self.environment,
            "sources": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in sorted(set(sources))
            ],
            "output": {"path": target.name, "sha256": sha256(target)},
        }
        provenance_path = self.output / f"{name}.json"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        self.figures.append(
            {
                "figure": name,
                "png": target.name,
                "provenance": provenance_path.name,
                "claim": claim,
                "limitation": limitation,
            }
        )

    def finish(self):
        (self.output / "manifest.json").write_text(
            json.dumps(
                {
                    "figures": self.figures,
                    "environment": self.environment,
                    "interpretation": (
                        "Claim-specific numerical evidence; no completeness, "
                        "biological-state, or clinical inference."
                    ),
                },
                indent=2,
            )
            + "\n"
        )


def response_figure(package, config, source):
    inhibitory = config["model"]["inhibitory"]
    slope = float(inhibitory["slope"])
    onset = float(inhibitory["threshold"])
    failure = FIGURE3_FAILURE_THRESHOLD
    midpoint = (onset + failure) / 2.0
    inputs = np.linspace(onset - 3.0, failure + 3.0, 500)
    values = inhibitory_response(inputs, slope, onset, failure)

    figure, axis = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    axis.plot(inputs, values, color="#7c3aed", linewidth=2.4)
    axis.axvline(midpoint, color="#334155", linestyle="--", linewidth=1.2)
    axis.scatter([midpoint], [inhibitory_response(midpoint, slope, onset, failure)],
                 color="#111827", zorder=3)
    axis.set(
        xlabel="Effective inhibitory input",
        ylabel="Response",
        title="Implemented equal-slope failure response",
        xlim=(inputs[0], inputs[-1]),
        ylim=(0.0, 1.02),
    )
    axis.text(
        midpoint + 0.18,
        0.04,
        f"midpoint = {midpoint:g}",
        color="#334155",
    )
    axis.grid(alpha=0.18)
    package.save(
        figure,
        "response_curve",
        [source],
        "The implemented equal-slope response rises and then falls symmetrically.",
        "This does not validate independently adjustable slopes or historical fits.",
    )


def phase_portrait_figure(
    package, config, config_path, equilibrium_path, evidence
):
    anchor = next(item for item in config["anchors"] if item["name"] == "figure3")
    excitatory = config["model"]["excitatory"]
    inhibitory = config["model"]["inhibitory"]
    parameters = {
        "e_to_e": float(anchor["e_to_e"]),
        "i_to_e": float(anchor["i_to_e"]),
        "e_to_i": FIGURE3_E_TO_I,
        "i_to_i": float(anchor["i_to_i"]),
        "a_e": float(excitatory["slope"]),
        "a_i": float(inhibitory["slope"]),
        "theta_e": float(excitatory["threshold"]),
        "theta_on": float(inhibitory["threshold"]),
        "theta_off": FIGURE3_FAILURE_THRESHOLD,
    }
    e_grid, i_grid = np.meshgrid(
        np.linspace(0.0, 0.5, 201), np.linspace(0.0, 0.5, 201)
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), layout="constrained")
    for axis, condition in zip(axes, CONDITIONS):
        e_balance, i_balance = balance(e_grid, i_grid, parameters, condition)
        axis.contour(e_grid, i_grid, e_balance, levels=[0], colors=["#2166ac"])
        axis.contour(e_grid, i_grid, i_balance, levels=[0], colors=["#b2182b"])
        for row in evidence:
            if row["condition"] != condition:
                continue
            marker = "o" if row["stability"] == "Attracting" else "x"
            axis.scatter(
                float(row["E"]),
                float(row["I"]),
                color="#111827",
                marker=marker,
                s=36,
                zorder=3,
            )
        axis.set(
            xlabel="E active fraction",
            ylabel="I active fraction",
            title=CONDITIONS[condition][0],
            xlim=(-0.01, 0.51),
            ylim=(-0.01, 0.51),
            aspect="equal",
        )
        axis.grid(alpha=0.12)
    figure.suptitle(
        f"Figure 3 anchor · E-to-I = {FIGURE3_E_TO_I:g} · "
        f"failure threshold = {FIGURE3_FAILURE_THRESHOLD:g}\n"
        "blue: E nullcline · red: I nullcline · circle: attracting · cross: other",
        fontsize=11,
    )
    foi_attracting = [
        row
        for row in evidence
        if row["condition"] == "failure_of_inhibition"
        and row["stability"] == "Attracting"
    ]
    if foi_attracting:
        highest_e = max(foi_attracting, key=lambda row: float(row["E"]))
        claim = (
            "The supplied Figure 3 FoI evidence contains "
            f"{len(foi_attracting)} locally attracting roots; its highest-E "
            f"attracting root is (E={float(highest_e['E']):g}, "
            f"I={float(highest_e['I']):g})."
        )
    else:
        claim = (
            "The supplied Figure 3 FoI evidence contains no locally "
            "attracting root."
        )
    package.save(
        figure,
        "figure3_phase_portraits",
        [config_path, equilibrium_path],
        claim,
        "The roots and local spectra are parameter-specific and do not certify completeness.",
    )


def coexistence_figure(package, map_path, metadata_path, evidence):
    all_counts = [int(row["attracting_equilibria"]) for row in evidence]
    foi_counts = [
        int(row["attracting_equilibria"])
        for row in evidence
        if row["condition"] == "failure_of_inhibition"
    ]
    color_maximum = max(4, max(all_counts))
    foi_maximum = max(foi_counts)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    last_scatter = None
    for axis, condition in zip(axes, CONDITIONS):
        selected = [row for row in evidence if row["condition"] == condition]
        last_scatter = axis.scatter(
            [float(row["e_to_i"]) for row in selected],
            [float(row["failure_threshold"]) for row in selected],
            c=[int(row["attracting_equilibria"]) for row in selected],
            vmin=0,
            vmax=color_maximum,
            cmap="viridis",
            s=18,
        )
        axis.set(
            xlabel="E-to-I coupling",
            ylabel="Failure threshold",
            title=CONDITIONS[condition][0],
        )
    figure.colorbar(
        last_scatter,
        ax=axes,
        ticks=list(range(color_maximum + 1)),
        label="Discovered locally attracting equilibria",
    )
    figure.suptitle("Figure 3 sampled coexistence plane", fontsize=11)
    package.save(
        figure,
        "figure3_coexistence",
        [map_path, metadata_path],
        (
            "The supplied Figure 3 FoI slice has a sampled maximum of "
            f"{foi_maximum} discovered attracting roots."
        ),
        "The finite search is not an exhaustive attractor count.",
    )


def render(coexistence, output):
    sources = verify_inputs(
        coexistence, ("config.toml", "equilibria.csv", "map.csv", "metadata.toml")
    )
    config_path, equilibrium_path, map_path, metadata_path = sources
    config = tomllib.loads(config_path.read_text())
    map_evidence = validate_coexistence_artifact(metadata_path, map_path)
    phase_evidence = validate_phase_evidence(equilibrium_path, map_evidence)
    package = FigurePackage(output)
    response_figure(package, config, config_path)
    phase_portrait_figure(
        package, config, config_path, equilibrium_path, phase_evidence
    )
    coexistence_figure(package, map_path, metadata_path, map_evidence)
    package.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coexistence", type=Path, default=Path("output/coexistence_study")
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.coexistence, arguments.output)


if __name__ == "__main__":
    main()
