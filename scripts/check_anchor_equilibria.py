"""Independent SciPy multistart check of the Figure 3 Julia anchor.

This implements the documented balance and Jacobian directly, without calling
the Julia model. Agreement is a numerical cross-check, not completeness proof.

The check uses a 51-by-51 uniform seed grid on [0, 0.5]^2, SciPy's hybr
solver with xtol=1e-11 and maxfev=500, a balance-residual acceptance threshold
of 1e-10, and coordinate and spectrum matching tolerances of 1e-7. These
script-local conventions are also written to metadata.json; they do not change
the package model or its public numerical API.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import tomllib

import numpy as np
import scipy
from scipy.optimize import root
from scipy.special import expit


FIGURE3_E_TO_I = 19.0
FIGURE3_FAILURE_THRESHOLD = 8.0
SEED_GRID_POINTS = 51
SEED_MINIMUM = 0.0
SEED_MAXIMUM = 0.5
SOLVER_XTOL = 1e-11
SOLVER_MAXFEV = 500
RESIDUAL_ATOL = 1e-10
COORDINATE_MATCH_ATOL = 1e-7
SPECTRUM_MATCH_ATOL = 1e-7


def equations(state, parameters, failure):
    e, i = state
    p = parameters
    ue = p["e_to_e"] * e - p["i_to_e"] * i
    ui = p["e_to_i"] * e - p["i_to_i"] * i
    fe = expit(p["a_e"] * (ue - p["theta_e"]))
    onset = expit(p["a_i"] * (ui - p["theta_on"]))
    offset = expit(p["a_i"] * (ui - p["theta_off"])) if failure else 0.0
    fi = onset - offset
    de = p["a_e"] * fe * (1 - fe)
    di = p["a_i"] * (onset * (1 - onset) - offset * (1 - offset))
    balance = np.array([-e + (1 - e) * fe, -i + (1 - i) * fi])
    jac = np.array([
        [-1 - fe + (1 - e) * de * p["e_to_e"], -(1 - e) * de * p["i_to_e"]],
        [(1 - i) * di * p["e_to_i"], -1 - fi - (1 - i) * di * p["i_to_i"]],
    ])
    return balance, jac


ROOT_FIELDS = ("anchor", "condition", "root", "E", "I", "residual",
               "eigenvalue_1_real", "eigenvalue_1_imaginary", "eigenvalue_2_real",
               "eigenvalue_2_imaginary", "reference_matches", "state_error", "spectrum_error")


def write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames if fieldnames is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path, reference_path, output):
    config = tomllib.loads(config_path.read_text())
    with reference_path.open() as stream:
        reference = list(csv.DictReader(stream))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.toml")
    shutil.copy2(reference_path, output / "reference.csv")
    shutil.copy2(__file__, output / "check_anchor_equilibria.py")
    attempts, roots, comparisons = [], [], []
    anchors = [anchor for anchor in config["anchors"] if anchor["name"] == "figure3"]
    if len(anchors) != 1:
        raise ValueError("configuration must contain exactly one figure3 anchor")
    for anchor in anchors:
        p = dict(anchor, e_to_i=FIGURE3_E_TO_I,
                 theta_off=FIGURE3_FAILURE_THRESHOLD,
                 a_e=config["model"]["excitatory"]["slope"],
                 a_i=config["model"]["inhibitory"]["slope"],
                 theta_e=config["model"]["excitatory"]["threshold"],
                 theta_on=config["model"]["inhibitory"]["threshold"])
        for condition in ("control", "failure_of_inhibition"):
            failure = condition == "failure_of_inhibition"
            found = []
            seeds = np.linspace(SEED_MINIMUM, SEED_MAXIMUM, SEED_GRID_POINTS)
            for e, i in itertools.product(seeds, repeat=2):
                result = root(lambda x: equations(x, p, failure)[0], [e, i],
                              jac=lambda x: equations(x, p, failure)[1],
                              method="hybr",
                              options={"xtol": SOLVER_XTOL,
                                       "maxfev": SOLVER_MAXFEV})
                residual = float(np.max(np.abs(equations(result.x, p, failure)[0])))
                accepted = bool(np.all(np.isfinite(result.x))
                                and residual <= RESIDUAL_ATOL
                                and np.all(result.x >= 0) and np.all(result.x <= 1))
                attempts.append(dict(anchor=anchor["name"], condition=condition,
                                     seed_E=e, seed_I=i, E=result.x[0], I=result.x[1],
                                     solver_success=bool(result.success), solver_status=int(result.status),
                                     residual=residual, accepted=accepted))
                if accepted and not any(
                    np.max(np.abs(result.x - x)) <= COORDINATE_MATCH_ATOL
                    for x in found
                ):
                    found.append(result.x.copy())
            found.sort(key=lambda x: (x[0], x[1]))
            expected = [r for r in reference if r["anchor"] == anchor["name"]
                        and r["condition"] == condition
                        and float(r["e_to_i"]) == FIGURE3_E_TO_I
                        and float(r["failure_threshold"]) == FIGURE3_FAILURE_THRESHOLD]
            matched_ids = set()
            for index, state in enumerate(found):
                balance, jac = equations(state, p, failure)
                eig = np.sort_complex(np.linalg.eigvals(jac / np.array([
                    config["model"]["excitatory"]["timescale"],
                    config["model"]["inhibitory"]["timescale"]])[:, None]))
                candidates = [r for r in expected if max(abs(state[0] - float(r["E"])),
                              abs(state[1] - float(r["I"])))
                              <= COORDINATE_MATCH_ATOL]
                state_error, spectrum_error = None, None
                if len(candidates) == 1:
                    row = candidates[0]
                    matched_ids.add(row["equilibrium"])
                    state_error = float(max(abs(state[0] - float(row["E"])), abs(state[1] - float(row["I"]))))
                    expected_eig = np.sort_complex([complex(float(row[f"eigenvalue_{j}_real"]),
                                                     float(row[f"eigenvalue_{j}_imaginary"])) for j in (1, 2)])
                    spectrum_error = float(np.max(np.abs(eig - expected_eig)))
                roots.append(dict(anchor=anchor["name"], condition=condition, root=index + 1,
                                  E=state[0], I=state[1], residual=float(np.max(np.abs(balance))),
                                  eigenvalue_1_real=eig[0].real, eigenvalue_1_imaginary=eig[0].imag,
                                  eigenvalue_2_real=eig[1].real, eigenvalue_2_imaginary=eig[1].imag,
                                  reference_matches=len(candidates), state_error=state_error,
                                  spectrum_error=spectrum_error))
            current = [r for r in roots if r["anchor"] == anchor["name"] and r["condition"] == condition]
            passed = bool(len(found) == len(expected) == len(matched_ids)
                          and len(expected) > 0
                          and all(r["reference_matches"] == 1
                                  and r["spectrum_error"] <= SPECTRUM_MATCH_ATOL
                                  for r in current))
            comparisons.append(dict(anchor=anchor["name"], condition=condition,
                                    independent_count=len(found), reference_count=len(expected), passed=passed))
    write_csv(output / "attempts.csv", attempts)
    write_csv(output / "roots.csv", roots, ROOT_FIELDS)
    write_csv(output / "comparisons.csv", comparisons)
    metadata = dict(python=platform.python_version(), scipy=scipy.__version__, numpy=np.__version__,
                    seed_grid=[SEED_GRID_POINTS, SEED_GRID_POINTS],
                    seed_bounds=[SEED_MINIMUM, SEED_MAXIMUM],
                    solver="scipy.optimize.root:hybr",
                    solver_xtol=SOLVER_XTOL, solver_maxfev=SOLVER_MAXFEV,
                    residual_atol=RESIDUAL_ATOL,
                    coordinate_match_atol=COORDINATE_MATCH_ATOL,
                    spectrum_match_atol=SPECTRUM_MATCH_ATOL,
                    completeness="not_certified", passed=all(r["passed"] for r in comparisons),
                    replay="python check_anchor_equilibria.py --config config.toml --reference reference.csv --output replay")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparisons, indent=2))
    return metadata["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.config, args.reference, args.output) else 1)
