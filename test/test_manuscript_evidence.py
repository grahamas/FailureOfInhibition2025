"""Tests for the standalone manuscript-evidence utilities."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "foi-matplotlib-tests")
)
ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_script("check_anchor_equilibria")
figures = load_script("plot_manuscript_evidence")


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ManuscriptEvidenceUtilityTests(unittest.TestCase):
    def make_archive(
        self,
        directory,
        *,
        execution_success=True,
        map_status="completed",
        foi_count=3,
        onset_threshold=4.0,
        include_foi_phase_root=True,
    ):
        config = directory / "config.toml"
        config.write_text(
            textwrap.dedent(
                """
                [model.excitatory]
                timescale = 7.8
                slope = 5.0
                threshold = 1.5

                [model.inhibitory]
                timescale = 34.32
                slope = 5.0
                threshold = {onset_threshold}

                [[anchors]]
                name = "figure3"
                e_to_e = 17.0
                i_to_e = 9.0
                i_to_i = 4.0
                """.format(onset_threshold=onset_threshold)
            ).lstrip()
        )
        equilibria = directory / "equilibria.csv"
        equilibrium_rows = [
            {
                "anchor": "figure3",
                "condition": "control",
                "e_to_i": 19,
                "failure_threshold": 8,
                "E": 0.1,
                "I": 0.1,
                "stability": "Attracting",
            }
        ]
        if include_foi_phase_root:
            equilibrium_rows.append(
                {
                    "anchor": "figure3",
                    "condition": "failure_of_inhibition",
                    "e_to_i": 19,
                    "failure_threshold": 8,
                    "E": 0.5,
                    "I": 0.001,
                    "stability": "Attracting",
                }
            )
        write_csv(
            equilibria,
            (
                "anchor",
                "condition",
                "e_to_i",
                "failure_threshold",
                "E",
                "I",
                "stability",
            ),
            equilibrium_rows,
        )
        coexistence = directory / "map.csv"
        write_csv(
            coexistence,
            (
                "anchor",
                "condition",
                "status",
                "e_to_i",
                "failure_threshold",
                "discovered_equilibria",
                "attracting_equilibria",
            ),
            [
                {
                    "anchor": "figure3",
                    "condition": "control",
                    "status": map_status,
                    "e_to_i": 19,
                    "failure_threshold": 8,
                    "discovered_equilibria": 1,
                    "attracting_equilibria": 1,
                },
                {
                    "anchor": "figure3",
                    "condition": "failure_of_inhibition",
                    "status": map_status,
                    "e_to_i": 19,
                    "failure_threshold": 8,
                    "discovered_equilibria": 1,
                    "attracting_equilibria": 1,
                },
                {
                    "anchor": "figure3",
                    "condition": "failure_of_inhibition",
                    "status": map_status,
                    "e_to_i": 20,
                    "failure_threshold": 8,
                    "discovered_equilibria": foi_count,
                    "attracting_equilibria": foi_count,
                },
            ],
        )
        metadata = directory / "metadata.toml"
        metadata.write_text(
            f"execution_success = {str(execution_success).lower()}\n"
            "failed_search_contexts = []\n"
        )
        checksums = {
            path.name: figures.sha256(path)
            for path in (config, equilibria, coexistence, metadata)
        }
        (directory / "checksums.toml").write_text(
            'algorithm = "SHA-256"\n\n[files]\n'
            + "".join(f'"{name}" = "{digest}"\n' for name, digest in checksums.items())
        )

    def test_independent_jacobian_matches_finite_difference(self):
        parameters = {
            "e_to_e": 17.0,
            "i_to_e": 9.0,
            "e_to_i": 19.0,
            "i_to_i": 4.0,
            "a_e": 5.0,
            "a_i": 5.0,
            "theta_e": 1.5,
            "theta_on": 4.0,
            "theta_off": 8.0,
        }
        state = np.array([0.31, 0.42])
        step = 1e-6
        for failure in (False, True):
            _, analytical = checker.equations(state, parameters, failure)
            numerical = np.empty((2, 2))
            for column in range(2):
                offset = np.zeros(2)
                offset[column] = step
                plus = checker.equations(state + offset, parameters, failure)[0]
                minus = checker.equations(state - offset, parameters, failure)[0]
                numerical[:, column] = (plus - minus) / (2 * step)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-7, atol=1e-8)

    def test_extreme_balance_inputs_remain_finite(self):
        parameters = {
            "e_to_e": 17.0,
            "i_to_e": 9.0,
            "e_to_i": 19.0,
            "i_to_i": 4.0,
            "a_e": 5.0,
            "a_i": 5.0,
            "theta_e": 1.5,
            "theta_on": 4.0,
            "theta_off": 8.0,
        }
        for condition in figures.CONDITIONS:
            result = figures.balance(
                np.array([-1e8, 1e8]), 0.0, parameters, condition
            )
            self.assertTrue(np.isfinite(result).all())

    def test_renderer_writes_only_three_figures_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            self.make_archive(archive)
            output = root / "figures"
            figures.render(archive, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "response_curve.png",
                    "response_curve.json",
                    "figure3_phase_portraits.png",
                    "figure3_phase_portraits.json",
                    "figure3_coexistence.png",
                    "figure3_coexistence.json",
                    "manifest.json",
                },
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                [item["figure"] for item in manifest["figures"]],
                [
                    "response_curve",
                    "figure3_phase_portraits",
                    "figure3_coexistence",
                ],
            )
            for item in manifest["figures"]:
                provenance = json.loads((output / item["provenance"]).read_text())
                self.assertEqual(
                    provenance["output"]["sha256"],
                    figures.sha256(output / item["png"]),
                )
            phase = json.loads(
                (output / "figure3_phase_portraits.json").read_text()
            )
            self.assertEqual(
                {Path(source["path"]).name for source in phase["sources"]},
                {"config.toml", "equilibria.csv"},
            )
            self.assertIn("highest-E attracting root is (E=0.5, I=0.001)", phase["claim"])
            coexistence = json.loads(
                (output / "figure3_coexistence.json").read_text()
            )
            self.assertIn("sampled maximum of 3", coexistence["claim"])
            self.assertEqual(
                {Path(source["path"]).name for source in coexistence["sources"]},
                {"map.csv", "metadata.toml"},
            )
            with self.assertRaises(FileExistsError):
                figures.render(archive, output)

    def test_renderer_rejects_changed_consumed_input_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            self.make_archive(archive)
            with (archive / "map.csv").open("a") as stream:
                stream.write("\n")
            output = root / "figures"
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                figures.render(archive, output)
            self.assertFalse(output.exists())

    def test_renderer_rejects_failed_or_incomplete_archives_before_writing(self):
        cases = (
            (
                {"execution_success": False},
                "did not complete successfully",
            ),
            ({"map_status": "execution_failed"}, "incomplete cells"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    archive = root / "archive"
                    archive.mkdir()
                    self.make_archive(archive, **arguments)
                    output = root / "figures"
                    with self.assertRaisesRegex(ValueError, message):
                        figures.render(archive, output)
                    self.assertFalse(output.exists())

    def test_coexistence_claim_and_color_scale_follow_observed_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            self.make_archive(archive, foi_count=5)
            output = root / "figures"
            figures.render(archive, output)
            provenance = json.loads(
                (output / "figure3_coexistence.json").read_text()
            )
            self.assertIn("sampled maximum of 5", provenance["claim"])
            self.assertNotIn("No sampled cell has four", provenance["limitation"])

    def test_phase_portrait_rejects_root_count_mismatch_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            self.make_archive(archive, include_foi_phase_root=False)
            output = root / "figures"
            with self.assertRaisesRegex(ValueError, "rows do not match map count"):
                figures.render(archive, output)
            self.assertFalse(output.exists())

    def test_response_annotation_uses_computed_midpoint(self):
        class CapturingPackage:
            def save(self, figure, *args):
                self.figure = figure

        package = CapturingPackage()
        config = {
            "model": {
                "inhibitory": {
                    "slope": 5.0,
                    "threshold": 2.0,
                }
            }
        }
        figures.response_figure(package, config, Path("config.toml"))
        labels = [text.get_text() for text in package.figure.axes[0].texts]
        self.assertIn("midpoint = 5", labels)
        figures.plt.close(package.figure)


if __name__ == "__main__":
    unittest.main()
