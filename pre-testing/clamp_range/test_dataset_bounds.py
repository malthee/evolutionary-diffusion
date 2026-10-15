"""Deterministic tests for experiment reductions, sampling and deadlines."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import dataset_bounds as experiment
import nbformat
import run_notebooks as runner
import torch
from run_notebooks import run_with_timeout


class BoundsTests(unittest.TestCase):
    def test_public_runner_modes_propagate_notebook_deadline(self):
        for selection in ("--one", "--start"):
            with (
                self.subTest(selection=selection),
                tempfile.TemporaryDirectory() as directory,
                patch.object(runner, "ROOT", Path(directory)),
                patch.object(sys, "argv", ["run_notebooks.py", selection, "drawbench"]),
                patch.object(runner, "execute") as execute,
                patch.object(
                    runner,
                    "run_with_timeout",
                    side_effect=subprocess.TimeoutExpired("notebook", 3600),
                ) as supervised,
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.main()
                supervised.assert_called_once()
                command, _ = supervised.call_args.args
                self.assertEqual(command[-2:], ["--_execute", "drawbench"])
                execute.assert_not_called()

    def test_single_notebook_does_not_run_following_datasets(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "ROOT", Path(directory)),
            patch.object(sys, "argv", ["run_notebooks.py", "--one", "drawbench"]),
            patch.object(runner, "execute") as execute,
            patch.object(runner, "run_with_timeout") as supervised,
        ):
            runner.main()
            supervised.assert_called_once()
            self.assertEqual(supervised.call_args.args[0][-1], "drawbench")
            execute.assert_not_called()

    def test_streaming_matches_stacked_negative_and_constant_values(self):
        tensors = [
            torch.tensor([[[-3.0, 2.0, 0.0]]]),
            torch.tensor([[[-1.0, 2.0, -5.0]]]),
            torch.tensor([[[-4.0, 2.0, 4.0]]]),
        ]
        bounds = experiment.Bounds()
        for tensor in tensors:
            bounds.update(tensor)
        stacked = torch.stack(tensors)
        torch.testing.assert_close(bounds.minimum, stacked.min(dim=0).values)
        torch.testing.assert_close(bounds.maximum, stacked.max(dim=0).values)
        self.assertEqual(bounds.summary()["constant_coordinates"], 1)
        self.assertEqual(bounds.summary()["global_min"], -5.0)
        self.assertEqual(bounds.summary()["global_max"], 4.0)

    def test_rejects_shape_change_nonfinite_and_empty_summary(self):
        bounds = experiment.Bounds()
        with self.assertRaises(ValueError):
            bounds.summary()
        bounds.update(torch.ones(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            bounds.update(torch.ones(1, 3, 3))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bounds.update(torch.full((1, 2, 3), float("nan")))

    def test_sampling_is_deterministic_unique_and_nested(self):
        records = [{"id": i} for i in range(30)]
        small = experiment.select_records(records, 5)
        large = experiment.select_records(records, 20)
        self.assertEqual(small, large[:5])
        self.assertEqual(large, experiment.select_records(records, 20))
        self.assertEqual(len({x["id"] for x in large}), 20)

    def test_cleaning_preserves_exact_text_and_source_id(self):
        rows = [
            {"id": 1, "prompt": " a cat "},
            {"id": 2, "prompt": " a cat "},
            {"id": 3, "prompt": ""},
            {"id": 4, "prompt": "a cat"},
        ]
        records, counts = experiment.clean_records(rows)
        self.assertEqual([x["id"] for x in records], [1, 4])
        self.assertEqual(
            counts,
            {"raw_rows": 4, "blank_rows": 1, "duplicate_rows": 1, "unique_prompts": 2},
        )

    def test_budget_accounts_for_preparation_and_headroom(self):
        self.assertEqual(experiment.planned_count(100, 20, [2.0] * 32, 100), 32)
        self.assertEqual(experiment.planned_count(100, 110, [2.0] * 32, 100), 0)
        with self.assertRaises(ValueError):
            experiment.planned_count(100, 0, [], 100)

    def test_wall_clock_timeout_terminates_worker(self):
        with (
            tempfile.TemporaryFile() as output,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            run_with_timeout(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                output,
                seconds=0.1,
            )

    def test_wall_clock_timeout_also_shuts_down_notebook_kernel(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            tempfile.TemporaryFile() as output,
        ):
            directory = Path(temporary)
            marker = directory / "kernel_pid"
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(
                        "from pathlib import Path\nimport os, time\n"
                        + f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                        + "time.sleep(60)"
                    )
                ]
            )
            nbformat.write(notebook, directory / "sdxl_turbo_parti.ipynb")
            command = [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; "
                + f"sys.path.insert(0, {str(experiment.ROOT)!r}); "
                + f"import run_notebooks; run_notebooks.ROOT = Path({str(directory)!r}); "
                + "run_notebooks.execute('parti')",
            ]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_with_timeout(command, output, seconds=4)
            self.assertTrue(marker.exists(), "Notebook kernel never started")
            pid = int(marker.read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail("Notebook kernel survived the wall-clock timeout")

    def test_comparison_handles_zero_span_and_extensions(self):
        baseline = {"min": torch.tensor([-2.0, 5.0]), "max": torch.tensor([2.0, 5.0])}
        candidate = {"min": torch.tensor([-3.0, 5.0]), "max": torch.tensor([4.0, 5.0])}
        metrics = experiment.compare_bounds(candidate, baseline)
        self.assertEqual(metrics["span_change_percent"], 75.0)
        self.assertEqual(metrics["either_outside_percent"], 50.0)
        self.assertEqual(metrics["largest_lower_extension"], 1.0)
        constant = {"min": torch.tensor([5.0]), "max": torch.tensor([5.0])}
        self.assertIsNone(
            experiment.compare_bounds(constant, constant)["span_change_percent"]
        )

    def test_checkpoint_roundtrip(self):
        bounds = experiment.Bounds()
        bounds.update(torch.tensor([[1.0, -2.0]]))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            experiment.save_checkpoint(
                directory, {"token": bounds}, {"dataset": "fixture"}, []
            )
            saved = torch.load(directory / "bounds.pt", weights_only=True)
            torch.testing.assert_close(saved["token"]["min"], bounds.minimum)
            self.assertTrue((directory / "summary.json").is_file())

    def test_first_invalid_pooled_embedding_preserves_failure_status(self):
        class InvalidPooledEncoder:
            def __init__(self, device=None):
                self.device = "cpu"
                self.calls = 0

            def verify_creator_parity(self):
                return []

            def encode(self, prompt):
                self.calls += 1
                pooled = torch.ones(1, 2)
                if self.calls > 5:
                    pooled[0, 0] = float("nan")
                return torch.ones(1, 2, 3), pooled, [False, False], [3, 3]

        rows = [
            {"id": str(i), "prompt": f"prompt {i}", "category": "test"}
            for i in range(5)
        ]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(experiment, "RESULTS", Path(temporary)),
            patch.object(experiment, "raw_records", side_effect=lambda _: iter(rows)),
            patch.object(experiment, "TextEncoder", InvalidPooledEncoder),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                experiment.run_dataset("geneval")
            import json

            summary = json.loads(
                (Path(temporary) / "geneval" / "summary.json").read_text()
            )
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["encoded_prompts"], 0)
            self.assertIn("non-finite", summary["error"])

    def test_run_encodes_once_for_both_outputs_and_stops_on_deadline(self):
        instances = []

        class FakeEncoder:
            def __init__(self, device=None):
                self.device = "cpu"
                self.calls = 0
                instances.append(self)

            def verify_creator_parity(self):
                return []

            def encode(self, prompt):
                self.calls += 1
                return torch.ones(1, 2, 3), torch.ones(1, 2), [False, False], [3, 3]

        rows = [
            {"id": str(i), "prompt": f"prompt {i}", "category": "test"}
            for i in range(5)
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(experiment, "RESULTS", Path(tmp)),
            patch.object(experiment, "raw_records", side_effect=lambda _: iter(rows)),
            patch.object(experiment, "TextEncoder", FakeEncoder),
        ):
            result = experiment.run_dataset("geneval")
            self.assertEqual(result["encoded_prompts"], 5)
            self.assertEqual(instances[-1].calls, 10)  # Five probes, five main encodes.
            saved = torch.load(
                Path(tmp) / "geneval" / "matched_bounds.pt", weights_only=True
            )
            self.assertEqual(set(saved), {"token", "pooled"})
            result = experiment.run_dataset("geneval", stop_seconds=-1)
            self.assertEqual(result["status"], "partial_timeout")
            self.assertEqual(result["encoded_prompts"], 0)


if __name__ == "__main__":
    unittest.main()
