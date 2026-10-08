"""Campaign policy/paired controls are validated without CUDA or inference."""

from evolutionary_extensions.experiments.prompt_embedding.adapter import (
    PromptEmbeddingExperimentAdapter,
)


import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import nbformat
import pytest

from evolutionary_extensions.execution.campaign import (
    campaign_deadline,
    campaign_reproducibility,
    run_campaign,
)
from evolutionary_extensions.execution.notebooks import execute_cells

ROOT = Path(__file__).resolve().parents[1]


def config():
    document = json.loads((ROOT / "configs/examples/osga.json").read_text())
    document["campaign"].update(
        seeds=[42, 42, 42],
        initial_budget_seconds=None,
        trials=[
            {
                "label": label,
                "overrides": {
                    "success_ratio": ratio,
                    "comparison_factor": factor,
                    "max_selection_pressure": 100,
                },
            }
            for label, ratio, factor in [("A", 0.6, 1), ("B", 0.4, 1), ("C", 0.4, 0.9)]
        ],
    )
    return document


def test_disabled_paired_recipe_and_reproducibility_grouping(tmp_path):
    c = config()
    trials = PromptEmbeddingExperimentAdapter.resolve_trials(c, c["campaign"]["seeds"])
    assert [t["experiment"]["success_ratio"] for t in trials] == [0.6, 0.4, 0.4]
    assert [t["experiment"]["comparison_factor"] for t in trials] == [1, 1, 0.9]
    assert all(t["experiment"]["max_selection_pressure"] == 100 for t in trials)
    assert campaign_deadline(c["campaign"]) is None
    results = [
        {
            "run_id": f"trial-{i}-seed-42",
            "seed": 42,
            "scientific_hashes": {"embeddings": str(i)},
        }
        for i in range(1, 4)
    ]
    assert campaign_reproducibility(results, trials) == (None, None)
    same = PromptEmbeddingExperimentAdapter.resolve_trials(
        {**c, "campaign": {"seeds": [42, 42]}}, [42, 42]
    )
    assert campaign_reproducibility(results[:2], same) == (False, None)
    results[1]["scientific_hashes"] = results[0]["scientific_hashes"]
    assert campaign_reproducibility(results[:2], same) == (True, None)
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="disabled"):
        asyncio.run(run_campaign(path))
    assert not list(tmp_path.glob("**/managed_sessions.json"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_budget_seconds", 0),
        ("initial_budget_seconds", -1),
        ("initial_budget_seconds", True),
        ("initial_budget_seconds", float("inf")),
        ("initial_deadline_unix", float("nan")),
    ],
)
def test_invalid_clock_policy(key, value):
    with pytest.raises(ValueError):
        campaign_deadline({key: value})


def test_finite_policy_preserves_explicit_deadline_and_relative_budget():
    before = time.time()
    assert (
        before + 30
        <= campaign_deadline({"initial_budget_seconds": 30})
        <= time.time() + 30
    )
    assert (
        campaign_deadline(
            {"initial_deadline_unix": 123, "initial_budget_seconds": None}
        )
        == 123
    )


@pytest.mark.parametrize(
    "trials",
    [
        [{}],
        [{"overrides": {"max_evaluations": 9999}}, {}, {}],
        [{"overrides": {"success_ratio": 1.1}}, {}, {}],
        [{"overrides": {"comparison_factor": True}}, {}, {}],
        [{"label": "A"}, {"label": "A"}, {}],
        [{"seed": 43}, {}, {}],
    ],
)
def test_invalid_overrides_rejected_before_owned_sessions(tmp_path, trials):
    c = config()
    c["execution"]["enabled"] = True
    c["campaign"]["trials"] = trials
    c["execution"]["campaign_root"] = str(tmp_path / "campaign")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        asyncio.run(run_campaign(path))
    assert not (tmp_path / "campaign").exists()


def test_notebook_cell_without_deadline_runs_and_expired_deadline_does_not(tmp_path):
    class WS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send_json(self, msg):
            self.ident = msg["header"]["msg_id"]
            self.count = 0

        async def receive(self):
            await asyncio.sleep(0.01)
            self.count += 1
            kind = "execute_input" if self.count == 1 else "status"
            content = (
                {"execution_count": 1}
                if self.count == 1
                else {"execution_state": "idle"}
            )
            data = {
                "parent_header": {"msg_id": self.ident},
                "msg_type": kind,
                "content": content,
            }
            return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data))

    class HTTP:
        def ws_connect(self, *args, **kwargs):
            return WS()

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")])
    asyncio.run(
        execute_cells(
            HTTP(),
            "http://server",
            {},
            "owned-kernel",
            "owned-session",
            notebook,
            tmp_path / "executed.ipynb",
            None,
        )
    )
    assert notebook.cells[0].execution_count == 1
    with pytest.raises(TimeoutError):
        asyncio.run(
            execute_cells(
                HTTP(),
                "http://server",
                {},
                "owned-kernel",
                "owned-session",
                notebook,
                tmp_path / "expired.ipynb",
                time.time() - 1,
            )
        )
    assert not (tmp_path / "expired.ipynb").exists()
