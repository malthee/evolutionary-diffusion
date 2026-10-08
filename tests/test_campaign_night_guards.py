"""Controller admission and finalization clocks need no models or network."""

import asyncio
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from evolutionary_extensions.execution import campaign


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "200", 99, None])
def test_invalid_finalization_deadline(value):
    with pytest.raises(ValueError):
        campaign.campaign_finalization_deadline(
            {"finalization_deadline_unix": value}, 100
        )


def test_finalization_clock_fallback():
    assert campaign.campaign_finalization_deadline({}, 100) == 100
    assert campaign.campaign_finalization_deadline({}, None) is None
    assert (
        campaign.campaign_finalization_deadline(
            {"finalization_deadline_unix": 200}, 100
        )
        == 200
    )


def setup_campaign(tmp_path, monkeypatch):
    template = tmp_path / "template.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("pass", metadata={"tags": ["parameters"]})]
        ),
        template,
    )
    document = {
        "execution": {
            "enabled": True,
            "campaign_root": str(tmp_path / "campaign"),
            "notebook": str(template),
            "jupyter_root": str(tmp_path),
            "jupyter_url": "http://unused",
            "kernel_name": "python",
            "runner_count": 2,
        },
        "campaign": {
            "seeds": [42, 43],
            "initial_deadline_unix": 100,
            "finalization_deadline_unix": 200,
        },
        "experiment": {
            "output_root": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path),
        },
        "drive": {"enabled": False},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))

    class Adapter:
        report_files = ()
        report_directories = ()

        def resolve_trials(self, config, seeds):
            return [
                {"label": f"trial-{i}", "overrides": {}, "experiment": {}}
                for i, _ in enumerate(seeds, 1)
            ]

        def validate_execution(self, *args):
            pass

        def validate_template(self, *args):
            pass

        def validate_parameters(self, *args):
            pass

        def parameters(self, params):
            return json.dumps(params)

        def estimate_run_bytes(self, params):
            return 1

        def campaign_report(self, summary):
            return summary["status"]

    class Response:
        url = "http://unused"

        async def read(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class HTTP(Response):
        def __init__(self, **kwargs):
            self.cookie_jar = SimpleNamespace(filter_cookies=lambda url: {})

        def get(self, url):
            return Response()

    class Pool(Response):
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return [{"session_id": "session", "kernel_id": "kernel"}] * 2

    clocks = []

    class Finalizer:
        reserved_bytes = 0

        def __init__(self, **kwargs):
            self.jobs = []

        async def wait_for_slot(self, deadline):
            clocks.append(deadline)

        def submit(self, run_id, function, size):
            future = asyncio.get_running_loop().create_future()
            future.set_result({"transfer": None, "output_bytes": 1})
            self.jobs.append({"run_id": run_id, "future": future})

        def summary(self):
            return []

        async def drain(self, deadline):
            clocks.append(deadline)
            return [j["future"].result() for j in self.jobs]

        def close(self):
            pass

    executed = []

    async def execute(http, url, headers, kernel, session, notebook, current, deadline):
        params = json.loads(notebook.cells[0].source)
        executed.append((params["seed"], deadline, params["deadline_unix"]))
        run = Path(params["output_root"]) / params["run_id"]
        run.mkdir()
        (run / "result.json").write_text(
            json.dumps(
                {
                    "run_id": params["run_id"],
                    "seed": params["seed"],
                    "validation_passed": True,
                    "scientific_hashes": {"embeddings": str(params["seed"])},
                }
            )
        )
        nbformat.write(notebook, current)

    monkeypatch.setattr(campaign.aiohttp, "ClientSession", HTTP)
    monkeypatch.setattr(campaign, "ManagedKernelPool", Pool)
    monkeypatch.setattr(campaign, "BackgroundFinalizer", Finalizer)
    monkeypatch.setattr(campaign, "execute_cells", execute)
    monkeypatch.setattr(
        campaign, "require_campaign_space", lambda *args, **kwargs: None
    )
    return path, Adapter(), executed, clocks


@pytest.mark.parametrize("all_skipped", [False, True])
def test_skip_is_persisted_without_optimization_or_sibling_failure(
    tmp_path, monkeypatch, all_skipped
):
    path, adapter, executed, clocks = setup_campaign(tmp_path, monkeypatch)

    async def before(index, params, results, status):
        if all_skipped or index == 1:
            raise campaign.SkipTrial("Admission closed")
        return params

    asyncio.run(campaign.run_campaign(path, adapter=adapter, before_trial=before))
    summary = json.loads((tmp_path / "campaign/campaign_summary.json").read_text())
    status = json.loads((tmp_path / "campaign/campaign_status.json").read_text())
    assert executed == ([] if all_skipped else [(43, 100, 100)])
    assert summary["initial_trials_passed"] is (not all_skipped)
    assert summary["status"] == ("all_trials_skipped" if all_skipped else "passed")
    assert len(summary["skipped_trials"]) == (2 if all_skipped else 1)
    assert status["stage"] == "complete"
    assert status["skipped_trials"] == summary["skipped_trials"]
    skipped = summary["skipped_trials"][0]
    assert {k: skipped[k] for k in ("index", "label", "seed", "reason")} == {
        "index": 1,
        "label": "trial-1",
        "seed": 42,
        "reason": "Admission closed",
    }
    assert isinstance(skipped["skipped_unix"], float)
    assert set(clocks) == {200}


def test_callback_errors_remain_campaign_failures(tmp_path, monkeypatch):
    path, adapter, executed, _ = setup_campaign(tmp_path, monkeypatch)

    async def before(*args):
        raise RuntimeError("Broken guard")

    with pytest.raises(builtins.ExceptionGroup, match="TaskGroup"):
        asyncio.run(campaign.run_campaign(path, adapter=adapter, before_trial=before))
    status = json.loads((tmp_path / "campaign/campaign_status.json").read_text())
    assert status["stage"] == "failed"
    assert executed == []


def test_invalid_deadline_rejected_before_adapter_or_network(tmp_path, monkeypatch):
    path, adapter, _, _ = setup_campaign(tmp_path, monkeypatch)
    document = json.loads(path.read_text())
    document["campaign"]["finalization_deadline_unix"] = 99
    path.write_text(json.dumps(document))
    adapter.resolve_trials = lambda *args: pytest.fail(
        "adapter called before clock validation"
    )
    with pytest.raises(ValueError, match=">="):
        asyncio.run(campaign.run_campaign(path, adapter=adapter))
    assert not (tmp_path / "campaign").exists()


def test_skip_exception_outside_admission_remains_failure(tmp_path, monkeypatch):
    path, adapter, _, _ = setup_campaign(tmp_path, monkeypatch)

    async def execute(*args):
        raise campaign.SkipTrial("Unexpected execution exception")

    monkeypatch.setattr(campaign, "execute_cells", execute)
    with pytest.raises(builtins.ExceptionGroup):
        asyncio.run(campaign.run_campaign(path, adapter=adapter))
    status = json.loads((tmp_path / "campaign/campaign_status.json").read_text())
    assert status["stage"] == "failed"
    assert status["skipped_trials"] == []
