from evolutionary_extensions.experiments.prompt_embedding.adapter import (
    PromptEmbeddingExperimentAdapter,
)

import hashlib
import json
from pathlib import Path

import pytest

from evolutionary_extensions.persistence import google_drive as d


class FakeDrive:
    def __init__(self):
        self.files = {}
        self.sessions = {}
        self.allocations = 0
        self.fail_once = False

    def check_account(self):
        pass

    def metadata(self, ident):
        if ident == "folder":
            return {"id": ident, "mimeType": "application/vnd.google-apps.folder"}
        data = self.files.get(ident)
        if data is None:
            return None
        return {
            "id": ident,
            "size": str(len(data)),
            "md5Checksum": hashlib.md5(data, usedforsecurity=False).hexdigest(),
            "parents": ["folder"],
        }

    def request(self, url, *, method="GET", data=None, headers=None):
        if "generateIds" in url:
            self.allocations += 1
            return 200, {}, json.dumps({"ids": ["id1"]}).encode()
        if "uploadType" in url:
            ident = json.loads(data).get("id", "id1")
            self.sessions["https://www.googleapis.com/session"] = ident
            return 200, {"Location": "https://www.googleapis.com/session"}, b""
        ident = self.sessions[url]
        if not data:
            if ident in self.files:
                return 200, {}, b""
            return 308, {}, b""
        self.files[ident] = data
        if self.fail_once:
            self.fail_once = False
            raise OSError("lost acknowledgement")
        return 200, {}, b""


def prepared(tmp_path):
    run = tmp_path / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "experiment_complete.json").write_text('{"status":"complete"}')
    (run / "science.txt").write_text("fitness, lineage, embeddings")
    fake = FakeDrive()
    return (
        run,
        fake,
        d.GoogleDrivePersistor(
            tmp_path / "credentials",
            "folder",
            tmp_path / "zip",
            tmp_path / "receipts",
            run.parent,
            client=fake,
        ),
    )


def verify_fake(monkeypatch, fake):
    def verify(client, ident, manifest):
        value = fake.files[ident]
        if (
            len(value) != manifest["bytes"]
            or hashlib.sha256(value).hexdigest() != manifest["sha256"]
        ):
            raise RuntimeError("download corruption")
        return manifest["sha256"]

    monkeypatch.setattr(d, "verify_download", verify)
    monkeypatch.setattr(d.time, "sleep", lambda _: None)


def test_idempotent_upload_lost_ack_and_verified_cleanup(tmp_path, monkeypatch):
    run, fake, persistor = prepared(tmp_path)
    verify_fake(monkeypatch, fake)
    fake.fail_once = True
    stages = []
    first = persistor.persist(run, on_stage=stages.append)
    assert stages == ["packaging", "uploading_and_verifying"]
    assert first["transfer_seconds"] == pytest.approx(
        sum(
            first[k]
            for k in ("packaging_seconds", "upload_verify_seconds", "cleanup_seconds")
        )
    )
    assert run.exists() and first["verified"]
    second = persistor.persist(run, delete_after_verification=True)
    assert fake.allocations == 1 and len(fake.files) == 1
    assert not run.exists() and not (tmp_path / "zip/trial.zip").exists()
    assert second["runner_copies_deleted"]
    assert "session_url" not in Path(first["receipt_path"]).read_text()


def test_corrupt_remote_refuses_cleanup(tmp_path, monkeypatch):
    run, fake, persistor = prepared(tmp_path)
    verify_fake(monkeypatch, fake)
    archive, _ = persistor.package(run)
    state, _ = persistor.upload(archive)
    fake.files["id1"] = b"corrupted"
    with pytest.raises(RuntimeError, match="Remote archive verification"):
        persistor.cleanup(run, archive, state)
    assert run.exists() and archive.exists()


def test_changed_source_refuses_cleanup(tmp_path, monkeypatch):
    run, fake, persistor = prepared(tmp_path)
    verify_fake(monkeypatch, fake)
    archive, _ = persistor.package(run)
    state, _ = persistor.upload(archive)
    (run / "science.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        persistor.cleanup(run, archive, state)
    assert run.exists() and archive.exists()


def test_download_verification_required(tmp_path, monkeypatch):
    run, fake, persistor = prepared(tmp_path)
    archive, _ = persistor.package(run)
    state = tmp_path / "receipts/trial.json"
    d.upload(archive, tmp_path / "credentials", "folder", state, client=fake)
    with pytest.raises(ValueError, match="verified download"):
        persistor.cleanup(run, archive, state)
    assert run.exists()


def test_upload_corruption_retains_everything(tmp_path, monkeypatch):
    run, fake, persistor = prepared(tmp_path)
    verify_fake(monkeypatch, fake)
    archive, _ = persistor.package(run)
    with archive.open("ab") as handle:
        handle.write(b"change")
    with pytest.raises(ValueError, match="Archive changed"):
        persistor.upload(archive)
    assert fake.allocations == 0 and run.exists()


def test_packaging_rejects_secrets_and_unfinished_outputs(tmp_path):
    run, _, persistor = prepared(tmp_path)
    (run / "experiment_complete.json").unlink()
    with pytest.raises(ValueError, match="completion marker"):
        persistor.package(run)
    (run / "experiment_complete.json").write_text('{"status":"complete"}')
    (run / "private.pem").write_text("not a real key")
    with pytest.raises(ValueError, match="Credential-like"):
        persistor.package(run)


def decision_fixture():
    config = json.loads(Path("configs/examples/osga.json").read_text())
    config["execution"]["enabled"] = True
    result = {
        "validation_passed": True,
        "scientific_hashes": {"embeddings": "abc"},
        "loop_seconds": 10,
        "evaluation_count": 100,
        "export_seconds": 1,
    }
    transfers = [
        {"verified": True, "transfer_seconds": 1, "bytes": 1000000} for _ in range(3)
    ]
    return config, [dict(result) for _ in range(3)], transfers


def test_streamed_download_hash_rejects_corruption(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace

    payload = b"scientific archive" * 17
    manifest = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    client = SimpleNamespace(
        access_token=lambda: "test-token", timeout=lambda seconds: seconds
    )
    monkeypatch.setattr(
        d.urllib.request,
        "build_opener",
        lambda *handlers: SimpleNamespace(open=lambda *a, **k: io.BytesIO(payload)),
    )
    assert d.verify_download(client, "id1", manifest) == manifest["sha256"]
    manifest["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Downloaded SHA-256"):
        d.verify_download(client, "id1", manifest)


def test_notebook_errors_cannot_finalize_a_run(tmp_path):
    import nbformat

    from evolutionary_extensions.execution.campaign import finalize_run

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "run_result = run_experiment(config, runtime)",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "error", ename="RuntimeError", evalue="failure", traceback=[]
                    )
                ],
            )
        ]
    )
    path = tmp_path / "executed.ipynb"
    nbformat.write(notebook, path)
    with pytest.raises(ValueError, match="contains an error"):
        finalize_run(tmp_path, path)
    assert not (tmp_path / "experiment_complete.json").exists()


def test_completed_archive_packages_without_drive_credentials(tmp_path, monkeypatch):
    def unexpected_credentials(*args, **kwargs):
        raise AssertionError("Offline packaging must not access Drive credentials")

    monkeypatch.setattr(d, "DriveClient", unexpected_credentials)
    run = tmp_path / "trial"
    run.mkdir()
    (run / "experiment_complete.json").write_text('{"status":"complete"}')
    (run / "records.json").write_text('[{"fitness":6.5}]')
    archive = tmp_path / "offline.zip"
    manifest = d.package(run, archive)
    assert d.hashes(archive) == {k: manifest[k] for k in ("bytes", "md5", "sha256")}
    with d.zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert json.loads(zipped.read("trial/records.json"))[0]["fitness"] == 6.5
    assert run.is_dir()


@pytest.mark.parametrize("unlimited", [False, True])
def test_background_finalizer_overlaps_work_bounds_backlog_and_drains(unlimited):
    import asyncio
    import threading
    import time

    from evolutionary_extensions.execution.campaign import BackgroundFinalizer

    entered, release = threading.Event(), threading.Event()

    def first():
        entered.set()
        assert release.wait(5)
        return "verified first"

    async def scenario():
        queue = BackgroundFinalizer(limit=2)
        try:
            queue.submit("first", first, 100)
            assert await asyncio.to_thread(entered.wait, 5)
            queue.submit("second", lambda: "verified second", 200)
            assert queue.reserved_bytes == 300
            with pytest.raises(RuntimeError, match="slot"):
                queue.submit("third", lambda: None, 1)
            waiting = asyncio.create_task(
                queue.wait_for_slot(None if unlimited else time.time() + 5)
            )
            await asyncio.sleep(0)
            assert (
                not waiting.done()
            )  # Backpressure applies only when both slots are full.
            release.set()
            await waiting
            assert await queue.drain(None if unlimited else time.time() + 5) == [
                "verified first",
                "verified second",
            ]
            assert queue.reserved_bytes == 0
            assert all(j["state"] == "complete" for j in queue.summary())
        finally:
            release.set()
            queue.close()

    asyncio.run(scenario())


def test_background_failure_and_deadline_retain_outputs(tmp_path):
    import asyncio
    import threading
    import time

    from evolutionary_extensions.execution.campaign import BackgroundFinalizer

    evidence = tmp_path / "completed-run"
    evidence.write_text("scientific output")

    def failing():
        raise OSError("upload failed")

    async def scenario():
        queue = BackgroundFinalizer()
        try:
            queue.submit("trial", failing, 10)
            with pytest.raises(OSError, match="upload failed"):
                await queue.drain(time.time() + 5)
            with pytest.raises(OSError, match="upload failed"):
                await queue.wait_for_slot(time.time() + 5)
            assert queue.summary()[0]["state"] == "failed" and evidence.exists()
        finally:
            queue.close()
        release = threading.Event()
        queue = BackgroundFinalizer()
        try:
            queue.submit("trial", lambda: release.wait(5), 10)
            with pytest.raises(TimeoutError):
                await queue.drain(time.time() - 1)
            assert evidence.exists()
        finally:
            release.set()
            await queue.drain(time.time() + 5)
            queue.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "runner_count,wait_for_space,seeds,automatic_root,unlimited,variant_controls",
    [
        (1, False, [42, 42, 43], False, False, False),
        (1, True, [42, 42, 43], False, False, False),
        (2, False, [42, 42, 43], False, False, False),
        (3, False, [42, 42, 43], False, False, False),
        (1, False, [42], False, False, False),
        (1, False, [42, 42], False, False, False),
        (2, False, [42, 42, 43, 43, 44], False, False, False),
        (1, False, [42, 42, 43], True, False, False),
        (1, True, [42, 42, 43], False, True, False),
        (3, False, [42, 42, 42], False, True, True),
    ],
)
def test_background_campaign_freezes_notebook_and_starts_next_trial(
    tmp_path,
    monkeypatch,
    runner_count,
    wait_for_space,
    seeds,
    automatic_root,
    unlimited,
    variant_controls,
):
    import ast
    import asyncio
    import threading
    from types import SimpleNamespace

    import nbformat

    from evolutionary_extensions.execution import campaign as managed

    config = json.loads(Path("configs/examples/osga.json").read_text())
    config["execution"]["enabled"] = True
    config["deployment"].update(
        output_root=str(tmp_path / "runs"), cache_dir=str(tmp_path)
    )
    config["execution"].update(
        campaign_root=str(tmp_path / "reports"),
        jupyter_root=str(tmp_path),
        runner_count=runner_count,
    )
    config["campaign"]["seeds"] = seeds
    if unlimited:
        config["campaign"]["initial_budget_seconds"] = None
    if variant_controls:
        config["campaign"]["trials"] = [
            {
                "label": label,
                "overrides": {
                    "success_ratio": ratio,
                    "comparison_factor": factor,
                    "max_selection_pressure": 100,
                },
            }
            for label, ratio, factor in [("A", 0.6, 1), ("B", 0.4, 1), ("C", 0.4, 0.9)]
        ]
    if automatic_root:
        config["execution"].pop("campaign_root")
    config["experiment"].update(
        candidate_batch_size=4,
        initial_embeddings_file=".local/inputs/anchor.safetensors",
        initial_embeddings_sha256="a" * 64,
    )
    config["execution"].update(enabled=True, warm_runners=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    first_started, second_executed = threading.Event(), threading.Event()
    closed = []
    created = []
    executed_paths = {}

    class Response:
        status = 201
        url = "http://127.0.0.1:8888/lab"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            return b""

        async def json(self):
            return {"id": self.ident, "kernel": {"id": self.ident + "-kernel"}}

    class HTTP(Response):
        def __init__(self, **kwargs):
            self.cookie_jar = SimpleNamespace(filter_cookies=lambda url: {})

        def get(self, *args, **kwargs):
            return Response()

        def post(self, *args, **kwargs):
            response = Response()
            response.ident = "owned-session-" + str(len(created) + 1)
            created.append(kwargs["json"]["path"])
            return response

        def delete(self, url, **kwargs):
            closed.append(url)
            response = Response()
            response.status = 204
            return response

    def parameters(notebook):
        cell = next(
            c for c in notebook.cells if "parameters" in c.metadata.get("tags", [])
        )
        return ast.literal_eval(ast.parse(cell.source).body[0].value)

    async def execute(
        http, base, headers, kernel, session, notebook, current, deadline
    ):
        params = parameters(notebook)
        assert params["initial_embeddings_file"] == str(
            Path(managed.checkout_root()) / ".local/inputs/anchor.safetensors"
        )
        if unlimited:
            assert deadline is None and params["deadline_unix"] is None
        if variant_controls:
            index = int(params["run_id"].split("-")[1]) - 1
            for name, value in config["campaign"]["trials"][index]["overrides"].items():
                assert params[name] == value
        await asyncio.sleep(0)
        executed_paths[session] = str(current)
        if params["run_id"] == "trial-2-seed-42":
            assert await asyncio.to_thread(first_started.wait, 5)
            second_executed.set()
        run = Path(params["output_root"]) / params["run_id"]
        run.mkdir(parents=True)
        result = {
            "run_id": params["run_id"],
            "seed": params["seed"],
            "population_size": 10,
            "completed_generations": 8,
            "evaluation_count": 100,
            "best_fitness": 7.0,
            "loop_seconds": 10,
            "export_seconds": 1,
            "validation_passed": False,
            "termination_reason": "max_selection_pressure",
            "status": "terminated",
            "scientific_hashes": {
                "embeddings": str(params["seed"])
                + (
                    str(params["success_ratio"]) + str(params["comparison_factor"])
                    if variant_controls
                    else ""
                )
            },
        }
        (run / "result.json").write_text(json.dumps(result))
        for name in (
            "config.json",
            "parity.json",
            "environment.json",
            "model_manifest.json",
            "generation_statistics.csv",
            "stage_timings.csv",
            "operator_statistics.csv",
            "embedding_statistics.csv",
        ):
            (run / name).write_text("{}")
        for name in ("plots", "media"):
            (run / name).mkdir()
        nbformat.write(notebook, current)

    def finalize(run, notebook, *, adapter=None):
        if run.name == "trial-1-seed-42":
            first_started.set()
            if len(seeds) > 1:
                assert second_executed.wait(5)
        # Check after the following experiment has overwritten current.ipynb.
        assert parameters(nbformat.read(notebook, as_version=4))["run_id"] == run.name
        (run / "artifact_integrity.json").write_text('{"passed":true}')
        (run / "experiment_complete.json").write_text('{"status":"complete"}')
        return json.loads((run / "result.json").read_text())

    original_space_check = managed.require_campaign_space
    waited = []

    def check_space(settings, pending_bytes=0, *, estimated_bytes):
        if wait_for_space and settings["run_id"] == "trial-2-seed-42" and not waited:
            waited.append(True)
            second_executed.set()  # Allow the immutable first run to finish and free space.
            raise OSError("Backlog consumes disk reserve")
        original_space_check(settings, pending_bytes, estimated_bytes=estimated_bytes)

    monkeypatch.setattr(managed, "require_campaign_space", check_space)
    monkeypatch.setattr(managed.aiohttp, "ClientSession", HTTP)
    monkeypatch.setattr(managed, "execute_cells", execute)
    monkeypatch.setattr(managed, "finalize_run", finalize)
    monkeypatch.setattr(
        managed.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3)
    )
    asyncio.run(managed.run_campaign(path))
    assert bool(waited) == wait_for_space
    assert len(closed) == runner_count
    assert len(set(created)) == runner_count
    assert len(set(executed_paths.values())) == runner_count
    assert all("owned-session-" in url for url in closed)
    campaign_dir = tmp_path / Path(created[0]).parent
    summary = json.loads((campaign_dir / "campaign_summary.json").read_text())
    assert len(summary["results"]) == len(seeds)
    assert summary["same_seed_identical"] is (
        True if len(set(seeds)) < len(seeds) and not variant_controls else None
    )
    assert summary["different_seed_distinct"] is (True if len(set(seeds)) > 1 else None)
    if automatic_root:
        assert campaign_dir.parent.name.startswith("campaign-")
    for index, seed in enumerate(seeds, 1):
        name = f"trial-{index}-seed-{seed}"
        report = campaign_dir / "reports" / name
        assert (
            json.loads((report / "finalization.json").read_text())["stage"]
            == "complete"
        )
        assert (report / "local_archive.json").exists()


def test_campaign_space_reserves_next_run_and_pending_archives(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evolutionary_extensions.execution import campaign as managed

    config = json.loads(Path("configs/examples/osga.json").read_text())
    config["execution"]["enabled"] = True
    settings = {
        **config["deployment"],
        **config["experiment"],
        "output_root": str(tmp_path),
        "cache_dir": str(tmp_path),
    }
    monkeypatch.setattr(
        managed.shutil, "disk_usage", lambda path: SimpleNamespace(free=24 * 1024**3)
    )
    settings.update(population_size=10, num_generations=10, max_evaluations=910)
    managed.require_campaign_space(
        settings,
        estimated_bytes=PromptEmbeddingExperimentAdapter.estimate_run_bytes(settings),
    )
    with pytest.raises(OSError, match="backlog"):
        managed.require_campaign_space(
            settings,
            pending_bytes=2 * 1024**3,
            estimated_bytes=PromptEmbeddingExperimentAdapter.estimate_run_bytes(
                settings
            ),
        )


def test_managed_pool_closes_created_sessions_when_creation_fails(tmp_path):
    import asyncio

    from evolutionary_extensions.execution.campaign import ManagedKernelPool

    closed = []

    class Response:
        def __init__(self, status):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return {"id": "first-owned", "kernel": {"id": "first-kernel"}}

    class HTTP:
        created = 0

        def post(self, *args, **kwargs):
            self.created += 1
            return Response(201 if self.created == 1 else 500)

        def delete(self, url, **kwargs):
            closed.append(url)
            return Response(204)

    async def scenario():
        with pytest.raises(RuntimeError, match="creation"):
            async with ManagedKernelPool(
                HTTP(), "http://server", {}, ["first", "second"], "python", tmp_path
            ):
                pytest.fail("Pool admitted failed creation")

    asyncio.run(scenario())
    assert closed == ["http://server/api/sessions/first-owned"]
    assert not json.loads((tmp_path / "session_closed.json").read_text())["failures"]


@pytest.mark.parametrize("seeds", [[], [True], ["42"], [2**32]])
def test_campaign_rejects_invalid_seed_queue_before_creating_session(tmp_path, seeds):
    import asyncio

    from evolutionary_extensions.execution.campaign import run_campaign

    config = json.loads(Path("configs/examples/osga.json").read_text())
    config["execution"]["enabled"] = True
    config["deployment"]["output_root"] = str(tmp_path / "runs")
    config["campaign"]["seeds"] = seeds
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="seeds"):
        asyncio.run(run_campaign(path))


def test_campaign_refuses_reusing_owned_session_directory(tmp_path):
    import asyncio

    from evolutionary_extensions.execution.campaign import run_campaign

    root = tmp_path / "previous-campaign"
    root.mkdir()
    marker = root / "managed_sessions.json"
    marker.write_text('[{"session_id":"prior-owned"}]')
    config = json.loads(Path("configs/examples/osga.json").read_text())
    config["execution"]["enabled"] = True
    config["deployment"]["output_root"] = str(tmp_path / "runs")
    config["execution"]["campaign_root"] = str(root)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fresh campaign_root"):
        asyncio.run(run_campaign(path))
    assert marker.read_text() == '[{"session_id":"prior-owned"}]'


def test_managed_pool_refuses_shared_kernel_and_closes_all_owned_sessions(tmp_path):
    import asyncio

    from evolutionary_extensions.execution.notebooks import ManagedKernelPool

    closed = []

    class Response:
        status = 201

        def __init__(self, ident):
            self.ident = ident

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return {"id": self.ident, "kernel": {"id": "shared-kernel"}}

    class HTTP:
        count = 0

        def post(self, *args, **kwargs):
            self.count += 1
            return Response(f"owned-{self.count}")

        def delete(self, url, **kwargs):
            closed.append(url)
            response = Response("")
            response.status = 204
            return response

    async def scenario():
        with pytest.raises(RuntimeError, match="independent kernels"):
            async with ManagedKernelPool(
                HTTP(), "http://server", {}, ["first", "second"], "python", tmp_path
            ):
                pytest.fail("Shared global RNG was admitted")

    asyncio.run(scenario())
    assert closed == [
        "http://server/api/sessions/owned-1",
        "http://server/api/sessions/owned-2",
    ]
    with pytest.raises(ValueError, match="distinct notebook"):
        ManagedKernelPool(
            HTTP(), "http://server", {}, ["same", "same"], "python", tmp_path
        )
