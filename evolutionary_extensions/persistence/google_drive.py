"""Package completed experiments and resumably upload to an authorized Drive folder.

No login/scopes are changed. OAuth credentials and resumable URLs stay outside
experiment folders. Cleanup requires a live remote checksum verification and an
unchanged local snapshot within an explicitly allowed runner root.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://www.googleapis.com/drive/v3"
SCOPE = "https://www.googleapis.com/auth/drive.file"
CHUNK = 8 * 1024 * 1024


from .packaging import atomic_json, hashes, package, snapshot


class DriveClient:
    def __init__(self, credentials, *, deadline_unix=None):
        self.deadline_unix = deadline_unix
        path = Path(credentials)
        if path.stat().st_mode & 0o077:
            raise ValueError("OAuth credential file must have mode 0600")
        self.credentials = json.loads(path.read_text())
        if self.credentials.get("scope") != SCOPE:
            raise ValueError(
                "Use dedicated drive.file credentials; do not reuse/broaden the read-only grant"
            )
        self.token = None
        self.expires = 0

    def timeout(self, seconds):
        if self.deadline_unix is None:
            return seconds
        remaining = self.deadline_unix - time.time()
        if remaining <= 0:
            raise TimeoutError("Upload deadline reached; retain local outputs")
        return min(seconds, remaining)

    def access_token(self):
        if time.time() >= self.expires:
            payload = urllib.parse.urlencode(
                {
                    "client_id": self.credentials["client_id"],
                    "client_secret": self.credentials["client_secret"],
                    "refresh_token": self.credentials["refresh_token"],
                    "grant_type": "refresh_token",
                }
            ).encode()
            request = urllib.request.Request(
                "https://oauth2.googleapis.com/token", data=payload
            )
            with urllib.request.urlopen(request, timeout=self.timeout(60)) as response:
                token = json.load(response)
            self.token = token["access_token"]
            self.expires = time.time() + token.get("expires_in", 3600) - 60
        return self.token

    def request(self, url, *, method="GET", data=None, headers=None):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "www.googleapis.com":
            raise ValueError(
                "Unexpected upload endpoint; refuse sending OAuth credentials"
            )
        request = urllib.request.Request(
            url,
            method=method,
            data=data,
            headers={
                "Authorization": f"Bearer {self.access_token()}",
                **(headers or {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout(180)) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            if error.code in {308, 404, 409, 429, 500, 502, 503, 504}:
                return error.code, dict(error.headers), error.read()
            raise RuntimeError(
                f"Drive request failed with HTTP {error.code}; credentials/output not printed"
            ) from None

    def metadata(self, file_id):
        status, _, body = self.request(
            f"{API}/files/{urllib.parse.quote(file_id, safe='')}?fields=id,name,mimeType,size,md5Checksum,parents,trashed,webViewLink&supportsAllDrives=true"
        )
        return json.loads(body) if status == 200 else None

    def check_account(self):
        status, _, body = self.request(f"{API}/about?fields=user(emailAddress)")
        if (
            status != 200
            or json.loads(body).get("user", {}).get("emailAddress")
            != self.credentials["expected_account"]
        ):
            raise ValueError(
                "Authenticated account differs from the configured uploader account"
            )


def verify_remote(client, state, manifest):
    remote = client.metadata(state["file_id"])
    if (
        not remote
        or remote.get("trashed")
        or int(remote.get("size", -1)) != manifest["bytes"]
        or remote.get("md5Checksum") != manifest["md5"]
        or state["folder_id"] not in remote.get("parents", [])
    ):
        raise RuntimeError("Remote archive verification failed; local results retained")
    return remote


def remote_offset(headers):
    value = headers.get("Range", headers.get("range", ""))
    return int(value.rsplit("-", 1)[1]) + 1 if value else 0


def upload(archive, credentials, folder_id, state_file, *, client=None):
    archive = Path(archive).resolve(strict=True)
    manifest = json.loads(Path(str(archive) + ".manifest.json").read_text())
    if hashes(archive) != {key: manifest[key] for key in ["bytes", "md5", "sha256"]}:
        raise ValueError("Archive changed since packaging")
    state_path = Path(state_file).resolve()
    if state_path.is_relative_to(Path(manifest["experiment"])):
        raise ValueError("Resumable state must be outside the archived experiment")
    client = client or DriveClient(credentials)
    client.check_account()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if (
            state.get("sha256") != manifest["sha256"]
            or state.get("folder_id") != folder_id
        ):
            raise ValueError("Upload state belongs to a different archive/destination")
    else:
        status, _, body = client.request(
            f"{API}/files/generateIds?count=1&space=drive&type=files"
        )
        if status != 200:
            raise RuntimeError("Unable to allocate an idempotent Drive file ID")
        state = {
            "file_id": json.loads(body)["ids"][0],
            "folder_id": folder_id,
            "sha256": manifest["sha256"],
        }
        atomic_json(state_path, state, private=True)
    if state.get("verified"):
        return verify_remote(client, state, manifest)
    existing = client.metadata(state["file_id"])
    if existing and existing.get("md5Checksum"):
        remote = verify_remote(client, state, manifest)
        state.update(verified=True, webViewLink=remote.get("webViewLink"))
        state.pop("session_url", None)
        atomic_json(state_path, state, private=True)
        return remote
    if "session_url" not in state:
        payload = json.dumps(
            {
                "id": state["file_id"],
                "name": archive.name,
                "parents": [folder_id],
                "appProperties": {"archive_sha256": manifest["sha256"]},
            }
        ).encode()
        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true"
        method = "POST"
        if existing:
            # An expired session may have left an empty file; replace its content
            # using the persisted ID instead of creating a duplicate.
            url = f"https://www.googleapis.com/upload/drive/v3/files/{urllib.parse.quote(state['file_id'], safe='')}?uploadType=resumable&supportsAllDrives=true"
            method = "PATCH"
            payload = json.dumps(
                {
                    "name": archive.name,
                    "appProperties": {"archive_sha256": manifest["sha256"]},
                }
            ).encode()
        status, headers, _ = client.request(
            url,
            method=method,
            data=payload,
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "application/zip",
                "X-Upload-Content-Length": str(manifest["bytes"]),
            },
        )
        if status not in {200, 201}:
            raise RuntimeError(
                "Unable to create resumable upload; retain state and retry"
            )
        state["session_url"] = headers.get("Location", headers.get("location"))
        atomic_json(state_path, state, private=True)
    failures = 0
    with archive.open("rb") as handle:
        while failures < 6:
            try:
                status, headers, _ = client.request(
                    state["session_url"],
                    method="PUT",
                    data=b"",
                    headers={
                        "Content-Length": "0",
                        "Content-Range": f"bytes */{manifest['bytes']}",
                    },
                )
                if status in {200, 201}:
                    break
                if status == 404:
                    state.pop("session_url", None)
                    atomic_json(state_path, state, private=True)
                    raise RuntimeError(
                        "Upload session expired; rerun to resume using the same file ID"
                    )
                if status != 308:
                    raise OSError(f"Upload status HTTP {status}")
                offset = remote_offset(headers)
                if not 0 <= offset < manifest["bytes"]:
                    raise RuntimeError("Invalid remote byte range")
                handle.seek(offset)
                chunk = handle.read(CHUNK)
                status, _, _ = client.request(
                    state["session_url"],
                    method="PUT",
                    data=chunk,
                    headers={
                        "Content-Type": "application/zip",
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{offset + len(chunk) - 1}/{manifest['bytes']}",
                    },
                )
                if status in {200, 201}:
                    break
                if status != 308:
                    raise OSError(f"Upload chunk HTTP {status}")
                failures = 0
            except (OSError, urllib.error.URLError):
                failures += 1
                time.sleep(min(30, 2**failures))
        else:
            raise RuntimeError(
                "Resumable upload retry limit reached; retain all local outputs"
            )
    remote = verify_remote(client, state, manifest)
    state.update(
        verified=True, webViewLink=remote.get("webViewLink"), verified_unix=time.time()
    )
    state.pop("session_url", None)
    atomic_json(state_path, state, private=True)
    return remote


def cleanup(experiment, archive, state_file, runner_root, client):
    experiment = Path(experiment).resolve(strict=True)
    allowed = Path(runner_root).resolve(strict=True)
    manifest = json.loads(Path(str(archive) + ".manifest.json").read_text())
    state = json.loads(Path(state_file).read_text())
    if (
        experiment == allowed
        or not experiment.is_relative_to(allowed)
        or str(experiment) != manifest["experiment"]
    ):
        raise ValueError(
            "Cleanup target must match the packaged experiment inside the allowed runner root"
        )
    if not state.get("verified") or state.get("sha256") != manifest["sha256"]:
        raise ValueError("No verified upload receipt; retain experiment")
    verify_remote(client, state, manifest)
    if snapshot(experiment) != manifest["files"]:
        raise ValueError("Experiment changed since packaging; refuse cleanup")
    shutil.rmtree(experiment)
    state["runner_directory_deleted_unix"] = time.time()
    atomic_json(state_file, state, private=True)


def verify_download(client, file_id, manifest):
    """Hash a Drive download incrementally; retain no duplicate archive in memory."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise RuntimeError("Unexpected download redirect; retain local outputs")

    url = f"{API}/files/{urllib.parse.quote(file_id, safe='')}?alt=media&supportsAllDrives=true"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {client.access_token()}"}
    )
    digest, size = hashlib.sha256(), 0
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=client.timeout(180)) as response:
        while True:
            client.timeout(180)
            block = response.read(CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    if size != manifest["bytes"] or digest.hexdigest() != manifest["sha256"]:
        raise RuntimeError("Downloaded SHA-256 differs; retain all local outputs")
    return digest.hexdigest()


class GoogleDrivePersistor:
    """Optional completed-run persistence, with explicit independently verified cleanup."""

    def __init__(
        self,
        credentials,
        folder_id,
        archive_root,
        receipt_root,
        runner_root,
        *,
        deadline_unix=None,
        client=None,
    ):
        self.credentials = Path(credentials).expanduser()
        self.folder_id = folder_id
        self.archive_root = Path(archive_root).expanduser().resolve()
        self.receipt_root = Path(receipt_root).expanduser().resolve()
        self.runner_root = Path(runner_root).expanduser().resolve()
        self.client = client or DriveClient(
            self.credentials, deadline_unix=deadline_unix
        )

    def check_destination(self):
        self.client.check_account()
        folder = self.client.metadata(self.folder_id)
        if (
            not folder
            or folder.get("trashed")
            or folder.get("mimeType") != "application/vnd.google-apps.folder"
        ):
            raise ValueError("Destination is not an accessible Drive folder")

    def package(self, experiment):
        experiment = Path(experiment).resolve(strict=True)
        archive = self.archive_root / f"{experiment.name}.zip"
        manifest_path = Path(str(archive) + ".manifest.json")
        if archive.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if snapshot(experiment) != manifest["files"]:
                raise ValueError("Completed run changed; refuse reuse of archive")
        else:
            manifest = package(experiment, archive)
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
        return archive, manifest

    def upload(self, archive):
        state_path = self.receipt_root / f"{Path(archive).stem}.json"
        remote = upload(
            archive, self.credentials, self.folder_id, state_path, client=self.client
        )
        manifest = json.loads(Path(str(archive) + ".manifest.json").read_text())
        sha = verify_download(self.client, remote["id"], manifest)
        state = json.loads(state_path.read_text())
        state.update(download_sha256=sha, downloaded_verified_unix=time.time())
        atomic_json(state_path, state, private=True)
        return state_path, remote

    def cleanup(self, experiment, archive, state_path):
        manifest = json.loads(Path(str(archive) + ".manifest.json").read_text())
        state = json.loads(Path(state_path).read_text())
        if state.get("download_sha256") != manifest["sha256"]:
            raise ValueError("No verified download; retain local outputs")
        cleanup(experiment, archive, state_path, self.runner_root, self.client)
        Path(archive).unlink()

    def persist(self, experiment, *, delete_after_verification=False, on_stage=None):
        self.check_destination()
        started = time.perf_counter()
        if on_stage:
            on_stage("packaging")
        archive, manifest = self.package(experiment)
        packaged = time.perf_counter()
        if on_stage:
            on_stage("uploading_and_verifying")
        state_path, remote = self.upload(archive)
        uploaded = time.perf_counter()
        if delete_after_verification:
            if on_stage:
                on_stage("verified_cleanup")
            self.cleanup(experiment, archive, state_path)
        finished = time.perf_counter()
        return {
            "file_id": remote["id"],
            "webViewLink": remote.get("webViewLink"),
            "bytes": manifest["bytes"],
            "sha256": manifest["sha256"],
            "md5": manifest["md5"],
            "transfer_seconds": finished - started,
            "packaging_seconds": packaged - started,
            "upload_verify_seconds": uploaded - packaged,
            "cleanup_seconds": finished - uploaded,
            "verified": True,
            "runner_copies_deleted": delete_after_verification,
            "receipt_path": str(state_path),
        }
