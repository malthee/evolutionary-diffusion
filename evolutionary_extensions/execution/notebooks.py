"""Optional Jupyter execution helpers shared by evolutionary notebook adapters.

Requires the existing Jupyter environment (aiohttp/nbformat) and Python 3.11+.
The caller owns model/resource preflight and an external compute-stop deadline.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import aiohttp
import nbformat

from evolutionary_extensions.persistence.packaging import atomic_json


class ManagedKernelPool:
    """Independent server-owned processes; never share a kernel's global RNG."""

    def __init__(self, http, base_url, headers, paths, kernel_name, evidence):
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Require distinct notebook paths for independent kernels")
        self.http, self.base_url, self.headers = http, base_url, headers
        self.paths, self.kernel_name, self.evidence = paths, kernel_name, evidence
        self.runners = []

    async def __aenter__(self):
        try:
            for index, path in enumerate(self.paths):
                async with self.http.post(
                    f"{self.base_url}/api/sessions",
                    headers=self.headers,
                    json={
                        "path": path,
                        "name": f"Evolutionary runner {index + 1}",
                        "type": "notebook",
                        "kernel": {"name": self.kernel_name},
                    },
                ) as response:
                    if response.status != 201:
                        raise RuntimeError(
                            f"Managed session creation HTTP {response.status}"
                        )
                    session = await response.json()
                self.runners.append(
                    {
                        "session_id": session["id"],
                        "kernel_id": session["kernel"]["id"],
                        "notebook": path,
                    }
                )
                atomic_json(self.evidence / "managed_sessions.json", self.runners)
                if len({r["kernel_id"] for r in self.runners}) != len(self.runners):
                    raise RuntimeError(
                        "Server must create independent kernels per runner"
                    )
            return self.runners
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *args):
        failures = []
        for runner in self.runners:
            try:
                async with self.http.delete(
                    f"{self.base_url}/api/sessions/{runner['session_id']}",
                    headers=self.headers,
                ) as response:
                    if response.status not in (204, 404):
                        failures.append({**runner, "http_status": response.status})
            except (aiohttp.ClientError, OSError, TimeoutError) as error:
                failures.append({**runner, "error_type": type(error).__name__})
        atomic_json(
            self.evidence / "session_closed.json",
            {
                "sessions": self.runners,
                "failures": failures,
                "closed_unix": time.time(),
            },
        )
        if failures:
            raise RuntimeError("Owned managed session cleanup failed; inspect evidence")


async def execute_cells(
    http, base_url, headers, kernel_id, session_id, notebook, path, deadline
):
    async with http.ws_connect(
        f"{base_url}/api/kernels/{kernel_id}/channels",
        headers=headers,
        heartbeat=20,
        max_msg_size=32 * 1024**2,
    ) as ws:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            remaining = None if deadline is None else deadline - time.time() - 120
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Notebook deadline reached; preserve outputs")
            message_id = uuid.uuid4().hex
            await ws.send_json(
                {
                    "header": {
                        "msg_id": message_id,
                        "username": "experiment-controller",
                        "session": session_id,
                        "msg_type": "execute_request",
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "channel": "shell",
                    "buffers": [],
                    "content": {
                        "code": cell.source,
                        "silent": False,
                        "store_history": True,
                        "user_expressions": {},
                        "allow_stdin": False,
                        "stop_on_error": True,
                    },
                }
            )
            failed = False
            async with asyncio.timeout(remaining):
                while True:
                    event = await ws.receive()
                    if event.type != aiohttp.WSMsgType.TEXT:
                        raise ConnectionError("Managed kernel channel ended")
                    message = json.loads(event.data)
                    if message.get("parent_header", {}).get("msg_id") != message_id:
                        continue
                    content = message.get("content", {})
                    kind = message.get("msg_type") or message["header"]["msg_type"]
                    if kind == "execute_input":
                        cell.execution_count = content["execution_count"]
                    elif kind == "stream":
                        cell.outputs.append(nbformat.v4.new_output("stream", **content))
                        print(content["text"], end="", flush=True)
                    elif kind in {"display_data", "execute_result", "error"}:
                        fields = {
                            k: v
                            for k, v in content.items()
                            if k
                            in {
                                "data",
                                "metadata",
                                "execution_count",
                                "ename",
                                "evalue",
                                "traceback",
                            }
                        }
                        cell.outputs.append(nbformat.v4.new_output(kind, **fields))
                        if kind == "error":
                            failed = True
                            print("\n".join(content["traceback"]), flush=True)
                    nbformat.write(notebook, path)
                    if kind == "status" and content.get("execution_state") == "idle":
                        break
            if failed:
                raise RuntimeError(f"Notebook failed in cell {index}")
