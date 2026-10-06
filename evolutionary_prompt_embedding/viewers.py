"""Local notebook viewers. No viewer dependency is needed to write archives."""

import hashlib
import json
import mimetypes
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

from evolutionary_prompt_embedding.archive import sha256_file

DECK_VERSION = "9.4.0"
DECK_SHA256 = "2eb6a1ae0d58604b1378682cd1136f8793478ba801e43dae48b3807e48758a6b"


def prepare_deck_asset(cache_dir):
    """Download once, verify its pinned digest, then serve entirely locally."""
    path = Path(cache_dir) / f"deck.gl-{DECK_VERSION}.min.js"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(
            f"https://unpkg.com/deck.gl@{DECK_VERSION}/dist.min.js", timeout=60
        ) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != DECK_SHA256:
            raise ValueError("deck.gl asset checksum mismatch")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    if sha256_file(path) != DECK_SHA256:
        raise ValueError("Cached deck.gl asset checksum mismatch")
    return path


class LocalViewerServer:
    """Serve only registered assets/images on loopback; close when finished."""

    def __init__(self, archives):
        self.archives = {str(a.run_dir): a for a in archives}
        self.routes = {}
        self.token = uuid.uuid4().hex
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                entry = owner.routes.get(self.path.split("?", 1)[0])
                if entry is None:
                    self.send_error(404)
                    return
                content, mime = entry
                try:
                    if isinstance(content, Path):
                        content = content.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(content)
                except FileNotFoundError:
                    self.send_error(404)
                except BrokenPipeError:
                    pass

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"

    def register(self, content, mime):
        route = f"/{self.token}/{len(self.routes)}"
        self.routes[route] = (content, mime)
        return self.base_url + route

    def image_urls(self, row):
        archive = self.archives[str(row.archive_dir)]
        urls = []
        for relative in row.image_paths:
            path = archive.resolve_path(relative)
            if path.is_file():
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                urls.append(self.register(path, mime))
            else:
                urls.append(None)
        return urls

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.routes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def atlas_table(projection, server):
    """Adapt IDs and neighbours to Atlas's documented dictionary format."""
    frame = projection.copy()
    included = set(frame.record_id)
    frame["neighbors"] = [
        {
            "ids": [rid for rid in ids if rid in included],
            "distances": [
                float(d) for rid, d in zip(ids, distances) if rid in included
            ],
        }
        for ids, distances in zip(frame.neighbors, frame.neighbor_distances)
    ]
    frame["image_urls"] = [server.image_urls(row) for row in frame.itertuples()]
    frame["image"] = [
        next((url for url in urls if url), None) for urls in frame.image_urls
    ]
    frame["image_status"] = [
        "No images"
        if len(paths) == 0
        else ("Available" if all(urls) else "Some images missing")
        for paths, urls in zip(frame.image_paths, frame.image_urls)
    ]
    # Atlas gets metadata and coordinates, never the raw tensor arrays.
    # File locations and duplicate JSON encodings are available through notebook
    # inspection; they should not overwhelm the viewer's automatic charts.
    frame = frame.drop(
        columns=[
            "archive_dir",
            "tensor_file",
            "tensor_row",
            "image_paths",
            "metadata_json",
            "fitness_json",
            "neighbor_distances",
            "projection_key",
        ]
    )
    leading = [
        key
        for key in (
            "generation",
            "island_id",
            "candidate_slot",
            "fitness",
            "prompt",
            "category",
            "island_description",
            "image",
            "image_status",
        )
        if key in frame
    ]
    return frame[leading + [key for key in frame if key not in leading]]


def show_atlas(projection, server):
    from embedding_atlas.widget import EmbeddingAtlasWidget

    frame = atlas_table(projection, server)
    return EmbeddingAtlasWidget(
        frame,
        row_id="record_id",
        x="x",
        y="y",
        neighbors="neighbors",
        image="image",
        text="prompt" if "prompt" in frame.columns else None,
        labels="disabled",
        show_table=True,
        show_charts=True,
    )


def show_gallery(projection, server, record_id):
    """Inspect all images for an Atlas selection, including missing-file markers."""
    import html

    from IPython.display import HTML

    matches = projection[projection.record_id == record_id]
    if matches.empty:
        raise KeyError(record_id)
    row = next(matches.itertuples())
    urls = server.image_urls(row)
    parts = [f"<h4>{html.escape(record_id)}</h4>"]
    if not urls:
        parts.append("<p>No images accompany this embedding.</p>")
    for i, url in enumerate(urls):
        parts.append(
            f'<img src="{url}" alt="Candidate image {i}" style="max-width:320px">'
            if url
            else f"<p>Image {i}: file missing</p>"
        )
    return HTML("".join(parts))


def _thumbnail_selection(frame, limit):
    eligible = frame[frame.image_paths.map(len) > 0].sort_values("record_id")
    # Round-robin groups keep generations/islands represented deterministically.
    groups = [
        group.index.tolist()
        for _, group in eligible.groupby(
            ["run_id", "generation", "island_id"], dropna=False, sort=True
        )
    ]
    selected = []
    depth = 0
    while groups and len(selected) < limit:
        remaining = []
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                remaining.append(group)
                if len(selected) == limit:
                    break
        groups = remaining
        depth += 1
    return selected


def show_3d(projection, server, cache_dir=None, thumbnail_limit=2000):
    """All points remain selectable; bounded texture pages add image markers."""
    from IPython.display import IFrame

    if "z" not in projection:
        raise ValueError("Compute a three-dimensional projection first")
    if not 0 <= thumbnail_limit <= 2000:
        raise ValueError("Thumbnail limit must be between 0 and 2000")
    cache = Path(
        cache_dir or next(iter(server.archives.values())).run_dir / "analysis_cache"
    )
    asset = prepare_deck_asset(cache)
    asset_url = server.register(asset, "text/javascript")
    frame = projection.reset_index(drop=True)
    coordinates = frame[["x", "y", "z"]].to_numpy(dtype=float)
    center = coordinates.mean(axis=0)
    scale = max(float(np.abs(coordinates - center).max()), 1e-12)
    coordinates = (coordinates - center) / scale * 100
    entries = []
    for i, row in enumerate(frame.itertuples()):
        archive = server.archives[str(row.archive_dir)]
        entries.append(
            {
                "record_id": row.record_id,
                "position": coordinates[i].tolist(),
                "generation": int(row.generation),
                "island": str(row.island_id),
                "fitness": None
                if row.fitness is None or np.isnan(row.fitness)
                else float(row.fitness),
                "fitness_original": json.loads(row.fitness_json),
                "objective_names": archive.manifest["run_metadata"].get(
                    "objective_names", []
                ),
                "images": server.image_urls(row),
                "image_count": len(row.image_paths),
                "metadata": json.loads(row.metadata_json),
                "tensor_spec": archive.manifest["tensor_spec"],
                "neighbors": list(row.neighbors),
                "neighbor_distances": list(row.neighbor_distances),
                "neighbor_space": row.neighbor_space,
            }
        )
    pages = []
    selected = _thumbnail_selection(frame, thumbnail_limit)
    icons = []
    for index in selected:
        row = frame.iloc[index]
        archive = server.archives[str(row.archive_dir)]
        if not len(row.image_paths):
            continue
        path = archive.resolve_path(row.image_paths[0])
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                icons.append((index, ImageOps.fit(image.convert("RGBA"), (64, 64))))
        except OSError:
            continue
    for start in range(0, len(icons), 1024):
        group = icons[start : start + 1024]
        canvas = Image.new("RGBA", (2048, 2048))
        mapping = {}
        points = []
        for slot, (index, image) in enumerate(group):
            x, y = slot % 32 * 64, slot // 32 * 64
            canvas.paste(image, (x, y))
            mapping[str(index)] = {
                "x": x,
                "y": y,
                "width": 64,
                "height": 64,
                "mask": False,
            }
            points.append(dict(entries[index], icon=str(index)))
        import io

        output = io.BytesIO()
        canvas.save(output, format="PNG")
        pages.append(
            {
                "image": server.register(output.getvalue(), "image/png"),
                "mapping": mapping,
                "points": points,
            }
        )
    payload = {"points": entries, "pages": pages, "thumbnail_count": len(icons)}
    data_url = server.register(
        json.dumps(payload, allow_nan=False).encode(), "application/json"
    )
    template = (
        files("evolutionary_prompt_embedding")
        .joinpath("assets/viewer_3d.html")
        .read_text(encoding="utf-8")
    )
    page = template.replace("__DECK_URL__", asset_url).replace("__DATA_URL__", data_url)
    url = server.register(page.encode(), "text/html")
    return IFrame(url, width="100%", height=720)
