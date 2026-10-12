"""Camera-source selection, paired codecs and immutable image identities."""

import csv
import hashlib
import os
import random
import re
import urllib.request
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_photos(rows, test_count=100, calibration_count=200, seed=42):
    rows = sorted(rows, key=lambda row: row["File"])
    if any(not re.fullmatch(r"r[0-9a-z]+", row["File"]) for row in rows):
        raise ValueError("Invalid RAISE source filename")
    if len({r["File"] for r in rows}) != len(rows):
        raise ValueError("Duplicate RAISE source identities")
    if len(rows) < test_count + calibration_count:
        raise ValueError("Insufficient disjoint real photographs")
    indices = random.Random(seed).sample(
        range(len(rows)), test_count + calibration_count
    )
    return [
        {**rows[index], "group": "real" if i < test_count else "calibration"}
        for i, index in enumerate(indices)
    ]


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        temporary.open("wb") as out,
    ):
        while block := response.read(1024**2):
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(target)


def canonical_photo(path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        # TIFF decoding, orientation and 8-bit conversion are explicit provenance.
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = 512 / min(width, height)
        image = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
        )
        left, top = (image.width - 512) // 2, (image.height - 512) // 2
        return image.crop((left, top, left + 512, top + 512))


def write_variants(image, root, image_id, group, **metadata):
    root = Path(root)
    records = []
    for codec, extension in (("png", "png"), ("jpeg95", "jpg")):
        relative = Path("images") / codec / f"{image_id}.{extension}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if codec == "png":
            image.save(temporary, format="PNG")
        else:
            image.save(temporary, format="JPEG", quality=95, subsampling=0)
        temporary.replace(path)
        records.append(
            {
                "image_id": image_id,
                "group": group,
                "codec": codec,
                "path": relative.as_posix(),
                "sha256": hashes(path)["sha256"],
                **metadata,
            }
        )
    return records


def patch_coordinates(image_id, width=512, height=512, seed=42, count=64):
    """Torch RandomCrop coordinate semantics, independent of ordering and batching."""
    import torch

    key = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "big") % (2**63))
    coordinates = []
    for _ in range(count):
        top = int(torch.randint(0, height - 31, (), generator=generator))
        left = int(torch.randint(0, width - 31, (), generator=generator))
        coordinates.append((left, top))
    return coordinates


def simplest_patch(image, coordinates):
    import numpy as np

    def diversity(patch):
        a = np.asarray(patch, dtype=np.int64)
        return sum(
            np.abs(x - y).sum()
            for x, y in (
                (a[:, :-1], a[:, 1:]),
                (a[:-1], a[1:]),
                (a[:-1, :-1], a[1:, 1:]),
                (a[1:, :-1], a[:-1, 1:]),
            )
        )

    patches = [image.crop((x, y, x + 32, y + 32)) for x, y in coordinates]
    index = min(range(len(patches)), key=lambda i: diversity(patches[i]))
    return patches[index], index


def prepare_real(config):
    root = Path(config["output"])
    selected = select_photos(
        read_csv(config["raise_csv"]),
        config["test_count"],
        config["calibration_count"],
        config["sample_seed"],
    )
    selection_path = root / "selection.json"
    if selection_path.exists():
        import json

        if json.loads(selection_path.read_text()) != selected:
            raise ValueError("Frozen photo selection changed")
    atomic_json(selection_path, selected)
    result = []
    for row in selected:
        source = root / "originals" / f"{row['File']}.tif"
        if not source.exists():
            download(row["TIFF"], source)
        result.extend(
            write_variants(
                canonical_photo(source),
                root,
                row["File"],
                row["group"],
                pair_id=row["File"],
                source_url=row["TIFF"],
                source_sha256=hashes(source)["sha256"],
            )
        )
    atomic_json(root / "real_images.json", result)
    return result
