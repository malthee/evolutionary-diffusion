"""Official RAISE metadata and unchanged Synthbuster descriptions, CPU-only."""

import argparse
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else count
        count = min(count, self.size - self.position)
        if count <= 0:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{self.position + count - 1}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError(
                    "Official archive does not support ranges; refuse full 12 GB download"
                )
            result = response.read(count)
        if len(result) != count:
            raise ValueError("Incomplete archive range")
        self.position += count
        return result


def prepare_metadata(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raise_url = "https://loki.disi.unitn.it/RAISE/getFile.php?p=1k"
    with urllib.request.urlopen(raise_url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    (root / "RAISE_1k.csv").write_bytes(archive.read("RAISE_1k.csv"))
    zenodo = "https://zenodo.org/api/records/10066460"
    with urllib.request.urlopen(zenodo, timeout=60) as response:
        release = json.load(response)
    entry = next(f for f in release["files"] if f["key"] == "synthbuster.zip")
    with zipfile.ZipFile(RangeFile(entry["links"]["self"], entry["size"])) as archive:
        raw = archive.read("synthbuster/prompts.csv")
        (root / "Synthbuster-LICENCE.txt").write_bytes(
            archive.read("synthbuster/LICENCE.txt")
        )
    (root / "synthbuster-prompts.csv").write_bytes(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    with (root / "normalized-prompts.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=("image_id", "prompt"))
        writer.writeheader()
        writer.writerows(
            {
                "image_id": r["image name (matching Raise-1k)"].split(".")[0],
                "prompt": r["Prompt"],
            }
            for r in rows
        )
    atomic_json(
        root / "dataset_sources.json",
        {
            "raise_url": raise_url,
            "synthbuster_record": zenodo,
            "release_archive_checksum": entry["checksum"],
            "prompts": len(rows),
            "files": {
                p.name: hashes(p)
                for p in root.iterdir()
                if p.suffix in (".csv", ".txt")
            },
            "registration_submitted": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare_metadata(parser.parse_args().output)
