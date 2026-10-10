"""Bind reviewed exports to a prepared bundled Data app; never publish it."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json


def build_report(root, project, node, plugin):
    root, project, plugin = Path(root), Path(project), Path(plugin)
    existing = json.loads((project / "src/data.json").read_text())
    data = json.loads((root / "reports/reviewed-data.json").read_text())
    data["id"] = existing["id"]
    data["status"] = "ready"
    atomic_json(project / "src/data.json", data)
    content = Path(__file__).parent / "report_content"
    for source in content.iterdir():
        shutil.copy2(source, project / "src/content/report" / source.name)
    subprocess.run(
        [
            str(node),
            str(plugin / "scripts/data-app.mjs"),
            "build",
            "--project-dir",
            str(project),
            "--separate-data",
        ],
        check=True,
    )
    shutil.copytree(project / "dist", root / "reports", dirs_exist_ok=True)
    return root / "reports/index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "project", "node", "plugin"):
        parser.add_argument("--" + key, required=True)
    build_report(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
