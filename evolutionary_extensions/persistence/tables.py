"""CSV evidence exports with structured values retained as JSON."""

import csv
import json
from pathlib import Path


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot export empty evidence")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )
