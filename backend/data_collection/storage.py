from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def append_unique_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    key: str,
) -> int:
    """Append records to a JSONL file while avoiding duplicate keys.

    This is intentionally dependency-free so collection can run on a cheap
    CPU/VPS without pandas or a GPU.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if key in item:
                    seen.add(str(item[key]))

    added = 0
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            if key not in record:
                raise KeyError(f"record is missing deduplication key: {key}")
            record_key = str(record[key])
            if record_key in seen:
                continue
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            seen.add(record_key)
            added += 1

    return added
