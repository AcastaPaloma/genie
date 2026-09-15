"""Restore the pinned official NeuroMechFly browser assets, verifying every hash.

Run from the repository root: .venv/bin/python ui/scripts/prepare_body.py
The manifest is committed in public/body/provenance.json; no live branch is used.
"""
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request

import certifi

OUT = Path(__file__).resolve().parents[1] / "public" / "body"


def main():
    manifest = json.loads((OUT / "provenance.json").read_text())
    revision = manifest["revision"]
    base = f"https://raw.githubusercontent.com/NeLy-EPFL/flygym/{revision}/"
    context = ssl.create_default_context(cafile=certifi.where())
    for entry in manifest["files"]:
        target = OUT / entry["file"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
            continue
        with urllib.request.urlopen(base + entry["source"], context=context, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {entry['file']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"Restored {entry['file']}")
    print(f"Verified {len(manifest['files'])} official assets at {revision}.")


if __name__ == "__main__":
    main()
