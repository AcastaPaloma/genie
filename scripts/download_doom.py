"""Download the exact pinned train/val/test subset. Uses only Python's stdlib."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name('doom_split.json')
    manifest = json.loads(manifest_path.read_text())

    def verify(path, entry):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')

    def download(entry):
        path = args.output / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = (f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/"
                   f"{manifest['revision']}/{entry['path']}?download=true")
            temporary = path.with_suffix('.array_record.partial')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as handle:
                shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
            verify(temporary, entry)
            temporary.replace(path)
        else:
            verify(path, entry)  # Reuse only matching files; never replace mismatched local data.
        return entry['path']

    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, path in enumerate(pool.map(download, manifest['files']), 1):
            print(f"{count}/{len(manifest['files'])} verified: {path}", flush=True)
    shutil.copyfile(manifest_path, args.output / 'download_manifest.json')
    print(f"Done: 42 train / 7 val / 7 test shards in {args.output.resolve()}")


if __name__ == '__main__':
    main()
