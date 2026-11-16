"""Fetch pinned research inputs and safely restore versioned training artifacts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def download(row, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha(destination) == row['sha256']:
        return
    temporary = destination.with_name(destination.name + '.download')
    url = urllib.parse.quote(row['url'], safe='/:?=&')
    request = urllib.request.Request(url, headers={'User-Agent': 'SemAudit-TKDE-reproduction'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            if row.get('line_endings') == 'crlf':
                content = temporary.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                temporary.write_bytes(content)
            if sha(temporary) != row['sha256']:
                raise ValueError('SHA-256 mismatch for ' + str(destination))
            temporary.replace(destination)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def extract(path):
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            destination = (ROOT / item.filename).resolve()
            if not destination.is_relative_to(ROOT):
                raise ValueError('Archive member is outside the repository')
            if item.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError('Archive contains a symbolic link')
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        archive.extractall(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['training', 'data', 'encoders', 'all'], default='training')
    parser.add_argument('--asset-dir', type=Path, default=ROOT / '.downloads')
    parser.add_argument('--offline', action='store_true', help='Use already downloaded release ZIPs')
    args = parser.parse_args()
    if args.kind in ['training', 'all']:
        for row in read('release_manifest.json')['assets']:
            path = args.asset_dir / row['name']
            if args.offline:
                if not path.is_file() or sha(path) != row['sha256']:
                    raise ValueError('Missing or changed release asset ' + row['name'])
            else:
                download(row, path)
            extract(path)
            print('Restored ' + row['name'], flush=True)
    if args.kind in ['data', 'all']:
        rows = read('provenance/BENCHMARK_DOWNLOAD_MANIFEST.json')
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda row: download(row, ROOT / row['file']), rows))
        # Match the two encoder workspaces to the same pinned benchmark files.
        base = ROOT / 'runtime/phase3_prototype/external/Time-IMM/data'
        for encoder in ['minilm', 'finbert']:
            for source in base.rglob('*.csv'):
                target = ROOT / 'experiments/review_followup_20261105/local_data' / encoder / source.relative_to(base)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        print('Verified benchmark files ' + str(len(rows)), flush=True)
    if args.kind in ['encoders', 'all']:
        rows = read('provenance/ENCODER_DOWNLOAD_MANIFEST.json')
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda row: download(row, ROOT / row['destination']), rows))
        print('Verified pretrained encoder files ' + str(len(rows)), flush=True)


if __name__ == '__main__':
    main()
