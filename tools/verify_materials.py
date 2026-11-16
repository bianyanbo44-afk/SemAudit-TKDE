"""Verify released training files and the preserved execution sources."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true')
    args = parser.parse_args()
    manifests = ['EXECUTION_SOURCE_MANIFEST.json', 'TRAINING_FILE_MANIFEST.json']
    if args.data:
        manifests.append('BENCHMARK_DOWNLOAD_MANIFEST.json')
    counts = {}
    for name in manifests:
        rows = json.loads((ROOT / 'provenance' / name).read_text(encoding='utf-8'))
        for row in rows:
            path = ROOT / row['file']
            if not path.is_file() or sha(path) != row['sha256']:
                raise ValueError('Missing or changed file ' + row['file'])
        counts[name] = len(rows)
    followup = ROOT / 'experiments/review_followup_20261105/results'
    phases = dict(encoders=270, neural_calibration=675, source_execution=150)
    for phase, expected in phases.items():
        actual = len(list((followup / phase).rglob('*_seed*.pt')))
        if actual != expected:
            raise ValueError(f'{phase} checkpoint count {actual}, expected {expected}')
    predictions = len(list(followup.rglob('*_predictions.npz')))
    if predictions != 2175:
        raise ValueError('Expected 2175 follow-up prediction sets')
    print(json.dumps(dict(verified_files=counts, followup_checkpoints=sum(phases.values()),
                          followup_prediction_sets=predictions, all_passed=True), indent=2))


if __name__ == '__main__':
    main()
