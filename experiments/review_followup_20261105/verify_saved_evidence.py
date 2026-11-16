"""Offline NumPy-only verification of every saved follow-up prediction set."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def load(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def mse(arrays):
    squared = (arrays['prediction'].astype(np.float64) - arrays['target'].astype(np.float64)) ** 2
    count = arrays['mask'].sum((0, 1))
    available = count > 0
    return float(((squared * arrays['mask']).sum((0, 1))[available] / count[available]).mean())


def main():
    files = sorted((ROOT / 'results').rglob('*_predictions.npz'))
    assert len(files) == 2175, len(files)
    checks = 0
    largest = 0.
    for path in files:
        arrays = load(path)
        assert np.isfinite(arrays['prediction']).all(), path
        assert arrays['prediction'].shape == arrays['target'].shape == arrays['mask'].shape
        assert len(arrays['sample_id']) == len(arrays['prediction'])
        expected_path = path.with_name(path.name.replace('_predictions.npz', '.json'))
        if expected_path.exists():
            expected = json.loads(expected_path.read_text(encoding='utf8'))
            expected_mse = expected['test_metrics']['mse'] if 'test_metrics' in expected else expected['mse']
            error = abs(mse(arrays) - expected_mse)
            largest = max(largest, error)
            assert error < 2e-5 * max(1., expected_mse), (path, error)
        for field in ['prediction', 'target', 'mask']:
            assert np.isfinite(arrays[field]).all(), (path, field)
        checks += 1
    # The 450 fixed-model calibration arrays are independently matched against
    # the same targets and masks as their selected aligned checkpoint.
    for path in (ROOT / 'results/neural_calibration').rglob('fixed_*_predictions.npz'):
        seed = path.name.split('_seed')[1].split('_')[0]
        reference = load(path.with_name(f'aligned_seed{seed}_predictions.npz'))
        arrays = load(path)
        for field in ['target', 'mask', 'sample_id', 'entity', 'start_time']:
            np.testing.assert_array_equal(reference[field], arrays[field])
    for path in (ROOT / 'results/encoder_history').rglob('*_predictions.npz'):
        relative = path.relative_to(ROOT / 'results/encoder_history')
        seed = path.name.split('_seed')[1].split('_')[0]
        reference = load(ROOT / 'results/encoders' / relative.parent / f'aligned_seed{seed}_predictions.npz')
        arrays = load(path)
        for field in ['target', 'mask', 'sample_id', 'entity', 'start_time']:
            np.testing.assert_array_equal(reference[field], arrays[field])
    manifest_path = ROOT / 'EVIDENCE_MANIFEST.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        for record in manifest:
            path = ROOT / record['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    result = dict(prediction_sets=checks, maximum_loss_replay_error=largest,
                  all_saved_values_finite=True, fixed_checkpoint_target_invariants=True,
                  scope='Saved-prediction and optional file-hash verification. Checkpoint reload records are supplied separately.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
