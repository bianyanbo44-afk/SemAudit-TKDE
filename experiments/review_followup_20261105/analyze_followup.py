"""Recompute all follow-up losses and paired summaries from saved predictions."""
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'analysis'
SEEDS = [53, 67, 79]
DATASETS = ['ClusterTrace', 'CESNET', 'EPA-Air', 'FNSPID', 'RepoHealth']
FAMILIES = ['immtsf_recavg_gradd', 'immtsf_t2v_xattn', 'selective_ssm']
SLUGS = dict(zip(DATASETS, ['clustertrace', 'cesnet', 'epa_air', 'fnspid', 'repohealth']))


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + '.json')).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf8')
    if isinstance(rows, list) and rows:
        with (OUT / (name + '.csv')).open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_array(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def mse(array):
    errors = (array['prediction'].astype(np.float64) - array['target'].astype(np.float64)) ** 2
    counts = array['mask'].sum((0, 1))
    available = counts > 0
    return float(((errors * array['mask']).sum((0, 1))[available] / counts[available]).mean())


def paired_invariants(reference, other):
    for key in ['sample_id', 'target', 'mask', 'entity', 'start_time']:
        np.testing.assert_array_equal(reference[key], other[key])


def gain(aligned, control):
    return 100 * (np.mean(control) - np.mean(aligned)) / np.mean(control)


def seed_summary(aligned, shuffle, timing):
    return dict(aligned=float(np.mean(aligned)), shuffle=float(np.mean(shuffle)),
                time_only=float(np.mean(timing)), aligned_sd=float(np.std(aligned, ddof=1)),
                sag=gain(aligned, shuffle), cbtg=gain(aligned, timing),
                wins_shuffle=int(np.sum(np.array(shuffle) > aligned)),
                wins_time=int(np.sum(np.array(timing) > aligned)), n=len(aligned))


def encoders():
    assert (RESULTS / 'encoders/COMPLETE.json').is_file()
    rows = []
    checks = 0
    maximum_error = 0.
    for encoder in ['minilm', 'finbert']:
        for dataset in DATASETS:
            for family in FAMILIES:
                folder = RESULTS / 'encoders' / encoder / SLUGS[dataset] / family
                losses = {condition: [] for condition in ['aligned', 'shuffle', 'time_only']}
                for seed in SEEDS:
                    initials = []
                    reference = None
                    for condition in losses:
                        record = read(folder / f'{condition}_seed{seed}.json')
                        array = load_array(folder / f'{condition}_seed{seed}_predictions.npz')
                        value = mse(array)
                        error = abs(value - record['test_metrics']['mse'])
                        assert error < 2e-5 * max(1., value)
                        maximum_error = max(maximum_error, error)
                        losses[condition].append(value)
                        initials.append(record['initial_state_sha256'])
                        if reference is None:
                            reference = array
                        else:
                            paired_invariants(reference, array)
                        checks += 1
                    assert len(set(initials)) == 1
                row = dict(encoder=encoder, dataset=dataset, family=family,
                           **seed_summary(losses['aligned'], losses['shuffle'], losses['time_only']))
                row['joint_support'] = bool(row['sag'] > 0 and row['cbtg'] > 0 and row['wins_shuffle'] >= 2 and row['wins_time'] >= 2)
                rows.append(row)
    save('encoder_summary', rows)
    save('encoder_verification', dict(prediction_cells=checks, maximum_mse_replay_error=maximum_error,
                                     all_initializations_paired=True, all_sample_target_mask_invariants=True))
    return rows


def historical():
    assert (RESULTS / 'strict_history/COMPLETE.json').is_file()
    rows = []
    for dataset in DATASETS:
        for family in FAMILIES:
            folder = RESULTS / 'strict_history' / SLUGS[dataset] / family
            for condition in ['history_rotation', 'history_nearest_swap']:
                values = []
                controls = []
                for seed in SEEDS + [83, 97, 101, 113, 127, 139, 151]:
                    a = load_array(folder / f'aligned_seed{seed}_predictions.npz')
                    b = load_array(folder / f'{condition}_seed{seed}_predictions.npz')
                    paired_invariants(a, b)
                    values.append(mse(a))
                    controls.append(mse(b))
                diagnostics = read(folder / f'{condition}_diagnostics.json')
                mapping = read(folder / f'{condition}_mapping.json')
                assert all(record['donor_arrival'] <= record['recipient_cutoff'] for record in mapping)
                assert diagnostics['donor_after_cutoff'] == 0
                rng = np.random.default_rng(20261105)
                index = rng.integers(0, 10, (10000, 10))
                a = np.array(values)
                b = np.array(controls)
                distribution = 100 * (b[index].mean(1) - a[index].mean(1)) / b[index].mean(1)
                low, high = np.quantile(distribution, [.025, .975])
                rows.append(dict(dataset=dataset, family=family, condition=condition, n=10,
                    aligned=float(a.mean()), altered=float(b.mean()), gain=gain(a, b),
                    gain_seed_low=float(low), gain_seed_high=float(high), seed_wins=int(np.sum(b > a)),
                    **diagnostics))
    save('strict_history_summary', rows)
    return rows


def entity_losses(array):
    per_query = np.mean((array['prediction'].astype(np.float64) - array['target'].astype(np.float64)) ** 2, axis=(1, 2))
    entities = sorted(set(array['entity']))
    return entities, np.array([per_query[array['entity'] == entity].mean() for entity in entities])


def holm(pvalues):
    values = np.array(pvalues)
    order = np.argsort(values)
    adjusted = np.zeros(len(values))
    highest = 0.
    for rank, index in enumerate(order):
        highest = max(highest, (len(values) - rank) * values[index])
        adjusted[index] = min(1., highest)
    return adjusted


def sign_test(differences):
    effective = np.abs(differences) > 1e-10
    nonzero = int(effective.sum())
    wins = int(np.sum(differences[effective] > 0))
    return float(stats.binomtest(wins, nonzero, .5, alternative='greater').pvalue) if nonzero else 1.


def neural_calibration():
    assert (RESULTS / 'neural_calibration/COMPLETE.json').is_file()
    config = read(ROOT / 'config.json')['neural_calibration']
    rows = []
    entity_rows = []
    maximum_error = 0.
    checks = 0
    rng = np.random.default_rng(20261105)
    for world in config['world_seeds']:
        for strength in config['strengths']:
            for family in FAMILIES:
                folder = RESULTS / 'neural_calibration' / f'world_{world}' / f'strength_{strength}' / family
                losses = {name: [] for name in ['aligned', 'shuffle', 'time_only', 'fixed_shuffle', 'fixed_zero']}
                validation = {name: [] for name in ['aligned', 'time_only']}
                for seed in SEEDS:
                    reference = load_array(folder / f'aligned_seed{seed}_predictions.npz')
                    initials = []
                    for condition in losses:
                        array = load_array(folder / f'{condition}_seed{seed}_predictions.npz')
                        paired_invariants(reference, array)
                        entities, values = entity_losses(array)
                        losses[condition].append(values)
                        if condition in ['aligned', 'shuffle', 'time_only']:
                            record = read(folder / f'{condition}_seed{seed}.json')
                            initials.append(record['initial_state_sha256'])
                            delta = abs(mse(array) - record['test_metrics']['mse'])
                            maximum_error = max(maximum_error, delta)
                            assert delta < 2e-5 * max(1., mse(array))
                            if condition in validation:
                                validation[condition].append(record['val_metrics']['mse'])
                        checks += 1
                    assert len(set(initials)) == 1
                averaged = {name: np.mean(values, axis=0) for name, values in losses.items()}
                delta = averaged['fixed_shuffle'] - averaged['aligned']
                zero_delta = averaged['fixed_zero'] - averaged['aligned']
                index = rng.integers(0, len(entities), (10000, len(entities)))
                bootstrap = 100 * (averaged['fixed_shuffle'][index].mean(1) - averaged['aligned'][index].mean(1)) / averaged['fixed_shuffle'][index].mean(1)
                low, high = np.quantile(bootstrap, [.025, .975])
                row = dict(world=world, strength=strength, family=family, entities=len(entities),
                    aligned=float(averaged['aligned'].mean()),
                    fixed_shuffle=float(averaged['fixed_shuffle'].mean()),
                    fixed_zero=float(averaged['fixed_zero'].mean()),
                    retrained_shuffle=float(averaged['shuffle'].mean()),
                    retrained_time_only=float(averaged['time_only'].mean()),
                    fixed_sag=gain(averaged['aligned'], averaged['fixed_shuffle']),
                    fixed_cbtg=gain(averaged['aligned'], averaged['fixed_zero']),
                    retrained_sag=gain(averaged['aligned'], averaged['shuffle']),
                    retrained_cbtg=gain(averaged['aligned'], averaged['time_only']),
                    fixed_sag_entity_low=float(low), fixed_sag_entity_high=float(high),
                    p_sag=sign_test(delta), p_cbtg=sign_test(zero_delta),
                    validation_capability_gain=gain(validation['aligned'], validation['time_only']))
                row['validation_capable'] = bool(row['validation_capability_gain'] > 1.)
                rows.append(row)
                for i, entity in enumerate(entities):
                    entity_rows.append(dict(world=world, strength=strength, family=family, entity=entity,
                                            **{name: float(values[i]) for name, values in averaged.items()}))
    for contrast in ['sag', 'cbtg']:
        adjusted = holm([row['p_' + contrast] for row in rows])
        for row, value in zip(rows, adjusted):
            row['holm_p_' + contrast] = float(value)
    for row in rows:
        row['nominal_detection'] = bool(row['p_sag'] < .05 and row['fixed_sag'] > 0)
        row['holm_detection'] = bool(row['holm_p_sag'] < .05 and row['fixed_sag'] > 0)
        row['screened_detection'] = bool(row['nominal_detection'] and row['validation_capable'])
    summaries = []
    for family in FAMILIES:
        for strength in config['strengths']:
            selected = [row for row in rows if row['family'] == family and row['strength'] == strength]
            summaries.append(dict(family=family, strength=strength, worlds=len(selected),
                mean_fixed_sag=float(np.mean([row['fixed_sag'] for row in selected])),
                min_fixed_sag=float(min(row['fixed_sag'] for row in selected)),
                max_fixed_sag=float(max(row['fixed_sag'] for row in selected)),
                mean_retrained_sag=float(np.mean([row['retrained_sag'] for row in selected])),
                nominal_detections=sum(row['nominal_detection'] for row in selected),
                holm_detections=sum(row['holm_detection'] for row in selected),
                validation_capable=sum(row['validation_capable'] for row in selected),
                screened_detections=sum(row['screened_detection'] for row in selected)))
    save('neural_calibration_worlds', rows)
    save('neural_calibration_entities', entity_rows)
    save('neural_calibration_summary', summaries)
    save('neural_calibration_verification', dict(prediction_cells=checks, maximum_mse_replay_error=maximum_error,
        all_initializations_paired=True, independent_entity_inference=True, holm_family_size=75))
    return summaries


def source_effects():
    assert (RESULTS / 'source_execution/COMPLETE.json').is_file()
    rows = []
    maximum_target_error = 0.
    for dataset in DATASETS:
        for family in FAMILIES:
            variants = ['full_entity_normalization', 'cache_text_origin', 'retain_empty_training_windows']
            if family == 'immtsf_recavg_gradd':
                variants.append('released_mixed_units')
            for variant in variants:
                reference_losses = []
                variant_losses = []
                own_losses = []
                for seed in SEEDS:
                    folder = RESULTS / 'source_execution' / variant / SLUGS[dataset] / family
                    record = read(folder / f'aligned_seed{seed}.json')
                    array = load_array(folder / f'aligned_seed{seed}_predictions.npz')
                    baseline = load_array(RESULTS / 'strict_history' / SLUGS[dataset] / family / f'aligned_seed{seed}_predictions.npz')
                    np.testing.assert_array_equal(array['sample_id'], baseline['sample_id'])
                    np.testing.assert_array_equal(array['mask'], baseline['mask'])
                    if variant == 'full_entity_normalization':
                        # Canonical scales are saved independently by the source
                        # runner so this NumPy replay never imports training code.
                        scales = read(RESULTS / 'source_execution' / 'canonical_scales' / f'{SLUGS[dataset]}_{family}.json')
                        new_scales = record['scales']
                        means = np.array([scales[entity]['mean'] for entity in array['entity']])[:, None, :]
                        stds = np.array([scales[entity]['std'] for entity in array['entity']])[:, None, :]
                        variant_means = np.array([new_scales[entity]['mean'] for entity in array['entity']])[:, None, :]
                        variant_stds = np.array([new_scales[entity]['std'] for entity in array['entity']])[:, None, :]
                        prediction = (array['prediction'] * variant_stds + variant_means - means) / stds
                        targets = (array['target'] * variant_stds + variant_means - means) / stds
                        observed = array['mask'] > 0
                        target_error = float(np.max(np.abs(targets[observed] - baseline['target'][observed])))
                        maximum_target_error = max(maximum_target_error, target_error)
                        np.testing.assert_allclose(targets[observed], baseline['target'][observed], rtol=2e-4, atol=2e-4)
                        canonical_array = dict(array, prediction=prediction, target=baseline['target'])
                    else:
                        paired_invariants(baseline, array)
                        canonical_array = array
                    reference_losses.append(mse(baseline))
                    variant_losses.append(mse(canonical_array))
                    own_losses.append(mse(array))
                    assert abs(own_losses[-1] - record['test_metrics']['mse']) < 2e-5 * max(1., own_losses[-1])
                change = 100 * (np.mean(variant_losses) - np.mean(reference_losses)) / np.mean(reference_losses)
                rows.append(dict(dataset=dataset, family=family, variant=variant, n=3,
                    canonical_mse=float(np.mean(reference_losses)), variant_canonical_scale_mse=float(np.mean(variant_losses)),
                    variant_own_scale_mse=float(np.mean(own_losses)), canonical_scale_change_percent=float(change),
                    paired_delta_sd=float(np.std(np.array(variant_losses) - reference_losses, ddof=1)),
                    variant_lower_loss_seeds=int(np.sum(np.array(variant_losses) < reference_losses)),
                    added_empty_training_windows=sum(record['added_empty_training_windows'].values())))
    save('source_effect_summary', rows)
    save('source_effect_verification', dict(cells=len(rows),paired_fits=150,maximum_canonical_target_roundtrip_error=maximum_target_error))
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['history', 'encoders', 'calibration', 'source', 'all'], default='all')
    phase = parser.parse_args().phase
    functions = {'history': historical, 'encoders': encoders, 'calibration': neural_calibration, 'source': source_effects}
    for name, function in functions.items():
        if phase in ['all', name]:
            rows = function()
            print(name, 'verified summary cells', len(rows), flush=True)
