"""Reload every new neural checkpoint and compare independent inference outputs."""
from pathlib import Path
import argparse
import gc
import os
from types import SimpleNamespace
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
import common as c
import neural_calibration as calibration
import source_execution as source
from strict_history import transform


def compare(checkpoint, predictions_path, family, data, original_config, prepared, device,
            report, time_mode=None, counter='verified_checkpoints'):
    model = c.build(family, data, original_config, device, time_mode=time_mode)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['state_dict'], strict=True)
    arrays = c.predictions(model, prepared.test, data)
    with np.load(predictions_path) as saved:
        np.testing.assert_allclose(arrays['prediction'], saved['prediction'], rtol=1e-6, atol=1e-6)
        for key in ['target', 'mask', 'sample_id', 'entity', 'start_time']:
            np.testing.assert_array_equal(arrays[key], saved[key])
        maximum = float(np.max(np.abs(arrays['prediction'] - saved['prediction'])))
    report['maximum_absolute_prediction_difference'] = max(report['maximum_absolute_prediction_difference'], maximum)
    report[counter] = report.get(counter, 0) + 1
    del model


def prepare_test(family, data, original_config, windows, device):
    probe = c.build(family, data, original_config, device)
    split = c.prepare_timeimm_split(probe, windows, data.input_dims, len(data.feature_names))
    del probe
    return SimpleNamespace(test=split)


def encoder_checkpoints(report, device):
    for encoder in c.CONFIG['encoders']:
        for dataset in c.CONFIG['datasets']:
            for family in c.FAMILIES:
                _, _, original = c.source(dataset, family)
                data = c.load_timeimm(c.HERE / 'local_data' / encoder / dataset, c.DATASET_SPECS[dataset],
                    normalization='train_only', text_cache_name='frozen_encoder.pt',
                    text_preprocessing='layer_norm' if family == 'selective_ssm' else 'none')
                aligned = {name: getattr(data, name + '_windows') for name in ['train', 'val', 'test']}
                variants = dict(aligned=aligned,
                    shuffle={name: c.shuffle_text_values_within_entities(windows, data.window_info, data.text_modality, seed=9101 + i)
                             for i, (name, windows) in enumerate(aligned.items())},
                    time_only={name: c.zero_text_values(windows, data.text_modality) for name, windows in aligned.items()})
                folder = c.HERE / 'results/encoders' / encoder / c.audit.SLUGS[dataset] / family
                for condition, windows in variants.items():
                    prepared = prepare_test(family, data, original['config'], windows['test'], device)
                    for seed in c.SEEDS:
                        compare(folder / f'{condition}_seed{seed}.pt', folder / f'{condition}_seed{seed}_predictions.npz',
                                family, data, original['config'], prepared, device, report)
                    del prepared
                print('RELOAD encoder', encoder, dataset, family, report['verified_checkpoints'], flush=True)
                del data
                gc.collect()
                torch.cuda.empty_cache()


def calibration_checkpoints(report, device):
    settings = c.CONFIG['neural_calibration']
    out = c.HERE / 'results/neural_calibration'
    for world in settings['world_seeds']:
        records = torch.load(out / 'inputs' / f'world_{world}.pt', weights_only=True)
        for strength in settings['strengths']:
            for family in c.FAMILIES:
                _, _, original = c.source('EPA-Air', family)
                data = calibration.make_data(records, strength, family)
                aligned = {name: getattr(data, name + '_windows') for name in ['train', 'val', 'test']}
                shuffled = {name: calibration.transform(windows, data.window_info, data.text_modality, 'history_rotation')[0]
                            for name, windows in aligned.items()}
                variants = dict(aligned=aligned, shuffle=shuffled,
                    time_only={name: c.zero_text_values(windows, data.text_modality) for name, windows in aligned.items()})
                folder = out / f'world_{world}' / f'strength_{strength}' / family
                for condition, windows in variants.items():
                    prepared = prepare_test(family, data, original['config'], windows['test'], device)
                    for seed in c.SEEDS:
                        compare(folder / f'{condition}_seed{seed}.pt', folder / f'{condition}_seed{seed}_predictions.npz',
                                family, data, original['config'], prepared, device, report)
                    del prepared
                for condition, windows in [('fixed_shuffle', shuffled['test']),
                                            ('fixed_zero', variants['time_only']['test'])]:
                    prepared = prepare_test(family, data, original['config'], windows, device)
                    for seed in c.SEEDS:
                        compare(folder / f'aligned_seed{seed}.pt', folder / f'{condition}_seed{seed}_predictions.npz',
                                family, data, original['config'], prepared, device, report,
                                counter='fixed_evaluations_checked')
                    del prepared
                print('RELOAD calibration', world, strength, family, report['verified_checkpoints'], flush=True)
                del data
                gc.collect()
                torch.cuda.empty_cache()


def encoder_history_checkpoints(report, device):
    for encoder in c.CONFIG['encoders']:
        for dataset in c.CONFIG['datasets']:
            for family in c.FAMILIES:
                _, _, original = c.source(dataset, family)
                data = c.load_timeimm(c.HERE / 'local_data' / encoder / dataset,
                    c.DATASET_SPECS[dataset], normalization='train_only',
                    text_cache_name='frozen_encoder.pt',
                    text_preprocessing='layer_norm' if family == 'selective_ssm' else 'none')
                trained = c.HERE / 'results/encoders' / encoder / c.audit.SLUGS[dataset] / family
                folder = c.HERE / 'results/encoder_history' / encoder / c.audit.SLUGS[dataset] / family
                for condition in ['history_rotation', 'history_nearest_swap']:
                    windows, _, _ = transform(data.test_windows, data.window_info,
                                              data.text_modality, condition)
                    prepared = prepare_test(family, data, original['config'], windows, device)
                    for seed in c.SEEDS:
                        compare(trained / f'aligned_seed{seed}.pt',
                            folder / f'{condition}_seed{seed}_predictions.npz', family, data,
                            original['config'], prepared, device, report,
                            counter='fixed_evaluations_checked')
                    del prepared
                print('RELOAD encoder history', encoder, dataset, family,
                      report['fixed_evaluations_checked'], flush=True)
                del data
                gc.collect()
                torch.cuda.empty_cache()
    assert report['fixed_evaluations_checked'] == 180
    report['unique_referenced_checkpoints'] = 90


def source_checkpoints(report, device):
    out = c.HERE / 'results/source_execution'
    for dataset in c.CONFIG['datasets']:
        for family in c.FAMILIES:
            _, _, original = c.source(dataset, family)
            canonical = c.load_original(dataset, family)
            full = c.load_timeimm(c.PROJECT / 'external/Time-IMM/data' / dataset, c.DATASET_SPECS[dataset],
                normalization='official_full_entity', text_cache_name=canonical.text_cache_name,
                text_preprocessing=canonical.text_preprocessing)
            origin = source.cache_origin(canonical, dataset)
            membership, _ = source.retain_empty_training(canonical, dataset)
            variants = dict(full_entity_normalization=full, cache_text_origin=origin,
                            retain_empty_training_windows=membership)
            if family == 'immtsf_recavg_gradd':
                variants['released_mixed_units'] = canonical
            for variant, data in variants.items():
                windows = {name: getattr(data, name + '_windows') for name in ['train', 'val', 'test']}
                prepared = prepare_test(family, data, original['config'], windows['test'], device)
                folder = out / variant / c.audit.SLUGS[dataset] / family
                for seed in c.SEEDS:
                    compare(folder / f'aligned_seed{seed}.pt', folder / f'aligned_seed{seed}_predictions.npz',
                            family, data, original['config'], prepared, device, report,
                            time_mode='released_mixed_units' if variant == 'released_mixed_units' else None)
                del prepared
            print('RELOAD source', dataset, family, report['verified_checkpoints'], flush=True)
            del canonical, full, origin, membership
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['encoders', 'calibration', 'source', 'encoder_history', 'all'], default='all')
    phase = parser.parse_args().phase
    device = c.setup()
    for name, function, expected in [('encoders', encoder_checkpoints, 270),
                                      ('calibration', calibration_checkpoints, 675),
                                      ('source', source_checkpoints, 150),
                                      ('encoder_history', encoder_history_checkpoints, 0)]:
        if phase not in ['all', name]:
            continue
        if name == 'encoder_history' and not (c.HERE / 'results/encoder_history/COMPLETE.json').is_file():
            if phase == 'encoder_history':
                raise RuntimeError('Historical encoder evaluations have not completed')
            continue
        report = dict(phase=name, verified_checkpoints=0, maximum_absolute_prediction_difference=0.,
                      device='cuda', rtol=1e-6, atol=1e-6)
        function(report, device)
        assert report['verified_checkpoints'] == expected
        c.save(c.HERE / 'results/verification' / f'{name}_checkpoint_replay.json', report)
