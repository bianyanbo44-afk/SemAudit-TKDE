"""Cross full-grid encoder replacements with strict historical interventions."""
import gc
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
import common as c
from strict_history import transform
from analyze_followup import gain, load_array, paired_invariants, save


def main():
    assert (c.HERE / 'results/ALL_EXPERIMENTS_VERIFIED.json').is_file()
    out = c.freeze('encoder_history', [__file__, c.HERE / 'strict_history.py'])
    device = c.setup()
    records = []
    summaries = []
    for encoder in c.CONFIG['encoders']:
        for dataset in c.CONFIG['datasets']:
            for family in c.FAMILIES:
                _, _, original = c.source(dataset, family)
                data = c.load_timeimm(c.HERE / 'local_data' / encoder / dataset,
                    c.DATASET_SPECS[dataset], normalization='train_only',
                    text_cache_name='frozen_encoder.pt',
                    text_preprocessing='layer_norm' if family == 'selective_ssm' else 'none')
                trained = c.HERE / 'results/encoders' / encoder / c.audit.SLUGS[dataset] / family
                folder = out / encoder / c.audit.SLUGS[dataset] / family
                folder.mkdir(parents=True, exist_ok=True)
                model = c.build(family, data, original['config'], device)
                for condition in ['history_rotation', 'history_nearest_swap']:
                    windows, mapping, diagnostics = transform(data.test_windows, data.window_info,
                                                             data.text_modality, condition)
                    c.save(folder / f'{condition}_mapping.json', mapping)
                    c.save(folder / f'{condition}_diagnostics.json', diagnostics)
                    packed = c.prepare_timeimm_split(model, windows, data.input_dims,
                                                    len(data.feature_names))
                    aligned_losses, perturbed_losses = [], []
                    for seed in c.SEEDS:
                        checkpoint = trained / f'aligned_seed{seed}.pt'
                        record_path = folder / f'{condition}_seed{seed}.json'
                        prediction_path = folder / f'{condition}_seed{seed}_predictions.npz'
                        aligned = load_array(trained / f'aligned_seed{seed}_predictions.npz')
                        if record_path.is_file():
                            record = c.read(record_path)
                            arrays = load_array(prediction_path)
                        else:
                            state = torch.load(checkpoint, map_location=device, weights_only=True)
                            model.load_state_dict(state['state_dict'], strict=True)
                            before = c.audit.state_digest(model)
                            arrays = c.predictions(model, packed, data)
                            assert c.audit.state_digest(model) == before
                            np.savez_compressed(prediction_path, **arrays)
                            record = dict(encoder=encoder, dataset=dataset, family=family,
                                condition=condition, seed=seed, mse=c.array_mse(arrays),
                                checkpoint_sha256=c.sha(checkpoint), diagnostics=diagnostics)
                            c.save(record_path, record)
                        paired_invariants(aligned, arrays)
                        aligned_losses.append(c.array_mse(aligned))
                        perturbed_losses.append(c.array_mse(arrays))
                        records.append(record)
                    summaries.append(dict(encoder=encoder, dataset=dataset, family=family,
                        condition=condition, n=3, fixed_gain=gain(aligned_losses, perturbed_losses),
                        aligned_mse=float(np.mean(aligned_losses)),
                        control_mse=float(np.mean(perturbed_losses)),
                        wins=int(np.sum(np.array(perturbed_losses) > aligned_losses)),
                        changed_fraction=diagnostics['changed_fraction'], donor_after_cutoff=0))
                    del packed
                print('ENCODER HISTORY', encoder, dataset, family, len(records), flush=True)
                del model, data
                gc.collect()
                torch.cuda.empty_cache()
    assert len(records) == 180
    c.save(out / 'manifest.json', records)
    save('encoder_history_summary', summaries)
    c.save(out / 'COMPLETE.json', dict(evaluations=len(records), checkpoints=90,
        all_donors_historical=True, classification='Exploratory crossover after full encoder results'))


if __name__ == '__main__':
    main()
