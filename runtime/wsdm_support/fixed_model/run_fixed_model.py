"""WSDM model constructors and invariant checks for relocated experiment replay."""
from pathlib import Path
import hashlib
import torch
from eventmamba.forecasting import ResidualEventForecast
from eventmamba.immtsf_baseline import IMMTSFRecAvgGRAddBaseline, IMMTSFT2VXAttnBaseline
from eventmamba.model import EventMambaConfig
from eventmamba.selective import SelectiveEventMamba

ROOT = Path(__file__).resolve().parents[2] / 'phase3_prototype'
SLUGS = {'ClusterTrace': 'clustertrace', 'CESNET': 'cesnet', 'EPA-Air': 'epa_air',
         'FNSPID': 'fnspid', 'RepoHealth': 'repohealth'}


def sources_for(dataset, architecture):
    slug = SLUGS[dataset]
    if architecture == 'selective_ssm':
        relative = ('fnspid_untouched_gate/multimodal_models' if dataset == 'FNSPID'
                    else f'{slug}_multimodal_ln_fresh3_cuda')
        return ROOT / 'results' / relative, 'selective_eventmamba_no_reliability'
    if architecture == 'immtsf_recavg_gradd':
        return ROOT / 'results' / f'{slug}_immtsf_fusion_audit_fresh3', 'immtsf_recavg_gradd_aligned_unit_consistent'
    if architecture == 'immtsf_t2v_xattn':
        return ROOT / 'results' / f'{slug}_immtsf_xattn_audit_fresh3', 'immtsf_t2v_xattn_aligned'
    raise ValueError(architecture)


def build_model(architecture, data, config, device):
    numeric = tuple(range(len(data.feature_names)))
    parameters = dict(input_dims=data.input_dims, numeric_modalities=numeric,
        text_modality=int(data.text_modality), target_dim=len(data.feature_names), total_window=data.spec.total)
    if architecture == 'immtsf_recavg_gradd':
        return IMMTSFRecAvgGRAddBaseline(**parameters, time_mode='unit_consistent',
            d_txt=768, recency_sigma=1., dropout=.1).to(device)
    if architecture == 'immtsf_t2v_xattn':
        return IMMTSFT2VXAttnBaseline(**parameters, d_txt=128, n_heads=1, dropout=.1, kappa=1.).to(device)
    if architecture != 'selective_ssm':
        raise ValueError(architecture)
    settings = config['model']
    backbone = SelectiveEventMamba(EventMambaConfig(input_dims=data.input_dims,
        event_width=int(settings['event_width']), state_dim=int(settings['state_dim']),
        readout_width=int(settings['readout_width']), target_dim=len(data.feature_names), use_reliability=False))
    return ResidualEventForecast(backbone, data.input_dims, persistence_modalities=numeric).to(device)


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def verify_invariants(before, after, text_modality):
    assert len(before) == len(after)
    for original, altered in zip(before, after):
        for field in ['sample_id', 'cutoff']:
            assert getattr(original, field) == getattr(altered, field)
        for field in ['query_horizons', 'targets', 'target_masks']:
            assert torch.equal(getattr(original, field), getattr(altered, field)), field
        assert len(original.events) == len(altered.events)
        for old_event, new_event in zip(original.events, altered.events):
            for field in ['event_time', 'arrival_time', 'modality', 'quality', 'event_id']:
                assert getattr(old_event, field) == getattr(new_event, field), field
            if old_event.modality != text_modality:
                assert torch.equal(old_event.value, new_event.value)
