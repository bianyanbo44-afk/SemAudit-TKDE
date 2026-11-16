"""Portable path binding and unchanged training/evaluation primitives."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE.parent
PROJECT=Path(os.environ.get('SEMAUDIT_PROJECT_ROOT',ARCHIVE/'03_原始代码_数据_训练结果/phase3_prototype')).resolve()
EXTENSION=Path(os.environ.get('SEMAUDIT_EXTENSION_ROOT',ARCHIVE/'05_扩展实验与期刊草稿/journal_extension_20261012')).resolve()
PREVIOUS=ARCHIVE/'02_新增实验与数据/SemAudit_added_experiments_20261104'
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
spec=importlib.util.spec_from_file_location('pinned_fixed_audit',EXTENSION/'fixed_model/run_fixed_model.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.ROOT=PROJECT
from eventmamba.batching import index_packed_grouped
from eventmamba.timeimm import load_timeimm,DATASET_SPECS,GPT2_TEXT_CACHE,GPT2_TEXT_CACHE_1024
from eventmamba.timeimm_training import PreparedTimeIMMData,prepare_timeimm_split,train_timeimm_model
from eventmamba.multimodal import shuffle_text_values_within_entities,zero_text_values
from eventmamba.training import set_deterministic_seed,append_jsonl

FAMILIES=['immtsf_recavg_gradd','immtsf_t2v_xattn','selective_ssm']
SEEDS=[53,67,79]
ALL_SEEDS=SEEDS+[83,97,101,113,127,139,151]
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf8'))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');temp.replace(p)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def freeze(phase,files):
    out=HERE/'results'/phase;out.mkdir(parents=True,exist_ok=True)
    identity=dict(phase=phase,classification='Exploratory post-review experiments; original test sets previously inspected.',
        utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,
        source_hashes={str(Path(p).relative_to(HERE)):sha(p) for p in [HERE/'config.json',HERE/'common.py',*files]},
        torch=torch.__version__,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    p=out/'FROZEN_PROTOCOL.json'
    if p.exists():
        old=read(p)
        if old['source_hashes']!=identity['source_hashes']:
            corrections=out/'EXECUTION_SOURCE_UPDATES.jsonl'
            with corrections.open('a',encoding='utf8') as stream:stream.write(json.dumps(identity,ensure_ascii=False)+'\n')
    else:save(p,identity)
    return out
def setup():
    torch.set_num_threads(4)
    return torch.device('cuda')
def source(dataset,family):
    folder,name=audit.sources_for(dataset,family)
    r=read(folder/'results.json')
    return folder,name,r
def training(r):return dict(r.get('effective_training',r['config']['training']))
def build(family,data,config,device,time_mode=None):
    width=data.input_dims[data.text_modality]
    proxy=replace(data,text_embedding_dim=768) if width!=768 and family!='selective_ssm' else data
    model=audit.build_model(family,proxy,config,device)
    if family!='selective_ssm' and width!=768:
        model.input_dims=data.input_dims
        model.ttf.input_proj=torch.nn.Linear(width,model.ttf.d_txt).to(device)
    if time_mode is not None:
        assert family=='immtsf_recavg_gradd';model.time_mode=time_mode
    # An all-masked attention row otherwise produces NaN gradients before the
    # upstream presence mask zeroes its result. Nonempty rows are unchanged.
    def finite_empty_keys(module,args,kwargs):
        mask=kwargs.get('key_padding_mask')
        if mask is not None and mask.dtype==torch.bool and mask.shape[1] and bool(mask.all(1).any()):
            mask=mask.clone();mask[mask.all(1),0]=False;kwargs={**kwargs,'key_padding_mask':mask}
        return args,kwargs
    for module in model.modules():
        if isinstance(module,torch.nn.MultiheadAttention):module.register_forward_pre_hook(finite_empty_keys,with_kwargs=True)
    return model
def load_original(dataset,family):
    cache=GPT2_TEXT_CACHE_1024 if dataset=='RepoHealth' else GPT2_TEXT_CACHE
    return load_timeimm(PROJECT/'external/Time-IMM/data'/dataset,DATASET_SPECS[dataset],normalization='train_only',text_cache_name=cache,
                        text_preprocessing='layer_norm' if family=='selective_ssm' else 'none')
@torch.inference_mode()
def predictions(model,split,data,batch_size=32):
    model.eval();pieces=[]
    for start in range(0,len(split.windows),batch_size):
        ix=torch.arange(start,min(start+batch_size,len(split.windows)),device=split.query_times.device)
        pieces.append(model.forward_packed_queries(index_packed_grouped(split.packed,ix),split.query_times.index_select(0,ix)).cpu().numpy())
    return dict(prediction=np.concatenate(pieces),target=split.targets.cpu().numpy(),mask=split.target_masks.cpu().numpy(),
        sample_id=np.array([w.sample_id for w in split.windows]),
        entity=np.array([data.window_info[w.sample_id].entity for w in split.windows]),
        start_time=np.array([data.window_info[w.sample_id].start_time for w in split.windows]))
def array_mse(z):
    e=(z['prediction'].astype(np.float64)-z['target'].astype(np.float64))**2
    c=z['mask'].sum((0,1));valid=c>0
    return float(((e*z['mask']).sum((0,1))[valid]/c[valid]).mean())
def prepared(family,data,config,sets,device):
    probe=build(family,data,config,device)
    result=PreparedTimeIMMData(**{k:prepare_timeimm_split(probe,v,data.input_dims,len(data.feature_names)) for k,v in sets.items()})
    del probe;return result
def fit_one(out,name,seed,family,data,config,budget,packed,device,metadata,time_mode=None):
    path=out/f'{name}_seed{seed}.json'
    if path.exists():
        result=read(path)
        assert (out/f'{name}_seed{seed}.pt').is_file()
        assert (out/f'{name}_seed{seed}_predictions.npz').is_file()
        return result
    out.mkdir(parents=True,exist_ok=True);set_deterministic_seed(seed)
    model=build(family,data,config,device,time_mode=time_mode)
    initial=audit.state_digest(model)
    result=train_timeimm_model(name=name,model=model,data=data,prepared=packed,output_dir=out,
        log_record=lambda r:append_jsonl(out/'epochs.jsonl',r),epochs=int(budget['epochs']),patience=int(budget['patience']),
        batch_size=int(budget['batch_size']),learning_rate=float(budget['learning_rate']),weight_decay=float(budget['weight_decay']),
        grad_clip=float(budget['grad_clip']),seed=seed)
    # Generated splits have independent entities. Omit only groups that are
    # absent from the evaluated split, never an observed or nonfinite loss.
    for split_name,split_data in [('val',packed.val),('test',packed.test)]:
        observed={data.window_info[w.sample_id].entity for w in split_data.windows if bool(w.target_masks.any())}
        values=result[split_name+'_metrics']['entity_mse']
        assert all(math.isfinite(float(values[e]))for e in observed)
        result[split_name+'_metrics']['entity_mse']={e:v for e,v in values.items()if e in observed}
    arrays=predictions(model,packed.test,data)
    assert abs(array_mse(arrays)-result['test_metrics']['mse'])<2e-5*max(1.,result['test_metrics']['mse'])
    np.savez_compressed(out/f'{name}_seed{seed}_predictions.npz',**arrays)
    result.update(metadata,family=family,seed=seed,condition=name,training=budget,initial_state_sha256=initial,
                  input_dims=list(data.input_dims),source_sha256=sha(HERE/'common.py'))
    save(path,result)
    print(json.dumps(dict(phase=metadata.get('phase'),family=family,seed=seed,condition=name,mse=result['test_metrics']['mse'],seconds=result['duration_seconds']),ensure_ascii=False),flush=True)
    del model;return result
def original_checkpoint(dataset,family,seed):
    if seed in SEEDS:
        folder,name,_=source(dataset,family);return folder/f'{name}_seed{seed}.pt'
    return PREVIOUS/'results/ten_seed'/audit.SLUGS[dataset]/family/f'aligned_seed{seed}.pt'
