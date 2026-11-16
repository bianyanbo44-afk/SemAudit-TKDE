"""Empirical irregular-clock calibration using the three original neural families."""
from __future__ import annotations
from dataclasses import replace
import gc
import importlib.util
import math
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
import common as c
from strict_history import transform
from eventmamba.data import AsyncEvent
from eventmamba.timeimm import TimeIMMData,TimeIMMSpec,TimeIMMScale,TimeIMMWindowInfo,TimeIMMWindowSample

def calibration_source():
    path=c.PREVIOUS/'experiments/irregular_fine/run_calibration.py'
    spec=importlib.util.spec_from_file_location('prior_clock_generator',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cfg=c.read(path.parent/'config.json');cfg['source_root']=str(c.PROJECT/'external/Time-IMM/data/EPA-Air/processed')
    cfg['max_queries_per_entity']=c.CONFIG['neural_calibration']['queries_per_entity']
    return mod,cfg,path

def phrases(out,oldcfg,device):
    path=out/'phrase_embeddings.pt'
    if path.exists():return torch.load(path,weights_only=True)
    snapshot=c.EXTENSION/'external_validation/models/minilm'
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False).to(device).eval()
    texts=[x for pair in oldcfg['train_phrase_pairs']+oldcfg['heldout_phrase_pairs']for x in pair]
    with torch.inference_mode():
        batch=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt').to(device)
        hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
        vectors=torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1).cpu()
    result=dict(vectors=vectors,phrases=texts,model_files_sha256={p.name:c.sha(p)for p in snapshot.iterdir()if p.is_file()})
    torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result

def generate(world,schedules,vectors):
    settings=c.CONFIG['neural_calibration'];records={};sid=0
    for split_index,(split,n) in enumerate(settings['entities_per_split'].items()):
        group=[]
        for entity_index in range(n):
            streams=np.random.SeedSequence([world,split_index,entity_index]).spawn(5)
            numeric_rng,sign_rng,delay_rng,phrase_rng,noise_rng=[np.random.default_rng(x)for x in streams]
            schedule_index=entity_index%len(schedules);schedule=schedules[schedule_index]
            cutoffs=schedule['cutoffs'][:settings['queries_per_entity']]
            nt=schedule['numeric'][schedule['numeric']<=cutoffs[-1]]
            dimensions=settings['target_dimensions']
            values=np.empty((len(nt),dimensions));values[0]=numeric_rng.normal(size=dimensions)
            gaps=np.maximum(np.diff(nt),0);unit=np.median(gaps[gaps>0])
            for i,gap in enumerate(gaps,start=1):
                phi=.985**(gap/unit)
                values[i]=phi*values[i-1]+numeric_rng.normal(0,.08*np.sqrt((1-phi*phi)/(1-.985**2)),size=dimensions)
            tt=schedule['text'];directions=sign_rng.choice([-1.,1.],size=len(tt))
            arrival=tt+delay_rng.exponential(settings['arrival_delay_mean_days'],len(tt))
            pair=phrase_rng.integers(0,4,len(tt))+(0 if split=='train' else 4)
            phrase=2*pair+(directions<0).astype(int)
            for cutoff in cutoffs:
                start=cutoff-settings['history_days'];ni=np.flatnonzero((nt>=start)&(nt<cutoff))
                ti=np.flatnonzero((tt>=start)&(tt<cutoff)&(arrival<=cutoff))
                assert len(ni)>0
                weights=np.exp(-(cutoff-arrival[ti]))
                z=float(np.dot(weights,directions[ti])/np.sqrt(np.sum(weights**2)))if len(ti) else 0.
                events=[]
                for i in ni:
                    for modality in range(dimensions):
                        events.append(dict(event_time=float(nt[i]-start),arrival_time=float(nt[i]-start),modality=modality,
                            value=torch.tensor([values[i,modality]],dtype=torch.float32),quality=1.,event_id=int(i*dimensions+modality)))
                for i in ti:events.append(dict(event_time=float(tt[i]-start),arrival_time=float(arrival[i]-start),modality=dimensions,
                    value=vectors[int(phrase[i])].clone(),quality=1.,event_id=int(dimensions*len(nt)+i)))
                events.sort(key=lambda e:(e['arrival_time'],e['modality'],e['event_id']))
                for i,e in enumerate(events):e['event_id']=i
                group.append(dict(sample_id=sid,entity=f'{split}_{entity_index:03}',start_time=float(start),schedule=schedule_index,
                    events=events,baseline=(.8*values[ni[-1]]+.2*np.mean(values[ni],axis=0)).tolist(),signal=z,
                    noise=noise_rng.normal(0,settings['noise_sd'],size=dimensions).tolist(),text_count=len(ti)))
                sid+=1
        records[split]=group
    return records

def make_data(records,strength,family):
    settings=c.CONFIG['neural_calibration'];info={};scales={};sets={}
    dimensions=settings['target_dimensions']
    width=int(next(e['value'].numel()for group in records.values()for r in group for e in r['events']if e['modality']==dimensions))
    for split,group in records.items():
        windows=[]
        for r in group:
            events=[]
            for e in r['events']:
                payload=e['value']
                if e['modality']==dimensions and family=='selective_ssm':payload=torch.nn.functional.layer_norm(payload,(width,))
                events.append(AsyncEvent(**{**e,'value':payload}))
            window=TimeIMMWindowSample(events=tuple(events),cutoff=settings['history_days'],
                query_horizons=torch.tensor([settings['horizon_days']]),
                targets=torch.tensor((np.array(r['baseline'])+strength*r['signal']*np.array(settings['signal_coefficients'])+np.array(r['noise']))[None],dtype=torch.float32),
                target_masks=torch.ones(1,dimensions),sample_id=r['sample_id'])
            windows.append(window);info[window.sample_id]=TimeIMMWindowInfo(r['entity'],0,r['start_time'])
            scales[r['entity']]=TimeIMMScale(np.zeros(dimensions),np.ones(dimensions),0,np.zeros(dimensions,dtype=int),0.)
        sets[split]=tuple(windows)
    return TimeIMMData(train_windows=sets['train'],val_windows=sets['val'],test_windows=sets['test'],window_info=info,
        scales=scales,feature_names=tuple(f'generated_numeric_{i}'for i in range(dimensions)),window_counts={},target_counts={},dropped_no_text={},
        text_cache_name='actual MiniLM template vectors',text_embedding_dim=width,text_metadata={},
        text_preprocessing='layer_norm'if family=='selective_ssm'else'none',normalization='train_only',
        spec=TimeIMMSpec('generated_irregular',settings['history_days'],settings['horizon_days'],settings['history_days'],'days'))

def main():
    mod,oldcfg,oldpath=calibration_source()
    out=c.freeze('neural_calibration',[__file__,c.HERE/'strict_history.py'])
    device=c.setup();schedules=mod.load_schedules(oldcfg)
    c.save(out/'schedule_identity.json',dict(source_generator_sha256=c.sha(oldpath),templates=mod.schedule_identity(schedules)))
    emb=phrases(out,oldcfg,device);manifest=[]
    settings=c.CONFIG['neural_calibration']
    for world in settings['world_seeds']:
        rawpath=out/'inputs'/f'world_{world}.pt';rawpath.parent.mkdir(exist_ok=True)
        if rawpath.exists():records=torch.load(rawpath,weights_only=True)
        else:
            records=generate(world,schedules,emb['vectors']);torch.save(records,rawpath)
        for strength in settings['strengths']:
            for family in c.FAMILIES:
                folder,_,original=c.source('EPA-Air',family)
                data=make_data(records,strength,family);aligned={s:getattr(data,s+'_windows')for s in ['train','val','test']}
                shuffled={};diags={}
                for split,windows in aligned.items():
                    shuffled[split],maps,diags[split]=transform(windows,data.window_info,data.text_modality,'history_rotation')
                variants=dict(aligned=aligned,shuffle=shuffled,
                    time_only={s:c.zero_text_values(w,data.text_modality)for s,w in aligned.items()})
                run=out/f'world_{world}'/f'strength_{strength}'/family;run.mkdir(parents=True,exist_ok=True)
                c.save(run/'input_identity.json',dict(raw_sha256=c.sha(rawpath),strength=strength,diagnostics=diags,
                    source_config_sha256=c.sha(folder/'results.json'),feature_scaling='known generator units, identity scale; per-event LayerNorm only for selective family'))
                for condition,sets in variants.items():
                    packed=c.prepared(family,data,original['config'],sets,device)
                    for seed in c.SEEDS:
                        result=c.fit_one(run,condition,seed,family,data,original['config'],c.training(original),packed,device,
                            dict(phase='neural_calibration',world=world,strength=strength,representation='MiniLM actual held-out phrase templates'))
                        manifest.append(dict(file=str((run/f'{condition}_seed{seed}.json').relative_to(out)),
                            world=world,strength=strength,family=family,condition=condition,seed=seed))
                        c.save(out/'manifest.json',manifest)
                    del packed
                # Fixed-checkpoint reliance uses exactly the aligned selected weights.
                model=c.build(family,data,original['config'],device)
                for condition,windows in [('fixed_shuffle',shuffled['test']),('fixed_zero',variants['time_only']['test'])]:
                    packed=c.prepare_timeimm_split(model,windows,data.input_dims,len(data.feature_names))
                    for seed in c.SEEDS:
                        file=run/f'{condition}_seed{seed}_predictions.npz'
                        if file.exists():continue
                        ckpt=torch.load(run/f'aligned_seed{seed}.pt',map_location=device,weights_only=True)
                        model.load_state_dict(ckpt['state_dict'],strict=True)
                        before=c.audit.state_digest(model);z=c.predictions(model,packed,data)
                        assert c.audit.state_digest(model)==before
                        np.savez_compressed(file,**z)
                    del packed
                del model,data;gc.collect();torch.cuda.empty_cache()
    assert len(manifest)==675
    c.save(out/'COMPLETE.json',dict(fits=len(manifest),planned=675,worlds=5,strengths=5,families=3,seeds=3,conditions=3))

if __name__=='__main__':main()
