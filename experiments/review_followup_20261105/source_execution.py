"""Execute pinned upstream paths, then measure isolated learned protocol effects."""
from __future__ import annotations
import argparse
from dataclasses import replace
import contextlib
import csv
import difflib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import common as c
import eventmamba.timeimm as adapter

try:
    import prettytable
except ModuleNotFoundError:
    # The pinned loader uses this pure-Python package only for console tables.
    # Resolve the installed copy without changing NumPy/sklearn imports.
    system_packages = Path.home()/'AppData/Local/Programs/Python/Python312/Lib/site-packages'
    sys.path.insert(0, str(system_packages))
    try:
        import prettytable
    finally:
        sys.path.remove(str(system_packages))

def module_from(path,name,text=None):
    if text is None:
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    mod=type(os)(name);mod.__file__=str(path);exec(compile(text,str(path),'exec'),mod.__dict__);return mod

def fixture(root,future=13.):
    entity=root/'processed/e';entity.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(dict(date_time=pd.date_range('2020-01-01',periods=14,freq='D'),record_id=['e']*14,
        x=[*map(float,range(13)),future],y=np.linspace(0,1,14)))
    frame.to_csv(entity/'time_series.csv',index=False)
    pd.DataFrame(dict(date_time=pd.to_datetime(['2020-01-02','2020-01-04','2020-01-08','2020-01-12']),
                      record_id=['e']*4,note=['a','b','c','d'])).to_csv(entity/'text.csv',index=False)
    return root

def regression():
    out=c.freeze('source_execution',[__file__]);upstream=c.PROJECT/'external/IMM-TSF'
    parsepath=upstream/'lib/parse_datasets.py';cachepath=upstream/'compute_text_embeddings.py'
    raw=parsepath.read_text(encoding='utf8');cachetext=cachepath.read_text(encoding='utf8')
    # Execute complete real modules. Repairs target the cited source statements.
    released=module_from(parsepath,'released_upstream_parser')
    normblock='''            if normalize:
                df[feat_cols] = df[feat_cols].apply('''
    normrepair='''            if normalize:
                fit_mask = df["_ts_raw"] < pd.Timestamp(args.normalization_fit_end)
                df[feat_cols] = df[feat_cols].apply('''
    assert raw.count(normblock)==1
    normtext=raw.replace(normblock,normrepair).replace('col.mean()','col[fit_mask].mean()').replace('col.std()','col[fit_mask].std()')
    membertext=raw.replace('if len(selected) == 0:','if False and len(selected) == 0:')
    assert membertext!=raw
    cacherepair=cachetext.replace('base_ts = df["date_time"].min()',
        'base_ts = pd.to_datetime(pd.read_csv(os.path.join(rec_dir, "time_series.csv"))["date_time"]).min()')
    assert cacherepair!=cachetext
    patches={'train_only_normalization':(raw,normtext),'retain_no_text':(raw,membertext),'shared_cache_origin':(cachetext,cacherepair)}
    for label,(before,after)in patches.items():
        (out/f'{label}.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='released',tofile=label)),encoding='utf8')
    norm=module_from(parsepath,'repaired_normalization',normtext)
    member=module_from(parsepath,'repaired_membership',membertext)
    def load(module,root,**kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return module.ChunkedTimeSeriesDataset(str(root),history=2,pred_window=2,stride=2,normalize=kw.pop('normalize',False),**kw)
    results=[]
    with tempfile.TemporaryDirectory(prefix='semaudit_actual_upstream_')as temporary:
        temp=Path(temporary);a=fixture(temp/'ordinary');b=fixture(temp/'future_changed',future=1000.)
        args=argparse.Namespace(normalization_fit_end='2020-01-07')
        ra,rb=load(released,a,normalize=True),load(released,b,normalize=True)
        fa,fb=load(norm,a,normalize=True,args=args),load(norm,b,normalize=True,args=args)
        before_delta=float((ra[0][2]-rb[0][2]).abs().max());after_delta=float((fa[0][2]-fb[0][2]).abs().max())
        assert before_delta>0 and after_delta==0.
        results.append(dict(issue='normalization',executed=str(parsepath),released_failure_observed=True,
            future_changes_earlier_chunk_max=before_delta,repaired_future_changes_earlier_chunk_max=after_delta))
        released_no=load(released,a,enable_text=False);fixed_no=load(member,a,enable_text=False)
        released_yes=load(released,a,enable_text=True);fixed_yes=load(member,a,enable_text=True)
        assert len(released_no)==len(released_yes)<len(fixed_no)==len(fixed_yes)
        assert any(not item[4]for item in fixed_yes)
        old={x[0]:x for x in released_no};new={x[0]:x for x in fixed_no}
        for key,x in old.items():
            for i in [1,2,3]:torch.testing.assert_close(x[i],new[key][i],rtol=0,atol=0)
        results.append(dict(issue='membership',executed=str(parsepath),released_chunks=len(released_no),repaired_chunks=len(fixed_no),
            released_numeric_only_deletes_empty_text=True,paired_numeric_tensors_unchanged=True))
        # Use a deterministic embedding fixture to isolate timestamp handling;
        # actual model encoding is separately exercised in encoder_coverage.py.
        def embeddings(notes,*args,**kw):
            ids={'a':1.,'b':2.,'c':3.,'d':4.}
            return torch.tensor([[[ids[t],1.,2.]for t in notes[0]]]),torch.ones(1,len(notes[0]),dtype=torch.bool)
        def run_cache(text,tag):
            wd=temp/tag;path=wd/'data/fixture';fixture(path)
            mod=module_from(cachepath,'cache_'+tag,text);mod.load_llm=lambda *args,**kw:(None,None);mod.embed_notes=embeddings
            previous=Path.cwd()
            try:
                os.chdir(wd)
                with contextlib.redirect_stdout(io.StringIO()):mod.compute_text_embeddings('fixture','GPT2',6,320,'cpu')
            finally:os.chdir(previous)
            return path,torch.load(path/'processed/e/text_embeddings_model=GPT2_layers=6_maxlen=320.pt',weights_only=True)
        badpath,bad=run_cache(cachetext,'released_cache');goodpath,good=run_cache(cacherepair,'repaired_cache')
        torch.testing.assert_close(bad['embeddings'],good['embeddings'],rtol=0,atol=0)
        np.testing.assert_allclose(bad['rel_times'].numpy(),[0.,2.,6.,10.])
        np.testing.assert_allclose(good['rel_times'].numpy(),[1.,3.,7.,11.])
        rr=load(released,goodpath,enable_text=True)
        rc=load(released,goodpath,enable_text=True,use_text_embeddings=True,llm_model_fusion='GPT2',llm_layers_fusion=6,max_length=320)
        assert len(rr)==len(rc)
        for rawchunk,cachedchunk in zip(rr,rc):
            assert rawchunk[0]==cachedchunk[0]
            assert [x[0]for x in rawchunk[4]]==[x[0]for x in cachedchunk[4]]
        results.append(dict(issue='cache_origin',executed=[str(cachepath),str(parsepath)],released_offsets=bad['rel_times'].tolist(),
            repaired_offsets=good['rel_times'].tolist(),embedding_payloads_identical=True,repaired_raw_cached_membership_and_times_match=True,
            encoder_fixture='deterministic payload fixture; no claim of a real language encoder in this isolated timestamp test'))
    from fusions.TTF_RecAvg import TTF_RecAvg
    torch.manual_seed(42);fusion=TTF_RecAvg('GPT2',6,use_text_embeddings=True,recency_sigma=.3,dropout=0.,d_txt=16).eval()
    notes=torch.randn(2,3,768);tau=torch.tensor([[1.,3.,5.],[.5,1.5,6.]])
    queries=torch.tensor([[7.,8.],[7.,8.]]);total=14.
    collate_args = argparse.Namespace(history=7., pred_window=7., device=torch.device('cpu'))
    numeric_batch = [
        (f'fixture_{i}', torch.tensor([1., 3., 5., 7., 8.]),
         torch.ones(5, 4), torch.ones(5, 4))
        for i in range(2)
    ]
    collated = released.variable_time_collate_fn(numeric_batch, collate_args)
    torch.testing.assert_close(collated['tp_to_predict'], queries / total, rtol=0, atol=0)
    with torch.inference_mode():
        released_values=fusion(notes,tau,collated['tp_to_predict'])[0]
        repaired_values=fusion(notes,tau/total,queries/total)[0]
        changed_units=fusion(notes,tau*24,queries/total)[0]
        correct_units=fusion(notes,(tau*24)/(total*24),(queries*24)/(total*24))[0]
    assert not torch.allclose(released_values,repaired_values)
    torch.testing.assert_close(repaired_values,correct_units,rtol=1e-6,atol=1e-6)
    results.append(dict(issue='recavg_units',executed=str(upstream/'fusions/TTF_RecAvg.py'),
        released_repaired_max_difference=float((released_values-repaired_values).abs().max()),
        repaired_physical_unit_conversion_max_difference=float((correct_units-repaired_values).abs().max()),
        actual_pinned_module_executed=True,actual_upstream_query_collate_executed=True))
    c.save(out/'UPSTREAM_REGRESSION.json',dict(passed=True,checks=results,source_sha256={str(p.relative_to(upstream)):c.sha(p)for p in [parsepath,cachepath,upstream/'fusions/TTF_RecAvg.py']}))
    print('ACTUAL UPSTREAM: four released/repaired path regressions passed',flush=True)

def entity_rows(data,dataset,entity):
    return adapter._load_entity(c.PROJECT/'external/Time-IMM/data'/dataset/'processed'/entity,data.feature_names,
        adapter.UNIT_SECONDS[data.spec.time_unit],data.text_cache_name,data.text_preprocessing)

def cache_origin(data,dataset):
    rows={entity:entity_rows(data,dataset,entity)for entity in data.scales}
    wrong={entity:replace(r,text_times=r.text_times-r.text_times.min())for entity,r in rows.items()}
    sets={}
    for split in ['train','val','test']:
        windows=[]
        for w in getattr(data,split+'_windows'):
            info=data.window_info[w.sample_id];r=wrong[info.entity];start=info.start_time
            h=np.flatnonzero((r.times>=start)&(r.times<start+data.spec.history))
            q=np.flatnonzero((r.times>=start+data.spec.history)&(r.times<start+data.spec.total))
            raw=adapter._Window(info.window_index,start,h,q);scale=data.scales[info.entity]
            events=adapter._history_events(r,raw,(r.values-scale.mean)/scale.std,data.spec.history)
            windows.append(replace(w,events=events))
        sets[split+'_windows']=tuple(windows)
    return replace(data,**sets)

def retain_empty_training(data,dataset):
    # Execute the original adapter window builder with only its has-text gate
    # changed. Split boundaries and training scales remain canonical and frozen.
    raw=inspect.getsource(adapter._make_windows)
    changed=raw.replace('if has_text:','if True:')
    assert changed!=raw
    namespace=dict(adapter.__dict__);exec(compile(changed,str(Path(adapter.__file__)),'exec'),namespace)
    make_all=namespace['_make_windows'];existing={(data.window_info[w.sample_id].entity,data.window_info[w.sample_id].start_time)for s in ['train','val','test']for w in getattr(data,s+'_windows')}
    train=list(data.train_windows);info=dict(data.window_info);sid=max(info)+1;counts={}
    for entity in data.scales:
        original=[data.window_info[w.sample_id].start_time for w in data.train_windows if data.window_info[w.sample_id].entity==entity]
        if not original:continue
        boundary=max(original);r=entity_rows(data,dataset,entity);all_windows,_=make_all(r,data.spec);scale=data.scales[entity];n=0
        for window in all_windows:
            if (entity,window.start)in existing or window.start>boundary:continue
            sample=adapter._window_sample(r,window,(r.values-scale.mean)/scale.std,data.spec,sid)
            assert not any(e.modality==data.text_modality for e in sample.events)
            train.append(sample);info[sid]=adapter.TimeIMMWindowInfo(entity,window.index,window.start);sid+=1;n+=1
        counts[entity]=n
    return replace(data,train_windows=tuple(train),window_info=info),counts

def train():
    out=c.freeze('source_execution',[__file__]);device=c.setup();manifest=[]
    for dataset in c.CONFIG['datasets']:
        for family in c.FAMILIES:
            folder,_,original=c.source(dataset,family);canonical=c.load_original(dataset,family)
            c.save(out/'canonical_scales'/f'{c.audit.SLUGS[dataset]}_{family}.json',
                {entity:dict(mean=scale.mean.tolist(),std=scale.std.tolist()) for entity,scale in canonical.scales.items()})
            full=c.load_timeimm(c.PROJECT/'external/Time-IMM/data'/dataset,c.DATASET_SPECS[dataset],
                normalization='official_full_entity',text_cache_name=canonical.text_cache_name,text_preprocessing=canonical.text_preprocessing)
            origin=cache_origin(canonical,dataset);membership,counts=retain_empty_training(canonical,dataset)
            variants=dict(full_entity_normalization=full,cache_text_origin=origin,retain_empty_training_windows=membership)
            if family=='immtsf_recavg_gradd':variants['released_mixed_units']=canonical
            for variant,data in variants.items():
                # Forecast population, channels and raw targets stay paired.
                assert [w.sample_id for w in data.test_windows]==[w.sample_id for w in canonical.test_windows]
                sets={s:getattr(data,s+'_windows')for s in ['train','val','test']}
                packed=c.prepared(family,data,original['config'],sets,device)
                run=out/variant/c.audit.SLUGS[dataset]/family
                for seed in c.SEEDS:
                    r=c.fit_one(run,'aligned',seed,family,data,original['config'],c.training(original),packed,device,
                        dict(phase='source_effects',variant=variant,dataset=dataset,source_config_sha256=c.sha(folder/'results.json'),
                             added_empty_training_windows=counts if variant=='retain_empty_training_windows'else{},
                             fixed_test_population=True,scales={e:dict(mean=s.mean.tolist(),std=s.std.tolist())for e,s in data.scales.items()}),
                        time_mode='released_mixed_units'if variant=='released_mixed_units'else None)
                    manifest.append(dict(file=str((run/f'aligned_seed{seed}.json').relative_to(out)),variant=variant,dataset=dataset,family=family,seed=seed))
                    c.save(out/'manifest.json',manifest)
                del packed
            del canonical,full,origin,membership;torch.cuda.empty_cache()
    assert len(manifest)==150
    c.save(out/'COMPLETE.json',dict(fits=len(manifest),planned=150,source_regressions=c.read(out/'UPSTREAM_REGRESSION.json')['passed']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['regression','train','all'],default='all');args=parser.parse_args()
    if args.phase in ['regression','all']:regression()
    if args.phase in ['train','all']:train()
