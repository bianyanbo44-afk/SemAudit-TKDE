"""Prepare traceable figure data from retained predictions; no model training."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/review_followup_20261105/results'
OUT = ROOT/'delivery/figures_v2_20261106'
DATA = OUT/'data'
SEEDS = [53, 67, 79]
ALL_SEEDS = SEEDS+[83,97,101,113,127,139,151]
DOMAINS = ['ClusterTrace','CESNET','EPA-Air','FNSPID','RepoHealth']
SLUG = dict(zip(DOMAINS,['clustertrace','cesnet','epa_air','fnspid','repohealth']))
FAMILIES = ['selective_ssm','immtsf_recavg_gradd','immtsf_t2v_xattn']
SOURCES = {}

def record(path):
    path=Path(path)
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(path):
    return json.loads(record(path).read_text(encoding='utf-8'))

def arrays(path):
    with np.load(record(path)) as a:return {k:a[k] for k in a.files}

def mse(a):
    error=(a['prediction'].astype(np.float64)-a['target'].astype(np.float64))**2
    counts=a['mask'].sum((0,1));keep=counts>0
    return float(((error*a['mask']).sum((0,1))[keep]/counts[keep]).mean())

def save(name, rows):
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if isinstance(rows,list) and rows:
        with (DATA/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def encoder_data():
    path=ROOT/'experiments/added_experiments_20261104/results/analysis/per_seed.csv'
    with record(path).open(newline='',encoding='utf-8-sig') as f:
        records=[dict(encoder='gpt2',dataset=r['dataset'],family=r['family'],seed=int(r['seed']),
            **{k:float(r[k])for k in ['aligned','shuffle','time_only','sag','cbtg']})
            for r in csv.DictReader(f) if int(r['seed']) in SEEDS]
    for encoder in ['minilm','finbert']:
        for d in DOMAINS:
            for family in FAMILIES:
                folder=RESULTS/'encoders'/encoder/SLUG[d]/family
                for seed in SEEDS:
                    losses={c:read(folder/f'{c}_seed{seed}.json')['test_metrics']['mse'] for c in ['aligned','shuffle','time_only']}
                    records.append(dict(encoder=encoder,dataset=d,family=family,seed=seed,**losses,
                        sag=100*(losses['shuffle']-losses['aligned'])/losses['shuffle'],
                        cbtg=100*(losses['time_only']-losses['aligned'])/losses['time_only']))
    summaries=[]
    for encoder in ['gpt2','minilm','finbert']:
        for d in DOMAINS:
            for family in FAMILIES:
                rows=[r for r in records if (r['encoder'],r['dataset'],r['family'])==(encoder,d,family)]
                assert len(rows)==3
                means={c:float(np.mean([r[c] for r in rows]))for c in ['aligned','shuffle','time_only']}
                sag=100*(means['shuffle']-means['aligned'])/means['shuffle']
                cbtg=100*(means['time_only']-means['aligned'])/means['time_only']
                ws=sum(r['aligned']<r['shuffle'] for r in rows);wt=sum(r['aligned']<r['time_only'] for r in rows)
                summaries.append(dict(encoder=encoder,dataset=d,family=family,**means,sag=sag,cbtg=cbtg,
                    wins_shuffle=ws,wins_time=wt,support=bool(sag>0 and cbtg>0 and ws>=2 and wt>=2)))
    expected=read(ROOT/'manuscript/source_data/encoder_summary.json')
    for r in expected:
        s=next(x for x in summaries if all(x[k]==r[k]for k in ['encoder','dataset','family']))
        assert abs(s['sag']-r['sag'])<2e-5 and abs(s['cbtg']-r['cbtg'])<2e-5
        assert s['support']==r['joint_support']
    thresholds=[]
    for encoder in ['gpt2','minilm','finbert']:
        for t in [0,.01,.1,.5,1,2]:
            thresholds.append(dict(encoder=encoder,threshold=t,count=sum(r['sag']>t and r['cbtg']>t and r['wins_shuffle']>=2 and r['wins_time']>=2 for r in summaries if r['encoder']==encoder)))
    save('encoder_seeds',records);save('encoder_summary',summaries);save('encoder_thresholds',thresholds)

def history_data():
    summaries=[];records=[];coverage=[]
    for encoder in ['gpt2','minilm','finbert']:
        for d in DOMAINS:
            for family in FAMILIES:
                folder=RESULTS/'strict_history'/SLUG[d]/family if encoder=='gpt2' else RESULTS/'encoder_history'/encoder/SLUG[d]/family
                for condition in ['history_rotation','history_nearest_swap']:
                    diag=read(folder/f'{condition}_diagnostics.json')
                    seeds=ALL_SEEDS if encoder=='gpt2' else SEEDS
                    aa=[];cc=[]
                    for seed in seeds:
                        if encoder=='gpt2':
                            a=read(folder/f'aligned_seed{seed}.json')['mse']
                        else:
                            a=read(RESULTS/'encoders'/encoder/SLUG[d]/family/f'aligned_seed{seed}.json')['test_metrics']['mse']
                        c=read(folder/f'{condition}_seed{seed}.json')['mse']
                        aa.append(a);cc.append(c)
                        records.append(dict(encoder=encoder,dataset=d,family=family,condition=condition,seed=seed,
                            aligned=a,control=c,gain=100*(c-a)/c))
                    value=100*(np.mean(cc)-np.mean(aa))/np.mean(cc)
                    summaries.append(dict(encoder=encoder,dataset=d,family=family,condition=condition,n=len(seeds),
                        gain=float(value),seed_min=min(r['gain']for r in records[-len(seeds):]),seed_max=max(r['gain']for r in records[-len(seeds):]),**diag))
                    if encoder=='gpt2' and family==FAMILIES[0]:coverage.append(dict(dataset=d,condition=condition,**diag))
    # Coverage can be shared only because all three representations have equal counts.
    for r in summaries:
        ref=next(c for c in coverage if c['dataset']==r['dataset'] and c['condition']==r['condition'])
        for k in ['histories','changed_histories','slots','changed_vectors','zero_or_singleton_histories']:
            assert r[k]==ref[k]
    save('history_summary',summaries);save('history_seeds',records);save('history_coverage',coverage)

def source_data():
    summaries=read(ROOT/'manuscript/source_data/source_effect_summary.json')
    records=[]
    for row in summaries:
        d,family,variant=(row[k]for k in ['dataset','family','variant'])
        for seed in SEEDS:
            baseline=arrays(RESULTS/'strict_history'/SLUG[d]/family/f'aligned_seed{seed}_predictions.npz')
            folder=RESULTS/'source_execution'/variant/SLUG[d]/family
            altered=arrays(folder/f'aligned_seed{seed}_predictions.npz')
            np.testing.assert_array_equal(baseline['sample_id'],altered['sample_id'])
            np.testing.assert_array_equal(baseline['mask'],altered['mask'])
            if variant=='full_entity_normalization':
                canonical=read(RESULTS/'source_execution/canonical_scales'/f'{SLUG[d]}_{family}.json')
                scales=read(folder/f'aligned_seed{seed}.json')['scales']
                mean=np.array([canonical[e]['mean']for e in altered['entity']])[:,None,:]
                std=np.array([canonical[e]['std']for e in altered['entity']])[:,None,:]
                vm=np.array([scales[e]['mean']for e in altered['entity']])[:,None,:]
                vs=np.array([scales[e]['std']for e in altered['entity']])[:,None,:]
                converted=(altered['target']*vs+vm-mean)/std
                np.testing.assert_allclose(converted[altered['mask']>0],baseline['target'][baseline['mask']>0],rtol=2e-4,atol=2e-4)
                altered=dict(altered,prediction=(altered['prediction']*vs+vm-mean)/std,target=baseline['target'])
            else:np.testing.assert_array_equal(altered['target'],baseline['target'])
            a,b=mse(baseline),mse(altered)
            records.append(dict(dataset=d,family=family,variant=variant,seed=seed,canonical=a,altered=b,change=100*(b-a)/a))
        selected=records[-3:]
        aggregate=100*(np.mean([r['altered']for r in selected])-np.mean([r['canonical']for r in selected]))/np.mean([r['canonical']for r in selected])
        assert abs(aggregate-row['canonical_scale_change_percent'])<1e-8
    save('source_summary',summaries);save('source_seeds',records)

def case_data():
    family='immtsf_t2v_xattn';encoder='minilm';d='ClusterTrace'
    folder=RESULTS/'encoder_history'/encoder/SLUG[d]/family
    mapping=read(folder/'history_rotation_mapping.json')
    source=RESULTS/'encoders'/encoder/SLUG[d]/family
    aligned=[arrays(source/f'aligned_seed{s}_predictions.npz')for s in SEEDS]
    altered=[arrays(folder/f'history_rotation_seed{s}_predictions.npz')for s in SEEDS]
    def unit_loss(a):
        e=(a['prediction'].astype(float)-a['target'].astype(float))**2
        counts=a['mask'].sum(1);keep=counts>0
        per=np.divide((e*a['mask']).sum(1),counts,out=np.zeros_like(counts,dtype=float),where=keep)
        return np.array([v[k].mean()for v,k in zip(per,keep)])
    av=np.array([unit_loss(a)for a in aligned]);bv=np.array([unit_loss(b)for b in altered])
    effects=100*(bv.mean(0)-av.mean(0))/bv.mean(0)
    changed={m['window_id']for m in mapping if m['changed']}
    eligible=[i for i,w in enumerate(aligned[0]['sample_id'])if int(w) in changed]
    median=float(np.median(np.abs(effects[eligible])))
    index=min(eligible,key=lambda i:(abs(abs(effects[i])-median),int(aligned[0]['sample_id'][i])))
    ident=int(aligned[0]['sample_id'][index]);entity=str(aligned[0]['entity'][index]);start=float(aligned[0]['start_time'][index])
    maps=[m for m in mapping if m['window_id']==ident]
    raw=ROOT/'runtime/phase3_prototype/external/Time-IMM/data/ClusterTrace/processed'/entity
    with record(raw/'time_series.csv').open(encoding='utf-8-sig',newline='')as f:
        numeric=list(csv.DictReader(f))
    base=min(datetime.fromisoformat(r['date_time'])for r in numeric)
    with record(raw/'text.csv').open(encoding='utf-8-sig',newline='')as f:texts=list(csv.DictReader(f))
    notes=[]
    for m in maps:
        arrival=next(x['donor_arrival']for x in maps if x['donor_event']==m['recipient_event'])
        moment=base+timedelta(hours=start+arrival)
        candidate=min(texts,key=lambda r:abs((datetime.fromisoformat(r['date_time'])-moment).total_seconds()))
        assert abs((datetime.fromisoformat(candidate['date_time'])-moment).total_seconds())<.01
        notes.append(dict(recipient_event=m['recipient_event'],donor_event=m['donor_event'],time_hours=arrival,
            donor_time_hours=m['donor_arrival'],text=candidate['summary'],timestamp=candidate['date_time']))
    notes.sort(key=lambda r:r['time_hours'])
    a=aligned[0];b=altered[0]
    candidates=[j for j in range(a['target'].shape[-1])if a['mask'][index,:,j].sum()>=5]
    channel=max(candidates,key=lambda j:(np.std(a['target'][index,a['mask'][index,:,j]>0,j]),-j))
    keep=a['mask'][index,:,channel]>0
    positions=np.flatnonzero(keep)
    traces=[dict(observed_index=int(k+1),target=float(a['target'][index,k,channel]),aligned=float(a['prediction'][index,k,channel]),rotation=float(b['prediction'][index,k,channel]))for k in positions]
    losses=[dict(seed=s,aligned=float(av[n,index]),rotation=float(bv[n,index]),gain=float(100*(bv[n,index]-av[n,index])/bv[n,index]))for n,s in enumerate(SEEDS)]
    case=dict(dataset=d,encoder=encoder,family=family,window_id=ident,entity=entity,start_time_hours=start,cutoff_hours=12,
        selection='Window closest to the median absolute three-seed fixed-rotation effect among all changed test histories; ties by sample ID.',
        eligible_histories=len(eligible),total_histories=len(effects),median_absolute_gain=median,case_gain=float(effects[index]),
        prediction_seed=53,channel_index=channel,channel_name=list(numeric[0])[2+channel],notes=notes,traces=traces,seed_losses=losses)
    save('real_case',case)
    save('case_all_window_effects',[dict(window_id=int(w),effect=float(effects[i]),changed=int(w)in changed,selected=i==index)for i,w in enumerate(a['sample_id'])])
    save('case_prediction_trace',traces);save('case_seed_losses',losses)

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    encoder_data();print('encoder data ready',flush=True)
    history_data();print('history data ready',flush=True)
    source_data();print('source data ready',flush=True)
    case_data();print('case data ready',flush=True)
    save('calibration_worlds',read(ROOT/'manuscript/source_data/neural_calibration_worlds.json'))
    (DATA/'input_hashes.json').write_text(json.dumps(SOURCES,indent=2),encoding='utf-8')
    print('All figure data verified.',flush=True)

if __name__=='__main__':main()
