"""Historical-only, same-window content reassignment on ten paired checkpoints."""
from __future__ import annotations
from dataclasses import replace
import gc
import random
import numpy as np
import torch
import common as c

def transform(windows,info,text_modality,condition):
    result=[];maps=[];changed=0;slots=0;changed_histories=0;singletons=0
    rng=random.Random(c.CONFIG['strict_history']['mapping_seed'])
    for window in windows:
        positions=[i for i,e in enumerate(window.events)if e.modality==text_modality]
        positions.sort(key=lambda i:(window.events[i].arrival_time,i))
        donors=list(range(len(positions)))
        if len(positions)<2:singletons+=1
        if condition=='history_rotation':
            for start in range(0,len(positions),c.CONFIG['strict_history']['block_size']):
                n=min(c.CONFIG['strict_history']['block_size'],len(positions)-start)
                shift=rng.randrange(1,n) if n>1 else 0
                for i in range(n):donors[start+i]=start+(i+shift)%n
        elif condition=='history_nearest_swap':
            remaining=list(range(len(positions)))
            if remaining:
                values=torch.stack([window.events[i].value for i in positions]);values=torch.nn.functional.normalize(values,dim=-1)
                similarity=(values@values.T).numpy()
                pairs=sorted(((float(similarity[i,j]),-abs(i-j),i,j)for i in remaining for j in remaining if i<j),reverse=True)
                available=set(remaining)
                for _,_,i,j in pairs:
                    if i in available and j in available:
                        donors[i]=j;donors[j]=i;available.remove(i);available.remove(j)
        elif condition!='aligned':raise ValueError(condition)
        replacements={};this_changed=0
        for recipient,donor in enumerate(donors):
            ri,di=positions[recipient],positions[donor];a,b=window.events[ri],window.events[di]
            assert a.arrival_time<=window.cutoff and b.arrival_time<=window.cutoff
            assert a.event_time<=window.cutoff and b.event_time<=window.cutoff
            replacements[ri]=b.value
            is_changed=not torch.equal(a.value,b.value);this_changed+=is_changed
            maps.append(dict(window_id=window.sample_id,entity=info[window.sample_id].entity,recipient_event=ri,donor_event=di,
                donor_arrival=b.arrival_time,recipient_cutoff=window.cutoff,changed=is_changed))
        altered=replace(window,events=tuple(replace(e,value=replacements.get(i,e.value))for i,e in enumerate(window.events)))
        # A within-history permutation preserves every payload's multiplicity.
        assert sorted(tuple(window.events[i].value.tolist())for i in positions)==sorted(tuple(altered.events[i].value.tolist())for i in positions)
        result.append(altered);changed+=this_changed;slots+=len(positions);changed_histories+=this_changed>0
    c.audit.verify_invariants(windows,result,text_modality)
    return tuple(result),maps,dict(slots=slots,changed_vectors=int(changed),changed_fraction=changed/slots if slots else 0.,
        histories=len(windows),changed_histories=int(changed_histories),zero_or_singleton_histories=singletons,
        donor_after_cutoff=0,invariants=True,donor_scope='same entity, split and recipient history; all donor arrivals by cutoff')

def main():
    out=c.freeze('strict_history',[__file__]);device=c.setup();records=[]
    for dataset in c.CONFIG['datasets']:
        for family in c.FAMILIES:
            _,_,original=c.source(dataset,family);data=c.load_original(dataset,family)
            run=out/c.audit.SLUGS[dataset]/family;run.mkdir(parents=True,exist_ok=True)
            model=c.build(family,data,original['config'],device)
            for condition in c.CONFIG['strict_history']['conditions']:
                windows,maps,diag=transform(data.test_windows,data.window_info,data.text_modality,condition)
                c.save(run/f'{condition}_mapping.json',maps);c.save(run/f'{condition}_diagnostics.json',diag)
                packed=c.prepare_timeimm_split(model,windows,data.input_dims,len(data.feature_names))
                for seed in c.ALL_SEEDS:
                    path=run/f'{condition}_seed{seed}.json'
                    if path.exists():records.append(c.read(path));continue
                    checkpoint=c.original_checkpoint(dataset,family,seed)
                    state=torch.load(checkpoint,map_location=device,weights_only=True)
                    model.load_state_dict(state['state_dict'],strict=True);before=c.audit.state_digest(model)
                    z=c.predictions(model,packed,data);mse=c.array_mse(z)
                    if condition=='aligned':assert abs(mse-state['test_metrics']['mse'])<2e-5*max(1.,mse)
                    assert c.audit.state_digest(model)==before
                    np.savez_compressed(run/f'{condition}_seed{seed}_predictions.npz',**z)
                    record=dict(phase='strict_history',dataset=dataset,family=family,condition=condition,seed=seed,mse=mse,
                        diagnostics=diag,checkpoint_sha256=c.sha(checkpoint),checkpoint=str(checkpoint))
                    c.save(path,record);records.append(record)
                print(f'HISTORY {dataset} {family} {condition}: changed {diag["changed_vectors"]}/{diag["slots"]}, future donors=0',flush=True)
                del packed
            del model,data;gc.collect();torch.cuda.empty_cache()
            c.save(out/'manifest.json',records)
    assert len(records)==450
    c.save(out/'COMPLETE.json',dict(evaluations=len(records),planned=450,all_donors_historical=True))

if __name__=='__main__':main()
