"""All-domain, all-family frozen encoder replacement with paired retraining."""
from __future__ import annotations
import argparse
import gc
import shutil
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
import common as c
from cache_timeimm_text_embeddings import read_entity

def encode():
    out=c.freeze('encoders',[__file__]);device=c.setup();records=[]
    for encoder,settings in c.CONFIG['encoders'].items():
        modelpath=c.EXTENSION/'external_validation/models'/settings['directory']
        hashes={p.name:c.sha(p) for p in modelpath.iterdir() if p.is_file()}
        identity=dict(encoder=encoder,model_files_sha256=hashes,pooling='masked mean, L2 normalization',max_length=settings['max_length'])
        c.save(out/f'{encoder}_model_identity.json',identity)
        tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False).to(device).eval()
        for dataset in c.CONFIG['datasets']:
            for source in sorted((c.PROJECT/'external/Time-IMM/data'/dataset/'processed').iterdir()):
                if not source.is_dir():continue
                target=c.HERE/'local_data'/encoder/dataset/'processed'/source.name
                target.mkdir(parents=True,exist_ok=True)
                for name in ['time_series.csv','text.csv']:
                    if not (target/name).exists():shutil.copy2(source/name,target/name)
                cache=target/'frozen_encoder.pt';record=target/'encoding_identity.json'
                if cache.exists() and record.exists():records.append(c.read(record));continue
                parsed=read_entity(source,3600. if dataset=='ClusterTrace' else 86400.)
                vectors=[];lengths=[]
                with torch.inference_mode():
                    for start in range(0,len(parsed['notes']),settings['batch_size']):
                        notes=parsed['notes'][start:start+settings['batch_size']]
                        tokenaudit=tokenizer(notes,truncation=False,padding=False)
                        lengths.extend(len(x) for x in tokenaudit['input_ids'])
                        batch=tokenizer(notes,padding=True,truncation=True,max_length=settings['max_length'],return_tensors='pt').to(device)
                        hidden=model(**batch).last_hidden_state
                        mask=batch['attention_mask'].unsqueeze(-1)
                        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                        vectors.append(torch.nn.functional.normalize(pooled,dim=-1).cpu())
                embeddings=torch.cat(vectors)
                metadata=dict(model_id=encoder,resolved_revision=hashes['config.json'],pooling='masked mean L2',
                    max_length=settings['max_length'],embedding_dim=int(embeddings.shape[1]),timestamp_base='first numeric timestamp',text_sha256=parsed['text_sha256'])
                torch.save(dict(embeddings=embeddings,relative_times=torch.tensor(parsed['times'],dtype=torch.float64),
                    source_row_indices=torch.tensor(parsed['row_indices'],dtype=torch.int64),metadata=metadata),cache)
                row=dict(encoder=encoder,dataset=dataset,entity=source.name,notes=len(lengths),
                    truncated_notes=int(np.sum(np.array(lengths)>settings['max_length'])),max_tokens=max(lengths),
                    median_tokens=float(np.median(lengths)),embedding_dim=int(embeddings.shape[1]),cache_sha256=c.sha(cache),source_text_sha256=parsed['text_sha256'])
                c.save(record,row);records.append(row)
                print(f'ENCODE {encoder} {dataset} {source.name}: {len(lengths)} notes',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    c.save(out/'encoding_manifest.json',records)

def train():
    out=c.freeze('encoders',[__file__]);device=c.setup();manifest=[]
    for encoder in c.CONFIG['encoders']:
        for dataset in c.CONFIG['datasets']:
            for family in c.FAMILIES:
                folder,_,original=c.source(dataset,family)
                data=c.load_timeimm(c.HERE/'local_data'/encoder/dataset,c.DATASET_SPECS[dataset],normalization='train_only',
                    text_cache_name='frozen_encoder.pt',text_preprocessing='layer_norm' if family=='selective_ssm' else 'none')
                aligned={s:getattr(data,s+'_windows') for s in ['train','val','test']}
                variants=dict(aligned=aligned,shuffle={s:c.shuffle_text_values_within_entities(w,data.window_info,data.text_modality,seed=9101+i)
                    for i,(s,w) in enumerate(aligned.items())},time_only={s:c.zero_text_values(w,data.text_modality)for s,w in aligned.items()})
                run=out/encoder/c.audit.SLUGS[dataset]/family
                for condition,sets in variants.items():
                    packed=c.prepared(family,data,original['config'],sets,device)
                    for seed in c.SEEDS:
                        result=c.fit_one(run,condition,seed,family,data,original['config'],c.training(original),packed,device,
                            dict(phase='encoder_coverage',encoder=encoder,dataset=dataset,source_config_sha256=c.sha(folder/'results.json')))
                        manifest.append(dict(file=str((run/f'{condition}_seed{seed}.json').relative_to(out)),**{k:result[k]for k in ['encoder','dataset','family','seed','condition','initial_state_sha256']}))
                        c.save(out/'manifest.json',manifest)
                    del packed
                del data;gc.collect();torch.cuda.empty_cache()
    assert len(manifest)==270
    c.save(out/'COMPLETE.json',dict(fits=len(manifest),planned=270))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['encode','train','all'],default='all');args=parser.parse_args()
    if args.phase in ['encode','all']:encode()
    if args.phase in ['train','all']:train()
