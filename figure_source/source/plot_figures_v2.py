"""Claim-led KBS figure suite, rendered exclusively with Matplotlib."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import sys
import textwrap
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.collections import PathCollection

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent
OUT=ROOT/'delivery/figures_v2_20261106'if SCRIPT_DIR.name=='tools'else SCRIPT_DIR.parent
DATA=OUT/'data'
AUDIT_DIR=SCRIPT_DIR if(SCRIPT_DIR/'audit_panel_alignment.py').is_file()else Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0,str(AUDIT_DIR))
from audit_panel_alignment import require_matplotlib_panel_alignment

W=183/25.4
DOMAINS=['ClusterTrace','CESNET','EPA-Air','FNSPID','RepoHealth']
FAMILIES=['selective_ssm','immtsf_recavg_gradd','immtsf_t2v_xattn']
FNAME={'selective_ssm':'Selective','immtsf_recavg_gradd':'RecAvg','immtsf_t2v_xattn':'Attention'}
SHORT={'selective_ssm':'Sel.','immtsf_recavg_gradd':'Rec.','immtsf_t2v_xattn':'Att.'}
ENC=['gpt2','minilm','finbert']
ENAME={'gpt2':'GPT-2','minilm':'MiniLM','finbert':'FinBERT'}
EC={'gpt2':'#69747D','minilm':'#286A95','finbert':'#B96643'}
BLUE='#286A95';ORANGE='#B96643';INK='#25313B';LIGHT='#E9EFF3';GREY='#69747D'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.titlesize':9,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'lines.linewidth':1.1,'savefig.facecolor':'white','axes.facecolor':'white',
    'legend.frameon':False,'text.color':INK,'axes.labelcolor':INK,
    'xtick.color':INK,'ytick.color':INK,'figure.facecolor':'white'})

CAPTIONS={}

def load(name):return json.loads((DATA/(name+'.json')).read_text(encoding='utf-8'))

def label(ax,letter,title):
    trans=ax.transAxes+ScaledTranslation(-14/72,8/72,ax.figure.dpi_scale_trans)
    ax.text(0,1,letter,transform=trans,fontsize=10,fontweight='bold',va='bottom')
    ax.set_title(title,loc='left',fontweight='bold',pad=10)

def note(fig,text,y=.025):fig.text(.055,y,text,fontsize=7.5,ha='left',va='bottom',color=GREY)

def save(fig,name,axes,ids,rows=None,columns=None,caption=''):
    fig.canvas.draw()
    for ax in axes:
        for collection in ax.collections:
            if not isinstance(collection,PathCollection):continue
            offsets=np.asarray(collection.get_offsets(),dtype=float)
            if offsets.size==0:continue
            xl=sorted(ax.get_xlim());yl=sorted(ax.get_ylim())
            assert np.all((offsets[:,0]>=xl[0])&(offsets[:,0]<=xl[1])),name+' clips observed x positions'
            assert np.all((offsets[:,1]>=yl[0])&(offsets[:,1]<=yl[1])),name+' clips observed y positions'
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=rows or [],column_groups=columns or [],
        json_out=OUT/(name+'.alignment.json'),strict=True,require_panel_labels=True)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=600)
    plt.close(fig)
    if name in ['fig2_representation_decisions','fig3_historical_coverage','fig6_implementation_effects','figS1_encoder_seed_effects']:
        caption+=' Sel., Rec. and Att. denote Selective, RecAvg and Attention, respectively.'
    CAPTIONS[name]=caption
    (OUT/(name+'.caption.txt')).write_text(caption,encoding='utf-8')

def box(ax,x,y,w,h,text,fill=LIGHT,edge='#B8C7D1',fs=8,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.016',
        facecolor=fill,edgecolor=edge,linewidth=.8,transform=ax.transAxes))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,
        fontweight='bold'if bold else'normal',transform=ax.transAxes,linespacing=1.35)

def arrow(ax,start,end,color=GREY,style='-|>',lw=.9,rad=0):
    ax.add_patch(FancyArrowPatch(start,end,transform=ax.transAxes,arrowstyle=style,
        mutation_scale=8,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=2))

def fig1():
    fig=plt.figure(figsize=(W,5.7))
    gs=fig.add_gridspec(3,1,left=.075,right=.965,top=.93,bottom=.09,hspace=.55,height_ratios=[1,1.25,1.05])
    axes=[fig.add_subplot(gs[i])for i in range(3)]
    for ax in axes:ax.set_axis_off()
    a,b,c=axes
    label(a,'a','A shared prediction task, multiple text representations')
    items=[(.01,.59,.18,.28,'Numerical history'),(.01,.15,.18,.28,'Arrived text'),
           (.26,.15,.20,.28,'GPT-2 / MiniLM\n/ FinBERT'),(.53,.25,.20,.50,'Same fusion family\n+ training budget'),
           (.80,.25,.18,.50,'Paired predictions\non common units')]
    for x,y,w,h,t in items:box(a,x,y,w,h,t,bold=x==.26,fill='#DCEAF2'if x==.26 else LIGHT)
    a.plot([.20,.49],[.73,.73],transform=a.transAxes,color=GREY,lw=.9)
    arrow(a,(.49,.73),(.53,.65));arrow(a,(.19,.29),(.26,.29))
    arrow(a,(.46,.29),(.53,.38));arrow(a,(.73,.50),(.80,.50))
    a.text(.01,.01,'Forecast units, targets, numerical inputs and event slots are retained.',transform=a.transAxes,fontsize=7.5,color=GREY)
    label(b,'b','Content controls preserve the event clock')
    # Illustrative slots, not empirical observations.
    xpos=[.35,.55,.75];names=['Aligned','Split shuffle','Time-only','Within-history']
    contents=[['A','B','C'],['D','A','E'],['0','0','0'],['B','A','C']]
    ypos=[.83,.59,.35,.11]
    colors={'A':'#D5E6F1','B':'#E8D5C8','C':'#DCE7DC','D':'#DDD7E7','E':'#E3DFC8','0':'#EEF0F2'}
    for y,title,tokens in zip(ypos,names,contents):
        b.text(.01,y,title,transform=b.transAxes,ha='left',va='center',fontsize=8)
        cursor=.27
        for x in xpos:
            b.plot([cursor,x-.06],[y,y],transform=b.transAxes,color='#BBC6CD',lw=.8)
            cursor=x+.06
        b.plot([cursor,.86],[y,y],transform=b.transAxes,color='#BBC6CD',lw=.8)
        for x,t in zip(xpos,tokens):box(b,x-.035,y-.065,.07,.13,t,fill=colors[t],edge='white',fs=8,bold=True)
    b.plot([.88,.88],[.02,.95],transform=b.transAxes,color=INK,lw=.8)
    b.text(.90,.88,'Forecast\ncutoff',transform=b.transAxes,va='top',fontsize=7.5)
    b.text(.29,1.01,'Same recipient slots',transform=b.transAxes,fontsize=7.5,color=GREY)
    b.text(.01,-.13,'Schematic: historical donors A–C; split-level donors may come from other windows (D, E).',transform=b.transAxes,fontsize=7.5,color=GREY)
    label(c,'c','Complementary audit operations connect effects to decisions')
    box(c,.01,.18,.44,.62,'CONDITION-RETRAINED\nTrain aligned, shuffled and time-only models\nSAG + CBTG + paired wins',fill='#DCEAF2',edge='#A6BDCE',fs=8)
    box(c,.54,.18,.44,.62,'FIXED CHECKPOINT\nReassign historical content at fixed weights\nResponse + achieved coverage',fill='#EFE2D8',edge='#CEB59F',fs=8)
    c.text(.01,-.06,'Generated signal levels calibrate the fixed historical probe and matched retraining responses.',transform=c.transAxes,fontsize=7.5,color=GREY)
    caption='Representation-specific semantic audit. a, Text representations enter paired forecasting configurations while numerical histories, forecast units and event slots are retained. b, Illustrative content assignments distinguish within-entity/split permutation, occurrence-preserving zero content and historical reassignment. The historical family includes block rotation and nearest-vector swapping; the illustration shows one permitted permutation. c, Separately trained conditions quantify SAG and CBTG, whereas selected aligned checkpoints quantify historical perturbation responses with fixed weights. Natural data use recorded event times as availability times; the generator models arrival delays explicitly.'
    save(fig,'fig1_audit_mechanism',axes,list('abc'),columns=[list('abc')],caption=caption)

def fig2():
    rows=load('encoder_summary');seeds=load('encoder_seeds');thresholds=load('encoder_thresholds')
    fig=plt.figure(figsize=(W,6.6));gs=fig.add_gridspec(2,2,left=.09,right=.97,bottom=.12,top=.91,
        wspace=.72,hspace=.6,height_ratios=[2.25,1])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    label(a,'a','Alignment depends on representation\nClusterTrace / Attention')
    for n,encoder in enumerate(ENC):
        summary=next(r for r in rows if r['encoder']==encoder and r['dataset']=='ClusterTrace'and r['family']==FAMILIES[2])
        selected=[r for r in seeds if r['encoder']==encoder and r['dataset']=='ClusterTrace'and r['family']==FAMILIES[2]]
        for j,metric in enumerate(['sag','cbtg']):
            x=j+(n-1)*.20;vals=[r[metric]for r in selected]
            a.plot([x,x],[min(vals),max(vals)],color=EC[encoder],alpha=.35,lw=2.4)
            a.scatter(x+np.array([-.025,0,.025]),vals,s=16,facecolor='white',edgecolor=EC[encoder],linewidth=.8,zorder=3)
            a.scatter([x],[summary[metric]],s=30,marker='D',color=EC[encoder],zorder=4,label=ENAME[encoder]if j==0 else None)
    a.axhline(0,color='#BAC4CB',lw=.8);a.set_xticks([0,1],['SAG','CBTG']);a.set_xlim(-.55,1.55)
    a.set_ylabel('Gain (%)')
    a.legend(loc='upper left',bbox_to_anchor=(-.02,-.12),handletextpad=.3,labelspacing=.6,ncol=3,columnspacing=.7)
    label(b,'b','Matched audit decisions')
    order=[(d,f)for d in DOMAINS for f in FAMILIES]
    b.set_xlim(-.5,2.5);b.set_ylim(14.5,-.5)
    for j,e in enumerate(ENC):
        for i,(d,f)in enumerate(order):
            r=next(r for r in rows if(r['encoder'],r['dataset'],r['family'])==(e,d,f))
            b.add_patch(Rectangle((j-.44,i-.43),.88,.86,facecolor='#F0F3F5',edgecolor='white',lw=.5))
            if r['support']:b.scatter([j],[i],s=44,c=EC[e],marker='o',edgecolor='white',linewidth=.6,zorder=3)
            else:b.scatter([j],[i],s=17,c='#B6C0C8',marker='x',linewidth=.7)
    b.set_xticks(range(3),[ENAME[e]+'\n'+str(sum(r['support']for r in rows if r['encoder']==e))+'/15'for e in ENC])
    b.set_yticks(range(15),[d+' / '+SHORT[f]for d,f in order]);b.tick_params(length=0)
    for y in [2.5,5.5,8.5,11.5]:b.axhline(y,color='white',lw=2)
    for sp in b.spines.values():sp.set_visible(False)
    label(c,'c','Support resolves by effect size')
    for e in ENC:
        rs=[r for r in thresholds if r['encoder']==e]
        c.plot(range(len(rs)),[r['count']for r in rs],marker='o',ms=4,color=EC[e],label=ENAME[e])
    c.set_xticks(range(6),['0','0.01','0.1','0.5','1','2']);c.set_yticks(range(0,8,2));c.set_ylim(-.35,7)
    c.set_xlabel('Minimum gain required in both contrasts (%) · discrete settings');c.set_ylabel('Supported cells / 15')
    c.legend(loc='upper right',ncol=3,handlelength=1.4,columnspacing=1)
    note(fig,'All encoders: seeds 53, 67, 79. Filled map dots satisfy the joint descriptive rule; crosses do not.',.045)
    caption='Representation changes the descriptive audit decision. a, ClusterTrace attention configuration, showing all three paired seed gains (open points), their range, and gains computed from mean condition losses (diamonds). b, All fifteen domain/family configurations for each encoder under the same seeds and rule. A filled dot requires positive SAG and CBTG and at least two aligned wins against each of the two controls; crosses retain all other configurations. c, Support counts with the same win rule and increasing lower bounds on both mean-loss gains. The threshold settings are discrete and are sensitivity choices rather than universal practical-effect cutoffs. Common-seed zero-threshold counts are 2/15, 5/15 and 6/15. The separate ten-seed GPT-2 reference remains 0/15. Full paired seed values appear in Supplementary Fig. S1 and the accompanying CSV.'
    save(fig,'fig2_representation_decisions',[a,b,c],list('abc'),rows=[['a','b']],caption=caption)

def forest(ax,summary,points,key='gain',color=BLUE,offset=0,marker='o',rows=None):
    for i,r in enumerate(summary):
        p=points(r);values=[x[key]for x in p]
        ax.plot([min(values),max(values)],[i+offset,i+offset],color=color,alpha=.4,lw=1.1)
        ax.scatter(values,i+offset+np.linspace(-.045,.045,len(values)),s=8,color=color,alpha=.5,zorder=2)
        ax.scatter([r[key]],[i+offset],s=21,marker=marker,facecolor='white',edgecolor=color,linewidth=1,zorder=3)

def fig3():
    coverage=load('history_coverage');summaries=load('history_summary');seeds=load('history_seeds')
    fig=plt.figure(figsize=(W,6.35))
    gs=fig.add_gridspec(2,1,left=.19,right=.97,top=.92,bottom=.13,hspace=.62,height_ratios=[1,2.4])
    top=gs[0].subgridspec(1,2,wspace=.55);bottom=gs[1].subgridspec(1,3,wspace=.20)
    a,b=[fig.add_subplot(top[i])for i in range(2)]
    lower=[fig.add_subplot(bottom[i])for i in range(3)]
    label(a,'a','Changed forecast histories')
    rs=[next(r for r in coverage if r['dataset']==d and r['condition']=='history_rotation')for d in DOMAINS]
    for i,r in enumerate(rs):
        a.barh(i,100,color='#EDF1F4',height=.55)
        a.barh(i,100*r['changed_histories']/r['histories'],color=BLUE,height=.55)
        a.text(103,i,f"{r['changed_histories']}/{r['histories']}",ha='left',va='center',fontsize=7.5)
    a.set_yticks(range(5),DOMAINS);a.invert_yaxis();a.set_xlim(0,135);a.set_xticks([0,50,100]);a.set_xlabel('Changed histories (%)');a.tick_params(axis='y',length=0)
    label(b,'b','Changed text vectors')
    for cond,col,m in [('history_rotation',BLUE,'o'),('history_nearest_swap',ORANGE,'^')]:
        rs=[next(r for r in coverage if r['dataset']==d and r['condition']==cond)for d in DOMAINS]
        b.plot([100*r['changed_fraction']for r in rs],np.arange(5)+(-.10 if m=='o'else .10),marker=m,ls='',color=col,label='Rotation'if m=='o'else'Nearest swap',ms=4)
    b.set_yticks(range(5),[]);b.set_ylim(4.5,-.5);b.set_xlim(0,105);b.set_xticks([0,50,100]);b.set_xlabel('Changed vectors (%)');b.tick_params(axis='y',length=0)
    handles,names=b.get_legend_handles_labels();fig.legend(handles,names,loc='upper right',bbox_to_anchor=(.98,.995),ncol=2,columnspacing=.8,handletextpad=.4)
    order=[(d,f)for d in DOMAINS for f in FAMILIES]
    for j,(ax,e)in enumerate(zip(lower,ENC)):
        label(ax,chr(99+j),ENAME[e]+' · fixed model')
        for cond,col,m,off in [('history_rotation',BLUE,'o',-.15),('history_nearest_swap',ORANGE,'^',.15)]:
            rs=[next(r for r in summaries if(r['encoder'],r['dataset'],r['family'],r['condition'])==(e,d,f,cond))for d,f in order]
            forest(ax,rs,lambda r:[s for s in seeds if all(s[k]==r[k]for k in ['encoder','dataset','family','condition'])],color=col,offset=off,marker=m)
        ax.axvline(0,color='#B7C2CA',lw=.7);ax.set_xscale('symlog',linthresh=.001,linscale=.7)
        ax.set_xlim(-3,3);ax.set_xticks([-1,-.01,0,.01,1],['−1','−.01','0','.01','1']);ax.set_ylim(14.6,-.6)
        ax.set_yticks(range(15),[d+' / '+SHORT[f]for d,f in order]if j==0 else[])
        ax.tick_params(axis='y',length=0);ax.set_xlabel('Historical gain (%)')
        for y in [2.5,5.5,8.5,11.5]:ax.axhline(y,color='#E6ECF0',lw=.6)
    note(fig,'Points show all seeds; open markers show mean-loss gains; lines show seed ranges.',.056)
    note(fig,'GPT-2: 10 seeds. MiniLM / FinBERT: 3 seeds. Bottom axes use a symmetric-log scale (linear within ±0.001%).',.022)
    caption='Historical interventions connect achieved coverage to fixed-checkpoint response. a, Changed histories and total retained histories for each domain; rotation and nearest swapping change the same history counts. b, The fraction of text slots whose actual vector changes under each control. Coverage counts are identical across the three encoder configurations and three model families, verified before sharing the summaries. c–e, All fifteen domain/family configurations per encoder under both controls, with all seed gains, seed ranges and gains from mean losses. All donor arrivals satisfy the recipient cutoff. Unchanged histories remain in the population. Symmetric-log axes show the wide response range with a linear region of ±0.001%. Optimization seeds describe repeated training on the retained data; they are not independent-domain replicates.'
    save(fig,'fig3_historical_coverage',[a,b]+lower,list('abcde'),rows=[['a','b'],['c','d','e']],caption=caption)

def fig4(metric='sag',supp=False):
    rows=load('calibration_worlds');levels=[0,.05,.1,.25,1];worlds=sorted(set(r['world']for r in rows))
    fig,axes=plt.subplots(2,3,figsize=(W,5.85),gridspec_kw={'height_ratios':[1.25,1]})
    fig.subplots_adjust(left=.10,right=.97,top=.89,bottom=.20,wspace=.40,hspace=.75)
    for j,f in enumerate(FAMILIES):
        ax=axes[0,j];label(ax,chr(97+j),FNAME[f])
        for operation,color,m in [('fixed',BLUE,'o'),('retrained',ORANGE,'s')]:
            key=operation+'_'+metric;means=[];half=[]
            for k,l in enumerate(levels):
                vals=np.array([r[key]for r in rows if r['family']==f and r['strength']==l])
                assert len(vals)==5
                means.append(vals.mean());half.append(stats.t.ppf(.975,4)*stats.sem(vals))
                ax.scatter(k+np.linspace(-.07,.07,5)+( -.07 if operation=='fixed'else .07),vals,s=9,color=color,alpha=.4,zorder=2)
            ax.errorbar(np.arange(5)+(-.07 if operation=='fixed'else .07),means,yerr=half,marker=m,ms=3.4,
                color=color,capsize=2,lw=1,label='Fixed checkpoint'if operation=='fixed'else'Condition-retrained')
        ax.axhline(0,color='#B8C3CB',lw=.6);ax.set_yscale('symlog',linthresh=.1,linscale=.6)
        ax.set_ylim(-4,100);ax.set_yticks([-1,-.1,0,.1,1,10,100],['−1','−0.1','0','0.1','1','10','100'])
        ax.set_xticks(range(5),['0','.05','.1','.25','1']);ax.set_xlabel('Signal setting')
        if j==0:ax.set_ylabel(metric.upper()+' (%) · symlog')
        else:ax.set_yticklabels([])
        ax2=axes[1,j];label(ax2,chr(100+j),'Capability × detection')
        mat=np.zeros((5,5));counts=[]
        for k,l in enumerate(levels):
            counts.append(sum(r['holm_p_'+metric]<.05 and r['fixed_'+metric]>0 for r in rows if r['family']==f and r['strength']==l))
            for n,w in enumerate(worlds):
                r=next(r for r in rows if r['family']==f and r['strength']==l and r['world']==w)
                detect=r['holm_p_'+metric]<.05 and r['fixed_'+metric]>0
                mat[n,k]=int(r['validation_capable'])+2*int(detect)
        cmap=ListedColormap(['#EBEFF2','#AFC9DC','#E5C0A9','#286A95'])
        ax2.imshow(mat,cmap=cmap,vmin=-.5,vmax=3.5,aspect='auto',interpolation='nearest')
        ax2.set_yticks(range(5),[str(w)for w in worlds]if j==0 else[])
        ax2.set_xticks(range(5),[f'{n}/5'for n in counts]);ax2.tick_params(length=0)
        ax2.set_xlabel('Holm detections / 5 worlds',fontsize=7.5)
        if j==0:ax2.set_ylabel('Generated world')
        ax2.set_xticks(np.arange(-.5,5,1),minor=True);ax2.set_yticks(np.arange(-.5,5,1),minor=True)
        ax2.grid(which='minor',color='white',lw=2);ax2.tick_params(which='minor',length=0)
        for s in ax2.spines.values():s.set_visible(False)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,bbox_to_anchor=(.53,.985),columnspacing=2)
    state_labels=['Neither','Capable','Detected','Both'];state_colors=['#EBEFF2','#AFC9DC','#E5C0A9','#286A95']
    for x,t,c in zip([.14,.34,.54,.74],state_labels,state_colors):
        fig.add_artist(Rectangle((x,.091),.024,.022,transform=fig.transFigure,facecolor=c,edgecolor='white'))
        fig.text(x+.032,.102,t,va='center',fontsize=8)
    note(fig,'All 5 worlds per setting are retained. Responses: world means, 95% t intervals and individual world points.',.041)
    note(fig,'Signal levels are discrete; the response axis is linear within ±0.1%. Capability uses held-out validation gain.',.014)
    caption=f'Calibration resolves signal strength, model operation and learner capability. a–c, Fixed-checkpoint and condition-retrained {metric.upper()} at each tested signal setting. Points show all five independent generated worlds; error bars are 95% Student t intervals across worlds. The symmetric-log response axis is linear within ±0.1%. Strengths are discrete settings. d–f, Every world is classified by validation capability (greater than 1% gain over separately trained time-only on validation data) and positive-direction detection after Holm correction across 75 world/strength/family tests for this contrast. Column labels give detections out of five worlds and correspond to the signal settings directly above. Entity losses first average histories and optimization seeds within independent entities. The actual MiniLM generator and its training budgets define this calibration. The fixed historical probe and condition-retrained audit are shown as distinct quantities.'
    name='figS2_calibration_cbtg'if supp else'fig4_signal_calibration'
    save(fig,name,list(axes.flat),list('abcdef'),rows=[list('abc'),list('def')],columns=[['a','d'],['b','e'],['c','f']],caption=caption)

def fig5():
    case=load('real_case');fig=plt.figure(figsize=(W,7.1))
    gs=fig.add_gridspec(3,2,left=.10,right=.97,bottom=.12,top=.93,hspace=.70,wspace=.65,height_ratios=[1.15,1,1.15])
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:]);c=fig.add_subplot(gs[2,0]);d=fig.add_subplot(gs[2,1])
    a.set_axis_off();label(a,'a','A real history contains two arrived notes')
    for i,n in enumerate(case['notes']):
        x=.01+i*.51;col='#DCEAF2'if i==0 else'#F0E1D7'
        words=n['text'].split();excerpt=' '.join(words[:20])+'…'
        excerpt=textwrap.fill(excerpt,width=46)
        box(a,x,.05,.46,.84,'',fill=col)
        a.text(x+.025,.80,f"Note {chr(65+i)} · arrival {n['time_hours']:.2f} h",transform=a.transAxes,fontweight='bold',fontsize=8.2,va='top')
        a.text(x+.025,.59,excerpt,transform=a.transAxes,fontsize=8,va='top',linespacing=1.4)
    a.text(.01,-.12,'Excerpts from the benchmark notes; complete payloads and timestamps accompany the figure data.',transform=a.transAxes,fontsize=7.5,color=GREY)
    label(b,'b','The historical swap changes content while retaining both slots')
    b.set_xlim(0,13.7);b.set_ylim(-.5,1.5);b.set_yticks([1,0],['Aligned','Rotation'])
    b.set_xticks([0,2,4,6,8,10,12]);b.set_xlabel('Hours from history start')
    b.axvline(12,color=INK,lw=.8);b.text(12.15,1.25,'Cutoff',fontsize=8)
    times=[n['time_hours']for n in case['notes']]
    for y in [0,1]:
        cursor=0
        for t in times:
            b.plot([cursor,t-.30],[y,y],color='#CDD5DA',lw=.8)
            cursor=t+.30
        b.plot([cursor,12],[y,y],color='#CDD5DA',lw=.8)
    for i,t in enumerate(times):
        for y,j in [(1,i),(0,1-i)]:
            b.scatter([t],[y],s=210,marker='s',facecolor=BLUE if j==0 else ORANGE,edgecolor='white',zorder=3)
            b.text(t,y,chr(65+j),ha='center',va='center',fontsize=8,color='white',fontweight='bold',zorder=4)
    b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False)
    label(c,'c','Prediction displacement')
    trace=case['traces'];x=np.arange(1,len(trace)+1)
    delta=np.array([r['rotation']-r['aligned']for r in trace])
    c.plot(x,delta,color=BLUE,lw=1.3)
    c.axhline(0,color='#BAC5CD',lw=.7)
    low=min(0.,float(delta.min()));high=max(0.,float(delta.max()));padding=max((high-low)*.16,1e-9)
    c.set_ylim(low-padding,high+padding)
    c.set_xlabel('Observed forecast entry');c.set_ylabel('Δ prediction · n_job')
    c.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    label(d,'d','Historical response across seeds')
    for n,r in enumerate(case['seed_losses']):d.scatter([n],[r['gain']],color=BLUE,s=27,zorder=2)
    d.scatter([3],[case['case_gain']],color=INK,s=34,marker='D',zorder=3)
    d.axhline(0,color='#BAC5CD',lw=.7)
    d.set_xticks([0,1,2,3],['53','67','79','Pooled']);d.set_xlim(-.4,3.4)
    d.set_xlabel('Optimizer seed');d.set_ylabel('Historical gain (%)')
    note(fig,f"ClusterTrace / MiniLM / Attention · window {case['window_id']} · trajectory seed 53 · 3-seed historical gain {case['case_gain']:+.3f}%",.055)
    note(fig,f"Case selected closest to the median absolute effect among all {case['eligible_histories']} changed histories; {case['total_histories']} histories in the full population.",.025)
    caption=f'A traceable historical intervention on a real forecasting window. a, Excerpts of the two source notes from ClusterTrace window {case["window_id"]}; the figure data preserve full payloads, timestamps, entity identity and the donor mapping. b, The executed within-history rotation exchanges these two payloads while retaining their recipient slots, numerical inputs and cutoff. c, The difference between rotated and aligned seed-53 attention predictions with MiniLM for every retained n_job entry, in train-standardized coordinates. The horizontal coordinate is forecast-entry order rather than inferred event time. d, Paired historical gains on all channels for optimizer seeds 53, 67 and 79; the pooled diamond is computed from the three mean condition losses. This window is selected by a fixed median-absolute-effect rule among all {case["eligible_histories"]} changed test histories, with ties resolved by sample ID. The complete selection population, targets and original forecasts accompany the figure. Population responses appear in Fig. 3.'
    save(fig,'fig5_real_history_case',[a,b,c,d],list('abcd'),rows=[['c','d']],columns=[['a','b']],caption=caption)

def fig6():
    summaries=load('source_summary');seeds=load('source_seeds')
    variants=['full_entity_normalization','cache_text_origin','retain_empty_training_windows','released_mixed_units']
    titles=['Whole-entity\nscaling','Cached text\norigin','Empty-history\ninclusion','RecAvg\ntime units']
    fig,axes=plt.subplots(1,4,figsize=(W,6.2));fig.subplots_adjust(left=.20,right=.90,top=.84,bottom=.19,wspace=.27)
    order=[(d,f)for d in DOMAINS for f in FAMILIES]
    for j,(ax,var,title)in enumerate(zip(axes,variants,titles)):
        label(ax,chr(97+j),title)
        rs=[]
        actualorder=order if j<3 else[(d,FAMILIES[1])for d in DOMAINS]
        for d,f in actualorder:
            r=next(r for r in summaries if(r['dataset'],r['family'],r['variant'])==(d,f,var))
            rs.append(dict(r,change=r['canonical_scale_change_percent']))
        forest(ax,rs,lambda r:[s for s in seeds if all(s[k]==r[k]for k in ['dataset','family','variant'])],key='change',color=BLUE)
        ax.axvline(0,color='#BDC7CD',lw=.7);ax.set_ylim(len(actualorder)-.7,-.7)
        ax.set_yticks(range(len(actualorder)),[d+' / '+SHORT[f]for d,f in actualorder]if j==0 else(DOMAINS if j==3 else[]))
        ax.tick_params(axis='y',length=0,pad=3)
        ax.set_xlabel('MSE change (%)')
        if j<3:
            for y in [2.5,5.5,8.5,11.5]:ax.axhline(y,color='#E6ECF0',lw=.6)
        if j==0:ax.set_xlim(-85,42);ax.set_xticks([-75,0,25])
        if j==1:ax.set_xlim(-14,23);ax.set_xticks([-10,0,20])
        if j==2:ax.set_xlim(-13,13);ax.set_xticks([-10,0,10])
        if j==3:
            ax.yaxis.tick_right();ax.set_xlim(-1.95,.5);ax.set_xticks([-1,0])
    fig.suptitle('Implementation choices shift loss on common forecast units',x=.55,y=.97,fontsize=10,fontweight='bold')
    note(fig,'Open markers: ratio of mean losses. Small points and ranges: all 3 paired optimizer seeds.',.10)
    note(fig,'Positive values denote higher MSE. Panel a uses whole-entity statistics, including future observations.',.062)
    note(fig,'RepoHealth empty-history variants add 0 training windows (ΔN = 0). Each panel states its own numeric scale.',.025)
    caption='Implementation sensitivity with paired seeds and a common loss coordinate. a–c, All fifteen domain/family configurations under each of three general variants. d, The five applicable RecAvg mixed-time-unit configurations. Each small point is the percentage loss change for one paired optimizer seed; horizontal ranges span the three seed values. Open markers use the ratio of mean variant and canonical losses. Whole-entity-scale predictions are transformed back to train-only coordinates before scoring against canonical targets. Positive values denote higher MSE. All fifty applicable configurations and all 150 variant predictions are retained. The RepoHealth empty-history variant adds no windows, so its zero effect represents an unchanged training population. Numeric axis ranges differ across panels.'
    save(fig,'fig6_implementation_effects',list(axes),list('abcd'),rows=[list('abcd')],caption=caption)

def figS1():
    rows=load('encoder_summary');seeds=load('encoder_seeds');order=[(d,f)for d in DOMAINS for f in FAMILIES]
    fig,axes=plt.subplots(1,3,figsize=(W,7.0));fig.subplots_adjust(left=.21,right=.97,top=.88,bottom=.14,wspace=.23)
    for j,(ax,e)in enumerate(zip(axes,ENC)):
        label(ax,chr(97+j),ENAME[e]+' · matched seeds')
        rs=[next(r for r in rows if(r['encoder'],r['dataset'],r['family'])==(e,d,f))for d,f in order]
        for metric,color,m,off in [('sag',BLUE,'o',-.15),('cbtg',ORANGE,'s',.15)]:
            modified=[dict(r,gain=r[metric])for r in rs]
            ps=lambda r:[dict(s,gain=s[metric])for s in seeds if all(s[k]==r[k]for k in ['encoder','dataset','family'])]
            forest(ax,modified,ps,color=color,offset=off,marker=m)
        ax.axvline(0,color='#B9C5CE',lw=.7);ax.set_ylim(14.7,-.7)
        ax.set_yticks(range(15),[d+' / '+SHORT[f]for d,f in order]if j==0 else[]);ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Gain (%)');ax.set_xscale('symlog',linthresh=.01,linscale=.6);ax.set_xlim(-50,50)
        ax.set_xticks([-10,-.1,0,.1,10],['−10','−.1','0','.1','10'])
        for y in [2.5,5.5,8.5,11.5]:ax.axhline(y,color='#E6ECF0',lw=.6)
    fig.legend([plt.Line2D([],[],marker='o',color=BLUE,ls=''),plt.Line2D([],[],marker='s',color=ORANGE,ls='')],['SAG','CBTG'],loc='upper center',ncol=2,bbox_to_anchor=(.56,.98))
    note(fig,'All 45 domain/family/encoder configurations, using the same 3 seeds. Points: seed gains; open markers: mean-loss gains.',.06)
    note(fig,'Horizontal ranges span the paired seed values. Symmetric-log axes are linear within ±0.01%.',.025)
    caption='Complete matched-seed encoder effects. All five domains and three fusion families are shown for each representation using seeds 53, 67 and 79. SAG and CBTG are separately encoded by color and shape. Small points show seed-specific ratios, lines show their observed ranges, and open markers use the ratio of condition mean losses. Symmetric-log axes are linear within ±0.01%. The complete data are retained even when the joint descriptive support rule is not satisfied.'
    save(fig,'figS1_encoder_seed_effects',list(axes),list('abc'),rows=[list('abc')],caption=caption)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args()
    funcs={'1':fig1,'2':fig2,'3':fig3,'4':fig4,'5':fig5,'6':fig6,'S1':figS1,'S2':lambda:fig4('cbtg',True)}
    for key,func in funcs.items():
        if not args.only or key in args.only:func();print('Rendered figure '+key,flush=True)

if __name__=='__main__':main()
