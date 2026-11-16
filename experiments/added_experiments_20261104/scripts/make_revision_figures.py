"""Supplementary figures inheriting the original SemAudit visual grammar."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "source/figures"
sys.path.insert(0,"C:/Users/Lenovo/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.labelsize":6.8,
    "xtick.labelsize":6.2,"ytick.labelsize":6.2,"axes.linewidth":.8,
    "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
BLUE, AMBER, GRAY, INK = "#0F4D92", "#D98B2B", "#A5ABB2", "#24272B"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readcsv(p):
    with p.open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def save(fig,name,axes,inputs,exclude=()):
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_fontsize()<6:
            obj.set_fontsize(6)
    ids=[chr(97+i) for i in range(len(axes))]
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=[ids],
        exclude_axes=exclude,json_out=HERE/f"qa/{name}_alignment.json",strict=True)
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",pad_inches=.05)
    fig.savefig(OUT/f"{name}.svg",bbox_inches="tight",pad_inches=.05)
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",pad_inches=.05,dpi=600)
    plt.close(fig)
    (OUT/f"{name}_provenance.json").write_text(json.dumps(dict(
        source_files=[dict(file=str(p.relative_to(HERE)),sha256=sha(p)) for p in inputs],
        plotting_script_sha256=sha(Path(__file__)),all_declared_cells_retained=True,
        backend="Python/Matplotlib; original SemAudit Arial/blue/amber/gray style"),indent=2),encoding="utf8")


def calibration():
    source=HERE/"experiments/irregular_fine/results/summary.json"
    worlds=HERE/"experiments/irregular_fine/results/world_results.jsonl"
    data=json.loads(source.read_text())["conditions"]
    raw=[json.loads(line) for line in worlds.read_text().splitlines()]
    strengths=[0,.025,.05,.1,.25,.5,1]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.0))
    fig.subplots_adjust(left=.08,right=.99,top=.79,bottom=.24,wspace=.34)
    inset=axes[0].inset_axes([.10,.53,.55,.37])
    source_rows=[]
    for representation,label,color,marker,offset in [("oracle","Oracle signs",BLUE,"o",-.055),
                                                    ("sentence","MiniLM sentences",AMBER,"s",.055)]:
        cells=[next(c for c in data if c["representation"]==representation and c["strength"]==s) for s in strengths]
        means=[]; intervals=[]
        for s,c in zip(strengths,cells):
            vals=np.array([100*w["primary_fixed_shuffle"]["mean_loss_difference"]/w["test_entity_MSE"]["fixed_shuffle"]
                           for w in raw if w["representation"]==representation and w["strength"]==s])
            mean=float(vals.mean());half=float(t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals)))
            means.append(mean);intervals.append([mean-half,mean+half])
            source_rows.append(dict(representation=representation,strength=s,worlds=len(vals),gain_mean=mean,
                        gain_world_t95_low=mean-half,gain_world_t95_high=mean+half,
                        nominal_detections=c["detected_worlds"],gated_detections=c["gated_detected_worlds"]))
        means=np.array(means);intervals=np.array(intervals)
        x=np.arange(len(strengths))+offset
        axes[0].errorbar(x,means,yerr=np.array([means-intervals[:,0],intervals[:,1]-means]),
                        color=color,marker=marker,ms=3.2,lw=1,capsize=1.7,label=label)
        inset.errorbar(x[:4],means[:4],yerr=np.array([means[:4]-intervals[:4,0],intervals[:4,1]-means[:4]]),
                        color=color,marker=marker,ms=2.4,lw=.8,capsize=1.2)
        for ax,key,ci in [(axes[1],"detection_rate","detection_rate_wilson95"),
                           (axes[2],"gated_detection_rate","gated_rate_wilson95")]:
            y=100*np.array([c[key] for c in cells]);bounds=100*np.array([c[ci] for c in cells])
            ax.errorbar(x,y,yerr=np.maximum(0,np.array([y-bounds[:,0],bounds[:,1]-y])),
                        color=color,marker=marker,ms=3.2,lw=1,capsize=1.7)
    titles=["Achieved alignment effect","Nominal detection","Validation-screened detection"]
    for i,(ax,title) in enumerate(zip(axes,titles)):
        ax.set_title(title,loc="left",fontsize=8.2,fontweight="bold",pad=15)
        ax.text(-.16,1.12,chr(97+i),transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
        ax.set_xticks(np.arange(7),["0",".025",".05",".1",".25",".5","1"])
        ax.set_xlabel("Signal strength (declared grid)")
        ax.tick_params(length=2.5)
        ax.axhline(0,color=GRAY,lw=.6,zorder=0)
    axes[0].set_ylabel("Aligned-over-rotation gain (%)")
    axes[0].set_ylim(-3,100)
    axes[1].set_ylabel("Detected worlds (%)")
    axes[1].set_ylim(-4,106)
    axes[2].set_ylim(-4,106)
    axes[2].set_yticks([0,25,50,75,100])
    axes[1].set_yticks([0,25,50,75,100])
    inset.set_title("Low-signal zoom",fontsize=6.2,pad=2)
    inset.set_xticks(range(4),["0",".025",".05",".1"])
    inset.set_yticks([0,3,6])
    inset.set_ylim(-.3,7.8)
    inset.tick_params(length=2,pad=1)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.54,.015),ncol=2,fontsize=6.8)
    with (OUT/"figS1_calibration_source.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(source_rows[0]));writer.writeheader();writer.writerows(source_rows)
    save(fig,"figS1_irregular_calibration",axes,[source,worlds],exclude=[inset])


def seeds():
    source=HERE/"results/analysis/ten_seed.csv"
    perseed=HERE/"results/analysis/per_seed.csv"
    data,raw=readcsv(source),readcsv(perseed)
    datasets=["ClusterTrace","CESNET","EPA-Air","FNSPID","RepoHealth"]
    families=[("selective_ssm","Selective state-space"),("immtsf_recavg_gradd","RecAvg + GR-Add"),
              ("immtsf_t2v_xattn","T2V-XAttn + XAttn-Add")]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.25),sharey=True)
    fig.subplots_adjust(left=.145,right=.99,top=.79,bottom=.24,wspace=.30)
    for i,(ax,(family,title)) in enumerate(zip(axes,families)):
        for yi,dataset in zip(range(4,-1,-1),datasets):
            row=next(c for c in data if c["dataset"]==dataset and c["family"]==family)
            paired=sorted([c for c in raw if c["dataset"]==dataset and c["family"]==family],key=lambda c:int(c["seed"]))
            assert len(paired)==10
            ax.scatter([float(c["sag"]) for c in paired],yi+np.linspace(-.12,.12,10),
                       s=9,color=GRAY,alpha=.75,zorder=2,linewidths=0)
            center=float(row["sag"]);low=float(row["sag_low"]);high=float(row["sag_high"])
            ax.plot([low,high],[yi,yi],color=BLUE,lw=1.8,zorder=3)
            ax.scatter([center],[yi],s=24,color=BLUE,marker="D",edgecolor="white",linewidth=.5,zorder=4)
        ax.axvline(0,color=INK,ls="--",lw=.8,zorder=0)
        ax.set_title(title,loc="left",fontsize=8.0,fontweight="bold",pad=16)
        ax.text(-.13,1.12,chr(97+i),transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
        ax.text(0,1.025,"10 paired seeds; conditional on fixed data",transform=ax.transAxes,fontsize=6,color="#777777")
        ax.set_xlabel("SAG (%); positive favors aligned")
        ax.tick_params(length=2.5)
        ax.set_ylim(-.55,4.55)
        ax.tick_params(axis="y",length=0)
    axes[0].set_yticks(range(4,-1,-1),datasets)
    handles=[Line2D([],[],color=GRAY,marker="o",lw=0,ms=3,label="Individual paired seed"),
             Line2D([],[],color=BLUE,marker="D",lw=1.6,ms=3,label="Ratio of mean losses + 95% paired-seed interval")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.54,.02),ncol=2,fontsize=6.5)
    save(fig,"figS2_ten_seed_stability",axes,[source,perseed])


if __name__=="__main__":
    calibration();seeds()
    print("Two evidence figures exported from complete saved results.")
