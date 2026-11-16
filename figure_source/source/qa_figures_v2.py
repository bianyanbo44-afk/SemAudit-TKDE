"""Validate final figure geometry and assemble a captioned review booklet."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
import fitz

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent
OUT=ROOT/'delivery/figures_v2_20261106'if SCRIPT_DIR.name=='tools'else SCRIPT_DIR.parent
SKILL=SCRIPT_DIR if(SCRIPT_DIR/'audit_panel_alignment.py').is_file()else Path.home()/'.codex/skills/nature-figure/scripts'
NAMES=['fig1_audit_mechanism','fig2_representation_decisions','fig3_historical_coverage',
    'fig4_signal_calibration','fig5_real_history_case','fig6_implementation_effects',
    'figS1_encoder_seed_effects','figS2_calibration_cbtg']

def check(name):
    pdf=OUT/(name+'.pdf')
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    collision=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(pdf),
        '--json-out',str(OUT/(name+'.collision.json'))],capture_output=True,text=True,encoding='utf-8',env=env)
    glyph=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True,encoding='utf-8',env=env)
    if glyph.stdout:
        (OUT/(name+'.glyphs.json')).write_text(glyph.stdout,encoding='utf-8')
    c=json.loads((OUT/(name+'.collision.json')).read_text(encoding='utf-8'))
    g=json.loads(glyph.stdout)
    alignment=json.loads((OUT/(name+'.alignment.json')).read_text(encoding='utf-8'))
    result=dict(figure=name,collision_exit=collision.returncode,glyph_exit=glyph.returncode,
        collision_verdict=c.get('verdict'),collision_summary=c.get('summary'),minimum_font_pt=g.get('minimum_found_pt'),
        alignment_verdict=alignment.get('verdict'),width_mm=fitz.open(pdf)[0].rect.width*25.4/72)
    print(json.dumps(result,ensure_ascii=True),flush=True)
    return result

def booklet():
    doc=fitz.open()
    for i,name in enumerate(NAMES):
        source=fitz.open(OUT/(name+'.pdf'));page=doc.new_page(width=595.276,height=841.89)
        heading=('Figure '+str(i+1))if i<6 else('Supplementary Figure S'+str(i-5))
        page.insert_text((38,30),heading,fontsize=11,fontname='hebo',color=(.15,.20,.24))
        target_width=183/25.4*72
        height=source[0].rect.height*target_width/source[0].rect.width
        rect=fitz.Rect(38,48,38+target_width,48+height)
        page.show_pdf_page(rect,source,0)
        caption=(OUT/(name+'.caption.txt')).read_text(encoding='utf-8')
        # Helvetica supports the caption character set after simple typographic normalization.
        for old,new in [('–','-'),('—','-'),('−','-'),('×','x'),('±','+/-')]:caption=caption.replace(old,new)
        position=fitz.Rect(38,rect.y1+12,557,806)
        remaining=page.insert_textbox(position,caption,fontsize=8.5,lineheight=1.25,fontname='helv',color=(.15,.20,.24))
        if remaining<0:raise RuntimeError('Caption does not fit: '+name)
        page.insert_text((548,824),str(i+1),fontsize=8)
    doc.save(OUT/'KBS_Figure_Review_v2.pdf',garbage=4,deflate=True)
    tex=[]
    for name in NAMES:
        caption=(OUT/(name+'.caption.txt')).read_text(encoding='utf-8')
        caption=caption.replace('%',r'\%').replace('&',r'\&').replace('_',r'\_')
        tex.append('\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\textwidth]{'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:'+name+'}\n\\end{figure}\n')
    (OUT/'figure_insertion.tex').write_text('\n'.join(tex),encoding='utf-8')

def main():
    with ThreadPoolExecutor(max_workers=4)as pool:results=list(pool.map(check,NAMES))
    (OUT/'QA_SUMMARY.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(r['collision_exit']!=0 or r['glyph_exit']!=0 or r['alignment_verdict']!='PASS'for r in results):
        raise SystemExit('Resolve figure QA findings before creating delivery booklet.')
    preflight=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(SCRIPT_DIR/'plot_figures_v2.py'),'--json'],capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    (OUT/'SOURCE_PREFLIGHT.json').write_text(preflight.stdout,encoding='utf-8')
    if preflight.returncode:raise SystemExit('Source preflight needs repair.')
    booklet()
    files=[p for p in OUT.rglob('*')if p.is_file()and p.name!='MANIFEST.json']
    manifest=dict(backend='python-matplotlib',figures=8,main_candidates=6,supplementary=2,
        physical_width_mm=183,files=[dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in files])
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Figure QA and review booklet complete.',flush=True)

if __name__=='__main__':main()
