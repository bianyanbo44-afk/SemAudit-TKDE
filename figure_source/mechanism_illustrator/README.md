# TKDE Figure 1 Illustrator master

This directory contains the mechanism figure integrated into the TKDE manuscript on 17 November 2026.

- `fig1_audit_mechanism.ai` contains 278 native vector paths and 73 editable text frames.
- `fig1_audit_mechanism.svg` is the editable vector source.
- `fig1_audit_mechanism.pdf` is the publication vector export at 183 mm width.
- `fig1_audit_mechanism.png` is the 600 dpi preview.
- `caption.tex` and `caption.txt` contain the current manuscript caption.
- `verification.json` contains file hashes, font sizes and native-object checks.

`python tools/reproduce_figures.py` from the repository root copies these approved mechanism exports to the figure directory and regenerates the seven quantitative figures.

To revise the mechanism, open the AI or SVG in Illustrator, edit its native objects and export the updated PDF and PNG. The original artwork was constructed through Illustrator 30.0.0 with the scientific-illustrator-agent SVG workflow.

The source SVG can also be generated with `python figure_source/mechanism_illustrator/source/build_mechanism.py` on Windows with Python, Pillow and Arial installed. This writes the SVG and its design records beside the authoring script. The source consists entirely of vector primitives and editable text.
