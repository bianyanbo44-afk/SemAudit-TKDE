"""Rebuild the paired figure records and render the eight scientific figures."""
from pathlib import Path
import argparse
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'figure_source/source'


def load(name, text=None):
    path = SOURCES / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if text is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(text, str(path), 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-data', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCES))
    if args.rebuild_data:
        path = SOURCES / 'figure_v2_data.py'
        text = path.read_text(encoding='utf-8').replace("ROOT/'manuscript/source_data/", "ROOT/'figure_source/input_records/")
        module = load('figure_v2_data', text)
        module.ROOT = ROOT
        module.RESULTS = ROOT / 'experiments/review_followup_20261105/results'
        module.OUT = ROOT / 'figure_source'
        module.DATA = module.OUT / 'data'
        module.main()
    plot = load('plot_figures_v2')
    sys.argv = [str(SOURCES / 'plot_figures_v2.py')]
    plot.main()


if __name__ == '__main__':
    main()
