"""Run preserved follow-up scripts using the relocated project dependencies."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / 'experiments/review_followup_20261105'


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python runtime/run_followup.py SCRIPT.py [arguments]')
    script = (FOLLOWUP / sys.argv[1]).resolve()
    if script.parent != FOLLOWUP or not script.is_file():
        raise ValueError('Choose a top-level preserved follow-up script')
    os.environ['SEMAUDIT_PROJECT_ROOT'] = str(ROOT / 'runtime/phase3_prototype')
    os.environ['SEMAUDIT_EXTENSION_ROOT'] = str(ROOT / 'runtime/wsdm_support')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(FOLLOWUP))
    import common
    common.PREVIOUS = ROOT / 'experiments/added_experiments_20261104'
    sys.argv = [str(script), *sys.argv[2:]]
    os.chdir(FOLLOWUP)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
