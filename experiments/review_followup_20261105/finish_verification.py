"""Wait for every declared training phase, then verify all saved evidence."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EXPECTED = {'encoders': 270, 'neural_calibration': 675, 'source_execution': 150}


def main():
    while True:
        complete = True
        for phase, expected in EXPECTED.items():
            path = ROOT / 'results' / phase / 'COMPLETE.json'
            if not path.is_file():
                complete = False
                continue
            record = json.loads(path.read_text(encoding='utf8'))
            if record['fits'] != expected:
                raise RuntimeError(f'Unexpected completed fit count for {phase}')
        if complete:
            break
        time.sleep(20)
    commands = [
        ['analyze_followup.py', '--phase', 'all'],
        ['verify_checkpoints.py', '--phase', 'all'],
        ['test_followup_design.py'],
    ]
    for arguments in commands:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    (ROOT / 'results/ALL_EXPERIMENTS_VERIFIED.json').write_text(
        json.dumps(dict(completed_fits=sum(EXPECTED.values()), phases=EXPECTED,
                        all_loss_summaries_recomputed=True,
                        all_checkpoints_reloaded=True), indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
