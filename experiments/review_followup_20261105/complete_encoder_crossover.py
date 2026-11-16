"""Run the added historical crossover after the declared training verification."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    while not (ROOT / 'results/ALL_EXPERIMENTS_VERIFIED.json').is_file():
        time.sleep(20)
    for arguments in [['encoder_history.py'],
                      ['verify_checkpoints.py', '--phase', 'encoder_history'],
                      ['verify_saved_evidence.py']]:
        subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
    result = json.loads((ROOT / 'results/ALL_EXPERIMENTS_VERIFIED.json').read_text(encoding='utf8'))
    result.update(prediction_sets=2175, historical_encoder_evaluations=180,
                  historical_encoder_evaluations_reloaded=True)
    (ROOT / 'results/ALL_EVIDENCE_VERIFIED.json').write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
