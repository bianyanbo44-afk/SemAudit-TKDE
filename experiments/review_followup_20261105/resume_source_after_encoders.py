"""Resume the saved source-effects grid after the encoder phase releases GPU RAM."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
while not (ROOT / 'results/encoders/COMPLETE.json').is_file():
    time.sleep(20)

correction = dict(reason='Concurrent GPU source-effects run reached allocator OOM after 75 completed fits.',
                  action='Resume saved complete cells after encoder completion with unchanged budgets, seeds and data.',
                  preserved_log='source_training_execution.log')
(ROOT / 'results/source_execution/RESOURCE_RESUME.json').write_text(json.dumps(correction, indent=2), encoding='utf8')
result = subprocess.run([sys.executable, 'source_execution.py', '--phase', 'train'], cwd=ROOT)
raise SystemExit(result.returncode)
