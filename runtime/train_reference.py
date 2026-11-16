"""Run the preserved seven-seed extension with repository-local dependencies."""
from pathlib import Path
import hashlib
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/added_experiments_20261104/scripts/extend_seeds.py'


def main():
    text = SOURCE.read_text(encoding='utf-8')
    bindings = {
        'PROJECT = HERE.parents[1] / "phase3_prototype"':
            'PROJECT = Path(__file__).resolve().parents[3] / "runtime/phase3_prototype"',
        'EXTENSION = HERE.parent / "journal_extension_20261012"':
            'EXTENSION = Path(__file__).resolve().parents[3] / "runtime/wsdm_support"',
        'HERE / "REVISION_PROTOCOL.md"': 'HERE / "EXPERIMENT_PROTOCOL.md"',
    }
    for old, new in bindings.items():
        if text.count(old) != 1:
            raise ValueError('Preserved reference source changed')
        text = text.replace(old, new)
    module = types.ModuleType('semaudit_reference_training')
    module.__file__ = str(SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(text, str(SOURCE), 'exec'), module.__dict__)
    module.audit.sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    module.main()


if __name__ == '__main__':
    main()
