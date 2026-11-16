"""Replay independent representation evidence without modifying original artifacts."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE.parent / "journal_extension_20261012/external_validation/scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("representation_verifier", SCRIPTS / "verify_outputs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = HERE / "experiments/representations"
module.main()
