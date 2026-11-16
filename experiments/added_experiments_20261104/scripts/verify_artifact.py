"""Portable verification of hashes and every indexed neural prediction loss."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for record in manifest:
        file = root / record["file"]
        assert file.is_file(), str(file)
        assert hashlib.sha256(file.read_bytes()).hexdigest()==record["sha256"], str(file)
    index = json.loads((root / "PREDICTION_INDEX.json").read_text())
    errors = []
    for record in index:
        z = np.load(root / record["file"])
        error = (z["prediction"].astype(np.float64)-z["target"].astype(np.float64))**2
        count = z["mask"].sum((0,1))
        valid = count>0
        mse = ((error*z["mask"]).sum((0,1))[valid]/count[valid]).mean()
        delta = abs(float(mse)-record["expected_mse"])
        assert delta < 2e-5*max(1.,mse), (record,delta)
        assert len(z["sample_id"])==len(z["prediction"])
        errors.append(delta)
    report = dict(all_hashes_match=True, file_count=len(manifest), prediction_cells=len(index),
                  maximum_mse_replay_error=max(errors),
                  scope="Derived-loss and file-integrity replay; not a substitute for independently training a model.")
    (root / "VERIFICATION.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
