import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE = Path("experiments/final_scaleup/analysis/features_kvcf.csv")
OUTPUT = Path("experiments/e9_calibrated/frozen_splits.csv")
SEED = 2027

if OUTPUT.exists():
    raise SystemExit(
        f"Split already exists: {OUTPUT}. Refusing to overwrite."
    )

df = pd.read_csv(SOURCE, usecols=["request_id", "task"])

assert df["request_id"].nunique() == 1331
assert df.groupby("request_id")["task"].nunique().max() == 1

requests = (
    df.drop_duplicates("request_id")
    .sort_values("request_id")
    .reset_index(drop=True)
)

train, rest = train_test_split(
    requests,
    test_size=0.50,
    stratify=requests["task"],
    random_state=SEED
)

calibration, test = train_test_split(
    rest,
    test_size=0.50,
    stratify=rest["task"],
    random_state=SEED
)

splits = pd.concat([
    train.assign(split="train"),
    calibration.assign(split="calibration"),
    test.assign(split="test")
], ignore_index=True)

assert len(splits) == 1331
assert splits["request_id"].is_unique

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
splits.sort_values("request_id").to_csv(OUTPUT, index=False)

digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

manifest = {
    "seed": SEED,
    "source": str(SOURCE),
    "counts": splits["split"].value_counts().to_dict(),
    "sha256": digest
}

OUTPUT.with_suffix(".json").write_text(
    json.dumps(manifest, indent=2)
)

print("\n=== SPLIT COUNTS ===")
print(splits["split"].value_counts())

print("\n=== TASK DISTRIBUTION ===")
print(splits.groupby(["split", "task"]).size())

print("\nSHA256:", digest)
print("\nSUCCESS: Frozen splits created.")
