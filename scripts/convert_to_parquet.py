"""Convert the raw H&M CSVs in data/raw to Parquet in data/processed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_recsys.data import (  # noqa: E402
    ARTICLE_TYPES,
    CUSTOMER_TYPES,
    TRANSACTION_TYPES,
    csv_to_parquet,
)

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"

JOBS = [
    ("articles", ARTICLE_TYPES, 64 << 20),
    ("customers", CUSTOMER_TYPES, 64 << 20),
    ("transactions_train", TRANSACTION_TYPES, 64 << 20),
]

if __name__ == "__main__":
    for name, types, block in JOBS:
        n = csv_to_parquet(RAW / f"{name}.csv", OUT / f"{name}.parquet", types, block)
        print(f"{name}: {n:,} rows")
