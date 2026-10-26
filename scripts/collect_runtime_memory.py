"""Copy the real runtime / memory / size numbers of the project into the tracked file
evals/results/runtime_memory.json (the memory logs and the data files themselves are untracked).

Reads: data/processed/{freeze,dryrun,test_run}_memory.log, the parquet/CSV file sizes, /proc/meminfo, and the
tracked result files. Every block names its source.
"""
import json
import os
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys.memguard import MIN_AVAILABLE_KB  # noqa: E402
from run_reranker import CHUNK  # noqa: E402

PROC, RAW, RES = ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT / "evals" / "results"
BLOCK_MB = 64  # block_size=64 << 20 in scripts/convert_to_parquet.py


def mem_log(name):
    peaks, avail = [], []
    for line in (PROC / name).read_text().splitlines():
        m = re.search(r"peak\s+(\d+) MB\s+MemAvailable\s+(\d+) MB", line)
        if m:
            peaks.append(int(m.group(1)))
            avail.append(int(m.group(2)))
    return {"peak_rss_mb": max(peaks), "min_mem_available_mb": min(avail), "stages_logged": len(peaks), "source": f"data/processed/{name} (untracked)"}


def main():
    tables = ["articles", "customers", "transactions_train"]
    csv_b = {t: (RAW / f"{t}.csv").stat().st_size for t in tables}
    pq_b = {t: (PROC / f"{t}.parquet").stat().st_size for t in tables}
    rows = {t: pq.ParquetFile(PROC / f"{t}.parquet").metadata.num_rows for t in tables}
    meminfo = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    test, val = json.loads((RES / "test_final.json").read_text()), json.loads((RES / "validation_reference.json").read_text())
    tt, rr = json.loads((RES / "two_tower_validation.json").read_text()), json.loads((RES / "reranker_validation.json").read_text())
    main_tt = tt["final_runs"]["two_tower (chosen)"]
    out = {
        "machine": {"ram_mb": meminfo["MemTotal"] // 1024, "logical_cpus": os.cpu_count(), "cpu_threads_used_for_training_and_inference": 4,
                    "gpu": "none (CPU only)", "source": "/proc/meminfo, os.cpu_count(); torch.set_num_threads(4) in the scripts"},
        "data_scale": {"rows": rows, "source": "parquet metadata of data/processed/*.parquet (untracked)"},
        "conversion": {"csv_bytes": csv_b, "parquet_bytes": pq_b,
                       "total_csv_bytes": sum(csv_b.values()), "total_parquet_bytes": sum(pq_b.values()),
                       "transactions_reduction_pct": 100 * (1 - pq_b["transactions_train"] / csv_b["transactions_train"]),
                       "total_reduction_pct": 100 * (1 - sum(pq_b.values()) / sum(csv_b.values())),
                       "streamed_block_size_mb": BLOCK_MB, "parquet_compression": "zstd",
                       "source": "file sizes of data/raw/*.csv and data/processed/*.parquet (untracked); scripts/convert_to_parquet.py"},
        "memory_guard": {"abort_below_mem_available_mb": MIN_AVAILABLE_KB // 1000, "feature_build_chunk_customers": CHUNK,
                         "source": "src/fashion_recsys/memguard.py, scripts/run_reranker.py"},
        "freeze_run": {**mem_log("freeze_memory.log"), "note": "reconstruction of the reranker (24,000 customers, 7.7M candidate rows)"},
        "dry_run_validation": {**mem_log("dryrun_memory.log"), "runtime_s": val["runtime_s"], "reported_peak_rss_mb": val["peak_rss_mb"],
                               "customers": val["n_evaluated"], "source_runtime": "evals/results/validation_reference.json"},
        "final_test_run": {**mem_log("test_run_memory.log"), "runtime_s": test["runtime_s"], "reported_peak_rss_mb": test["peak_rss_mb"],
                           "customers": test["n_evaluated"], "source_runtime": "evals/results/test_final.json"},
        "two_tower_training_script": {"runtime_s": tt["runtime_s"], "peak_rss_mb": tt["peak_rss_mb"], "sec_per_epoch": main_tt["sec_per_epoch"],
                                      "n_params": main_tt["n_params"], "trials": len(tt["trials"]), "source": "evals/results/two_tower_validation.json"},
        "reranker_validation_script": {"runtime_s": rr["runtime_s"], "peak_rss_mb": rr["peak_rss_mb"], "source": "evals/results/reranker_validation.json"},
    }
    (RES / "runtime_memory.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k in ("machine", "data_scale", "conversion", "freeze_run", "final_test_run")}, indent=1))


if __name__ == "__main__":
    main()
