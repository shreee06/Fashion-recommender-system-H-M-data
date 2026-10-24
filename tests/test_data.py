import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from fashion_recsys.data import TRANSACTION_TYPES, csv_to_parquet

HEADER = "t_dat,customer_id,article_id,price,sales_channel_id\n"


def _write_csv(path, n):
    rows = [
        f"2020-01-{(i % 28) + 1:02d},cust{i:04d},{i:010d},{0.01 * (i + 1):.6f},{1 + i % 2}\n"
        for i in range(n)
    ]
    path.write_text(HEADER + "".join(rows))
    return rows


def test_article_id_keeps_leading_zeros(tmp_path):
    src, dst = tmp_path / "t.csv", tmp_path / "t.parquet"
    src.write_text(HEADER + "2020-01-01,c1,0108775015,0.05,2\n2020-01-02,c2,0000000001,0.1,1\n")
    csv_to_parquet(src, dst, TRANSACTION_TYPES)
    t = pq.read_table(dst)
    assert t.schema.field("article_id").type == pa.string()
    assert t["article_id"].to_pylist() == ["0108775015", "0000000001"]


def test_dtypes(tmp_path):
    src, dst = tmp_path / "t.csv", tmp_path / "t.parquet"
    _write_csv(src, 3)
    csv_to_parquet(src, dst, TRANSACTION_TYPES)
    s = pq.read_schema(dst)
    assert s.field("t_dat").type == pa.date32()
    assert s.field("customer_id").type == pa.string()
    assert s.field("price").type == pa.float32()
    assert s.field("sales_channel_id").type == pa.int8()
    assert pq.read_table(dst)["t_dat"][0].as_py() == datetime.date(2020, 1, 1)


def test_chunk_boundaries(tmp_path):
    src, dst = tmp_path / "t.csv", tmp_path / "t.parquet"
    n = 500
    _write_csv(src, n)
    # a tiny block size forces many chunks and rows that straddle block edges
    written = csv_to_parquet(src, dst, TRANSACTION_TYPES, block_size=1024)
    ref = tmp_path / "ref.parquet"
    csv_to_parquet(src, ref, TRANSACTION_TYPES, block_size=1 << 20)
    got, want = pq.read_table(dst), pq.read_table(ref)
    assert written == n and got.num_rows == n
    assert got.equals(want)
    assert got["article_id"].to_pylist() == [f"{i:010d}" for i in range(n)]
    assert got["customer_id"].to_pylist() == [f"cust{i:04d}" for i in range(n)]
