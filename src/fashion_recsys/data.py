"""CSV -> Parquet conversion for the H&M tables with explicit dtypes."""
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

TRANSACTION_TYPES = {
    "t_dat": pa.date32(),
    "customer_id": pa.string(),
    "article_id": pa.string(),  # leading zeros must survive
    "price": pa.float32(),
    "sales_channel_id": pa.int8(),
}
CUSTOMER_TYPES = {
    "customer_id": pa.string(),
    "FN": pa.float32(),
    "Active": pa.float32(),
    "club_member_status": pa.string(),
    "fashion_news_frequency": pa.string(),
    "age": pa.int16(),
    "postal_code": pa.string(),
}
ARTICLE_TYPES = {
    "article_id": pa.string(),
    "product_code": pa.string(),
    "prod_name": pa.string(),
    "detail_desc": pa.string(),
}


def csv_to_parquet(src, dst, column_types, block_size=64 << 20):
    """Stream `src` CSV into a Parquet file `dst` block by block.

    Memory use is bounded by `block_size`; returns the number of rows written.
    """
    reader = pacsv.open_csv(
        str(src),
        read_options=pacsv.ReadOptions(block_size=block_size),
        convert_options=pacsv.ConvertOptions(
            column_types=column_types, strings_can_be_null=True
        ),
    )
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with pq.ParquetWriter(str(dst), reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += batch.num_rows
    return rows


def load_transaction_codes(processed_dir, cache=True):
    """Load transactions as compact integer arrays (memory-light).

    Returns (day, cust, art, customer_ids, article_ids): `day` is days since epoch
    (int32), `cust`/`art` are row positions in customers/articles parquet (int32),
    and the id arrays map codes back to the original string ids. Only the three
    needed columns are read, row group by row group; the codes are cached next to
    the parquet files.
    """
    import numpy as np
    import pyarrow.compute as pc

    processed_dir = Path(processed_dir)
    customer_ids = pq.read_table(processed_dir / "customers.parquet", columns=["customer_id"])["customer_id"].combine_chunks()
    article_ids = pq.read_table(processed_dir / "articles.parquet", columns=["article_id"])["article_id"].combine_chunks()
    cache_file = processed_dir / "transactions_codes.npz"
    if cache and cache_file.exists():
        z = np.load(cache_file)
        return z["day"], z["cust"], z["art"], customer_ids, article_ids

    pf = pq.ParquetFile(processed_dir / "transactions_train.parquet")
    day, cust, art = [], [], []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["t_dat", "customer_id", "article_id"])
        day.append(t["t_dat"].cast(pa.int32()).to_numpy())
        for col, ids, out in (("customer_id", customer_ids, cust), ("article_id", article_ids, art)):
            idx = pc.index_in(t[col], value_set=ids)
            if idx.null_count:
                raise ValueError(f"{col} values missing from the dimension table")
            out.append(idx.to_numpy(zero_copy_only=False).astype(np.int32))
    day, cust, art = (np.concatenate(x) for x in (day, cust, art))
    if cache:
        np.savez(cache_file, day=day, cust=cust, art=art)
    return day, cust, art, customer_ids, article_ids


def load_price_channel(processed_dir, cache=True):
    """Price (float32) and sales channel (int8) of every transaction, in the same row order
    as `load_transaction_codes` (cached next to the parquet files)."""
    import numpy as np

    processed_dir = Path(processed_dir)
    cache_file = processed_dir / "transactions_price_channel.npz"
    if cache and cache_file.exists():
        z = np.load(cache_file)
        return z["price"], z["channel"]
    pf = pq.ParquetFile(processed_dir / "transactions_train.parquet")
    price, chan = [], []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["price", "sales_channel_id"])
        price.append(t["price"].to_numpy())
        chan.append(t["sales_channel_id"].to_numpy())
    price, chan = np.concatenate(price), np.concatenate(chan)
    if cache:
        np.savez(cache_file, price=price, channel=chan)
    return price, chan
