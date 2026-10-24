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
