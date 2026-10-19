"""Data discovery, schema mapping, inventory, and canonical master dataset.

Implements plan Phases 1-2 (sections 5-6):
- recursive discovery of csv/xlsx/xls under the immutable raw folders;
- robust, config-driven column alias mapping (never hard-coded filenames);
- per-file inventory + data dictionary with provenance fields;
- canonical master_reviews.parquet with source_file/source_folder/source_row/
  bank/bank_type and removal logging for every dropped row (Rule 5).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

import src.config as sc
from src.utils import (
    RemovalLog,
    save_json,
    setup_logging,
    sha1_short,
)

logger = setup_logging("data_loader")

CANONICAL_TEXT_COLS = [
    "review_text",
    "review_id",
    "rating",
    "review_date",
    "app_version",
    "reply_text",
    "reply_date",
    "user_name",
    "thumbs_up",
    "user_image",
    "review_version",
]

PROVENANCE_COLS = ["source_file", "source_folder", "source_row", "bank", "bank_type"]


def _map_columns(columns) -> dict[str, str]:
    """Map raw column names to canonical names via config aliases.

    Returns {canonical: raw_name}. Case-insensitive; first alias wins.
    """
    cfg = sc.get_config()["loader"]["column_aliases"]
    lowered = {}
    for col in columns:
        lowered.setdefault(str(col).strip().lower(), str(col))
    mapping: dict[str, str] = {}
    for canonical, aliases in cfg.items():
        for alias in aliases:
            key = str(alias).strip().lower()
            if key in lowered:
                mapping[canonical] = lowered[key]
                break
    return mapping


def _bank_name_from_file(path) -> str:
    stem = path.stem
    for suffix in ("_reviews", "-reviews", "_review"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.strip()


def _count_csv_rows(path) -> int:
    """Fast line count (data rows, header excluded). Returns -1 on failure."""
    try:
        with open(path, "rb") as fh:
            n = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
        return max(n - 1, 0)
    except OSError:
        return -1


def _count_csv_rows_exact(path) -> tuple[int, str]:
    """Exact data-row count via pyarrow (handles quoted multiline values).

    Returns (rows, rows_source) where rows_source is 'exact' or
    'estimated_from_lines' (fallback when pyarrow cannot parse).
    """
    try:
        import pyarrow.csv as pacsv

        opts = pacsv.ParseOptions(newlines_in_values=True)
        table = pacsv.read_csv(str(path), parse_options=opts)
        return int(table.num_rows), "exact"
    except Exception:
        return _count_csv_rows(path), "estimated_from_lines"


def read_any(path, nrows=None):
    """Read csv/xlsx/xls trying configured encodings. Returns DataFrame."""
    encodings = sc.get_config()["loader"]["encoding_candidates"]
    suffix = path.suffix.lower()
    last_err = None
    if suffix in (".xlsx", ".xls"):
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        return pd.read_excel(path, nrows=nrows, engine=engine)
    for enc in encodings:
        try:
            if nrows is None:
                # pyarrow engine: fastest, but strict about malformed lines
                try:
                    return pd.read_csv(path, encoding=enc, engine="pyarrow")
                except Exception:
                    pass  # fall back to tolerant c-engine below
            return pd.read_csv(
                path,
                nrows=nrows,
                encoding=enc,
                engine="c",
                on_bad_lines="warn",
            )
        except Exception as e:  # try next encoding
            last_err = e
    raise last_err if last_err else RuntimeError(f"Cannot read {path}")


def discover_files() -> list[dict]:
    """Recursively find supported raw files with their bank/type folders."""
    cfg = sc.get_config()
    exts = tuple(cfg["loader"]["supported_extensions"])
    found = []
    for raw_dir, bank_type in sc.resolve_raw_dirs():
        if not raw_dir.exists():
            logger.warning("Raw dir missing (reported, not invented): %s", raw_dir)
            continue
        for path in sorted(raw_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                found.append(
                    {
                        "path": path,
                        "source_file": path.name,
                        "source_folder": raw_dir.name,
                        "bank": _bank_name_from_file(path),
                        "bank_type": bank_type,
                    }
                )
    return found


def sniff_file(info: dict) -> dict:
    """Inspect header + small sample of one raw file; map its schema.

    Never reads the full large file here (plan section 5). Returns an
    inventory record including `usable` and a machine-readable `status`.
    """
    path = info["path"]
    cfg = sc.get_config()["loader"]
    rec = {
        "source_file": info["source_file"],
        "source_folder": info["source_folder"],
        "bank": info["bank"],
        "bank_type": info["bank_type"],
        "file_type": path.suffix.lower().lstrip("."),
        "file_size_bytes": path.stat().st_size,
    }
    try:
        head = read_any(path, nrows=cfg["sample_rows_for_sniff"])
    except Exception as e:  # unreadable file: report, never guess
        rec.update(
            usable=False,
            status=f"read_error:{type(e).__name__}",
            rows=0,
            columns=0,
            column_names="",
        )
        return rec

    mapping = _map_columns(list(head.columns))
    rec["columns"] = int(len(head.columns))
    rec["column_names"] = ";".join(str(c) for c in head.columns)
    for canon, field in [
        ("review_text", "review_column"),
        ("rating", "rating_column"),
        ("review_date", "date_column"),
        ("review_id", "review_id_column"),
        ("review_version", "review_version_column"),
        ("app_version", "app_version_column"),
        ("reply_text", "reply_column"),
        ("reply_date", "reply_date_column"),
    ]:
        rec[field] = mapping.get(canon, "")

    missing = [c for c in cfg["required_columns"] if c not in mapping]
    if missing:
        rec.update(usable=False, status="missing_required_columns:" + ",".join(missing))
    else:
        rec.update(usable=True, status="ok")

    if rec["file_type"] == "csv":
        rows, rows_source = _count_csv_rows_exact(path)
    else:
        rows, rows_source = -2, "unknown_xlsx"
    if rows == 0 and path.stat().st_size <= 8:
        rec.update(usable=False, status="empty_file", rows=0, rows_source=rows_source)
    elif rows == 0:
        rec.update(
            usable=False, status="header_only_or_empty", rows=0, rows_source=rows_source
        )
    else:
        rec["rows"] = rows
        rec["rows_source"] = rows_source
    return rec


def build_inventory(force: bool = False) -> pd.DataFrame:
    """Inventory every raw file -> data/inventory/data_inventory.csv
    plus data_dictionary.json describing the canonical schema and mapping."""
    paths = sc.get_paths()
    out_csv = paths["inventory_dir"] / "data_inventory.csv"
    out_dict = paths["inventory_dir"] / "data_dictionary.json"
    records = [sniff_file(info) for info in discover_files()]
    inv = pd.DataFrame.from_records(records)
    inv.to_csv(out_csv, index=False)

    canonical = {
        "review_id": "Stable review identifier if provided by source.",
        "review_text": "Raw review text (never modified in master).",
        "rating": "Star rating 1-5 if provided.",
        "review_date": "Review timestamp if provided (may be null).",
        "bank": "Derived from file name; never hard-coded.",
        "bank_type": "GOVERNMENT or PRIVATE from raw folder name.",
        "source_file": "Original file name (provenance).",
        "source_folder": "Original folder (provenance).",
        "source_row": "1-based row within source file (provenance).",
    }
    save_dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_schema": canonical,
        "files": [
            {
                "source_file": r["source_file"],
                "bank": r["bank"],
                "bank_type": r["bank_type"],
                "mapped_columns": {
                    c: r.get(f + "_column", "")
                    for c, f in [
                        ("review_text", "review"),
                        ("rating", "rating"),
                        ("review_date", "date"),
                        ("review_id", "review_id"),
                        ("reply_text", "reply"),
                        ("reply_date", "reply_date"),
                        ("app_version", "app_version"),
                    ]
                },
                "usable": bool(r.get("usable")),
                "status": r.get("status"),
                "rows": int(r.get("rows") or 0),
            }
            for r in records
        ],
    }
    save_json(save_dict, out_dict)
    n_ok = int((inv["usable"].eq(True)).sum())
    logger.info("Inventory: %d files (%d usable) -> %s", len(inv), n_ok, out_csv)
    return inv


MASTER_ORDER = [
    "bank",
    "bank_type",
    "review_id",
    "review_text",
    "rating",
    "review_date",
    "app_version",
    "review_version",
    "reply_text",
    "reply_date",
    "user_name",
    "thumbs_up",
    "user_image",
    "source_file",
    "source_folder",
    "source_row",
    "is_empty_text",
]


def _raw_path(source_folder: str, source_file: str) -> Path:
    for d, _ in sc.resolve_raw_dirs():
        if d.name == source_folder:
            return d / source_file
    raise FileNotFoundError(f"{source_folder}/{source_file}")


def _read_usable(path, mapping: dict) -> pd.DataFrame:
    """Read one usable file and project it onto the canonical schema."""
    df = read_any(path)
    df = df.rename(columns={raw: canon for canon, raw in mapping.items()})
    keep = [c for c in CANONICAL_TEXT_COLS if c in df.columns]
    return df.loc[:, keep].copy()


def _normalize_frame(
    df: pd.DataFrame,
    bank: str,
    bank_type: str,
    source_file: str,
    source_folder: str,
    rlog: RemovalLog,
) -> pd.DataFrame:
    """Add provenance, coerce types, flag empty text, dedupe in-file IDs."""
    before = len(df)
    df = df.copy()
    df["bank"] = bank
    df["bank_type"] = bank_type
    df["source_file"] = source_file
    df["source_folder"] = source_folder
    df["source_row"] = range(1, len(df) + 1)
    if "review_id" in df.columns:
        dup = df["review_id"].notna() & df["review_id"].duplicated(keep="first")
    else:
        dup = pd.Series(False, index=df.index)
    n_dup = int(dup.sum())
    if n_dup:
        rlog.add(
            "load:dedupe_review_id_infile",
            "duplicate reviewId within file",
            before,
            n_dup,
            before - n_dup,
            source_file,
        )
        df = df.loc[~dup].copy()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    if "review_text" in df.columns:
        txt = df["review_text"].astype("string")
        df["is_empty_text"] = txt.isna() | (txt.fillna("").str.strip() == "")
    else:
        df["is_empty_text"] = True
    return df


def load_master(force_inventory: bool = False):
    """Build the canonical master dataset with full provenance (plan s.6)."""
    paths = sc.get_paths()
    inv_csv = paths["inventory_dir"] / "data_inventory.csv"
    if force_inventory or not inv_csv.exists():
        inv = build_inventory()
    else:
        inv = pd.read_csv(inv_csv)
    usable = inv[inv["usable"].eq(True)]
    rlog = RemovalLog()
    frames, file_summaries = [], []
    for _, rec in usable.iterrows():
        path = _raw_path(str(rec["source_folder"]), str(rec["source_file"]))
        mapping = _map_columns(
            [c for c in str(rec["column_names"]).split(";") if c and c != "nan"]
        )
        df = _read_usable(path, mapping)
        b_before = len(df)
        df = _normalize_frame(
            df,
            rec["bank"],
            rec["bank_type"],
            rec["source_file"],
            rec["source_folder"],
            rlog,
        )
        frames.append(df)
        file_summaries.append(
            {
                "source_file": rec["source_file"],
                "bank": rec["bank"],
                "bank_type": rec["bank_type"],
                "rows_before": int(b_before),
                "rows_after": int(len(df)),
                "rows_removed_infile_dupes": int(b_before - len(df)),
            }
        )
        logger.info("Loaded %-34s rows=%d", rec["source_file"], len(df))

    if not frames:
        raise RuntimeError("No usable raw files found - check inventory report.")
    master = pd.concat(frames, ignore_index=True)

    before = len(master)
    dup = master["review_id"].notna() & master.duplicated(
        subset=["bank", "review_id"], keep="first"
    )
    n_dup = int(dup.sum())
    if n_dup:
        rlog.add(
            "load:dedupe_review_id_global",
            "duplicate (bank, review_id) across files",
            before,
            n_dup,
            before - n_dup,
            "ALL_FILES",
        )
        master = master.loc[~dup].copy()

    rid = master["review_id"].astype("string")
    missing_id = rid.isna() | (rid.fillna("").str.strip() == "")
    if missing_id.any():
        idx = master.index[missing_id.fillna(False)]
        master.loc[idx, "review_id"] = [
            sha1_short(b, f, r)
            for b, f, r in zip(
                master.loc[idx, "bank"],
                master.loc[idx, "source_file"],
                master.loc[idx, "source_row"],
            )
        ]

    # Normalize dtypes for reliable parquet conversion (pandas 3 + pyarrow):
    # object columns with mixed NaN floats/strings break arrow conversion.
    str_cols = [
        "review_id",
        "review_text",
        "app_version",
        "review_version",
        "reply_text",
        "user_name",
        "user_image",
        "bank",
        "bank_type",
        "source_file",
        "source_folder",
    ]
    for col in str_cols:
        if col in master.columns:
            master[col] = master[col].astype("string")
    if "source_row" in master.columns:
        master["source_row"] = master["source_row"].astype("int32")
    if "thumbs_up" in master.columns:
        master["thumbs_up"] = pd.to_numeric(
            master["thumbs_up"], errors="coerce"
        ).astype("Int64")
    for col in ("review_date", "reply_date"):
        if col in master.columns:
            master[col] = pd.to_datetime(master[col], errors="coerce")

    ordered = [c for c in MASTER_ORDER if c in master.columns]
    ordered += [c for c in master.columns if c not in ordered]
    master = master[ordered]
    master.to_parquet(paths["master_parquet"], index=False, engine="pyarrow")
    rlog.save()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": file_summaries,
        "rows_total": int(len(master)),
        "rows_removed_infile_dupes": int(
            sum(f["rows_removed_infile_dupes"] for f in file_summaries)
        ),
        "rows_removed_global_dupes": n_dup,
        "empty_text_rows": int(master["is_empty_text"].sum()),
        "banks": int(master["bank"].nunique()),
        "removal_log": str(rlog.path),
    }
    save_json(summary, paths["processed_dir"] / "load_summary.json")
    logger.info("Master dataset: %d rows -> %s", len(master), paths["master_parquet"])
    return master, summary


def build_demo_sample(master: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Stratified-by-bank demo sample for fast pipeline validation (s.42)."""
    seed = sc.get_seed()
    n = int(n or 2000)
    parts = []
    for bank, g in master.groupby("bank", sort=False, observed=True):
        take = max(10, int(round(n * len(g) / len(master))))
        parts.append(g.sample(n=min(take, len(g)), random_state=seed))
    demo = pd.concat(parts, ignore_index=True)
    demo = demo.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = sc.get_paths()["samples_dir"] / "demo_reviews.csv"
    demo.to_csv(out, index=False)
    logger.info("Demo sample: %d rows -> %s", len(demo), out)
    return demo


if __name__ == "__main__":
    inventory = build_inventory(force=True)
    cols = ["source_file", "bank", "bank_type", "rows", "usable", "status"]
    print(inventory[cols].to_string(index=False))
    master_df, load_summary = load_master()
    print(
        f"\nMaster rows: {load_summary['rows_total']} | "
        f"banks: {load_summary['banks']} | "
        f"empty text: {load_summary['empty_text_rows']}"
    )
    build_demo_sample(master_df)
