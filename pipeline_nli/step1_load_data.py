"""
STEP 1: Load data from train.csv and test.csv, normalize and save per-row JSON artifacts.
"""
import json
from pathlib import Path
import pandas as pd
from step0_config import TRAIN_CSV, TEST_CSV, INPUT_ROWS_DIR


def normalize_text(s: str):
    if s is None:
        return ""
    return s.strip()


def save_row_json(row: pd.Series, out_dir: Path):
    obj = row.to_dict()
    # Normalize textual fields
    for key in ["backstory", "content", "book_name", "char"]:
        if key in obj and obj[key] is not None:
            obj[key] = normalize_text(str(obj[key]))
    out_path = out_dir / f"{obj['id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path


def load_and_save(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    # Basic normalization
    df = df.fillna("")
    for _, row in df.iterrows():
        save_row_json(row, out_dir)
    return df


if __name__ == "__main__":
    print("Loading train and test CSVs and saving per-row artifacts...")
    if TRAIN_CSV.exists():
        print(f"Reading {TRAIN_CSV}")
        load_and_save(TRAIN_CSV, INPUT_ROWS_DIR)
    else:
        print(f"Train CSV not found at {TRAIN_CSV}")
    if TEST_CSV.exists():
        print(f"Reading {TEST_CSV}")
        load_and_save(TEST_CSV, INPUT_ROWS_DIR)
    else:
        print(f"Test CSV not found at {TEST_CSV}")
