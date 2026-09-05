import json
from pathlib import Path
import pandas as pd

try:
    from .step0_config import TRAIN_CSV, TEST_CSV, INPUT_ROWS_DIR
except (ImportError, ValueError):
    from pipeline_nli.step0_config import TRAIN_CSV, TEST_CSV, INPUT_ROWS_DIR

def normalize_text(s: str):
    return s.strip() if s else ""

def save_row_json(row: pd.Series, out_dir: Path):
    obj = row.to_dict()
    for key in ["backstory", "content", "book_name", "char"]:
        if key in obj and obj[key] is not None:
            obj[key] = normalize_text(str(obj[key]))
    out_path = out_dir / f"{obj['id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path

def load_and_save(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path, encoding='utf-8').fillna("")
    for _, row in df.iterrows():
        save_row_json(row, out_dir)
    return df

def run():
    if TRAIN_CSV.exists():
        load_and_save(TRAIN_CSV, INPUT_ROWS_DIR)
    if TEST_CSV.exists():
        load_and_save(TEST_CSV, INPUT_ROWS_DIR)

if __name__ == "__main__":
    run()
