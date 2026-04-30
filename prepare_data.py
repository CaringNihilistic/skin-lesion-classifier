"""
src/prepare_data.py — Download ISIC 2018 dataset from Kaggle and prepare folder structure.

NO CHANGES FROM ORIGINAL — this file was correct.
Included here for completeness so you have the full set of files.

Dataset: ISIC 2018 Task 3 — Skin Lesion Classification
Classes: 7 skin conditions, ~10,000 images total

Run: python src/prepare_data.py
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/raw")
DATA_SPLIT = Path("data/split")       # train/ val/ test/ with class subfolders

CLASS_NAMES = {
    "MEL":   "Melanoma",
    "NV":    "Melanocytic_Nevi",
    "BCC":   "Basal_Cell_Carcinoma",
    "AKIEC": "Actinic_Keratosis",
    "BKL":   "Benign_Keratosis",
    "DF":    "Dermatofibroma",
    "VASC":  "Vascular_Lesion",
}

# Short codes used in CSV → folder names
CODE_TO_FOLDER = {code: name for code, name in CLASS_NAMES.items()}

# ── STEP 1: Download from Kaggle ─────────────────────────────────────────────
def download_dataset():
    """
    You need a Kaggle API key: https://www.kaggle.com/docs/api
    Place kaggle.json at C:/Users/<you>/.kaggle/kaggle.json
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading ISIC 2018 dataset from Kaggle...")
    os.system(
        "kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 "
        f"-p {DATA_RAW} --unzip"
    )
    print("Download complete.")

# ── STEP 2: Parse labels CSV ──────────────────────────────────────────────────
def load_labels():
    """
    HAM10000 metadata CSV has columns:
    image_id, dx (diagnosis code), dx_type, age, sex, localization
    """
    csv_candidates = list(DATA_RAW.rglob("HAM10000_metadata.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            "HAM10000_metadata.csv not found. "
            "Make sure download completed successfully."
        )
    df = pd.read_csv(csv_candidates[0])
    print(f"Total images in metadata: {len(df)}")
    print(f"Class distribution:\n{df['dx'].value_counts()}\n")
    return df

# ── STEP 3: Build train/val/test splits ───────────────────────────────────────
def build_splits(df, train=0.70, val=0.15, test=0.15, seed=42):
    """
    Stratified split so all 7 classes are proportionally represented
    in every split.
    """
    df["label"] = df["dx"].str.upper()

    train_df, temp_df = train_test_split(
        df, test_size=(val + test), stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test / (val + test),
        stratify=temp_df["label"], random_state=seed
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

# ── STEP 4: Copy images into split/class subfolders ───────────────────────────
def copy_to_splits(train_df, val_df, test_df):
    """
    Creates:
      data/split/train/Melanoma/ISIC_0024306.jpg  ...
      data/split/val/Melanoma/...
      data/split/test/Melanoma/...

    PyTorch ImageFolder expects this exact structure.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Collect all image paths from raw download
    all_images = {
        p.stem: p
        for p in DATA_RAW.rglob("*.jpg")
    }
    print(f"Found {len(all_images)} images on disk.")

    missing = 0
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            img_id  = row["image_id"]
            code    = row["label"]
            folder  = CODE_TO_FOLDER.get(code, code)
            dest    = DATA_SPLIT / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)

            src = all_images.get(img_id)
            if src is None:
                missing += 1
                continue
            shutil.copy2(src, dest / src.name)

    if missing:
        print(f"Warning: {missing} images not found on disk (metadata mismatch).")

    # Summary
    for split_name in ["train", "val", "test"]:
        total = sum(
            len(list((DATA_SPLIT / split_name / c).glob("*.jpg")))
            for c in CODE_TO_FOLDER.values()
            if (DATA_SPLIT / split_name / c).exists()
        )
        print(f"{split_name}: {total} images copied")

# ── STEP 5: Save label map ────────────────────────────────────────────────────
def save_label_map():
    import json
    label_map = {i: name for i, name in enumerate(sorted(CODE_TO_FOLDER.values()))}
    with open("data/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved → data/label_map.json")
    print(label_map)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
    df = load_labels()
    train_df, val_df, test_df = build_splits(df)
    copy_to_splits(train_df, val_df, test_df)
    save_label_map()
    print("\nData preparation complete. Run train.py next.")
