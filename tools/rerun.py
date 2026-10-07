"""Apply the validation-split fix to the ECG notebook and re-execute it end to end.

Run from the repository root:  python tools/rerun.py
Use --no-run to apply the edits without executing.
"""
import re
import sys

import nbformat

NOTEBOOK = "MED_ECG_Classification_with_cnn.ipynb"

DOWNLOAD_CELL = '''# Data: ECG Heartbeat Categorization Dataset (MIT-BIH Arrhythmia), downloaded from Kaggle
import os
import kagglehub

DATA_DIR = kagglehub.dataset_download("shayanfazeli/heartbeat")
print(sorted(os.listdir(DATA_DIR)))'''

OLD_FIT = "history=model.fit(X_train, y_train,epochs=5,callbacks=callbacks, batch_size=32,validation_data=(X_test,y_test))"
NEW_FIT = '''# Hold out 10% of the training data for validation so the test set stays unseen until evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train.argmax(axis=1))
    history=model.fit(X_tr, y_tr,epochs=5,callbacks=callbacks, batch_size=32,validation_data=(X_val,y_val))'''

REPLACEMENTS = [
    ("from keras.utils.np_utils import to_categorical", "from tensorflow.keras.utils import to_categorical"),
    ('"/content/drive/MyDrive/medical imaging/medical imaging/mitbih_train.csv"', 'os.path.join(DATA_DIR, "mitbih_train.csv")'),
    ('"/content/drive/MyDrive/medical imaging/medical imaging/mitbih_test.csv"', 'os.path.join(DATA_DIR, "mitbih_test.csv")'),
    (OLD_FIT, NEW_FIT),
    ("import keras\nfrom keras.callbacks", "import keras\nkeras.utils.set_random_seed(42)  # reproducible runs\nfrom keras.callbacks"),
]


def apply_edits(nb):
    log = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "drive.mount(" in src:
            src = DOWNLOAD_CELL
            log.append("replaced Google Drive mount with dataset download")
        for old, new in REPLACEMENTS:
            if old in src and new not in src:
                src = src.replace(old, new)
                log.append("replaced: " + old[:60])
        cell.source = src
    return log


def summarise(nb):
    lines, errors = [], []
    pat = re.compile(r"^(Accuracy|\s+accuracy|\s+macro avg|\s+weighted avg|\s+\d\s)|val_accuracy", re.I)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                errors.append(f"cell {i}: {out.ename}: {out.evalue}")
            text = out.get("text", "") if out.output_type == "stream" else ""
            for line in text.splitlines():
                if pat.search(line):
                    lines.append(f"cell {i}: {line.strip()[-160:]}")
    return lines, errors


def main():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for entry in apply_edits(nb):
        print("EDIT:", entry)
    if "--no-run" in sys.argv:
        nbformat.write(nb, NOTEBOOK)
        return
    from nbclient import NotebookClient

    NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=True).execute()
    lines, errors = summarise(nb)
    print("\n".join(lines))
    with open("RESULTS.md", "w") as fh:
        fh.write("# Results from the latest notebook run\n\n```\n" + "\n".join(lines) + "\n```\n")
        if errors:
            fh.write("\n## Cells that raised errors\n\n```\n" + "\n".join(errors) + "\n```\n")
    if errors:
        print("ERRORS:\n" + "\n".join(errors))
    nbformat.write(nb, NOTEBOOK)
    with open("run_errors.txt", "w") as fh:
        fh.write("\n".join(errors))


if __name__ == "__main__":
    main()
