import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "backend" / "model" / "model.h5"
CONFUSION_OUT = SCRIPT_DIR / "confusion_matrix.png"


def main():
    os.chdir(SCRIPT_DIR)

    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}")
        return

    _, _, X_test, _, _, y_test, label_names, _ = load_dataset("data")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    names = [str(x) for x in label_names]
    print(classification_report(y_test, y_pred, target_names=names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick = np.arange(len(names))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_OUT, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {CONFUSION_OUT}")

    confused_pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((cm[i, j], names[i], names[j]))
    confused_pairs.sort(reverse=True, key=lambda x: x[0])
    print("Top-5 most confused sign pairs (count, true, predicted):")
    for row in confused_pairs[:5]:
        print(f"  {row[0]:4d}  {row[1]}  ->  {row[2]}")


if __name__ == "__main__":
    main()
