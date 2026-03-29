import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from dataset_loader import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_MODEL_DIR = SCRIPT_DIR.parent / "backend" / "model"
MODEL_OUT = BACKEND_MODEL_DIR / "model.h5"
LABELS_OUT = BACKEND_MODEL_DIR / "labels.txt"
CURVES_OUT = SCRIPT_DIR / "training_curves.png"


def main():
    os.chdir(SCRIPT_DIR)

    (
        X_train,
        X_val,
        _X_test,
        y_train,
        y_val,
        _y_test,
        label_names,
        encoder,
    ) = load_dataset("data")

    num_classes = len(encoder.classes_)
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(30, 258)),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(MODEL_OUT))

    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        for name in label_names:
            f.write(f"{name}\n")

    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])
    print(f"Final train accuracy: {train_acc:.4f}")
    print(f"Final val accuracy: {val_acc:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="train_loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(history.history["accuracy"], label="train_acc", linestyle="--")
    ax2.plot(history.history["val_accuracy"], label="val_acc", linestyle="--")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CURVES_OUT, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {CURVES_OUT}")


if __name__ == "__main__":
    main()
