import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LABELS = [
    "hello",
    "thanks",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "good",
    "bad",
    "more",
    "stop",
    "love",
    "what",
    "where",
    "who",
    "how",
    "name",
    "friend",
    "eat",
    "drink",
    "water",
    "home",
    "work",
    "learn",
]


class InferenceEngine:
    def __init__(
        self,
        model_path: str = "model/model.h5",
        sequence_length: int = 30,
    ):
        self.sequence_length = sequence_length
        self.sequence: List[np.ndarray] = []

        backend_dir = Path(__file__).resolve().parent
        self._model_path = (
            Path(model_path)
            if os.path.isabs(model_path)
            else backend_dir / model_path
        )
        labels_path = backend_dir / "model" / "labels.txt"

        self.labels: List[str] = list(DEFAULT_LABELS)
        if labels_path.is_file():
            with open(labels_path, "r", encoding="utf-8") as f:
                loaded = [line.strip() for line in f if line.strip()]
            if loaded:
                self.labels = loaded

        self.model = None
        if self._model_path.is_file():
            try:
                import tensorflow as tf

                self.model = tf.keras.models.load_model(str(self._model_path))
                logger.info("Loaded model from %s", self._model_path)
            except Exception as e:
                logger.warning("Could not load model from %s: %s", self._model_path, e)
                self.model = None
        else:
            logger.warning(
                "Model not found at %s — using mock predictions (demo mode)",
                self._model_path,
            )

    def add_frame(self, keypoints: np.ndarray) -> None:
        self.sequence.append(np.asarray(keypoints, dtype=np.float32).reshape(-1))
        if len(self.sequence) > self.sequence_length:
            self.sequence = self.sequence[-self.sequence_length :]

    def predict(self) -> Optional[Dict[str, Any]]:
        if len(self.sequence) < self.sequence_length:
            return None

        batch = np.expand_dims(
            np.stack(self.sequence[-self.sequence_length :], axis=0), axis=0
        )

        if self.model is None:
            n = len(self.labels)
            idx = random.randrange(n)
            word = self.labels[idx]
            confidence = random.uniform(0.6, 0.99)
            rest = (1.0 - confidence) / max(n - 1, 1) if n > 1 else 0.0
            all_scores = [rest] * n
            all_scores[idx] = float(confidence)
            return {
                "word": word,
                "confidence": float(confidence),
                "all_scores": all_scores,
            }

        preds = self.model.predict(batch, verbose=0)
        probs = np.asarray(preds[0], dtype=np.float64).reshape(-1)
        if np.any(probs < 0) or abs(np.sum(probs) - 1.0) > 0.01:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        top_idx = int(np.argmax(probs))
        word = self.labels[top_idx] if top_idx < len(self.labels) else "unknown"
        confidence = float(probs[top_idx])
        all_scores = [float(p) for p in probs]
        return {
            "word": word,
            "confidence": confidence,
            "all_scores": all_scores,
        }

    def reset(self) -> None:
        self.sequence = []
