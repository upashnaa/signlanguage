from typing import Optional


class PredictionSmoother:
    def __init__(
        self,
        threshold: float = 0.85,
        min_frames: int = 8,
        cooldown_frames: int = 15,
    ):
        self.threshold = threshold
        self.min_frames = min_frames
        self.cooldown_frames = cooldown_frames
        self._last_emitted: str = ""
        self._consecutive_word: Optional[str] = None
        self._consecutive_count: int = 0
        self._cooldown_remaining: int = 0

    @property
    def last_emitted(self) -> str:
        return self._last_emitted

    def reset(self) -> None:
        self._last_emitted = ""
        self._consecutive_word = None
        self._consecutive_count = 0
        self._cooldown_remaining = 0

    def tick(self) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def update(self, word: Optional[str], confidence: float) -> Optional[str]:
        if self._cooldown_remaining > 0:
            self._consecutive_word = None
            self._consecutive_count = 0
            return None

        if word is None or word == "":
            self._consecutive_word = None
            self._consecutive_count = 0
            return None

        if confidence < self.threshold:
            self._consecutive_word = None
            self._consecutive_count = 0
            return None

        if word == self._consecutive_word:
            self._consecutive_count += 1
        else:
            self._consecutive_word = word
            self._consecutive_count = 1

        if (
            self._consecutive_count >= self.min_frames
            and self._consecutive_word is not None
        ):
            emitted = self._consecutive_word
            self._last_emitted = emitted
            self._cooldown_remaining = self.cooldown_frames
            self._consecutive_word = None
            self._consecutive_count = 0
            return emitted

        return None
