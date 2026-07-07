from __future__ import annotations


class EmotionalState:
    """Adaptive internal emotion values driven by experience."""

    _SOFT_MIN = 0.03
    _SOFT_MAX = 0.97

    def __init__(
        self,
        curiosity: float = 0.5,
        fear: float = 0.0,
        confidence: float = 0.2,
        confusion: float = 0.2,
        learning_rate: float = 0.1,
    ) -> None:
        self.curiosity = self._clamp(curiosity)
        self.fear = self._clamp(fear)
        self.confidence = self._clamp(confidence)
        self.confusion = self._clamp(confusion)
        self.learning_rate = learning_rate

    def update(self, surprise: float, was_reset: bool, is_familiar: bool) -> dict[str, float]:
        surprise = self._clamp(surprise)
        safe_experience = not was_reset and is_familiar and surprise < 0.35

        curiosity_target = surprise
        confusion_target = surprise * (0.7 if is_familiar else 1.0)
        fear_target = 1.0 if was_reset else self.fear * 0.85
        confidence_target = 1.0 if safe_experience else self.confidence * 0.9

        self.curiosity = self._move_toward(self.curiosity, curiosity_target)
        self.confusion = self._move_toward(self.confusion, confusion_target)
        self.fear = self._move_toward(self.fear, fear_target)
        self.confidence = self._move_toward(self.confidence, confidence_target)

        if was_reset:
            self.confidence = self._move_toward(self.confidence, 0.0)
        elif safe_experience:
            self.fear = self._move_toward(self.fear, 0.0)
            self.confusion = self._move_toward(self.confusion, 0.0)

        return self.values()

    def dominant(self) -> str:
        return max(self.values(), key=self.values().get)

    def values(self) -> dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "fear": self.fear,
            "confidence": self.confidence,
            "confusion": self.confusion,
        }

    def _move_toward(self, current: float, target: float) -> float:
        return self._clamp(current + (target - current) * self.learning_rate)

    def _clamp(self, value: float) -> float:
        return max(self._SOFT_MIN, min(float(value), self._SOFT_MAX))
