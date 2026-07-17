from __future__ import annotations
from collections import deque
from statistics import fmean, pstdev
from typing import Any, Iterable


class QuestionEngine:
    """Asks simple pattern-matched questions when confusion persists."""

    DEFAULT_VOCABULARY = (
        "object", "event", "zone", "safe", "danger", "unknown",
        "red", "dark", "near",
    )

    def __init__(
        self,
        vocabulary: Iterable[str] | None = None,
        confusion_threshold: float = 0.08,   # ANTES: 0.2 — demasiado alto
        required_steps: int = 8,              # ANTES: 20 — demasiado alto
    ) -> None:
        self.vocabulary = list(vocabulary or self.DEFAULT_VOCABULARY)
        self.confusion_threshold = confusion_threshold
        self.required_steps = required_steps
        self.confusion_steps = 0
        self.last_confusion = 0.0
        self.last_surprise = 0.0
        self._surprise_history: deque[float] = deque(maxlen=200)
        self._steps_since_last_question = 0   # NUEVO: cooldown entre preguntas
        self._min_cooldown = 500              # NUEVO: esperar 500 pasos entre preguntas

    def observe(self, emotional_state: Any, surprise: float | None = None) -> bool:
        values = self._state_values(emotional_state)
        confusion = 0.0
        if isinstance(values, dict):
            confusion = self._clamp(values.get("confusion", 0.0))

        self.last_confusion = confusion
        if surprise is not None:
            self.last_surprise = self._clamp(surprise)
        elif isinstance(values, dict):
            self.last_surprise = self._clamp(values.get("surprise", self.last_surprise))

        surprise_is_unusual = self._surprise_above_baseline(self.last_surprise)
        self._surprise_history.append(self.last_surprise)

        if confusion > self.confusion_threshold or surprise_is_unusual:
            self.confusion_steps += 1
        else:
            self.confusion_steps = 0

        self._steps_since_last_question += 1
        return self.should_ask()

    def should_ask(self) -> bool:
        # Nunca dejar al agente en silencio durante más de 2000 pasos.
        if self._steps_since_last_question > 2000:
            return True

        # Cooldown: no preguntar si acaba de hacer una pregunta
        if self._steps_since_last_question < self._min_cooldown:
            return False

        # Condición principal: confusión sostenida
        if self.confusion_steps > self.required_steps:
            return True

        # Condición secundaria: sorpresa alta con algo de confusión acumulada
        if self.last_surprise > 0.01 and self.confusion_steps > 5:
            return True

        return False

    def formulate(self, emotional_state: Any, observation: Any, memory: Any) -> str:
        self._sync_vocabulary(memory)
        # Resetear cooldown al formular (el agente preguntó)
        self._steps_since_last_question = 0
        self.confusion_steps = 0

        values = self._state_values(emotional_state)
        fear = self._clamp(values.get("fear", 0.0)) if isinstance(values, dict) else 0.0
        surprise = self._current_surprise(values)

        description_subject = None
        if isinstance(observation, dict):
            description = str(observation.get("description") or "").strip()
            if description:
                description_subject = self._subject_from_description(description)

        if description_subject is None and isinstance(observation, dict):
            active_window = str(observation.get("active_window") or "").strip()
            clipboard_text = str(observation.get("clipboard_text") or "").strip()
            if active_window and "." not in active_window:
                description_subject = active_window
            elif clipboard_text:
                first_word = clipboard_text.split()[0].lower() if clipboard_text.split() else None
                if first_word and self._has_word(first_word):
                    description_subject = first_word

        room_object = self._object_from_observation(observation)
        event = self._event_from_observation(observation)
        zone = self._zone_from_observation(observation)

        if (fear > 0.45 or self._danger_near(observation)) and self._has_word("safe"):
            return f"is {description_subject or zone} safe?"
        if (surprise > 0.45 or event != "event") and (
            description_subject is not None or self._has_word(event)
        ):
            return f"why does {description_subject or event} happen?"
        return f"what is {description_subject or room_object}?"

    def reset(self) -> None:
        self.confusion_steps = 0
        self.last_confusion = 0.0
        self.last_surprise = 0.0
        self._steps_since_last_question = 0

    # ── métodos internos sin cambios ──────────────────────────────────────────

    def _sync_vocabulary(self, memory: Any) -> None:
        vocabulary = getattr(memory, "vocabulary", None)
        if vocabulary is None:
            language = getattr(memory, "language", None)
            vocabulary = getattr(language, "vocabulary", None)
        if vocabulary is None:
            return
        self.vocabulary = [str(word).lower() for word in vocabulary if str(word).strip()]

    def _object_from_observation(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return self._known_word("unknown", "object")
        for cell in observation.get("surroundings", []):
            if not isinstance(cell, dict):
                continue
            room_object = cell.get("object")
            if isinstance(room_object, dict):
                name = str(room_object.get("name") or room_object.get("type") or "").lower()
                color = str(room_object.get("color") or "").lower()
                if name and self._has_word(name):
                    return name
                if color and self._has_word(color):
                    return color
                return self._known_word("object", "unknown")
            if isinstance(room_object, str) and self._has_word(room_object):
                return room_object.lower()
        return self._known_word("unknown", "object")

    def _subject_from_description(self, description: str) -> str | None:
        description = description.lower().strip()
        if not description:
            return None
        object_names = ("lamp", "bell", "wanderer", "switch", "stone", "mirror", "plant", "portal", "crystal")
        colors = ("red", "blue", "gold", "green", "gray", "silver", "purple", "cyan", "yellow")
        events = ("danger", "noise", "moved", "glow", "grow", "teleport")

        for name in object_names:
            if name in description:
                for color in colors:
                    if f"{color} {name}" in description:
                        return f"{color} {name}"
                return name
        for event in events:
            if event in description:
                return event
        for color in colors:
            if color in description:
                return color
        return None

    def _event_from_observation(self, observation: Any) -> str:
        if isinstance(observation, dict):
            event = str(observation.get("event", "")).lower().strip()
            if event and self._has_word(event):
                return event
            if self._danger_near(observation):
                return self._known_word("danger", "event")
            if self._object_from_observation(observation) != "unknown":
                return self._known_word("touch", "event")
        return self._known_word("event", "surprise")

    def _zone_from_observation(self, observation: Any) -> str:
        if isinstance(observation, dict):
            position = observation.get("agent_position")
            if isinstance(position, list | tuple) and len(position) >= 2:
                return f"zone {position[0]} {position[1]}"
            zone = observation.get("zone")
            if zone is not None:
                return f"zone {zone}"
        return self._known_word("zone", "near")

    def _danger_near(self, observation: Any) -> bool:
        if not isinstance(observation, dict):
            return False
        for cell in observation.get("surroundings", []):
            if isinstance(cell, dict) and cell.get("danger", False):
                return True
        return False

    def _current_surprise(self, values: Any) -> float:
        if isinstance(values, dict):
            return self._clamp(values.get("surprise", self.last_surprise))
        return self.last_surprise

    def _surprise_above_baseline(self, surprise: float) -> bool:
        if not self._surprise_history:
            return surprise > 0.000050

        mean = fmean(self._surprise_history)
        deviation_threshold = max(1.5 * pstdev(self._surprise_history), 0.000050)
        return surprise - mean > deviation_threshold

    def _state_values(self, emotional_state: Any) -> Any:
        if isinstance(emotional_state, dict):
            return emotional_state
        if hasattr(emotional_state, "values"):
            return emotional_state.values()
        return emotional_state

    def _has_word(self, word: str) -> bool:
        return word.lower().strip() in {item.lower() for item in self.vocabulary}

    def _known_word(self, preferred: str, fallback: str) -> str:
        if self._has_word(preferred):
            return preferred
        if self._has_word(fallback):
            return fallback
        return self.vocabulary[0] if self.vocabulary else fallback

    def _clamp(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))
