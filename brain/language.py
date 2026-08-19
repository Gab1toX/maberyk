from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Any


class LanguageEngine:
    """Learns word associations from internal state patterns, without an LLM."""

    SEED_VOCABULARY = (
        "explore",
        "fear",
        "curious",
        "familiar",
        "strange",
        "red",
        "move",
        "touch",
        "danger",
        "safe",
        "unknown",
        "interesting",
        "boring",
        "surprise",
        "confused",
        "learning",
        "dark",
        "near",
        "far",
        "again",
    )

    def __init__(self, learning_rate: float = 0.1, seed: int | None = None) -> None:
        self.learning_rate = learning_rate
        self.random = random.Random(seed)
        self.vocabulary = list(self.SEED_VOCABULARY)
        self._vocabulary_lookup = {word.lower() for word in self.vocabulary}
        self.associations: dict[str, dict[str, float]] = {
            word: defaultdict(float) for word in self.vocabulary
        }
        self.expression_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._seed_associations()

    def express(self, emotional_state: Any, observation: Any, memory: Any) -> str:
        features = self._extract_features(emotional_state, observation, memory)
        # The step loop calls express() continuously; keep normalized vocabulary
        # cached instead of rebuilding a lowercase set on every generated thought.
        vocabulary = self._vocabulary_lookup

        def known_word(word: Any) -> str:
            word = str(word or "").lower().strip()
            if word in vocabulary:
                return word
            base = word.rstrip("_0123456789")
            return base if base and base in vocabulary else "algo"

        def first_object() -> tuple[str | None, str | None]:
            if not isinstance(observation, dict):
                return None, None

            for cell in observation.get("surroundings", []):
                if not isinstance(cell, dict):
                    continue

                room_object = cell.get("object")
                if isinstance(room_object, dict):
                    object_name = room_object.get("name") or room_object.get("type")
                    color = room_object.get("color")
                    return known_word(object_name), known_word(color)
                if isinstance(room_object, str):
                    return known_word(room_object), None

            return None, None

        def dominant_emotion() -> str:
            if hasattr(emotional_state, "dominant"):
                dominant = emotional_state.dominant()
                if isinstance(dominant, tuple):
                    dominant = dominant[0]
                return known_word(dominant)

            values = emotional_state.values() if hasattr(emotional_state, "values") else emotional_state
            if isinstance(values, dict) and values:
                emotion = max(values.items(), key=lambda item: item[1])[0]
                return known_word(emotion)

            return "algo"

        object_name, color = first_object()
        event = ""
        if isinstance(observation, dict):
            event = str(observation.get("event", "")).lower()

        if "danger" in event or "reset" in event:
            phrase = "siento peligro aqui"
        elif "touch" in event or "noise" in event or "changed_color" in event:
            object_name = object_name or "algo"
            phrase = f"toque {object_name} y algo paso"
        elif "moved" in event and object_name is None:
            phrase = "me muevo y exploro"
        elif object_name is not None:
            color = color or "algo"
            phrase = f"veo un {object_name} {color} cerca"
        else:
            emotion_word = dominant_emotion()
            phrase = f"siento {emotion_word} en este lugar"

        selected_words = [
            word
            for word in phrase.lower().split()
            if word in self.associations
        ]
        self.expression_history.append(
            {
                "phrase": phrase,
                "words": selected_words,
                "features": features,
                "accuracy": None,
            }
        )
        return phrase

    def learn(
        self,
        expression: str,
        emotional_state: Any,
        observation: Any,
        memory: Any,
        accuracy: float,
    ) -> None:
        features = self._extract_features(emotional_state, observation, memory)
        words = [word for word in expression.lower().split() if word in self.associations]
        self._update_words(words, features, accuracy)

    def improve_from_last_expression(self, accuracy: float) -> None:
        if not self.expression_history:
            return

        last_expression = self.expression_history[-1]
        last_expression["accuracy"] = accuracy
        self._update_words(
            last_expression["words"],
            last_expression["features"],
            accuracy,
        )

    def add_word(self, word: str) -> None:
        word = word.lower().strip()
        if word and word not in self.associations:
            self.vocabulary.append(word)
            self._vocabulary_lookup.add(word)
            self.associations[word] = defaultdict(float)

    def refresh_vocabulary_cache(self) -> None:
        """Rebuild normalized lookup after checkpoint restores mutate vocabulary."""
        self._vocabulary_lookup = {word.lower() for word in self.vocabulary}

    def _update_words(self, words: list[str], features: dict[str, float], accuracy: float) -> None:
        accuracy = self._clamp(accuracy)
        direction = (accuracy - 0.5) * 2.0

        for word in words:
            for feature, value in features.items():
                current = self.associations[word][feature]
                adjustment = self.learning_rate * direction * value
                self.associations[word][feature] = self._clamp_weight(current + adjustment)

    def _score_word(self, word: str, features: dict[str, float]) -> float:
        weights = self.associations[word]
        return sum(weights[feature] * value for feature, value in features.items())

    def _extract_features(self, emotional_state: Any, observation: Any, memory: Any) -> dict[str, float]:
        features: dict[str, float] = {}
        features.update(self._emotion_features(emotional_state))
        features.update(self._observation_features(observation))
        features.update(self._memory_features(memory, observation))
        return features

    def _emotion_features(self, emotional_state: Any) -> dict[str, float]:
        values = emotional_state.values() if hasattr(emotional_state, "values") else emotional_state
        if not isinstance(values, dict):
            return {}

        return {
            "emotion.curiosity": self._clamp(values.get("curiosity", 0.0)),
            "emotion.fear": self._clamp(values.get("fear", 0.0)),
            "emotion.confidence": self._clamp(values.get("confidence", 0.0)),
            "emotion.confusion": self._clamp(values.get("confusion", 0.0)),
        }

    def _observation_features(self, observation: Any) -> dict[str, float]:
        features = {
            "observation.object_near": 0.0,
            "observation.danger_near": 0.0,
            "observation.red_near": 0.0,
            "observation.dark": 0.0,
            "observation.unknown": 0.0,
        }

        if not isinstance(observation, dict):
            features["observation.unknown"] = 1.0
            return features

        surroundings = observation.get("surroundings", [])
        if not surroundings:
            features["observation.dark"] = 1.0

        known_cells = 0
        for cell in surroundings:
            if not isinstance(cell, dict):
                continue

            known_cells += float(cell.get("inside", False))
            if cell.get("danger", False):
                features["observation.danger_near"] = 1.0

            room_object = cell.get("object")
            if room_object is not None:
                features["observation.object_near"] = 1.0
                if room_object.get("color") == "red":
                    features["observation.red_near"] = 1.0

        if surroundings and known_cells / len(surroundings) < 0.5:
            features["observation.dark"] = 1.0

        return features

    def _memory_features(self, memory: Any, observation: Any) -> dict[str, float]:
        features = {
            "memory.familiar": 0.0,
            "memory.strange": 1.0,
            "memory.surprising": 0.0,
            "memory.repeated": 0.0,
        }

        if memory is None:
            return features

        if hasattr(memory, "memory_feature_snapshot"):
            try:
                snapshot = memory.memory_feature_snapshot(observation)
            except Exception:
                snapshot = None
            if isinstance(snapshot, dict):
                features.update(
                    {
                        "memory.familiar": self._clamp(snapshot.get("memory.familiar", 0.0)),
                        "memory.strange": self._clamp(snapshot.get("memory.strange", 1.0)),
                        "memory.surprising": self._clamp(snapshot.get("memory.surprising", 0.0)),
                        "memory.repeated": self._clamp(snapshot.get("memory.repeated", 0.0)),
                    }
                )
                return features

        try:
            similar = memory.recall_similar(observation, limit=3)
            summary = memory.summarize()
        except Exception:
            return features

        if similar:
            best_similarity = similar[0].get("similarity", 0.0)
            features["memory.familiar"] = self._clamp(best_similarity)
            features["memory.strange"] = self._clamp(1.0 - best_similarity)

        total_episodes = summary.get("total_episodes", 0)
        average_surprise = summary.get("average_surprise", 0.0)
        features["memory.surprising"] = self._clamp(average_surprise)
        features["memory.repeated"] = self._clamp(total_episodes / 100.0)
        return features

    def _seed_associations(self) -> None:
        seed_weights = {
            "explore": {"emotion.curiosity": 0.8, "memory.strange": 0.4},
            "fear": {"emotion.fear": 1.0, "observation.danger_near": 0.8},
            "curious": {"emotion.curiosity": 1.0, "memory.strange": 0.3},
            "familiar": {"memory.familiar": 1.0, "emotion.confidence": 0.5},
            "strange": {"memory.strange": 1.0, "emotion.confusion": 0.4},
            "red": {"observation.red_near": 1.0},
            "move": {"emotion.curiosity": 0.4},
            "touch": {"observation.object_near": 0.8, "emotion.curiosity": 0.3},
            "danger": {"observation.danger_near": 1.0, "emotion.fear": 0.7},
            "safe": {"emotion.confidence": 1.0, "memory.familiar": 0.5},
            "unknown": {"memory.strange": 0.8, "observation.unknown": 0.8},
            "interesting": {"emotion.curiosity": 0.8, "memory.surprising": 0.5},
            "boring": {"emotion.confidence": 0.5, "memory.familiar": 0.6},
            "surprise": {"memory.surprising": 1.0, "emotion.confusion": 0.4},
            "confused": {"emotion.confusion": 1.0},
            "learning": {"memory.repeated": 0.6, "emotion.curiosity": 0.4},
            "dark": {"observation.dark": 1.0, "memory.strange": 0.2},
            "near": {"observation.object_near": 0.6, "observation.danger_near": 0.4},
            "far": {"observation.object_near": -0.4, "observation.danger_near": -0.4},
            "again": {"memory.repeated": 1.0, "memory.familiar": 0.4},
        }

        for word, weights in seed_weights.items():
            for feature, weight in weights.items():
                self.associations[word][feature] = weight

    def _clamp(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    def _clamp_weight(self, value: float) -> float:
        return max(-1.0, min(float(value), 1.0))
