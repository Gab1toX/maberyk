from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import re
import torch
from torch import nn

from brain.curiosity import CuriosityModule
from brain.conversation_memory import ConversationMemory
from brain.emotion import EmotionalState
from brain.language import LanguageEngine
from brain.memory import EpisodicMemory
from brain.network import NeuralNetwork
from brain.questions import QuestionEngine
from brain.response import ResponseEngine


class Agent:
    """Autonomous curiosity-driven agent with no external reward."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_layers: Iterable[int] = (128, 128),
        learning_rate: float = 1e-3,
        exploration_rate: float = 0.1,
        device: str | torch.device = "cpu",
        memory_path: str | Path = "episodic_memory.sqlite3",
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.device = torch.device(device)
        self.memory_path = Path(memory_path)

        self.policy = NeuralNetwork(
            input_size=observation_size,
            hidden_layers=self.hidden_layers,
            output_size=action_size,
        ).to(self.device)
        self.curiosity = CuriosityModule(
            observation_size=observation_size,
            action_size=action_size,
            hidden_layers=self.hidden_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.curiosity.parameters()),
            lr=learning_rate,
        )
        self.action_loss = nn.CrossEntropyLoss(reduction="none")
        self.memory = EpisodicMemory(self.memory_path)
        self.conversation_memory = ConversationMemory(self.memory_path)
        self.emotional_state = EmotionalState()
        self.language = LanguageEngine()
        self.question_engine = QuestionEngine()
        self.response_engine = ResponseEngine(self.language.vocabulary)
        self.current_thought = ""
        self.current_question = ""
        self.current_question_observation: Any = None
        self.last_response = ""
        self.action_counts = [0 for _ in range(action_size)]
        self.danger_action_scores = [0.0 for _ in range(action_size)]
        self.step_count = 0
        self._recent_actions: deque[int] = deque(maxlen=1000)

    def act(self, observation: torch.Tensor | list[float] | tuple[float, ...]) -> int:
        observation_tensor = self._observation_tensor(observation)

        if torch.rand(1, device=self.device).item() < self.exploration_rate:
            scores = self._emotion_adjusted_scores(torch.zeros(1, self.action_size, device=self.device))
            probabilities = torch.softmax(scores, dim=-1)
            return torch.multinomial(probabilities.squeeze(0), 1).item()

        with torch.no_grad():
            action_scores = self.policy(observation_tensor)
            action_scores = self._emotion_adjusted_scores(action_scores)
            top_action = action_scores.argmax(dim=-1).item()

            # Break deterministic lock-in: if the greedy pick is also the
            # action that has dominated >80% of all steps so far, there is a
            # real risk the policy network has converged to always scoring it
            # highest regardless of observation. Rather than go fully random
            # (which would erase greedy behavior), only 30% of the time force
            # a sample weighted toward the other actions' own scores.
            total_steps = sum(self.action_counts)
            if total_steps > 0:
                dominant_action = max(range(self.action_size), key=lambda i: self.action_counts[i])
                dominant_share = self.action_counts[dominant_action] / total_steps
                if (
                    top_action == dominant_action
                    and dominant_share > 0.8
                    and torch.rand(1, device=self.device).item() < 0.3
                ):
                    non_dominant = [i for i in range(self.action_size) if i != dominant_action]
                    non_dominant_scores = action_scores[..., non_dominant]
                    probabilities = torch.softmax(non_dominant_scores, dim=-1).squeeze(0)
                    sampled_index = torch.multinomial(probabilities, 1).item()
                    return non_dominant[sampled_index]

            return top_action

    def learn(self, experience: dict[str, Any]) -> dict[str, float]:
        observation = self._observation_tensor(experience["observation"])
        action_index = self._action_index_tensor(experience["action"])
        action = self._one_hot_action(action_index)
        next_observation = self._observation_tensor(experience["next_observation"])

        action_scores = self.policy(observation)
        curiosity_loss = self.curiosity.loss(observation, action, next_observation)
        intrinsic_reward = self.curiosity.reward(observation, action, next_observation)

        policy_loss = (self.action_loss(action_scores, action_index) * intrinsic_reward.squeeze(-1)).mean()
        loss = curiosity_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        surprise = intrinsic_reward.mean().item()
        action_value = action_index[0].item()
        was_reset = self._was_reset(experience)
        is_familiar = self._is_familiar(experience["observation"])

        self.action_counts[action_value] += 1
        self.step_count += 1
        self._recent_actions.append(action_value)
        self._log_action_distribution_if_due()
        self._update_danger_memory(action_value, was_reset)
        self.emotional_state.update(
            surprise=surprise,
            was_reset=was_reset,
            is_familiar=is_familiar,
        )
        self._store_episode(experience, action_value, surprise)
        self._respond_to_human_message(experience)
        self.current_thought = self.language.express(
            self.emotional_state,
            self._plain_value(experience.get("next_observation", experience["observation"])),
            self.memory,
        )
        self.language.improve_from_last_expression(
            self._expression_accuracy(surprise, was_reset, is_familiar)
        )
        question_observation = self._plain_value(experience.get("outcome", experience["next_observation"]))
        self.question_engine.observe(self.emotional_state, surprise=surprise)
        if self.question_engine.should_ask():
            candidate = self.question_engine.formulate(
                self.emotional_state,
                question_observation,
                self.language,
            )
            if not self.conversation_memory.has_answered(candidate):
                self.current_question = candidate
                self.current_question_observation = question_observation
                self.question_engine.reset()

        return {
            "loss": loss.item(),
            "curiosity_loss": curiosity_loss.item(),
            "policy_loss": policy_loss.item(),
            "intrinsic_reward": surprise,
        }

    def get_inner_voice(self) -> str:
        return self.current_thought

    def get_question(self) -> str:
        return self.current_question

    def get_last_response(self) -> str:
        return self.last_response

    def receive_answer(self, text: str) -> None:
        answer = text.strip().lower()
        if not answer:
            return

        for word in answer.split():
            self.language.add_word(word)
        self.response_engine.update_vocabulary(self.language.vocabulary)
        self.language.learn(
            answer,
            self.emotional_state,
            self.current_question_observation,
            self.memory,
            accuracy=1.0,
        )
        self.conversation_memory.store(self.current_question, answer, source="human_taught")
        self.current_question = ""
        self.current_question_observation = None

    def get_conversation_context(self, question: str) -> str:
        results = self.conversation_memory.recall(question, limit=2)
        if not results:
            return ""

        most_recent = results[0]
        return f"I remember: '{most_recent['answer']}' about '{most_recent['question']}'"

    def save(self, path: str | Path) -> None:
        self.memory.flush()
        torch.save(
            {
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "memory_path": str(self.memory_path),
                "emotional_state": self.emotional_state.values(),
                "language_vocabulary": self.language.vocabulary,
                "language_associations": self._plain_associations(),
                "language_history": list(self.language.expression_history),
                "current_thought": self.current_thought,
                "current_question": self.current_question,
                "current_question_observation": self.current_question_observation,
                "last_response": self.last_response,
                "question_engine": self._question_engine_state(),
                "action_counts": self.action_counts,
                "danger_action_scores": self.danger_action_scores,
                "step_count": self.step_count,
                "policy_state_dict": self.policy.state_dict(),
                "curiosity_state_dict": self.curiosity.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "Agent":
        checkpoint = torch.load(path, map_location=map_location)
        agent = cls(
            observation_size=checkpoint["observation_size"],
            action_size=checkpoint["action_size"],
            hidden_layers=checkpoint["hidden_layers"],
            learning_rate=checkpoint["learning_rate"],
            exploration_rate=checkpoint["exploration_rate"],
            device=map_location or "cpu",
            memory_path=checkpoint.get("memory_path", "episodic_memory.sqlite3"),
        )
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.curiosity.load_state_dict(checkpoint["curiosity_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (KeyError, ValueError):
                print("⚠ optimizer_state_dict incompatible — se reinicia el optimizador (normal tras migración)")
        else:
            print("⚠ Sin optimizer_state_dict en checkpoint — optimizador iniciado desde cero")
        if "emotional_state" in checkpoint:
            agent.emotional_state = EmotionalState(**checkpoint["emotional_state"])
        if "language_vocabulary" in checkpoint and "language_associations" in checkpoint:
            agent.language.vocabulary = checkpoint["language_vocabulary"]
            agent.language.associations.clear()
            for word, associations in checkpoint["language_associations"].items():
                agent.language.associations[word] = defaultdict(float, associations)
            agent.language.expression_history = deque(
                checkpoint.get("language_history", []),
                maxlen=100,
            )
            agent.language.refresh_vocabulary_cache()
        agent.current_thought = checkpoint.get("current_thought", "")
        agent.current_question = checkpoint.get("current_question", "")
        agent.current_question_observation = checkpoint.get("current_question_observation")
        agent.last_response = checkpoint.get("last_response", "")
        agent.response_engine.update_vocabulary(agent.language.vocabulary)
        agent._load_question_engine_state(checkpoint.get("question_engine", {}))
        agent.action_counts = checkpoint.get("action_counts", agent.action_counts)
        agent.danger_action_scores = checkpoint.get("danger_action_scores", agent.danger_action_scores)
        agent.step_count = checkpoint.get("step_count", agent.step_count)
        return agent

    def close(self) -> None:
        self.memory.flush()
        self.memory.close()

    def _observation_tensor(self, observation: torch.Tensor | list[float] | tuple[float, ...]) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _action_index_tensor(self, action: int | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(action, dtype=torch.long, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _one_hot_action(self, action_index: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(action_index, num_classes=self.action_size).float()

    def _emotion_adjusted_scores(self, action_scores: torch.Tensor) -> torch.Tensor:
        emotion = self.emotional_state.values()
        adjusted_scores = action_scores.clone()

        curiosity = emotion["curiosity"]
        fear = emotion["fear"]
        max_count = max(self.action_counts, default=0) + 1
        # Laplace-smoothed total so an untrained agent (all counts 0) sees a
        # uniform 1/action_size share and contributes zero correction below.
        total_count = sum(self.action_counts) + self.action_size

        for action_index in range(self.action_size):
            unseen_bonus = 1.0 - (self.action_counts[action_index] / max_count)
            danger_penalty = self.danger_action_scores[action_index]

            # Root cause of the persistent "right" dominance: the curiosity
            # bonus below is the only term that pulls scores away from an
            # over-picked action, but it is scaled by `curiosity`, which
            # trends toward its 0.03 floor as prediction error drops over
            # long training -- exactly when action_counts imbalance is
            # largest and correction is needed most. That let learn()'s
            # imitation-style policy_loss (which reinforces whatever action
            # was just taken) run away unchecked. diversity_correction is a
            # curiosity-independent term: it is zero when an action's share
            # of total actions equals the uniform 1/action_size share, and
            # grows/shrinks proportionally to how far the action's actual
            # share has drifted from uniform, so it keeps correcting skew
            # even after curiosity has decayed.
            visit_share = (self.action_counts[action_index] + 1) / total_count
            diversity_correction = (1.0 / self.action_size - visit_share) * 2.0

            adjusted_scores[..., action_index] += curiosity * unseen_bonus * 0.25
            adjusted_scores[..., action_index] += diversity_correction
            adjusted_scores[..., action_index] -= fear * danger_penalty * 0.75

        # Even a 2.0x diversity_correction is an additive term and can still
        # be overwhelmed by a policy score that has run away after millions
        # of steps of imitation-style reinforcement. This hard suppression is
        # a multiplicative backstop applied after every additive adjustment
        # above: once an action's raw share of action_counts crosses 70%, its
        # score is crushed to 10% regardless of emotion state, guaranteeing
        # it can no longer win argmax against any other action with a
        # non-negative adjusted score.
        total_actions = sum(self.action_counts)
        if total_actions > 0:
            for action_index in range(self.action_size):
                if self.action_counts[action_index] / total_actions > 0.7:
                    adjusted_scores[..., action_index] *= 0.1

        return adjusted_scores

    def _log_action_distribution_if_due(self) -> None:
        if self.step_count % 10000 != 0:
            return

        total_recent = len(self._recent_actions)
        if not total_recent:
            return

        counts = [0 for _ in range(self.action_size)]
        for action_value in self._recent_actions:
            counts[action_value] += 1

        breakdown = ", ".join(
            f"{action_index}={counts[action_index] / total_recent * 100.0:.1f}%"
            for action_index in range(self.action_size)
        )
        print(f"[action-bias check] step {self.step_count} (last {total_recent} actions): {breakdown}")

    def _update_danger_memory(self, action: int, was_reset: bool) -> None:
        self.danger_action_scores = [score * 0.98 for score in self.danger_action_scores]
        if was_reset:
            self.danger_action_scores[action] = min(1.0, self.danger_action_scores[action] + 0.5)

    def _is_familiar(self, observation: Any) -> bool:
        similar = self.memory.recall_similar(self._plain_value(observation), limit=1)
        return bool(similar and similar[0].get("similarity", 0.0) >= 0.85)

    def _was_reset(self, experience: dict[str, Any]) -> bool:
        if "was_reset" in experience:
            return bool(experience["was_reset"])

        outcome = experience.get("outcome", experience.get("next_observation"))
        if isinstance(outcome, dict):
            event = str(outcome.get("event", "")).lower()
            return "reset" in event or "danger" in event

        return False

    def _store_episode(self, experience: dict[str, Any], action: int, surprise: float) -> None:
        if self.step_count % 50 != 0:
            return

        outcome = experience.get("outcome", experience.get("next_observation"))
        self.memory.store(
            {
                "observation": self._plain_value(experience["observation"]),
                "action": action,
                "outcome": self._plain_value(outcome),
                "surprise_level": surprise,
            }
        )

    def _respond_to_human_message(self, experience: dict[str, Any]) -> None:
        observation = experience.get("outcome")
        if not isinstance(observation, dict):
            observation = experience.get("next_observation")
        if not isinstance(observation, dict):
            observation = experience.get("observation")
        if not isinstance(observation, dict):
            return

        human_message = str(observation.get("human_message") or "").strip()
        if not human_message:
            return

        # Add every word from the human message to the agent vocabulary
        for word in human_message.lower().split():
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) > 2:
                self.language.add_word(clean)
        self.response_engine.update_vocabulary(self.language.vocabulary)

        # Retrieve conversation context to inform the response
        memory_context = self.conversation_memory.recall(human_message, limit=2)

        # Generate response using enriched vocabulary and emotional state
        self.last_response = self.response_engine.generate(
            human_message,
            self.emotional_state.values(),
            memory_context,
        )

        # Store the exchange and reinforce language associations
        if self.last_response:
            self.conversation_memory.store(human_message, self.last_response, source="agent_generated")
            self.language.learn(
                human_message,
                self.emotional_state,
                None,
                self.memory,
                accuracy=0.8,
            )

        # Curiosity boost for engaging in communication
        self.emotional_state.curiosity = min(
            1.0,
            self.emotional_state.curiosity + 0.3,
        )

    def _plain_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._plain_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._plain_value(item) for item in value]
        if isinstance(value, bool | int | float | str) or value is None:
            return value
        return str(value)

    def _plain_associations(self) -> dict[str, dict[str, float]]:
        return {
            word: dict(associations)
            for word, associations in self.language.associations.items()
        }

    def recall_similar(
        self, observation: Any, limit: int = 5, max_candidates: int = 200
    ) -> list[dict[str, Any]]:
        return self.memory.recall_similar(observation, limit, max_candidates)

    def summarize(self) -> dict[str, Any]:
        return self.memory.summarize()

    def memory_feature_snapshot(self, observation: Any) -> dict[str, float]:
        return self.memory.memory_feature_snapshot(observation)

    def _question_engine_state(self) -> dict[str, Any]:
        return {
            "vocabulary": self.question_engine.vocabulary,
            "confusion_threshold": self.question_engine.confusion_threshold,
            "required_steps": self.question_engine.required_steps,
            "confusion_steps": self.question_engine.confusion_steps,
            "last_confusion": self.question_engine.last_confusion,
            "last_surprise": self.question_engine.last_surprise,
            "steps_since_last_question": self.question_engine._steps_since_last_question,
        }

    def _load_question_engine_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.question_engine = QuestionEngine(
            vocabulary=state.get("vocabulary"),
            confusion_threshold=state.get("confusion_threshold", 0.08),
            required_steps=state.get("required_steps", 8),
        )
        self.question_engine.confusion_steps = int(state.get("confusion_steps", 0))
        self.question_engine.last_confusion = float(state.get("last_confusion", 0.0))
        self.question_engine.last_surprise = float(state.get("last_surprise", 0.0))
        self.question_engine._steps_since_last_question = int(state.get("steps_since_last_question", 0))

    def _expression_accuracy(self, surprise: float, was_reset: bool, is_familiar: bool) -> float:
        if was_reset:
            return 0.8
        if is_familiar:
            return 1.0 - min(surprise, 1.0)
        return min(1.0, 0.5 + surprise * 0.5)
