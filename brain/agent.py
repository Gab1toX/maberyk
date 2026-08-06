from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import re
import string
import torch
from torch import nn

from brain.curiosity import CuriosityModule
from brain.conversation_memory import ConversationMemory
from brain.emotion import EmotionalState
from brain.language import LanguageEngine
from brain.memory import EpisodicMemory
from brain.network import NeuralNetwork
from brain.permissions import PermissionManager
from brain.questions import QuestionEngine
from brain.response import ResponseEngine

if TYPE_CHECKING:
    from actions.desktop import DesktopEnvironment
    from brain.memory_retrieval import MemoryRetrieval
    from tutor.llm_tutor import LLMTutor
    from voice.ollama_voice import OllamaVoice

# Explicit override for free-form teaching: "aprende: <answer>" always stores
# (previous_user_message, answer) as human_taught, bypassing the heuristic.
_TEACH_PREFIX = "aprende:"
# Full words (not prefixes) whose presence as the FIRST token marks a message
# as a question, not an answer Gabito is volunteering — see
# _looks_like_question. Prefix matching previously misfired on words like
# "porque"/"pon" that merely started with a question stem such as "por".
_QUESTION_STARTS = frozenset((
    "que", "qué", "quien", "quién", "quienes", "quiénes", "como", "cómo",
    "donde", "dónde", "cuando", "cuándo", "cual", "cuál", "cuales", "cuáles",
    "cuanto", "cuánto", "cuanta", "cuánta", "cuantos", "cuántos", "por",
    "para", "sabes", "sabias", "sabías", "conoces", "recuerdas", "dime",
    "cuentame", "cuéntame", "explicame", "explícame", "puedes", "podrias",
    "podrías", "tienes", "hay", "existe", "what", "who", "how", "where",
    "when", "why",
))
# Phrases that undo the most recent free-form teaching entry — see the undo
# branch at the top of _respond_to_human_message.
_UNDO_PHRASES = frozenset(("olvida eso", "olvidalo", "olvídalo", "eso no"))


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

        from brain.memory_retrieval import MemoryRetrieval

        self.memory_retrieval: "MemoryRetrieval" = MemoryRetrieval(self.memory_path, self.language)
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

        # language_model.pt is trained offline (brain/language_model.py) and is
        # not part of agent_state.pt — it is looked up next to the checkpoint
        # and loaded lazily, only on the first human message received.
        self._checkpoint_dir = self.memory_path.parent
        self.permissions = PermissionManager(
            db_path=self._checkpoint_dir / "permissions.sqlite3"
        )
        self.desktop: "DesktopEnvironment | None" = None
        self._language_model = None
        self._language_model_tokenizer = None
        self._language_model_unavailable = False

        # Voice is opt-in and desktop-only: Kaggle training loops never call
        # enable_voice(), so this stays off and _voice is never touched there.
        self.voice_enabled = False
        self._voice: "OllamaVoice | None" = None
        self._voice_unavailable = False
        self._voice_history: deque[tuple[str, str]] = deque(maxlen=3)

        # Tutor is opt-in and desktop-only, same as voice: Kaggle training
        # loops never call enable_tutor(), so this stays off and _tutor is
        # never touched there.
        self.tutor_enabled = False
        self._tutor: "LLMTutor | None" = None
        self._tutor_unavailable = False
        self.pending_correction: dict[str, Any] | None = None
        self._tutor_corrections = 0

        # (previous_user_message, previous_retrieved_reply) — read at the
        # top of _respond_to_human_message to detect free-form teaching,
        # then overwritten with the current turn at the end of that call.
        self._last_exchange: tuple[str, str | None] | None = None

        # (question, answer, source) of the most recent teach-store, so a
        # follow-up "olvida eso" can undo exactly that row — see
        # _respond_to_human_message's undo branch and _store_taught_answer.
        self._last_taught: tuple[str, str, str] | None = None

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
            self._plain_value(experience.get("outcome", experience.get("next_observation", experience["observation"]))),
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
        self._checkpoint_dir = Path(path).parent
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
        memory_path = checkpoint.get("memory_path", "episodic_memory.sqlite3")
        if not Path(memory_path).exists():
            memory_path = str(Path(path).parent / Path(memory_path).name)
        agent = cls(
            observation_size=checkpoint["observation_size"],
            action_size=checkpoint["action_size"],
            hidden_layers=checkpoint["hidden_layers"],
            learning_rate=checkpoint["learning_rate"],
            exploration_rate=checkpoint["exploration_rate"],
            device=map_location or "cpu",
            memory_path=memory_path,
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
        agent._checkpoint_dir = Path(path).parent
        return agent

    def enable_desktop(self) -> None:
        from actions.desktop import DesktopEnvironment

        self.desktop = DesktopEnvironment(permissions=self.permissions)

    def enable_voice(self) -> None:
        """Opt in to Ollama-backed spoken replies (desktop/mind.py only)."""
        self.voice_enabled = True

    def enable_tutor(self) -> None:
        """Opt in to Groq-backed corrections of the agent's own raw language
        model output (desktop/mind.py only). Unlike voice, LLMTutor is built
        right here rather than lazily on first use, so its common_words
        snapshot is taken from the vocabulary at enable time; the
        never-raises/"unavailable" safety net still mirrors _get_voice.
        """
        if self._tutor is not None:
            self.tutor_enabled = True
            return
        if self._tutor_unavailable:
            return

        from tutor.llm_tutor import LLMTutor

        common_words = self._most_frequent_vocabulary(80)
        try:
            self._tutor = LLMTutor(common_words=common_words, debug=True)
        except Exception as exc:
            print(f"[tutor] failed to initialize LLMTutor: {exc}")
            self._tutor_unavailable = True
            return

        self.tutor_enabled = True

    def approve_correction(self) -> bool:
        """Human approves the pending tutor correction: it enters the corpus
        as tutor_approved and its new words join the agent's vocabulary.
        """
        if self.pending_correction is None:
            return False

        question = self.pending_correction["question"]
        corrected = self.pending_correction["corrected"]
        new_words = self.pending_correction["new_words"]

        self.conversation_memory.store(question, corrected, source="tutor_approved")
        for word in new_words:
            self.language.add_word(word)

        print(f"[tutor] approved: {question} -> {corrected}")
        self.pending_correction = None
        return True

    def reject_correction(self) -> bool:
        """Human discards the pending tutor correction: nothing is stored."""
        if self.pending_correction is None:
            return False

        question = self.pending_correction["question"]
        print(f"[tutor] discarded: {question}")
        self.pending_correction = None
        return True

    def close(self) -> None:
        self.memory.flush()
        self.memory.close()
        self.permissions.close()

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

        # The cumulative check above misses recent local bias: an action can
        # dominate 90%+ of the last 1000 steps without tripping the 70%
        # cumulative threshold if its share of the full history is still low
        # (e.g. early in a long run, or after a long earlier period of
        # balanced behavior). This second backstop looks only at
        # _recent_actions (a maxlen=1000 deque) so it reacts to short-term
        # runaway dominance regardless of cumulative history.
        recent_total = len(self._recent_actions)
        if recent_total >= 100:
            recent_counts = [0] * self.action_size
            for a in self._recent_actions:
                recent_counts[a] += 1
            for action_index in range(self.action_size):
                if recent_counts[action_index] / recent_total > 0.6:
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

    def _most_frequent_vocabulary(self, limit: int) -> list[str]:
        """Approximate word frequency for the tutor's common_words hint.

        LanguageEngine tracks no explicit per-word usage counter, so this
        ranks by cumulative association-weight magnitude -- the closest
        signal to "how often this word has been reinforced through use"
        that already exists in the model.
        """
        ranked = sorted(
            self.language.vocabulary,
            key=lambda word: sum(
                abs(weight) for weight in self.language.associations.get(word, {}).values()
            ),
            reverse=True,
        )
        return ranked[:limit]

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
            },
            thought=self.current_thought,
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

        if human_message.strip().lower() in _UNDO_PHRASES and self._last_taught is not None:
            question, answer, source = self._last_taught
            self.conversation_memory.delete_exact(question, answer, source)
            self._last_taught = None
            print(f"[teach] undone: {question}")
            self.last_response = "Listo, lo olvidé."
            return

        previous_user_message, previous_retrieved_reply = (
            self._last_exchange if self._last_exchange is not None else (None, None)
        )

        # Add every word from the human message to the agent vocabulary
        for word in human_message.lower().split():
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) > 2:
                self.language.add_word(clean)
        self.response_engine.update_vocabulary(self.language.vocabulary)

        retrieved_reply = self.memory_retrieval.retrieve(human_message)
        lm_reply = self._generate_language_model_response(human_message)

        just_learned = self._maybe_learn_from_teaching(
            previous_user_message, previous_retrieved_reply, human_message
        )
        self._last_exchange = (human_message, retrieved_reply)

        tutor_result = None
        if self.tutor_enabled and lm_reply:
            if self.pending_correction is not None:
                print("[tutor] skipped: correction still pending approval")
            else:
                tutor_result = self._generate_tutor_response(lm_reply, human_message)

        if tutor_result is not None:
            self.last_response, branch = tutor_result
            print(f"[response] branch={branch}")
        else:
            voice_reply = None
            if self.voice_enabled:
                voice_reply = self._generate_voice_response(
                    human_message, retrieved_reply, lm_reply, just_learned
                )

            if voice_reply:
                self.last_response, branch = voice_reply, "voice"
                print(f"[response] branch={branch}")
            else:
                self.last_response, branch = self._select_response(
                    human_message, retrieved_reply, lm_reply
                )

        # Store the exchange and reinforce language associations
        if self.last_response:
            # Verbatim human_taught echoes must not pollute the agent-voice
            # training corpus — only language_model replies count as the
            # agent's own generated language. Tutor branches are excluded
            # too: a correction only enters the corpus after explicit human
            # approval (approve_correction), and a rejection is never
            # stored. A silent branch (no retrieved/lm reply available)
            # produces an empty last_response and is never stored either.
            if branch == "retrieved":
                source = "retrieved"
            elif branch == "voice":
                source = "voice"
            elif branch in ("tutor", "tutor_rejected", "silent"):
                source = None
            else:
                source = "agent_generated"
            if source is not None:
                self.conversation_memory.store(human_message, self.last_response, source=source)
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

    def _load_language_model(self) -> bool:
        """Lazily load language_model.pt from the checkpoint directory.

        Silent no-op if the file is missing or fails to load — the model is
        an optional scaffold (see brain/language_model.py) and the agent must
        keep functioning on ResponseEngine alone until it exists.
        """
        path = self._checkpoint_dir / "language_model.pt"
        if not path.exists():
            self._language_model_unavailable = True
            return False

        try:
            from brain.language_model import LanguageModelTrainer

            model, tokenizer = LanguageModelTrainer.load_model(path, map_location=self.device)
            model.to(self.device)
            model.eval()
        except Exception as exc:
            print(f"[language_model] failed to load {path}: {exc}")
            self._language_model_unavailable = True
            return False

        self._language_model = model
        self._language_model_tokenizer = tokenizer
        return True

    def _generate_language_model_response(self, human_message: str) -> str | None:
        if self._language_model_unavailable:
            return None
        if self._language_model is None and not self._load_language_model():
            return None

        prompt_words = self._clean_words(human_message)
        if not prompt_words:
            return None

        try:
            generated = self._language_model.generate_reply(
                human_message, temperature=0.8
            )
        except Exception as exc:
            print(f"[language_model] generation failed: {exc}")
            return None

        tokenizer = self._language_model_tokenizer
        words = [word for word in generated if word not in (tokenizer.PAD, tokenizer.UNK)]
        if not words:
            return None
        return " ".join(words)

    def _generate_tutor_response(
        self, lm_reply: str, human_message: str
    ) -> tuple[str, str] | None:
        """Ask the tutor to rewrite the agent's own raw language-model reply.

        Returns None both when the tutor isn't actually available and on any
        API failure -- either way the caller falls through to
        voice/retrieved/language_model/response_engine exactly as if the
        tutor branch had never run.
        """
        if self._tutor is None:
            return None

        self._tutor_corrections += 1
        if self._tutor_corrections % 20 == 0:
            self._tutor.common_words = self._most_frequent_vocabulary(80)

        result = self._tutor.correct(lm_reply, set(self.language.vocabulary))
        if result is None:
            return None

        print(f"[tutor] raw: {lm_reply}")

        if result["status"] == "rejected":
            reason = result.get("reason", "unknown")
            if reason == "too-many-new-words":
                print(f"[tutor] rejected ({reason}: {result['new_words']})")
            else:
                print(f"[tutor] rejected ({reason})")
            return lm_reply, "tutor_rejected"

        corrected = result["corrected"]
        self.pending_correction = {
            "question": human_message,
            "raw": lm_reply,
            "corrected": corrected,
            "new_words": result["new_words"],
        }
        return corrected, "tutor"

    def _get_voice(self) -> "OllamaVoice | None":
        """Lazily construct the Ollama-backed voice, mirroring
        _load_language_model: optional, never raises, disables itself for
        the rest of the session if it cannot even be constructed.
        """
        if self._voice_unavailable:
            return None
        if self._voice is None:
            try:
                from voice.ollama_voice import OllamaVoice

                self._voice = OllamaVoice()
            except Exception as exc:
                print(f"[voice] failed to initialize OllamaVoice: {exc}")
                self._voice_unavailable = True
                return None
        return self._voice

    def _generate_voice_response(
        self,
        human_message: str,
        retrieved_reply: str | None,
        lm_reply: str | None,
        just_learned: bool = False,
    ) -> str | None:
        voice = self._get_voice()
        if voice is None:
            return None

        recent_thoughts = [
            entry["phrase"]
            for entry in list(self.language.expression_history)[-3:]
            if isinstance(entry, dict) and entry.get("phrase")
        ]
        state = {
            "emotions": self.emotional_state.values(),
            "recent_thoughts": recent_thoughts,
            "retrieved_answer": retrieved_reply,
            "lm_reply": lm_reply,
            "dominant_emotion": self.emotional_state.dominant(),
            "history": list(self._voice_history),
            "just_learned": just_learned,
        }

        try:
            reply = voice.verbalize(human_message, state)
        except Exception as exc:
            print(f"[voice] verbalize failed: {exc}")
            return None

        if reply:
            self._voice_history.append((human_message, reply))
        return reply

    def _maybe_learn_from_teaching(
        self,
        previous_user_message: str | None,
        previous_retrieved_reply: str | None,
        current_message: str,
    ) -> bool:
        """Turns free-form teaching in normal chat into human_taught data.

        Today only QuestionEngine answers (via receive_answer) become
        human_taught and thus retrievable. This lets a plain follow-up
        statement — or an explicit "aprende: ..." override — do the same,
        so knowledge Gabito volunteers unprompted isn't lost to a
        non-retrievable source like "voice"/"agent_generated".
        """
        stripped = current_message.strip()
        lowered = stripped.lower()

        # Explicit path: bypasses every heuristic gate below.
        if lowered.startswith(_TEACH_PREFIX):
            taught_answer = stripped[len(_TEACH_PREFIX):].strip()
            if previous_user_message and taught_answer:
                self._store_taught_answer(previous_user_message, taught_answer, source="human_taught")
                return True
            return False

        # Heuristic path: only fires when the previous turn read as Gabito
        # asking a question and this turn reads as a real, substantive answer
        # (not another question, not an echo of the question itself).
        if previous_user_message is None:
            return False
        if previous_retrieved_reply is not None:
            return False
        if not self._looks_like_question(previous_user_message.lower()):
            return False
        if self._looks_like_question(lowered):
            return False
        if len(stripped.split()) < 4:
            return False
        if lowered == previous_user_message.strip().lower():
            return False

        self._store_taught_answer(previous_user_message, current_message, source="taught_inferred")
        return True

    def _looks_like_question(self, lowered_message: str) -> bool:
        stripped = lowered_message.strip().strip('¿¡"\'').strip()
        if stripped.endswith("?"):
            return True

        tokens = stripped.split()
        if not tokens:
            return False

        first_token = tokens[0].rstrip(string.punctuation)
        return first_token in _QUESTION_STARTS

    def _store_taught_answer(self, question: str, answer: str, source: str) -> None:
        self.conversation_memory.store(question, answer, source=source)
        self._last_taught = (question, answer, source)
        print(f"[teach] {source}: {question} -> {answer}")

    def _select_response(
        self,
        human_message: str,
        retrieved_reply: str | None,
        lm_reply: str | None,
    ) -> tuple[str, str]:
        if retrieved_reply:
            branch = "retrieved"
            reply = retrieved_reply
        elif lm_reply:
            branch = "language_model"
            reply = lm_reply
        else:
            # An empty response is honest; ResponseEngine's SVO output is
            # noise that would otherwise enter the corpus as agent_generated.
            branch = "silent"
            reply = ""
        print(f"[response] branch={branch}")
        return reply, branch

    def _clean_words(self, text: str) -> list[str]:
        from brain.language_model import normalize_text

        words = []
        for word in normalize_text(str(text or "")).split():
            clean = re.sub(r"[^\w]", "", word)
            if clean:
                words.append(clean)
        return words

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
