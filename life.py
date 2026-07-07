from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from brain.agent import Agent
from entorno.room import Room


ACTIONS = ("up", "down", "left", "right", "touch")
BEHAVIORS = ("noise", "move_randomly", "change_color", "open_door", "nothing")
COLORS = ("gray", "gold", "blue", "red", "green", "yellow", "purple")

OBSERVATION_SIZE = 2 + 1 + 8 * (1 + 1 + len(BEHAVIORS) + len(COLORS))
ACTION_SIZE = len(ACTIONS)
SAVE_INTERVAL_SECONDS = 60 * 60
PRINT_INTERVAL_STEPS = 1000
STATE_PATH = Path("agent_state.pt")


def encode_observation(observation: dict[str, Any], room: Room) -> torch.Tensor:
    ax, ay = observation["agent_position"]
    features = [
        ax / (room.width - 1),
        ay / (room.height - 1),
        float(observation["door_open"]),
    ]

    for cell in observation["surroundings"]:
        room_object = cell["object"]
        features.append(float(cell["inside"]))
        features.append(float(room_object is not None))

        behavior = room_object["behavior"] if room_object is not None else None
        color = room_object["color"] if room_object is not None else None

        features.extend(float(behavior == known_behavior) for known_behavior in BEHAVIORS)
        features.extend(float(color == known_color) for known_color in COLORS)

    return torch.tensor(features, dtype=torch.float32)


def apply_action(room: Room, action_index: int) -> dict[str, Any]:
    action = ACTIONS[action_index]
    if action == "touch":
        return room.touch()
    return room.move(action)


def create_agent() -> Agent:
    if STATE_PATH.exists():
        return Agent.load(STATE_PATH)

    return Agent(
        observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE,
        hidden_layers=(128, 128),
    )


def main() -> None:
    room = Room()
    agent = create_agent()
    observation = room.get_observation(event="started")
    encoded_observation = encode_observation(observation, room)

    step = 0
    surprise_total = 0.0
    last_save_time = time.time()

    try:
        while True:
            action = agent.act(encoded_observation)
            next_observation = apply_action(room, action)
            encoded_next_observation = encode_observation(next_observation, room)

            stats = agent.learn(
                {
                    "observation": encoded_observation,
                    "action": action,
                    "next_observation": encoded_next_observation,
                    "outcome": next_observation,
                    "was_reset": next_observation.get("event") == "reset_by_danger",
                }
            )

            step += 1
            surprise_total += stats["intrinsic_reward"]

            if step % PRINT_INTERVAL_STEPS == 0:
                average_surprise = surprise_total / PRINT_INTERVAL_STEPS
                print(f"step={step} average_surprise={average_surprise:.6f}")
                surprise_total = 0.0

            now = time.time()
            if now - last_save_time >= SAVE_INTERVAL_SECONDS:
                agent.save(STATE_PATH)
                print(f"saved_state={STATE_PATH} step={step}")
                last_save_time = now

            encoded_observation = encoded_next_observation

    except KeyboardInterrupt:
        agent.save(STATE_PATH)
        print(f"saved_state={STATE_PATH} step={step}")


if __name__ == "__main__":
    main()
