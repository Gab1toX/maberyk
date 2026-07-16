from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pygame
import torch

from brain.agent import Agent
from entorno.room import Room
from human.interpreter import Interpreter


ACTIONS = ("up", "down", "left", "right", "touch")
BEHAVIORS = ("noise", "move_randomly", "change_color", "open_door", "nothing", "reflect", "grow", "teleport", "glow")
COLORS = ("gray", "gold", "blue", "red", "green", "yellow", "purple", "silver", "cyan")

OBSERVATION_SIZE = 2 + 1 + 8 * (1 + 1 + len(BEHAVIORS) + len(COLORS))
ACTION_SIZE = len(ACTIONS)
STATE_PATH = Path("agent_state.pt")

# Layout constants
WINDOW_HEIGHT = 820
PANEL_PADDING = 24
GRID_SIZE = 20
CELL_SIZE = 24
PANEL_GAP = 40
# Calculate window width so both grid panels and decorations fit without clipping.
# Each panel draws a small decoration offset of 12px on either side, so include
# an extra margin to ensure the right panel isn't clipped.
WINDOW_WIDTH = (
    2 * PANEL_PADDING + 2 * GRID_SIZE * CELL_SIZE + PANEL_GAP + 24
)
BOTTOM_PANEL_Y = 560
BOTTOM_PANEL_HEIGHT = 230
STEPS_PER_FRAME = 2
FPS = 30

BACKGROUND = (18, 20, 24)
PANEL = (32, 36, 42)
GRID_LINE = (58, 64, 72)
TEXT = (225, 230, 236)
MUTED_TEXT = (150, 158, 170)
INPUT_BG = (20, 23, 28)
INPUT_BORDER = (92, 104, 122)
QUESTION_CYAN = (80, 220, 230)
AGENT_COLOR = (245, 245, 245)
DANGER_COLOR = (210, 55, 120)
OBSTACLE_COLOR = (82, 86, 92)
OBJECT_COLORS = {
    "gray": (150, 155, 160),
    "gold": (230, 180, 72),
    "blue": (76, 140, 245),
    "red": (230, 72, 72),
    "green": (80, 190, 115),
    "yellow": (232, 220, 84),
    "purple": (178, 102, 235),
}
EMOTION_COLORS = {
    "curiosity": (80, 220, 230),
    "fear": (235, 72, 72),
    "confusion": (235, 210, 80),
    "confidence": (90, 210, 125),
}
PERMISSION_BG = (20, 28, 40)
PERMISSION_BORDER = (255, 180, 0)
APPROVE_COLOR = (80, 210, 120)
DENY_COLOR = (210, 60, 60)


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
    return Agent(OBSERVATION_SIZE, ACTION_SIZE, hidden_layers=(128, 128))


def curiosity_color(value: float, maximum: float) -> tuple[int, int, int]:
    if maximum <= 0:
        intensity = 0.0
    else:
        intensity = max(0.0, min(value / maximum, 1.0))

    blue = (54, 102, 220)
    red = (235, 62, 62)
    return (
        int(blue[0] + (red[0] - blue[0]) * intensity),
        int(blue[1] + (red[1] - blue[1]) * intensity),
        int(blue[2] + (red[2] - blue[2]) * intensity),
    )


def draw_grid_panel(
    surface: pygame.Surface,
    room: Room,
    origin: tuple[int, int],
    font: pygame.font.Font,
) -> None:
    x0, y0 = origin
    pygame.draw.rect(surface, PANEL, (x0 - 12, y0 - 42, GRID_SIZE * CELL_SIZE + 24, GRID_SIZE * CELL_SIZE + 58))
    surface.blit(font.render("Room", True, TEXT), (x0, y0 - 32))

    for y in range(room.height):
        for x in range(room.width):
            rect = pygame.Rect(x0 + x * CELL_SIZE, y0 + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, BACKGROUND, rect)
            if (x, y) in room.danger_zones:
                pygame.draw.rect(surface, DANGER_COLOR, rect)
            if (x, y) in room.obstacles:
                pygame.draw.rect(surface, OBSTACLE_COLOR, rect)
            pygame.draw.rect(surface, GRID_LINE, rect, 1)

    for room_object in room.objects:
        ox, oy = room_object.position
        rect = pygame.Rect(x0 + ox * CELL_SIZE + 4, y0 + oy * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
        color = OBJECT_COLORS.get(room_object.color, OBJECT_COLORS["gray"])
        pygame.draw.rect(surface, color, rect, border_radius=3)

    ax, ay = room.agent_position
    center = (x0 + ax * CELL_SIZE + CELL_SIZE // 2, y0 + ay * CELL_SIZE + CELL_SIZE // 2)
    pygame.draw.circle(surface, AGENT_COLOR, center, CELL_SIZE // 3)


def draw_heatmap_panel(
    surface: pygame.Surface,
    room: Room,
    zone_surprise: dict[tuple[int, int], float],
    object_surprise: dict[str, float],
    origin: tuple[int, int],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    x0, y0 = origin
    pygame.draw.rect(surface, PANEL, (x0 - 12, y0 - 42, GRID_SIZE * CELL_SIZE + 24, GRID_SIZE * CELL_SIZE + 58))
    surface.blit(font.render("Curiosity Heatmap", True, TEXT), (x0, y0 - 32))

    maximum_zone = max(zone_surprise.values(), default=0.0)
    maximum_object = max(object_surprise.values(), default=0.0)
    maximum = max(maximum_zone, maximum_object)

    for y in range(room.height):
        for x in range(room.width):
            surprise = zone_surprise.get((x, y), 0.0)
            rect = pygame.Rect(x0 + x * CELL_SIZE, y0 + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, curiosity_color(surprise, maximum), rect)
            pygame.draw.rect(surface, GRID_LINE, rect, 1)

    for room_object in room.objects:
        ox, oy = room_object.position
        surprise = object_surprise.get(room_object.name, zone_surprise.get(room_object.position, 0.0))
        color = curiosity_color(surprise, maximum)
        rect = pygame.Rect(x0 + ox * CELL_SIZE + 3, y0 + oy * CELL_SIZE + 3, CELL_SIZE - 6, CELL_SIZE - 6)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        label = small_font.render(room_object.name[:1].upper(), True, TEXT)
        surface.blit(label, label.get_rect(center=rect.center))


def draw_intervention_panel(
    surface: pygame.Surface,
    input_text: str,
    intervention_log: list[str],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    panel_width = GRID_SIZE * CELL_SIZE + 24
    panel_rect = pygame.Rect(PANEL_PADDING - 12, BOTTOM_PANEL_Y, panel_width, BOTTOM_PANEL_HEIGHT)
    pygame.draw.rect(surface, PANEL, panel_rect)

    surface.blit(font.render("Interventions", True, TEXT), (PANEL_PADDING, BOTTOM_PANEL_Y + 10))

    input_rect = pygame.Rect(PANEL_PADDING, BOTTOM_PANEL_Y + 42, GRID_SIZE * CELL_SIZE - 2, 34)
    pygame.draw.rect(surface, INPUT_BG, input_rect)
    pygame.draw.rect(surface, INPUT_BORDER, input_rect, 1)

    visible_text = input_text[-42:]
    input_surface = small_font.render(visible_text, True, TEXT)
    surface.blit(input_surface, (input_rect.x + 10, input_rect.y + 9))

    cursor_x = input_rect.x + 10 + input_surface.get_width() + 2
    pygame.draw.line(surface, TEXT, (cursor_x, input_rect.y + 8), (cursor_x, input_rect.y + 26), 1)

    log_x = PANEL_PADDING
    surface.blit(small_font.render("last 5", True, MUTED_TEXT), (log_x, BOTTOM_PANEL_Y + 88))
    for index, message in enumerate(intervention_log[-5:]):
        clipped = message[-62:]
        surface.blit(small_font.render(clipped, True, TEXT), (log_x, BOTTOM_PANEL_Y + 110 + index * 18))


def draw_inner_voice_panel(
    surface: pygame.Surface,
    agent: Agent,
    inner_voice_log: list[tuple[str, str]],
    origin: tuple[int, int],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    x0, y0 = origin
    panel_width = GRID_SIZE * CELL_SIZE + 24
    panel_rect = pygame.Rect(x0 - 12, y0, panel_width, BOTTOM_PANEL_HEIGHT)
    pygame.draw.rect(surface, PANEL, panel_rect)

    surface.blit(font.render("Inner Voice", True, TEXT), (x0, y0 + 10))

    emotion_values = agent.emotional_state.values()
    emotion_x = x0
    for emotion, value in emotion_values.items():
        color = EMOTION_COLORS.get(emotion, TEXT)
        label = f"{emotion} {value:.2f}"
        rendered = small_font.render(label, True, color)
        surface.blit(rendered, (emotion_x, y0 + 42))
        emotion_x += rendered.get_width() + 14

    surface.blit(small_font.render("thoughts", True, MUTED_TEXT), (x0, y0 + 72))
    for index, (thought, dominant_emotion) in enumerate(inner_voice_log[-10:]):
        color = EMOTION_COLORS.get(dominant_emotion, TEXT)
        clipped = thought[-60:]
        surface.blit(small_font.render(clipped, True, color), (x0, y0 + 94 + index * 13))


def draw_conversation_panel(
    surface: pygame.Surface,
    question: str,
    answer_text: str,
    conversation_log: list[tuple[str, str]],
    origin: tuple[int, int],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    x0, y0 = origin
    panel_width = GRID_SIZE * CELL_SIZE + 24
    panel_rect = pygame.Rect(x0 - 12, y0, panel_width, BOTTOM_PANEL_HEIGHT)
    pygame.draw.rect(surface, PANEL, panel_rect)

    surface.blit(font.render("Conversation", True, TEXT), (x0, y0 + 10))

    question_y = y0 + 42
    if question:
        surface.blit(font.render("?", True, QUESTION_CYAN), (x0, question_y - 2))
        clipped_question = question[-58:]
        surface.blit(small_font.render(clipped_question, True, QUESTION_CYAN), (x0 + 28, question_y + 4))

        input_rect = pygame.Rect(x0, y0 + 76, GRID_SIZE * CELL_SIZE - 2, 34)
        pygame.draw.rect(surface, INPUT_BG, input_rect)
        pygame.draw.rect(surface, QUESTION_CYAN, input_rect, 1)

        visible_text = answer_text[-54:]
        input_surface = small_font.render(visible_text, True, TEXT)
        surface.blit(input_surface, (input_rect.x + 10, input_rect.y + 9))

        cursor_x = input_rect.x + 10 + input_surface.get_width() + 2
        pygame.draw.line(surface, TEXT, (cursor_x, input_rect.y + 8), (cursor_x, input_rect.y + 26), 1)
    else:
        surface.blit(small_font.render("waiting for a question", True, MUTED_TEXT), (x0, question_y + 4))

    surface.blit(small_font.render("last 5 exchanges", True, MUTED_TEXT), (x0, y0 + 122))
    for index, (speaker, message) in enumerate(conversation_log[-10:]):
        color = QUESTION_CYAN if speaker == "Agent" else TEXT
        prefix = "Agent: " if speaker == "Agent" else "You: "
        clipped = f"{prefix}{message}"[-64:]
        surface.blit(small_font.render(clipped, True, color), (x0, y0 + 144 + index * 13))


def draw_permission_overlay(
    surface: pygame.Surface,
    pending: list[dict[str, Any]],
    selected_index: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    if not pending:
        return

    dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 160))
    surface.blit(dim, (0, 0))

    panel_width, panel_height = 500, 300
    panel_x = (surface.get_width() - panel_width) // 2
    panel_y = (surface.get_height() - panel_height) // 2
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(surface, PERMISSION_BG, panel_rect)
    pygame.draw.rect(surface, PERMISSION_BORDER, panel_rect, 2)

    surface.blit(font.render("Permission Request", True, PERMISSION_BORDER), (panel_x + 24, panel_y + 20))

    request = pending[selected_index]
    action_name = str(request.get("action_name", ""))
    category = str(request.get("category", ""))
    context = str(request.get("context", ""))[:60]

    detail_lines = [
        f"action: {action_name}",
        f"category: {category}",
        f"context: {context}",
    ]
    for index, line in enumerate(detail_lines):
        surface.blit(small_font.render(line, True, TEXT), (panel_x + 24, panel_y + 74 + index * 26))

    if len(pending) > 1:
        counter_surface = small_font.render(f"{selected_index + 1} of {len(pending)}", True, MUTED_TEXT)
        surface.blit(counter_surface, (panel_x + panel_width - counter_surface.get_width() - 24, panel_y + 24))

    instructions = [
        ("[ Y ] Approve", APPROVE_COLOR),
        ("   [ N ] Deny", DENY_COLOR),
        ("   [ Tab ] Next", MUTED_TEXT),
    ]
    instructions_x = panel_x + 24
    instructions_y = panel_y + panel_height - 40
    for text, color in instructions:
        rendered = small_font.render(text, True, color)
        surface.blit(rendered, (instructions_x, instructions_y))
        instructions_x += rendered.get_width()


def update_curiosity_maps(
    room: Room,
    observation: dict[str, Any],
    surprise: float,
    zone_surprise: dict[tuple[int, int], float],
    object_surprise: dict[str, float],
) -> None:
    position = observation["agent_position"]
    zone_surprise[position] = zone_surprise[position] * 0.95 + surprise * 0.05

    for cell in observation["surroundings"]:
        room_object = cell["object"]
        if room_object is None:
            continue

        name = room_object["name"]
        object_surprise[name] = object_surprise[name] * 0.95 + surprise * 0.05

    for room_object in room.objects:
        object_surprise.setdefault(room_object.name, 0.0)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Curiosity Mind")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    small_font = pygame.font.SysFont("consolas", 14)

    room = Room()
    agent = create_agent()
    agent.enable_desktop()
    interpreter = Interpreter()
    observation = room.get_observation(event="started")
    encoded_observation = encode_observation(observation, room)

    zone_surprise: dict[tuple[int, int], float] = defaultdict(float)
    object_surprise: dict[str, float] = defaultdict(float)
    input_text = ""
    answer_text = ""
    intervention_log: list[str] = []
    conversation_log: list[tuple[str, str]] = []
    inner_voice_log: list[tuple[str, str]] = []
    last_inner_voice = ""
    last_question = ""
    step = 0
    last_save_step = 0
    permission_selected = 0
    running = True

    try:
        while running:
            pending = agent.permissions.get_pending()
            if permission_selected >= len(pending):
                permission_selected = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if pending:
                        if event.key == pygame.K_y:
                            agent.permissions.answer_pending(pending[permission_selected]["id"], True)
                            permission_selected = 0
                        elif event.key == pygame.K_n:
                            agent.permissions.answer_pending(pending[permission_selected]["id"], False)
                            permission_selected = 0
                        elif event.key == pygame.K_TAB:
                            permission_selected = (permission_selected + 1) % len(pending)
                        continue

                    pending_question = agent.get_question()
                    if event.key == pygame.K_RETURN:
                        if pending_question:
                            answer = answer_text.strip()
                            if answer:
                                agent.receive_answer(answer)
                                conversation_log.append(("You", answer))
                                conversation_log = conversation_log[-10:]
                                last_question = ""
                            answer_text = ""
                        else:
                            command = input_text.strip()
                            if command:
                                result = interpreter.interpret(command, room)
                                room.receive_message(command)
                                conversation_log.append(("You", command))
                                conversation_log = conversation_log[-10:]
                                intervention_log.append(f"> {command}: {result['message']}")
                            input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        if pending_question:
                            answer_text = answer_text[:-1]
                        else:
                            input_text = input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        if pending_question:
                            answer_text = ""
                        else:
                            input_text = ""
                    elif event.unicode and event.unicode.isprintable():
                        if pending_question:
                            answer_text += event.unicode
                        else:
                            input_text += event.unicode

            for _ in range(STEPS_PER_FRAME):
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

                if agent.last_response:
                    conversation_log.append(("Agent", agent.last_response))
                    conversation_log = conversation_log[-10:]
                    agent.last_response = ""

                update_curiosity_maps(
                    room=room,
                    observation=next_observation,
                    surprise=stats["intrinsic_reward"],
                    zone_surprise=zone_surprise,
                    object_surprise=object_surprise,
                )
                encoded_observation = encoded_next_observation
                step += 1
                if step - last_save_step >= 500:
                    agent.save(STATE_PATH)
                    last_save_step = step
                    print(f"saved at step {step}")

            inner_voice = agent.get_inner_voice()
            if inner_voice and inner_voice != last_inner_voice:
                dominant_emotion = agent.emotional_state.dominant()
                inner_voice_log.append((inner_voice, dominant_emotion))
                inner_voice_log = inner_voice_log[-10:]
                last_inner_voice = inner_voice

            question = agent.get_question()
            if question and question != last_question:
                conversation_log.append(("Agent", question))
                conversation_log = conversation_log[-10:]
                last_question = question

            screen.fill(BACKGROUND)
            draw_grid_panel(screen, room, (PANEL_PADDING, 58), font)
            heatmap_x = PANEL_PADDING + GRID_SIZE * CELL_SIZE + PANEL_GAP
            draw_heatmap_panel(screen, room, zone_surprise, object_surprise, (heatmap_x, 58), font, small_font)
            draw_intervention_panel(screen, input_text, intervention_log, font, small_font)
            draw_conversation_panel(
                screen,
                question,
                answer_text,
                conversation_log,
                (heatmap_x, BOTTOM_PANEL_Y),
                font,
                small_font,
            )
            screen.blit(small_font.render(f"steps {step}", True, TEXT), (PANEL_PADDING, WINDOW_HEIGHT - 28))
            if pending:
                draw_permission_overlay(screen, pending, permission_selected, font, small_font)
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        agent.save(STATE_PATH)
        pygame.quit()


if __name__ == "__main__":
    main()
