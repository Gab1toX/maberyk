from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class RoomObject:
    name: str
    behavior: str
    position: tuple[int, int]
    color: str = "gray"


class Room:
    """A larger room with objects, zones, and observation-only interactions."""

    width = 60
    height = 60
    object_count = 30

    DIRECTIONS = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    OBSERVATION_OFFSETS = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.agent_position = (self.width // 2, self.height // 2)
        self.start_position = self.agent_position
        self.door_open = False
        self.visible_range = 1
        self.obstacles: set[tuple[int, int]] = set()
        self.rest_zones: set[tuple[int, int]] = set(
            self._random_positions_in_area(20, 0, 29, 0, 29)
        )
        self.knowledge_zones: set[tuple[int, int]] = set(
            self._random_positions_in_area(20, 30, 59, 0, 29)
        )
        self.danger_zones: set[tuple[int, int]] = set(
            self._random_positions_in_area(30, 0, 29, 30, 59)
        )
        self.chaos_zones: set[tuple[int, int]] = set(
            self._random_positions_in_area(20, 30, 59, 30, 59)
        )
        self._action_count: int = 0
        self._last_human_message: list[str] = []
        self.objects = self._create_objects()

    def move(self, direction: str) -> dict[str, Any]:
        """Move the agent in one of four directions and return observations."""
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")

        self._action_count += 1
        dx, dy = self.DIRECTIONS[direction]
        x, y = self.agent_position
        next_position = (x + dx, y + dy)
        event = f"moved_{direction}"

        if not self._inside_grid(next_position):
            event = "blocked_by_boundary"
        elif next_position not in self.obstacles:
            self.agent_position = next_position
            if self.agent_position in self.danger_zones:
                self.agent_position = self.start_position
                event = "reset_by_danger"
            elif self.agent_position in self.rest_zones:
                event = "entered_rest_zone"
            elif self.agent_position in self.knowledge_zones:
                event = "entered_knowledge_zone"
            elif self.agent_position in self.chaos_zones:
                self._move_agent_randomly()
                event = "chaos_moved"

        self._update_moving_objects()
        event = self._apply_random_event(event)
        return self.get_observation(event=event)

    def touch(self) -> dict[str, Any]:
        """Touch the first object in the 8 surrounding cells, if one exists."""
        self._action_count += 1
        touched_object = self._nearby_object()
        event = "nothing_touched"

        if touched_object is not None:
            event = self._apply_touch_behavior(touched_object)

        self._update_moving_objects()
        event = self._apply_random_event(event)
        return self.get_observation(event=event)

    def observe(self) -> list[dict[str, Any]]:
        """Return the 8 cells surrounding the agent."""
        cells = []
        ax, ay = self.agent_position

        for dx, dy in self._observation_offsets():
            position = (ax + dx, ay + dy)
            cells.append(
                {
                    "position": position,
                    "inside": self._inside_grid(position),
                    "obstacle": position in self.obstacles,
                    "danger": position in self.danger_zones,
                    "object": self._object_view_at(position),
                }
            )

        return cells

    def get_observation(self, event: str | None = None) -> dict[str, Any]:
        surroundings = self.observe()
        zone = self._current_zone()
        human_message = " ".join(self._last_human_message)
        self._last_human_message.clear()
        return {
            "agent_position": self.agent_position,
            "door_open": self.door_open,
            "event": event,
            "surroundings": surroundings,
            "zone": zone,
            "description": self._build_description(event, surroundings),
            "human_message": human_message,
        }

    def receive_message(self, text: str) -> None:
        message = str(text or "").strip()
        if message:
            self._last_human_message.append(message)

    def _build_description(
        self,
        event: str | None,
        surroundings: list[dict[str, Any]],
    ) -> str:
        def word_count(parts: list[str]) -> int:
            return len(", ".join(parts).split())

        def append_if_short(part: str) -> None:
            if word_count([*parts, part]) <= 15:
                parts.append(part)

        zone = self._current_zone()
        parts = [f"at position {self.agent_position}"]
        if zone != "normal":
            parts.append(f"in {zone} zone")
        event = event or ""

        nearby_object = next(
            (cell["object"] for cell in surroundings if cell["object"] is not None),
            None,
        )

        if "danger" in event or "reset" in event:
            parts.append("danger zone triggered")
            return ", ".join(parts)

        if nearby_object is not None:
            parts.append(
                f"a {nearby_object['color']} {nearby_object['name']} is nearby"
            )
        else:
            parts.append("nothing nearby")

        if self.door_open:
            append_if_short("the door is open")

        if "noise" in event:
            append_if_short(f"{event.removesuffix('_made_noise')} made noise")
        elif "changed_color" in event:
            append_if_short(f"{event.removesuffix('_changed_color')} changed color")
        elif event.startswith("moved_"):
            append_if_short(f"agent moved {event.removeprefix('moved_')}")

        return ", ".join(parts)

    def _create_objects(self) -> list[RoomObject]:
        behavior_specs = []
        behavior_specs.extend((f"bell_{index}", "noise", "gold") for index in range(1, 5))
        behavior_specs.extend((f"wanderer_{index}", "move_randomly", "blue") for index in range(1, 5))
        behavior_specs.extend((f"lamp_{index}", "change_color", "red") for index in range(1, 5))
        behavior_specs.extend((f"switch_{index}", "open_door", "green") for index in range(1, 5))
        behavior_specs.extend((f"stone_{index}", "nothing", "gray") for index in range(1, 5))
        behavior_specs.extend((f"mirror_{index}", "reflect", "silver") for index in range(1, 4))
        behavior_specs.extend((f"plant_{index}", "grow", "green") for index in range(1, 4))
        behavior_specs.extend((f"portal_{index}", "teleport", "purple") for index in range(1, 3))
        behavior_specs.extend((f"crystal_{index}", "glow", "cyan") for index in range(1, 3))

        positions = self._random_positions(len(behavior_specs))
        return [
            RoomObject(name=name, behavior=behavior, position=position, color=color)
            for (name, behavior, color), position in zip(behavior_specs, positions)
        ]

    def _random_positions_in_area(
        self,
        amount: int,
        min_x: int,
        max_x: int,
        min_y: int,
        max_y: int,
    ) -> list[tuple[int, int]]:
        blocked = {self.agent_position}
        positions = []

        while len(positions) < amount:
            position = (
                self.random.randint(min_x, max_x),
                self.random.randint(min_y, max_y),
            )
            if position not in blocked:
                blocked.add(position)
                positions.append(position)

        return positions

    def _random_positions(self, amount: int) -> list[tuple[int, int]]:
        blocked = {self.agent_position}
        positions = []

        while len(positions) < amount:
            position = (
                self.random.randrange(self.width),
                self.random.randrange(self.height),
            )
            if position not in blocked:
                blocked.add(position)
                positions.append(position)

        return positions

    def add_random_object(self, behavior: str | None = None) -> RoomObject:
        behavior_specs = {
            "noise": ("bell", "gold"),
            "move_randomly": ("wanderer", "blue"),
            "change_color": ("lamp", "red"),
            "open_door": ("switch", "green"),
            "nothing": ("stone", "gray"),
            "reflect": ("mirror", "silver"),
            "grow": ("plant", "green"),
            "teleport": ("portal", "purple"),
            "glow": ("crystal", "cyan"),
        }
        behavior = behavior or self.random.choice(list(behavior_specs))
        name_prefix, color = behavior_specs.get(behavior, ("object", "gray"))
        position = self._random_free_position()
        room_object = RoomObject(
            name=f"{name_prefix}_{len(self.objects) + 1}",
            behavior=behavior,
            position=position,
            color=color,
        )
        self.objects.append(room_object)
        return room_object

    def add_random_obstacles(self, amount: int) -> list[tuple[int, int]]:
        positions = []
        for _ in range(amount):
            position = self._random_free_position()
            self.obstacles.add(position)
            positions.append(position)
        return positions

    def add_random_danger_zones(self, amount: int) -> list[tuple[int, int]]:
        positions = []
        for _ in range(amount):
            position = self._random_free_position()
            self.danger_zones.add(position)
            positions.append(position)
        return positions

    def _nearby_object(self) -> RoomObject | None:
        nearby_positions = {
            (
                self.agent_position[0] + dx,
                self.agent_position[1] + dy,
            )
            for dx, dy in self.OBSERVATION_OFFSETS
        }
        return next(
            (room_object for room_object in self.objects if room_object.position in nearby_positions),
            None,
        )

    def _apply_touch_behavior(self, room_object: RoomObject) -> str:
        if room_object.behavior == "noise":
            return f"{room_object.name}_made_noise"

        if room_object.behavior == "move_randomly":
            self._move_object_randomly(room_object)
            return f"{room_object.name}_moved"

        if room_object.behavior == "change_color":
            room_object.color = self.random.choice(["red", "blue", "green", "yellow", "purple"])
            return f"{room_object.name}_changed_color"

        if room_object.behavior == "open_door":
            self.door_open = True
            return "door_opened"

        if room_object.behavior == "reflect":
            return f"{room_object.name}_reflected"

        if room_object.behavior == "grow":
            room_object.color = self.random.choice(["yellow", "green", "gold"])
            return f"{room_object.name}_grew"

        if room_object.behavior == "teleport":
            self.agent_position = self._random_free_position()
            return f"{room_object.name}_teleported_agent"

        if room_object.behavior == "glow":
            return f"{room_object.name}_glowed"

        return f"{room_object.name}_did_nothing"

    def _apply_random_event(self, event: str) -> str:
        if self._action_count % 50 != 0 or not self.objects:
            return event

        room_object = self.random.choice(self.objects)
        random_event = self._apply_touch_behavior(room_object)
        return f"{event}_{random_event}_random_event"

    def _move_agent_randomly(self) -> None:
        dx, dy = self.random.choice(list(self.DIRECTIONS.values()))
        position = (self.agent_position[0] + dx, self.agent_position[1] + dy)
        if self._inside_grid(position) and position not in self.obstacles:
            self.agent_position = position

    def _update_moving_objects(self) -> None:
        for room_object in self.objects:
            if room_object.behavior == "move_randomly":
                self._move_object_randomly(room_object)

    def _move_object_randomly(self, room_object: RoomObject) -> None:
        candidates = []
        occupied = {
            other.position
            for other in self.objects
            if other is not room_object
        }

        for dx, dy in self.DIRECTIONS.values():
            position = (room_object.position[0] + dx, room_object.position[1] + dy)
            if (
                self._inside_grid(position)
                and position != self.agent_position
                and position not in occupied
                and position not in self.obstacles
                and position not in self.danger_zones
            ):
                candidates.append(position)

        if candidates:
            room_object.position = self.random.choice(candidates)

    def _object_view_at(self, position: tuple[int, int]) -> dict[str, Any] | None:
        for room_object in self.objects:
            if room_object.position == position:
                return {
                    "name": room_object.name,
                    "behavior": room_object.behavior,
                    "color": room_object.color,
                }
        return None

    def _inside_grid(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def _current_zone(self) -> str:
        if self.agent_position in self.rest_zones:
            return "rest"
        if self.agent_position in self.knowledge_zones:
            return "knowledge"
        if self.agent_position in self.danger_zones:
            return "danger"
        if self.agent_position in self.chaos_zones:
            return "chaos"
        return "normal"

    def _observation_offsets(self) -> tuple[tuple[int, int], ...]:
        if self.visible_range <= 1:
            return self.OBSERVATION_OFFSETS

        offsets = []
        for dy in range(-self.visible_range, self.visible_range + 1):
            for dx in range(-self.visible_range, self.visible_range + 1):
                if dx != 0 or dy != 0:
                    offsets.append((dx, dy))
        return tuple(offsets)

    def _random_free_position(self) -> tuple[int, int]:
        occupied = {self.agent_position, *self.obstacles, *self.danger_zones}
        occupied.update(room_object.position for room_object in self.objects)

        while True:
            position = (
                self.random.randrange(self.width),
                self.random.randrange(self.height),
            )
            if position not in occupied:
                return position
