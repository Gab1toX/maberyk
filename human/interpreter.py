from __future__ import annotations

from typing import Any, Callable

from entorno.room import Room, RoomObject


class Interpreter:
    """Rule-based text interpreter for modifying the environment."""

    def __init__(self) -> None:
        self.intent_dictionary: dict[str, Callable[[Room], dict[str, Any]]] = {
            "more obstacles": self._more_obstacles,
            "obstacles": self._more_obstacles,
            "block": self._more_obstacles,
            "make it hard": self._make_it_hard,
            "hard": self._make_it_hard,
            "harder": self._make_it_hard,
            "new object": self._new_object,
            "add object": self._new_object,
            "object": self._new_object,
            "danger": self._danger,
            "dangerous": self._danger,
            "reset zones": self._danger,
        }

    def interpret(self, text: str, room: Room) -> dict[str, Any]:
        normalized_text = text.lower().strip()

        for phrase, handler in self.intent_dictionary.items():
            if phrase in normalized_text:
                return handler(room)

        return {
            "intent": "unknown",
            "changed": False,
            "message": "No matching environment modification found.",
        }

    def _more_obstacles(self, room: Room) -> dict[str, Any]:
        positions = room.add_random_obstacles(5)
        return {
            "intent": "more_obstacles",
            "changed": True,
            "positions": positions,
            "message": f"Added {len(positions)} blocking obstacles.",
        }

    def _make_it_hard(self, room: Room) -> dict[str, Any]:
        previous_range = room.visible_range
        room.visible_range = max(1, room.visible_range - 1)
        return {
            "intent": "make_it_hard",
            "changed": room.visible_range != previous_range,
            "visible_range": room.visible_range,
            "message": f"Visible range is now {room.visible_range}.",
        }

    def _new_object(self, room: Room) -> dict[str, Any]:
        room_object = room.add_random_object()
        return {
            "intent": "new_object",
            "changed": True,
            "object": self._object_view(room_object),
            "message": f"Added {room_object.name} with behavior {room_object.behavior}.",
        }

    def _danger(self, room: Room) -> dict[str, Any]:
        positions = room.add_random_danger_zones(4)
        return {
            "intent": "danger",
            "changed": True,
            "positions": positions,
            "message": f"Activated {len(positions)} reset zones.",
        }

    def _object_view(self, room_object: RoomObject) -> dict[str, Any]:
        return {
            "name": room_object.name,
            "behavior": room_object.behavior,
            "position": room_object.position,
            "color": room_object.color,
        }
