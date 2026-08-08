from __future__ import annotations

import re

import pygetwindow
import pyperclip


class DesktopEnv:
    """Observes the real desktop instead of a grid. Equivalent to Room, but
    the environment is the user's active window and clipboard rather than a
    simulated 60x60 grid."""

    CLIPBOARD_MAX_CHARS = 200

    KNOWN_APPS = (
        "spotify", "brave", "chrome", "firefox", "code", "discord", "notepad",
        "explorer", "word", "excel", "terminal", "youtube", "valorant",
        "herramienta recortes", "recortes", "teams", "zoom", "telegram",
        "whatsapp", "notion", "obsidian", "figma", "cursor",
    )
    # Multi-word names must be checked before their single-word substrings
    # (e.g. "herramienta recortes" before "recortes") so the more specific
    # match wins.
    _KNOWN_APPS_BY_SPECIFICITY = tuple(
        sorted(KNOWN_APPS, key=lambda name: -len(name.split()))
    )

    def __init__(self) -> None:
        self.last_window = ""
        self.last_title = ""
        self._pending_message = ""

    def get_observation(self) -> dict:
        active_window, window_title = self._read_active_window()
        clipboard_text = self._read_clipboard()

        event = "window_changed" if active_window != self.last_window else "idle"

        human_message = self._pending_message
        self._pending_message = ""

        observation = {
            "active_window": active_window,
            "window_title": window_title,
            "clipboard_text": clipboard_text,
            "event": event,
            "human_message": human_message,
            "description": self.description(active_window, event, clipboard_text),
        }

        self.last_window = active_window
        self.last_title = window_title
        return observation

    def receive_message(self, text: str) -> None:
        message = str(text or "").strip()
        if message:
            self._pending_message = message

    def description(self, active_window: str, event: str, clipboard_text: str) -> str:
        if clipboard_text:
            summary = f"using {active_window}, reading: {clipboard_text[:30]}"
        else:
            summary = f"using {active_window}, {event}"

        words = summary.split()
        return " ".join(words[:15])

    def _read_active_window(self) -> tuple[str, str]:
        try:
            window = pygetwindow.getActiveWindow()
            title = (window.title or "").strip() if window is not None else ""
            if not title:
                return "", ""

            lowered_title = title.lower()
            for app_name in self._KNOWN_APPS_BY_SPECIFICITY:
                if app_name in lowered_title:
                    return app_name, title

            # No known app matched: apps usually put their name at the end
            # of the title, after a " - " or " | " separator.
            segments = re.split(r" - | \| ", title)
            last_segment = segments[-1] if segments else title
            return last_segment.lower().strip(), title
        except Exception:
            return "", ""

    def _read_clipboard(self) -> str:
        try:
            text = pyperclip.paste()
            text = str(text or "").strip()
            return text[: self.CLIPBOARD_MAX_CHARS]
        except Exception:
            return ""
