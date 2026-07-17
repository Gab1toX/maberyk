from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from brain.permissions import PermissionManager


class DesktopEnvironment:
    """Maberyk's interface to the real desktop, gated by PermissionManager.

    Every action goes through permissions.request() first — nothing here
    touches mouse, keyboard, or screen without an ALLOWED verdict.
    """

    def __init__(self, permissions: "PermissionManager") -> None:
        self.permissions = permissions
        if not self.permissions.is_granted("desktop.screenshot"):
            self.permissions.grant_permanent("desktop.screenshot", "pc_control")

        import pyautogui

        self._pyautogui = pyautogui
        self._pyautogui.FAILSAFE = True
        self._pyautogui.PAUSE = 0.1

    def _authorized(self, action_name: str, context: str = "") -> bool:
        level, reason = self.permissions.request(action_name, "pc_control", context)
        if level.name != "ALLOWED":
            print(f"DesktopEnvironment: '{action_name}' blocked ({level.name}): {reason}")
            return False
        return True

    def screenshot(self) -> "PILImage | None":
        if not self._authorized("desktop.screenshot"):
            return None
        try:
            return self._pyautogui.screenshot()
        except Exception as exc:
            print(f"DesktopEnvironment: screenshot failed: {exc}")
            return None

    def move_mouse(self, x: int, y: int) -> bool:
        context = f"move to ({x}, {y})"
        if not self._authorized("desktop.move_mouse", context):
            return False
        try:
            self._pyautogui.moveTo(x, y)
            return True
        except Exception as exc:
            print(f"DesktopEnvironment: move_mouse failed: {exc}")
            return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        if button not in ("left", "right"):
            print(f"DesktopEnvironment: invalid button '{button}' (must be 'left' or 'right')")
            return False
        context = f"click {button} at ({x}, {y})"
        if not self._authorized("desktop.click", context):
            return False
        try:
            self._pyautogui.click(x=x, y=y, button=button)
            return True
        except Exception as exc:
            print(f"DesktopEnvironment: click failed: {exc}")
            return False

    def type_text(self, text: str) -> bool:
        context = f"type: {text[:50]}"
        if not self._authorized("desktop.type_text", context):
            return False
        try:
            self._pyautogui.typewrite(text)
            return True
        except Exception as exc:
            print(f"DesktopEnvironment: type_text failed: {exc}")
            return False

    def get_active_window(self) -> str | None:
        if not self._authorized("desktop.get_active_window"):
            return None
        try:
            window = self._pyautogui.getActiveWindow()
            return window.title if window else None
        except Exception as exc:
            print(f"DesktopEnvironment: get_active_window failed: {exc}")
            return None
