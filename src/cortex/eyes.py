"""Its eyes: what a window on the human's screen shows, or the whole screen, in words.

The picture is taken on the machine the screen belongs to, by fixed code here, and a vision model
on the local model server says what is in it. Only those words go to the brain: no picture enters
its container, its memory or its prompts (and a line over the bridge could not carry one anyway).
A window is taken by itself, not the whole screen scaled down: on a whole screen made small
enough for a small model, no word in any window can be read. A window that lies behind another
is taken as it is drawn, not as it is covered. Which window, and what to look for, the brain says;
the terminal it is shown in is passed over unless it names it, since in that it sees only itself.
"""

from __future__ import annotations

import struct
import sys
import time
import zlib
from pathlib import Path

import ollama

SIDE = 1024  # the longest side of the picture the model is shown, in pixels
SCREEN = ("screen", "bildschirm", "desktop", "whole screen", "the screen")  # names for all of it
LOOK = (
    "This is a window on a computer screen. Say in three to five plain sentences what it shows: "
    "which program, what is open in it, what is to be seen and read there. Name things, quote "
    "short words you can read, and say nothing you cannot see."
)
LOOK_SCREEN = (
    "This is a whole computer screen, made small. Say in three to five plain sentences which "
    "windows and programs are to be seen and where they lie, and what each shows. Say nothing "
    "you cannot see."
)
FOR = " Look in particular for this, and say plainly what you find, also when it is not there: "
OWN: set[int] = set()  # the windows the brain itself is shown in, on this machine


def png(width: int, height: int, rgb: bytes) -> bytes:
    """A picture as a PNG file, from rows of red, green and blue bytes."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[row * stride : (row + 1) * stride] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def shrink(width: int, height: int, bgra: bytes, side: int = SIDE) -> tuple[int, int, bytes]:
    """Rows of blue, green, red and spare bytes, made at most `side` pixels long and turned into
    red, green and blue: every n-th pixel of every n-th row, which keeps letters readable longer
    than it should and costs nothing."""
    step = max(1, -(-max(width, height) // side))
    out = bytearray()
    for y in range(0, height, step):
        row = bgra[y * width * 4 : (y + 1) * width * 4]
        pixels = row[:: 4 * step], row[1 :: 4 * step], row[2 :: 4 * step]
        line = bytearray(len(pixels[0]) * 3)
        line[0::3], line[1::3], line[2::3] = pixels[2], pixels[1], pixels[0]
        out += line
    return -(-width // step), -(-height // step), bytes(out)


def crop(width: int, bgra: bytes, left: int, top: int, wide: int, high: int) -> bytes:
    """A part of a picture of rows of four bytes a pixel."""
    return b"".join(bgra[(y * width + left) * 4 : (y * width + left + wide) * 4] for y in range(top, top + high))


def _win32():
    if sys.platform != "win32":
        raise OSError("this machine's screen cannot be looked at")
    import ctypes
    from ctypes import wintypes

    user, gdi, dwm, kernel = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.dwmapi, ctypes.windll.kernel32
    try:
        user.SetProcessDPIAware()  # windows' places in real pixels, not in scaled ones
    except (AttributeError, OSError):
        pass
    HWND, HANDLE = wintypes.HWND, wintypes.HANDLE
    user.GetForegroundWindow.restype = kernel.GetConsoleWindow.restype = user.GetAncestor.restype = HWND
    user.GetAncestor.argtypes = [HWND, wintypes.UINT]
    user.IsWindowVisible.argtypes = user.IsIconic.argtypes = user.GetWindowTextLengthW.argtypes = [HWND]
    user.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetWindowRect.argtypes = [HWND, ctypes.c_void_p]
    user.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(wintypes.DWORD)]
    user.PrintWindow.argtypes = [HWND, HANDLE, wintypes.UINT]
    user.GetDC.restype = gdi.CreateCompatibleDC.restype = gdi.CreateCompatibleBitmap.restype = HANDLE
    user.GetDC.argtypes = [HWND]
    user.ReleaseDC.argtypes = [HWND, HANDLE]
    gdi.CreateCompatibleDC.argtypes = [HANDLE]
    gdi.CreateCompatibleBitmap.argtypes = [HANDLE, ctypes.c_int, ctypes.c_int]
    gdi.SelectObject.argtypes = [HANDLE, HANDLE]
    gdi.SelectObject.restype = HANDLE
    gdi.BitBlt.argtypes = [HANDLE] + [ctypes.c_int] * 4 + [HANDLE] + [ctypes.c_int] * 2 + [wintypes.DWORD]
    gdi.GetDIBits.argtypes = [HANDLE, HANDLE, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
    gdi.DeleteObject.argtypes = gdi.DeleteDC.argtypes = [HANDLE]
    dwm.DwmGetWindowAttribute.argtypes = [HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [HANDLE]
    return ctypes, wintypes, user, gdi, dwm, kernel


def remember_own() -> None:
    """The windows the brain is shown in: the terminal it is being started from, which is in front
    while the human starts it. Called once by whatever starts the brain on this machine."""
    try:
        _, _, user, _, _, kernel = _win32()
    except OSError:
        return
    for window in (user.GetForegroundWindow(), kernel.GetConsoleWindow()):
        if window:
            OWN.add(window)
            OWN.add(user.GetAncestor(window, 3) or window)  # the terminal that owns a console


def program(window: int) -> str:
    """The program a window belongs to, by the name of its file."""
    ctypes, wintypes, user, _, _, kernel = _win32()
    pid = wintypes.DWORD()
    user.GetWindowThreadProcessId(window, ctypes.byref(pid))
    process = kernel.OpenProcess(0x1000, False, pid.value)  # enough to ask its name, nothing more
    if not process:
        return ""
    try:
        name, size = ctypes.create_unicode_buffer(520), wintypes.DWORD(520)
        return Path(name.value).stem if kernel.QueryFullProcessImageNameW(process, 0, name, ctypes.byref(size)) else ""
    finally:
        kernel.CloseHandle(process)


def windows() -> list[dict]:
    """Every window that can be seen on the screen, the topmost first: its handle, program, title,
    whether it is in front, whether it is minimised and whether the brain itself is shown in it."""
    ctypes, wintypes, user, _, dwm, _ = _win32()
    front = user.GetForegroundWindow()
    found: list[dict] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(window, _):
        length = user.GetWindowTextLengthW(window)
        hidden = wintypes.DWORD()  # a window of another desktop, or one the system keeps out of sight
        dwm.DwmGetWindowAttribute(window, 14, ctypes.byref(hidden), ctypes.sizeof(hidden))
        if length and user.IsWindowVisible(window) and not hidden.value:
            title = ctypes.create_unicode_buffer(length + 1)
            user.GetWindowTextW(window, title, length + 1)
            app = program(window)
            if title.value != "Program Manager" and app != "TextInputHost":  # the desktop itself, the keyboard
                found.append({
                    "handle": window, "app": app, "title": title.value[:90], "front": window == front,
                    "minimised": bool(user.IsIconic(window)), "own": window in OWN,
                })
        return True

    user.EnumWindows(each, 0)
    return found


def find(target: str) -> dict | None:
    """The window a name means. Empty: the one in front, or, when that is the terminal the brain is
    shown in, the topmost one behind it. A name for the whole screen: None. Otherwise the topmost
    window whose program or title holds every word of the name; the brain's own terminal only
    when nothing else does. Raises OSError, naming what is open, when no window fits."""
    wanted = " ".join(target.lower().split())
    if wanted in SCREEN:
        return None
    seen = windows()
    if not wanted or wanted in ("front", "in front", "foreground"):
        front = [w for w in seen if w["front"] and not w["own"]]
        rest = [w for w in seen if not w["own"] and not w["minimised"]]
        if front or rest:
            return (front or rest)[0]
        raise OSError("no window but the terminal you are shown in is open on the screen")
    words = wanted.split()
    fits = [w for w in seen if all(word in f"{w['app']} {w['title']}".lower() for word in words)]
    fits.sort(key=lambda w: (w["own"], w["minimised"], w["app"].lower() != wanted))  # stable: topmost first
    if fits:
        return fits[0]
    open_now = "; ".join(f"{w['app']} (“{w['title']}”)" for w in seen if not w["own"])[:400]
    raise OSError(f"no window on the screen is called “{target[:60]}”. Open are: {open_now or 'none'}")


def frame(window: int) -> tuple[int, int, int, int]:
    """Where a window is drawn on the screen, without the invisible border around it: left, top,
    width and height in pixels."""
    ctypes, wintypes, user, _, dwm, _ = _win32()
    box = wintypes.RECT()
    if dwm.DwmGetWindowAttribute(window, 9, ctypes.byref(box), ctypes.sizeof(box)):
        user.GetWindowRect(window, ctypes.byref(box))
    return box.left, box.top, box.right - box.left, box.bottom - box.top


def capture(window: dict | None) -> bytes:
    """A window as it is drawn now, even where another lies over it, or the whole screen (None),
    as a PNG at most SIDE pixels long."""
    ctypes, wintypes, user, gdi, _, _ = _win32()
    if window is not None and window["minimised"]:
        raise OSError(f"“{window['title']}” is minimised: nothing in it can be seen until it is opened again")
    if window is None:
        left, top = user.GetSystemMetrics(76), user.GetSystemMetrics(77)  # the whole of every screen
        width, height = user.GetSystemMetrics(78), user.GetSystemMetrics(79)
        inner = (0, 0, width, height)
    else:
        outer = wintypes.RECT()
        user.GetWindowRect(window["handle"], ctypes.byref(outer))
        left, top, width, height = outer.left, outer.top, outer.right - outer.left, outer.bottom - outer.top
        x, y, wide, high = frame(window["handle"])
        inner = (max(0, x - left), max(0, y - top), min(wide, width), min(high, height))
    if width < 8 or height < 8:
        raise OSError("that window is too small to see anything in")
    screen = user.GetDC(None)
    memory = gdi.CreateCompatibleDC(screen)
    bitmap = gdi.CreateCompatibleBitmap(screen, width, height)
    try:
        gdi.SelectObject(memory, bitmap)
        # Drawn by the window itself, so what covers it does not matter; 2 takes what the graphics
        # card draws too, such as a model in FreeCAD. Where a window will not, the screen is copied.
        drawn = window is not None and user.PrintWindow(window["handle"], memory, 2)
        if not drawn:
            gdi.BitBlt(memory, 0, 0, width, height, screen, left, top, 0x00CC0020)  # SRCCOPY
        info = struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, 0, 0, 0, 0, 0)  # top row first
        header = ctypes.create_string_buffer(info, 44)
        pixels = ctypes.create_string_buffer(width * height * 4)
        if not gdi.GetDIBits(memory, bitmap, 0, height, pixels, header, 0):
            raise OSError("the window could not be read")
        x, y, wide, high = inner
        wide, high = min(wide, width - x), min(high, height - y)
        small = shrink(wide, high, crop(width, pixels.raw, x, y, wide, high))
    finally:
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(memory)
        user.ReleaseDC(None, screen)
    return png(*small)


def describe(picture: bytes, model: str, host: str, ask: str = LOOK, question: str = "") -> str:
    """What a vision model on the model server says the picture shows. It runs on the processor,
    so that the thinking model keeps the graphics card to itself."""
    prompt = ask + (FOR + question.strip()[:300] if question.strip() else "")
    reply = ollama.Client(host=host).chat(
        model=model, messages=[{"role": "user", "content": prompt, "images": [picture]}],
        options={"num_gpu": 0, "temperature": 0.1, "num_predict": 260}, keep_alive="15m", think=False,
    )
    return " ".join(reply.message.content.split())


def see(model: str, host: str, target: str = "", question: str = "") -> dict:
    """Look at a window (`target` names it; empty is the one in front) or the whole screen, and
    say what it shows, with an eye to `question`. What the brain is lent, as data."""
    started = time.monotonic()
    window = find(target)
    words = describe(capture(window), model, host, LOOK if window else LOOK_SCREEN, question)
    if not words:
        raise OSError("the eyes saw nothing they could put into words")
    return {
        "window": window["title"][:90] if window else "the whole screen",
        "seen": words[:1000], "seconds": round(time.monotonic() - started, 1),
    }
