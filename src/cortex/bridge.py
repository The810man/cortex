"""What the machine that launched a container lends the brain inside it: its speakers and its screen.

A brain in a container has neither. It asks for them as data (these words in this voice, this
workspace page on the screen) and the launching machine carries the request out with code that is
fixed out here. Nothing a brain sends is ever run.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import socketserver
import threading
from collections.abc import Callable
from typing import Any

BRIDGE = "CORTEX_BRIDGE"  # host:port:key of the launching machine's bridge, set in a container
TIMEOUT = 150.0  # the longest speech allowed, in a voice that has to be fetched first


async def relay(bridge: str, do: str, **ask: Any) -> str:
    """Have the machine that launched this container do it."""
    try:
        host, port, key = bridge.rsplit(":", 2)
        async with asyncio.timeout(TIMEOUT):
            # A reply can carry a picture or some seconds of sound for a mind that sees and hears.
            reader, writer = await asyncio.open_connection(host, int(port), limit=2**25)
            try:
                writer.write(json.dumps({"key": key, "do": do, **ask}).encode() + b"\n")
                await writer.drain()
                reply = json.loads(await reader.readline())
            finally:
                writer.close()
    except ValueError:  # a malformed address, or a bridge that hung up without a word
        raise OSError("the bridge gave no answer") from None
    if not reply["ok"]:
        raise OSError(reply["text"])
    return reply["text"]


class _Request(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            ask = json.loads(self.rfile.readline(65536))
            if not secrets.compare_digest(str(ask.pop("key", "")), self.server.key):
                return
            reply = {"ok": True, "text": self.server.lends[ask.pop("do")](ask)}
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            reply = {"ok": False, "text": str(exc)[:200] or type(exc).__name__}
        self.wfile.write(json.dumps(reply).encode() + b"\n")


class Bridge(socketserver.ThreadingTCPServer):
    """Lends this machine to a brain in a container, for as long as the launcher lives.

    `lends` is everything on offer, by name. Each takes the request's data, checks it out here and
    answers in a line of text, so the container decides what is said or shown and nothing else.
    """

    daemon_threads = True

    def __init__(self, lends: dict[str, Callable[[dict[str, Any]], str]]) -> None:
        super().__init__(("127.0.0.1", 0), _Request)
        self.lends = lends
        self.key = secrets.token_hex(16)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def address(self, host: str = "127.0.0.1") -> str:
        """What goes in `BRIDGE` to reach this, given the name the asker knows this machine by."""
        return f"{host}:{self.server_address[1]}:{self.key}"
