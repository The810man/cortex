"""The one gateway to the local model server. Every region shares it."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import zlib
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

import httpx
import numpy as np
import ollama
from pydantic import BaseModel

from .config import ModelSettings

T = TypeVar("T", bound=BaseModel)
FALLBACK_DIM = 256
THINKING_ROOM = 1536  # tokens a model on llama.cpp may think for before it writes its reply


def lexical_vector(text: str) -> np.ndarray:
    """Hashed bag-of-words: a crude stand-in used when no embedding model is available."""
    vec = np.zeros(FALLBACK_DIM, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        vec[zlib.crc32(word.encode()) % FALLBACK_DIM] += 1.0
    return vec


class Truncated(Exception):
    """A reply that ran out of room before it ended: its JSON has no end, and nothing in it can be
    used. Told only that the JSON was broken, the mind took it for a parser's fault."""

    def __init__(self, tokens: int, characters: int, room: int) -> None:
        super().__init__(f"the reply ran out of room after {tokens} tokens ({characters} characters)")
        self.tokens, self.characters, self.room = tokens, characters, room


class Slots:
    """The model's slots, handed out in turn and by rank: the human's words first (2), then what
    the human raised (1), then everything else (0), each rank in the order it came."""

    def __init__(self, free: int) -> None:
        self._free = free
        self._waiting: tuple[deque[asyncio.Future[None]], ...] = (deque(), deque(), deque())

    @contextlib.asynccontextmanager
    async def take(self, rank: int | bool = 0) -> AsyncIterator[None]:
        line = self._waiting[2 if rank is True else int(rank)]
        if self._free:
            self._free -= 1
        else:
            turn: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            line.append(turn)
            try:
                await turn
            except asyncio.CancelledError:
                if turn.cancelled():
                    with contextlib.suppress(ValueError):
                        line.remove(turn)
                else:  # the slot arrived just as the caller was stopped: pass it on
                    self._release()
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        for line in reversed(self._waiting):
            while line:
                turn = line.popleft()
                if not turn.done():
                    turn.set_result(None)
                    return
        self._free += 1

    @property
    def taken(self) -> bool:
        return not self._free


class LLM:
    def __init__(self, models: ModelSettings) -> None:
        self.models = models
        self.client = ollama.AsyncClient(host=models.server)
        self._slots = Slots(max(1, models.parallel))
        self._inner = asyncio.Lock()  # the small model on the processor: one murmur at a time
        self._yielding: set[asyncio.Future] = set()  # thoughts in the making that make way for an urgent one
        self._no_think: set[str] = set()
        self.calls = 0
        self.tokens = 0
        self.speed = 0.0  # tokens per second, smoothed
        self.gpu_share: float | None = None  # how much of the thinking model is on the GPU; None until asked
        self.embed_error: str | None = None

    async def generate(
        self,
        role: str,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        temperature: float,
        max_tokens: int,
        on_start: Callable[[], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        urgent: bool = False,
        first: bool = False,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
    ) -> tuple[T, int]:
        """Stream one schema-constrained reply. Returns the parsed object and its token count.

        An urgent call does not wait for the thought in the making to finish: that thought is
        dropped where it stands and thought again from the start once the urgent one is out.
        `images` (PNG) and `audio` (WAV) go to the model with the words, as they are: a model that
        cannot take them is not handed them.
        """
        model = getattr(self.models, role)
        if urgent and self._slots.taken and self._yielding:
            next(iter(self._yielding)).cancel()
        while True:
            async with self._slots.take(2 if urgent else 1 if first else 0):
                if on_start:
                    on_start()
                reply = asyncio.ensure_future(
                    self._reply(model, system, prompt, schema, temperature, max_tokens, on_token, images, audio)
                )
                if not urgent:
                    self._yielding.add(reply)
                try:
                    text, count = await reply
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():  # the caller itself was stopped
                        raise
                    continue  # it made way; the slot is given up and asked for again
                finally:
                    self._yielding.discard(reply)
            break
        self.calls += 1
        self.tokens += count
        return schema.model_validate_json(text), count

    @property
    def senses(self) -> bool:
        """Whether the thinking model takes pictures and sound itself."""
        return self.models.engine == "llama"

    async def _reply(
        self, model: str, system: str, prompt: str, schema: type[BaseModel],
        temperature: float, max_tokens: int, on_token: Callable[[str], None] | None,
        images: list[bytes] | None = None, audio: list[bytes] | None = None,
    ) -> tuple[str, int]:
        if self.senses:
            return await self._stream_llama(system, prompt, schema, temperature, max_tokens, on_token, images, audio)
        try:
            return await self._stream(model, system, prompt, schema, temperature, max_tokens, on_token)
        except ollama.ResponseError as exc:
            if "think" not in str(exc).lower() or model in self._no_think:
                raise
            self._no_think.add(model)  # this model has no thinking switch; ask it plainly
            return await self._stream(model, system, prompt, schema, temperature, max_tokens, on_token)

    async def placement(self) -> float | None:
        """Ask the server how much of the thinking model sits on the GPU. Off it, thought crawls."""
        try:
            loaded = (await self.client.ps()).models
        except Exception:  # an older server, or none: nothing to report
            return self.gpu_share
        for entry in loaded:
            if entry.model in (self.models.main, f"{self.models.main}:latest") and entry.size:
                self.gpu_share = (entry.size_vram or 0) / entry.size
        return self.gpu_share

    async def murmur(self, system: str, prompt: str, schema: type[T], max_tokens: int = 140) -> T:
        """One short reply from the small model that runs on the processor beside the thinker: a
        slot of its own, so it never waits for the thinker, nor the thinker for it."""
        async with self._inner:
            reply = await self.client.chat(
                model=self.models.inner,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                format=schema.model_json_schema(),
                options={"temperature": 0.9, "num_predict": max_tokens, "num_gpu": 0, "num_ctx": 2048},
                keep_alive=self.models.keep_alive, think=False,
            )
        return schema.model_validate_json(reply.message.content)

    async def _stream(
        self, model: str, system: str, prompt: str, schema: type[BaseModel],
        temperature: float, max_tokens: int, on_token: Callable[[str], None] | None,
    ) -> tuple[str, int]:
        kwargs = {}
        if self.models.think is not None and model not in self._no_think:
            kwargs["think"] = self.models.think
        stream = await self.client.chat(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            stream=True,
            format=schema.model_json_schema(),
            options={"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.models.num_ctx},
            keep_alive=self.models.keep_alive,
            **kwargs,
        )
        pieces: list[str] = []
        count = 0
        async for chunk in stream:
            piece = chunk.message.content
            if piece:
                pieces.append(piece)
                if on_token:
                    on_token(piece)
            if chunk.done:
                count = chunk.eval_count or len(pieces)
                if chunk.eval_duration:
                    rate = count / (chunk.eval_duration / 1e9)
                    self.speed = rate if not self.speed else 0.8 * self.speed + 0.2 * rate
                if getattr(chunk, "done_reason", None) == "length":
                    raise Truncated(count, len("".join(pieces)), max_tokens)
        return "".join(pieces), count

    async def _stream_llama(
        self, system: str, prompt: str, schema: type[BaseModel], temperature: float, max_tokens: int,
        on_token: Callable[[str], None] | None, images: list[bytes] | None, audio: list[bytes] | None,
    ) -> tuple[str, int]:
        """The same reply from llama.cpp's server, which holds one model that sees and hears: the
        pictures and the sound go into the message beside the words."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for picture in images or []:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(picture).decode()}})
        for sound in audio or []:
            content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(sound).decode(), "format": "wav"}})
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "stream": True, "temperature": temperature,
            # Its thinking, when it thinks first, is written before the reply and has room of its own.
            "max_tokens": max_tokens + (THINKING_ROOM if self.models.think else 0),
            "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
            "chat_template_kwargs": {"enable_thinking": bool(self.models.think)},
            "timings_per_token": False,
        }
        pieces: list[str] = []
        count, ending = 0, ""
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            async with client.stream("POST", f"{self.models.llama}/v1/chat/completions", json=body) as response:
                if response.status_code != 200:
                    raise OSError(f"llama.cpp's server answered {response.status_code}: {(await response.aread())[:300]!r}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    for choice in chunk.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            pieces.append(piece)
                            if on_token:
                                on_token(piece)
                        ending = choice.get("finish_reason") or ending
                    timings = chunk.get("timings")
                    if timings and timings.get("predicted_n"):
                        count = int(timings["predicted_n"])
                        rate = float(timings.get("predicted_per_second") or 0)
                        if rate:
                            self.speed = rate if not self.speed else 0.8 * self.speed + 0.2 * rate
        count = count or len(pieces)
        if ending == "length":
            raise Truncated(count, len("".join(pieces)), max_tokens)
        return "".join(pieces), count

    async def embed(self, text: str) -> np.ndarray:
        """Unit-length embedding of the text."""
        vec = None
        if self.embed_error is None:
            try:
                reply = await self.client.embed(
                    model=self.models.embed, input=text[:4000], keep_alive=self.models.keep_alive,
                    options={"num_gpu": 0} if self.models.embed_on_cpu else None,
                )
                vec = np.asarray(reply.embeddings[0], dtype=np.float32)
            except Exception as exc:  # no embedding model, or no server: degrade, do not stop thinking
                self.embed_error = str(exc) or type(exc).__name__
        if vec is None:
            vec = lexical_vector(text)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec
