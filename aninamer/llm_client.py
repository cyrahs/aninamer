from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


ChatRole = Literal["system", "user"]


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


class LLMClient(Protocol):
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        ...
