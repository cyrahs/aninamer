from __future__ import annotations


class AninamerError(Exception):
    pass


class LLMOutputError(AninamerError):
    pass


class OpenAIError(AninamerError):
    pass
