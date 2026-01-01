from __future__ import annotations


class AninamerError(Exception):
    pass


class PlanValidationError(AninamerError):
    pass


class LLMOutputError(AninamerError):
    pass


class OpenAIError(AninamerError):
    pass


class ApplyError(AninamerError):
    pass
