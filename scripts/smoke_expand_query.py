import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _run_success_case() -> None:
    original_create = app.llm_client.chat.completions.create

    try:
        app.llm_client.chat.completions.create = lambda **kwargs: _FakeResponse("  career fairs job workshops  ")
        result = app.expand_query("I'm looking for career related stuff")
        assert result == "career fairs job workshops", f"Unexpected expanded query: {result!r}"
    finally:
        app.llm_client.chat.completions.create = original_create


def _run_fallback_case() -> None:
    original_create = app.llm_client.chat.completions.create

    def _raise_error(**kwargs):
        raise RuntimeError("mock llm failure")

    try:
        app.llm_client.chat.completions.create = _raise_error
        user_query = "any free food events on campus?"
        result = app.expand_query(user_query)
        assert result == user_query, f"Fallback failed; expected original query, got: {result!r}"
    finally:
        app.llm_client.chat.completions.create = original_create


def main() -> None:
    _run_success_case()
    _run_fallback_case()
    print("SMOKE_OK expand_query")


if __name__ == "__main__":
    main()
