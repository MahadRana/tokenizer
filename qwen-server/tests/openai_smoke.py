"""Call a running server with the official OpenAI Python SDK.

argv: base_url (already includes /v1) and max_tokens.
Prints one OK line. A non-zero exit is a failed assertion.
"""

import json
import sys
import urllib.request

from openai import OpenAI


def main() -> None:
    base = sys.argv[1].rstrip("/")
    max_tokens = int(sys.argv[2])
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    messages = [{"role": "user", "content": "Say hi"}]
    client = OpenAI(base_url=base, api_key="not-needed", max_retries=0, timeout=120.0)

    parts: list[str] = []
    objects: list[str] = []
    finish = None
    for chunk in client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0,
        max_tokens=max_tokens,
    ):
        assert chunk.object == "chat.completion.chunk", chunk.object
        objects.append(chunk.object)
        assert chunk.choices, "chunk has no choices"
        delta = chunk.choices[0].delta
        if delta is not None and delta.content:
            parts.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    assert objects, "SDK stream produced no chunks"
    assert finish in ("stop", "length"), finish

    # include_usage: ordinary chunks have usage is None (JSON null). The last
    # yielded chunk has the counts and an empty choices list. [DONE] is not
    # yielded. Docs: platform.openai.com/docs/api-reference/chat/create
    # stream_options.include_usage, read 2026-11-07.
    stream_usage = None
    stream_finish = None
    ordinary = 0
    for chunk in client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
    ):
        assert chunk.object == "chat.completion.chunk"
        if chunk.usage is None:
            assert chunk.choices, "a null-usage chunk should carry choices"
            ordinary += 1
            if chunk.choices[0].finish_reason:
                stream_finish = chunk.choices[0].finish_reason
        else:
            assert list(chunk.choices) == []
            stream_usage = chunk.usage
    assert ordinary >= 1
    assert stream_usage is not None, "SDK did not surface the final usage chunk"
    assert (
        stream_usage.total_tokens
        == stream_usage.prompt_tokens + stream_usage.completion_tokens
    )
    if stream_finish == "length":
        assert stream_usage.completion_tokens == max_tokens
    else:
        assert 1 <= stream_usage.completion_tokens <= max_tokens

    # The SDK consumes `data: [DONE]` internally and does not yield it.
    # Read the same request once as raw SSE and require that trailer.
    raw_body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    origin = base[: -len("/v1")] if base.endswith("/v1") else base
    request = urllib.request.Request(
        origin + "/v1/chat/completions",
        data=json.dumps(raw_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read().decode()
        content_type = response.headers.get("Content-Type", "")
    assert "text/event-stream" in content_type, content_type
    assert "chat.completion.chunk" in raw
    assert "data: [DONE]" in raw
    tail = [line for line in raw.splitlines() if line.startswith("data:")]
    assert tail[-1] == "data: [DONE]", tail[-1:]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=0,
        max_tokens=max_tokens,
    )
    assert resp.object == "chat.completion", resp.object
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].finish_reason in ("stop", "length")
    usage = resp.usage
    assert usage is not None
    assert usage.prompt_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    if resp.choices[0].finish_reason == "length":
        assert usage.completion_tokens == max_tokens, usage.completion_tokens
    else:
        assert 1 <= usage.completion_tokens <= max_tokens

    listed = client.models.list()
    ids = [item.id for item in listed.data]
    assert model in ids, ids

    print(
        f"OK prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
        f"finish={resp.choices[0].finish_reason} chunks={len(objects)} "
        f"stream_finish={finish} text={''.join(parts)!r} "
        f"include_usage_prompt={stream_usage.prompt_tokens} "
        f"include_usage_completion={stream_usage.completion_tokens}"
    )


if __name__ == "__main__":
    main()
