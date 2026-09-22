"""Telegram front-end: free-text messages -> LLM tool-calling -> results.

Long-polls Telegram for messages from TELEGRAM_CHAT_ID only, feeds each one
through OpenRouter's OpenAI-compatible chat completions endpoint (so
AGENT_MODEL can be any model OpenRouter carries, not just Anthropic's --
that's the whole reason this uses OpenAI-format tool calling rather than
Anthropic's tool_runner: OpenRouter's Anthropic-compatible endpoint only
works for Anthropic models, its universal OpenAI-compatible one works for
all of them), and replies with the final text.

Tool schemas and the name->callable dispatch are built here from tools.py's
existing @beta_tool-decorated functions -- Anthropic's `input_schema` and
OpenAI's `parameters` are both plain JSON Schema, so this is a reshape of
already-generated schemas, not a rewrite; tools.py itself needed no changes
to support this.

Conversation history is a plain list of OpenAI-format chat messages,
persisted to disk so a bot restart doesn't lose context.

State-changing tools gate on their own `confirmed` parameter (see tools.py)
-- there is no separate pending-action state machine here. The next message
is just the next turn in the same conversation; the model decides from
context whether it's a confirmation, a cancellation, or something unrelated.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notify import get_telegram_updates, send_telegram_message  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / ".agent_conversation.json"
# OpenRouter's own model slug (e.g. "anthropic/claude-opus-5",
# "openai/gpt-5", ...) -- see https://openrouter.ai/models.
MODEL = os.environ.get("AGENT_MODEL", "anthropic/claude-opus-5")
MAX_TOOL_ITERATIONS = 10  # guards against a runaway tool-call loop

SYSTEM_PROMPT = (
    "You are an operations assistant for a Persian ASR post-correction LoRA "
    "fine-tuning pipeline, controlled entirely through the tools you're "
    "given -- you have no other way to affect the system. The user is "
    "technical and will use precise/technical phrasing; you don't need to "
    "over-explain basic ML terms back to them. Several tools take a "
    "`confirmed` parameter: call with confirmed=False (or omit it) the "
    "first time, describe exactly what you're about to do, and wait for "
    "the user's explicit confirmation in their next message before calling "
    "the same tool again with confirmed=True. Never assume confirmation "
    "that wasn't given. Keep replies concise -- this is a chat interface, "
    "not a report."
)

# Reshape tools.py's @beta_tool schemas into OpenAI's function-calling
# format -- both input_schema (Anthropic) and parameters (OpenAI) are plain
# JSON Schema, so no tool logic is duplicated or re-described here.
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
    }
    for t in ALL_TOOLS
]
TOOL_DISPATCH = {t.name: t.func for t in ALL_TOOLS}


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except json.JSONDecodeError:
        return []


def _save_history(messages: list[dict]) -> None:
    tmp = HISTORY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(messages, indent=2, ensure_ascii=False))
    os.replace(tmp, HISTORY_PATH)


def execute_tool_call(name: str, arguments_json: str) -> str:
    """Runs one tool call, never raising -- a bad tool (invalid-JSON
    arguments the model hallucinated, or a bug in the tool itself) becomes
    an error string in the tool result instead of crashing the whole
    conversation turn. Pure enough to unit-test directly (given a name +
    a JSON args string) without a live model call.
    """
    func = TOOL_DISPATCH.get(name)
    if func is None:
        return f"Error: unknown tool {name!r}"
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return f"Error: model produced invalid JSON arguments for {name}: {e}"
    try:
        return str(func(**args))
    except Exception as e:
        return f"Error running {name}: {e}"


def handle_message(client: OpenAI, messages: list[dict], user_text: str) -> tuple[list[dict], str]:
    """Runs one full tool-use turn for `user_text` -- the standard
    request/execute/respond loop for OpenAI-format function calling (there's
    no SDK-provided auto-loop helper for this the way Anthropic's beta tool
    runner has one, so this is hand-written, following OpenAI's documented
    pattern: call, append the assistant's message, execute any tool_calls,
    append one role="tool" message per call, repeat until no more tool_calls).
    """
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages = messages + [{"role": "user", "content": user_text}]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=OPENAI_TOOLS, tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return messages, message.content or "(done, no text response)"

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call.function.name, tool_call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return messages, "(stopped after too many tool-call iterations in one turn -- something may be looping)"


def _make_client() -> OpenAI:
    """Routed through OpenRouter's OpenAI-compatible chat completions
    endpoint using OPENROUTER_API_KEY -- see AGENT_MODEL for which model.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)


def poll_forever() -> None:
    client = _make_client()
    allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not allowed_chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID must be set in .env")

    messages = _load_history()
    offset = None
    print("telegram_agent: polling started")

    while True:
        updates = get_telegram_updates(offset, timeout=30)
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message") or {}
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text")
            if not text:
                continue
            if chat_id != str(allowed_chat_id):
                print(f"telegram_agent: ignoring message from unauthorized chat_id={chat_id}")
                continue

            print(f"telegram_agent: received: {text!r}")
            try:
                messages, reply = handle_message(client, messages, text)
            except Exception as e:
                reply = f"Error handling that message: {e}"
                print(f"telegram_agent: {reply}")
            _save_history(messages)
            send_telegram_message(reply)

        time.sleep(1)


if __name__ == "__main__":
    poll_forever()
