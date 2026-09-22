"""Telegram notifications for the training supervisor/agent.

Best-effort by design: a failed notification must never crash the caller
(the training supervisor, or the chat agent) or block it waiting on a flaky
network. Every failure is caught and logged, never raised.
"""
from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_MAX_MESSAGE_LEN = 4096  # Telegram's sendMessage hard limit


def send_telegram_message(text: str) -> bool:
    """Sends `text` to TELEGRAM_CHAT_ID via TELEGRAM_BOT_TOKEN. Returns whether it
    was actually sent -- callers should treat a False return as "logged and
    move on", not as something to retry or raise over.

    Plain text (no parse_mode): log/traceback snippets often contain
    Markdown/MarkdownV2 special characters, and MarkdownV2 in particular
    requires escaping a long list of reserved characters that will appear in
    stack traces -- sending as plain text sidesteps that entirely rather
    than trying to escape it correctly.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("notify: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set, skipping message")
        return False

    if len(text) > _MAX_MESSAGE_LEN:
        text = text[: _MAX_MESSAGE_LEN - 20] + "\n...[truncated]"

    url = _API_BASE.format(token=token, method="sendMessage")
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        if resp.status_code == 429:
            # Telegram's own documented rate-limit response includes
            # retry_after in the body -- log it rather than blind-retrying,
            # since a notification is never worth blocking the caller over.
            retry_after = resp.json().get("parameters", {}).get("retry_after")
            print(f"notify: rate-limited by Telegram, retry_after={retry_after}s, dropping this message")
            return False
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"notify: failed to send Telegram message: {e}")
        return False


def get_telegram_updates(offset: int | None, timeout: int = 30) -> list[dict]:
    """Long-polls Telegram's getUpdates. `offset` should be the highest
    update_id seen so far + 1 (Telegram's own pagination convention -- passing
    it back acks every earlier update so they aren't redelivered). Returns an
    empty list (not an exception) on any network/timeout error, since the
    caller's poll loop should just try again next iteration.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return []
    url = _API_BASE.format(token=token, method="getUpdates")
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(url, params=params, timeout=timeout + 10)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except requests.RequestException as e:
        print(f"notify: getUpdates failed: {e}")
        return []
