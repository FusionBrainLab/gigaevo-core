"""
Telegram notification and async approval for experiment gates.

Usage:
    from tools.telegram_notify import notify, wait_for_approval

    # Gate 1: send + wait
    notify("Gate 1: experiment hover/my-exp design ready. Reply 'approved' or provide feedback.")
    reply = wait_for_approval(timeout_hours=24, keywords=["approved"])
    if reply.approved:
        proceed()
    else:
        handle_feedback(reply.text)

Environment variables (set in .env or shell):
    TELEGRAM_BOT_TOKEN   — Bot API token (from @BotFather)
    TELEGRAM_CHAT_ID     — Your personal chat ID or group ID

Setup: see docs/setup/telegram-bot.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import time

from dotenv import load_dotenv
import requests

load_dotenv()  # auto-load TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env

# Use HTTPS_PROXY from environment (set in ~/.claude/settings.json) so Telegram
# is reachable from servers where api.telegram.org is blocked.
_PROXIES: dict | None = None
if os.environ.get("HTTPS_PROXY"):
    _PROXIES = {
        "https": os.environ["HTTPS_PROXY"],
        "http": os.environ.get("HTTP_PROXY", os.environ["HTTPS_PROXY"]),
    }


def _bot_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. See docs/setup/telegram-bot.md for setup."
        )
    return token


def _chat_id() -> str:
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID not set. See docs/setup/telegram-bot.md for setup."
        )
    return chat_id


def notify(message: str, *, parse_mode: str = "Markdown") -> bool:
    """Send a Telegram message. Returns True on success."""
    token = _bot_token()
    chat_id = _chat_id()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": parse_mode}
    try:
        resp = requests.post(url, json=payload, timeout=10, proxies=_PROXIES)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[telegram_notify] WARNING: failed to send message: {e}")
        return False


def send_photo(
    image_path: str, caption: str = "", *, parse_mode: str = "Markdown"
) -> bool:
    """Send a PNG/JPG image to Telegram with an optional caption. Returns True on success."""
    from pathlib import Path

    token = _bot_token()
    chat_id = _chat_id()
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "caption": caption[:1024],
                    "parse_mode": parse_mode,
                },
                files={"photo": (Path(image_path).name, f, "image/png")},
                timeout=30,
                proxies=_PROXIES,
            )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[telegram_notify] WARNING: failed to send photo: {e}")
        return False


@dataclass
class ApprovalResult:
    approved: bool
    text: str
    timed_out: bool = False


def wait_for_approval(
    timeout_hours: float = 24.0,
    poll_interval_seconds: int = 30,
    keywords: list[str] | None = None,
    reject_keywords: list[str] | None = None,
) -> ApprovalResult:
    """
    Poll Telegram for a reply and return when one of the keywords is found.

    - If a keyword from `keywords` appears → ApprovalResult(approved=True)
    - If a keyword from `reject_keywords` appears → ApprovalResult(approved=False)
    - If timeout_hours elapses → ApprovalResult(timed_out=True, approved=False)

    Default approval keywords: ["approved", "approve", "yes", "ok", "go", "lgtm"]
    Default reject keywords: ["reject", "no", "stop", "cancel", "block"]
    """
    if keywords is None:
        keywords = ["approved", "approve", "yes", "ok", "go", "lgtm"]
    if reject_keywords is None:
        reject_keywords = ["reject", "no", "stop", "cancel", "block"]

    token = _bot_token()
    chat_id = _chat_id()
    deadline = time.time() + timeout_hours * 3600
    offset = None  # tracks last processed update_id

    # Get current update offset baseline (skip messages before this call)
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = requests.get(url, params={"timeout": 0}, timeout=10)
        updates = resp.json().get("result", [])
        if updates:
            offset = updates[-1]["update_id"] + 1
    except requests.RequestException:
        pass

    while time.time() < deadline:
        try:
            params: dict = {
                "timeout": poll_interval_seconds,
                "allowed_updates": ["message"],
            }
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(url, params=params, timeout=poll_interval_seconds + 5)
            updates = resp.json().get("result", [])
        except requests.RequestException as exc:
            print(f"[telegram_notify] poll error: {exc}")
            time.sleep(poll_interval_seconds)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            # Only process messages from the configured chat
            if str(msg.get("chat", {}).get("id", "")) != str(chat_id):
                continue
            text = (msg.get("text") or "").strip().lower()
            if any(kw in text for kw in keywords):
                return ApprovalResult(approved=True, text=text)
            if any(kw in text for kw in reject_keywords):
                return ApprovalResult(approved=False, text=text)

    return ApprovalResult(approved=False, text="", timed_out=True)


# ---------------------------------------------------------------------------
# Gate helpers used by run-experiment skill
# ---------------------------------------------------------------------------


def gate_design_approval(exp_name: str) -> ApprovalResult:
    """Gate 1: design review."""
    msg = (
        f"🔬 *Gate 1 — Design Approval*\n\n"
        f"Experiment: `{exp_name}`\n\n"
        f"Elena + Volkov have completed the design review.\n"
        f"Review `experiments/{exp_name}/01_design.md` and `02_review.md`.\n\n"
        f"Reply *approved* to proceed to implementation, or provide feedback."
    )
    notify(msg)
    return wait_for_approval(timeout_hours=48)


def gate_launch_confirmation(exp_name: str, cfg_summary: str) -> ApprovalResult:
    """Gate 2: launch confirmation."""
    msg = (
        f"🚀 *Gate 2 — Launch Confirmation*\n\n"
        f"Experiment: `{exp_name}`\n\n"
        f"Config dumps captured. Summary:\n```\n{cfg_summary[:600]}\n```\n\n"
        f"Check `cfg_run_*.txt` files for full config.\n"
        f"Reply *approved* to launch, or describe config issues."
    )
    notify(msg)
    return wait_for_approval(timeout_hours=24)


def gate_results_signoff(
    exp_name: str, verdict: str, effect_summary: str
) -> ApprovalResult:
    """Gate 3: results sign-off."""
    msg = (
        f"📊 *Gate 3 — Results Sign-off*\n\n"
        f"Experiment: `{exp_name}`\n"
        f"Verdict: *{verdict}*\n"
        f"Effect: {effect_summary}\n\n"
        f"Review `experiments/{exp_name}/05_results.md`.\n"
        f"Reply *approved* to merge the PR, or provide feedback."
    )
    notify(msg)
    return wait_for_approval(timeout_hours=72)


def post_fitness_update(
    exp_name: str, gen: int, fitness: float, label: str = ""
) -> None:
    """Hourly fitness update from watchdog."""
    ts = datetime.now(datetime.UTC).strftime("%H:%M UTC")
    run_label = f" ({label})" if label else ""
    msg = f"📈 *{exp_name}*{run_label}\nGen {gen} | Fitness {fitness:.1%} | {ts}"
    notify(msg)


def post_anomaly_alert(exp_name: str, issue: str) -> None:
    """Alert when anomaly detector finds something."""
    msg = (
        f"⚠️ *Anomaly: {exp_name}*\n\n"
        f"{issue}\n\n"
        f"Check `experiments/{exp_name}/04_issues_log.md`."
    )
    notify(msg)


if __name__ == "__main__":
    # Quick test: send a test message
    import sys

    test_msg = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "GigaEvo Telegram notify: test message"
    )
    ok = notify(test_msg)
    print("Sent." if ok else "FAILED.")
