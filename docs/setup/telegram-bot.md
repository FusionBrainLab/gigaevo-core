# Telegram Bot Setup

Two environment variables needed. Set them in `~/.env` or your shell profile.

## Step 1 — Create a bot

1. Open Telegram, message `@BotFather`
2. Send `/newbot`, follow the prompts
3. Copy the token (looks like `123456789:ABCdef...`)

```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdef..."
```

## Step 2 — Get your chat ID

1. Message your new bot (any text)
2. Open: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find `"chat":{"id":123456789}` in the response — that's your chat ID

```bash
export TELEGRAM_CHAT_ID="123456789"
```

## Step 3 — Test

```bash
gigaevo notify "GigaEvo test message"
```

You should receive the message on Telegram.

## Step 4 — Persist

Add both vars to `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export TELEGRAM_BOT_TOKEN="..."' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="..."' >> ~/.bashrc
```

## How gates work

When a gate fires (design, launch, results), `tools/telegram_notify.py` sends a formatted message and polls for your reply. Reply with:
- `approved` — proceed
- `reject` / `no` — stop and await further input
- Anything else — sent back to the skill as feedback for revision

The polling is long-poll (30s timeout per cycle) and runs up to 48h for Gate 1, 24h for Gate 2, 72h for Gate 3. After timeout, the gate is skipped with a warning logged to `04_issues_log.md`.

## Fallback (no Telegram)

If `TELEGRAM_BOT_TOKEN` is not set, `notify()` prints to stdout and `wait_for_approval()` blocks on `input()` — preserving the original terminal-blocking behavior as a fallback.
