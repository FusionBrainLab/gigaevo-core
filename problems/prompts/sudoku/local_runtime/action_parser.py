from __future__ import annotations

import re

from problems.prompts.sudoku.local_runtime.models import (
    Action,
    BacktrackAction,
    DoneAction,
    NodeAction,
)


class BasicActionParser:
    _RE_NODE_ID = re.compile(
        r"[<(\[]node[>)\]]\s*(\d+)\s+(.*?)(?:[<(\[]/node[>)\]]|\s*$)",
        re.S | re.I,
    )
    _RE_NODE_NO_ID = re.compile(
        r"[<(\[]node[>)\]]\s*(.*?)(?:[<(\[]/node[>)\]]|\s*$)",
        re.S | re.I,
    )
    _RE_BACK = re.compile(r"[<(\[]backtrack[>)\]]\s*(\d+)", re.I)
    _RE_DONE = re.compile(
        r"[<(\[]done[>)\]]\s*(.*?)(?:[<(\[]/done[>)\]]|\s*$)",
        re.S | re.I,
    )
    _RE_ANSWER = re.compile(r"answer\s*(?:is|:)\s*(.*?)(?:[.!]+\s*)?$", re.S | re.I)
    _RE_MALFORMED_TAG = re.compile(r"[<(\[](backtrack|done)", re.I)
    _RE_MALFORMED_CLOSING = re.compile(r"[<(\[]/[a-z]*[>)\]]?$", re.I)

    @staticmethod
    def _clean_content(content: str) -> str:
        content = content.strip()
        return BasicActionParser._RE_MALFORMED_CLOSING.sub("", content).strip()

    @staticmethod
    def _strip_trailing_punctuation(content: str) -> str:
        return content.rstrip(".!?").strip()

    @staticmethod
    def parse(text: str, node_id: int) -> Action:
        text = text.strip()

        if match := BasicActionParser._RE_BACK.search(text):
            return BacktrackAction(int(match.group(1)))

        if match := BasicActionParser._RE_DONE.search(text):
            content = BasicActionParser._clean_content(match.group(1))
            if content:
                return DoneAction(content)

        if match := BasicActionParser._RE_ANSWER.search(text):
            content = BasicActionParser._strip_trailing_punctuation(
                BasicActionParser._clean_content(match.group(1))
            )
            if content:
                return DoneAction(content)

        if BasicActionParser._RE_MALFORMED_TAG.search(text):
            return NodeAction(node_id, f'Malformed action attempt: "{text}"')

        if match := BasicActionParser._RE_NODE_ID.search(text):
            return NodeAction(int(match.group(1)), BasicActionParser._clean_content(match.group(2)))

        if match := BasicActionParser._RE_NODE_NO_ID.search(text):
            return NodeAction(node_id, BasicActionParser._clean_content(match.group(1)))

        return NodeAction(node_id, text)
