"""Instruction parser: extract target object from a natural language instruction.
Provides a function `extract_target_from_instruction` which tries to use OpenAI API
if OPENAI_API_KEY is set, otherwise falls back to a heuristic noun-extraction.
"""
import os
import re
from typing import Optional

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def heuristic_extract(instruction: str) -> Optional[str]:
    # Very small heuristic: find noun-like words after verbs like 'pick', 'find', 'grab', 'point'
    instruction = instruction.lower()
    # common verbs (allow matching verbs that may be followed by 'up')
    verbs = ["pick", "find", "grab", "point", "locate", "select", "place", "move", "push", "pull"]
    for v in verbs:
        # match 'pick the X' or 'pick up the X'
        m = re.search(rf"{v}(?:\s+up)?\s+the\s+([a-z0-9_-]+)", instruction)
        if m:
            return m.group(1)
    # fallback: first noun-ish word (simple)
    words = re.findall(r"[a-zA-Z]+", instruction)
    if not words:
        return None
    # remove stop words and common short prepositions
    stop = set(verbs + ["please", "the", "a", "an", "on", "in", "at", "to", "of", "up", "down", "over", "under"])
    for w in words:
        if w not in stop:
            return w
    return None

def extract_target_from_instruction(instruction: str) -> Optional[str]:
    """Return a single-word target token or None.

    If OPENAI_API_KEY is present, use the OpenAI API for more robust extraction.
    Otherwise use a lightweight heuristic.
    """
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            prompt = (
                "Extract the single target object (one word) from the following instruction. "
                "If ambiguous, choose the most likely manipulable object. Return only the word.\n\n"
                f"Instruction: \"{instruction.strip()}\"\n\nTarget:"
            )
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=8,
                temperature=0.0,
            )
            text = resp.choices[0].text.strip().split()[0]
            return re.sub(r"[^a-zA-Z0-9_-]", "", text).lower()
        except Exception:
            return heuristic_extract(instruction)
    else:
        return heuristic_extract(instruction)


if __name__ == "__main__":
    print(extract_target_from_instruction("Please pick up the red carrot on the table."))
