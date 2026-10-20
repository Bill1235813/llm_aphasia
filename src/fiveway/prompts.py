"""The elicitation prompts and the yes/no gate, shared with the earlier experiments.

The prompts and the yes/no parser are imported from ``interp/generate.py``
unchanged, so the five-way study asks every model exactly what the earlier
study asked:

    cinderella  Q1 "Do you know the story of Cinderella?" (the gate), then
                Q2 "Tell me as much of the story of Cinderella as you can."
    sandwich    "Tell me how you would make a peanut butter and jelly sandwich." (no gate)

Llama-2-chat gets the questions in its chat template; GPT-2 XL and Pythia get
bare "Question: ...\\nAnswer:" prompts. Qwen3.5 gets the same questions in its
own chat template, with thinking off (chat_prompt).
"""

from __future__ import annotations

import unicodedata

from interp.generate import LLAMA2_STORY_USER, LLAMA2_YESNO_USER, SANDWICH_USER, parse_yes_no  # noqa: F401
from interp.generate import task_prompts as earlier_task_prompts

STORY_TOKENS = 180   # the generation budget of every earlier search
YESNO_TOKENS = 12
MIN_STORY_CHARS = 20  # shorter answers are not scored, as before


def answer_length(text: str) -> int:
    """An answer's length in characters, not counting invisible format characters (Unicode category Cf).

    NOTE: some knockouts make Llama-2-13B-chat write a few
    letters and then hundreds of zero-width joiners (U+200D). Counted with
    len(), such an answer passes the 20-character rule, and the classifier
    calls one of them ("tOoOO" and 170 joiners) Broca with P = 0.70. No other
    answer in any search cache contains such characters, so for every other
    answer this length is len().
    """
    return sum(unicodedata.category(character) != "Cf" for character in text)


def long_enough(story: str) -> bool:
    """Whether an answer is long enough to be scored: at least MIN_STORY_CHARS characters (answer_length)."""
    return answer_length(story) >= MIN_STORY_CHARS


def can_think(tokenizer) -> bool:
    """Whether a chat template can let the model think before answering (Qwen3.5's can; Llama-2's cannot)."""
    return "enable_thinking" in (tokenizer.chat_template or "")


def chat_prompt(tokenizer, messages: list) -> str:
    """A chat model's prompt for these messages, ready for Engine.encode, with thinking off.

    NOTE: Qwen3.5 can write a "<think>...</think>" block
    before its answer. With enable_thinking=False its template closes an empty
    one ("<think>\\n\\n</think>\\n\\n"), so the answer, the part the classifier
    reads, starts at once. Llama-2's template has no such variable and renders
    exactly as before; it starts with a literal "<s>", which is removed here
    because Engine.encode puts the real beginning-of-sequence token in front.
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    return rendered.removeprefix("<s>")


def task_prompts(tokenizer, is_chat: bool, task: str) -> tuple:
    """(yes/no prompt or None, story prompt) of a task, as rendered for this model.

    The earlier study's prompts (interp/generate.py), unchanged, for every
    model whose template cannot think; Qwen3.5's template renders the same
    questions with thinking off.
    """
    if not (is_chat and can_think(tokenizer)):
        return earlier_task_prompts(tokenizer, is_chat, task)
    if task == "sandwich":
        return None, chat_prompt(tokenizer, [{"role": "user", "content": SANDWICH_USER}])
    return (chat_prompt(tokenizer, [{"role": "user", "content": LLAMA2_YESNO_USER}]),
            chat_prompt(tokenizer, [{"role": "user", "content": LLAMA2_STORY_USER}]))


def task_prompt_ids(engine, task: str) -> tuple:
    """(yes/no prompt ids or None, story prompt ids) of a task for this engine's model."""
    yesno_prompt, story_prompt = task_prompts(engine.tokenizer, engine.spec.is_chat, task)
    return (None if yesno_prompt is None else engine.encode(yesno_prompt)), engine.encode(story_prompt)


def cut_at_new_question(text: str) -> str:
    """A base model's answer up to the next question it writes itself.

    NOTE: GPT-2 XL and Pythia continue a "Question: ...
    Answer:" prompt by writing more question and answer turns of their own.
    Those turns are not part of their answer to our question, so the answer
    is cut before the first line that starts a new "Question:" (or "Q:").
    """
    for marker in ("\nQuestion:", "\nQ:", "\nquestion:"):
        position = text.find(marker)
        if position != -1:
            text = text[:position]
    return text.strip()
