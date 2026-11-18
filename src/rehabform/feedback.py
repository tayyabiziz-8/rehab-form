"""Feedback writers.

  template_feedback(findings)        deterministic text built from findings (the fallback)
  grounded_prompt(findings, ex)      prompt for a language model that may only reword findings
  baseline_prompt(bp, ex)            prompt for the comparison: the model sees only the pose sequence
  LLMClient                          any OpenAI-compatible chat endpoint (Ollama, LM Studio, Groq,
                                     OpenRouter, OpenAI, ...), standard library only, with a cache
  write_feedback(...)                LLM text -> claim check -> template fallback

Every grounded sentence ends with the id of the finding it uses, e.g. "[F1]".
The app strips the tags for the patient and uses them to link each sentence to
its measurement and time window on the clinician's replay.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .claims import CheckReport, check
from .findings import Finding

EXERCISES = {
    "CTK": "hiding face: raise both arms in front of the face, then lower them",
    "ELK": "flank stretch: raise one arm over the head and bend the upper body to the opposite side",
    "RTK": "torso rotation: arms held up in front, turn the upper body to one side and back",
}


def _fmt(x: float | None, unit: str) -> str:
    if x is None:
        return ""
    s = f"{x:g}"
    return f"{s}°" if unit == "°" else f"{s} {unit}"


def _when(f: Finding) -> str:
    if f.start_s is None:
        return ""
    if f.end_s is None or f.end_s == f.start_s:
        return f" (around {f.start_s:g} s)"
    return f" (around {f.start_s:g}–{f.end_s:g} s)"


def template_feedback(findings: list[Finding], max_errors: int = 3) -> str:
    """Deterministic, always passes the claim checker."""
    errors = [f for f in findings if f.status == "error"][:max_errors]
    oks = [f for f in findings if f.status == "ok"]
    sents = []
    for f in errors:
        advice = f.advice[0].upper() + f.advice[1:]
        if f.quotable and f.value is not None and f.typical is not None:
            sents.append(f"{advice}: the {f.measure_label} reached {_fmt(f.value, f.unit)}{_when(f)}, "
                         f"while correct repetitions typically reach about {_fmt(f.typical, f.unit)}. [{f.id}]")
        else:
            sents.append(f"{advice}{_when(f)}. [{f.id}]")
    if oks:
        f = oks[0]
        sents.append(f"Good: {f.advice}. [{f.id}]")
    if not errors:
        sents.append("Keep it up!")
    return " ".join(sents)


def strip_tags(text: str) -> str:
    return re.sub(r"\s*\[\s*F\d+(?:\s*,\s*F\d+)*\s*\]", "", text).strip()


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
GROUNDED_SYSTEM = """You write short exercise feedback for a physiotherapy patient at home.
You are given FINDINGS measured from their video. Rules:
1. Use ONLY the findings. Do not add any other observation, body part, side (left/right), number or time.
2. Quote a number only if it appears in that finding's "numbers" list, unchanged (you may drop the unit "shoulder widths" but not change the value).
3. End every sentence that uses a finding with its id in square brackets, e.g. "... higher. [F1]".
4. Write 2 to 4 sentences: the most important correction first, then one thing that went well if there is a finding with status "ok".
5. Plain, warm language for a patient. No medical advice, no diagnosis, no mention of pain or injury.
Output only the feedback text."""

BASELINE_SYSTEM = """You write short exercise feedback for a physiotherapy patient at home.
You are given the exercise and the patient's body-landmark positions over time (image pixels, y grows downwards).
Write 2 to 4 sentences: the most important correction first, then one thing that went well.
Plain, warm language for a patient. No medical advice, no diagnosis. Output only the feedback text."""


def grounded_prompt(findings: list[Finding], exercise: str) -> list[dict]:
    items = []
    for f in findings:
        item = {"id": f.id, "status": f.status, "checked": f.issue, "advice": f.advice,
                "body_part": {"Spine": "upper body", "BothArms": "both arms", "LeftArm": "left arm",
                              "RightArm": "right arm"}.get(f.body_part, f.body_part),
                "time_window_s": [f.start_s, f.end_s]}
        if f.side:
            item["side"] = f.side
        if f.status == "error" and f.severity:
            item["severity"] = f.severity
        if f.quotable and f.value is not None:
            item["measure"] = f.measure_label
            item["measured"] = _fmt(f.value, f.unit)
            item["typical_when_correct"] = _fmt(f.typical, f.unit)
        item["numbers"] = [x for x in f.allowed_numbers()]
        items.append(item)
    user = (f"Exercise: {EXERCISES.get(exercise, exercise)}\n"
            f"FINDINGS:\n{json.dumps(items, ensure_ascii=False, indent=1)}")
    return [{"role": "system", "content": GROUNDED_SYSTEM}, {"role": "user", "content": user}]


UPPER = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
         "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24}


def baseline_prompt(bp: np.ndarray, exercise: str, fps: float = 10.0, every_s: float = 0.5) -> list[dict]:
    """The pose sequence alone (upper body, every 0.5 s, rounded pixels), no measurements."""
    step = max(1, int(round(fps * every_s)))
    rows = []
    for t in range(0, len(bp), step):
        pts = {k: [int(round(v)) if np.isfinite(v) else None for v in bp[t, j, :2]] for k, j in UPPER.items()}
        rows.append({"t": round(t / fps, 1), **pts})
    user = (f"Exercise: {EXERCISES.get(exercise, exercise)}\n"
            f"Landmarks (x, y pixels; person's left = l_):\n{json.dumps(rows, separators=(',', ':'))}")
    return [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------
class LLMClient:
    """e.g. Ollama: base_url http://localhost:11434/v1, model qwen2.5:7b, no key.
    Groq / OpenRouter / OpenAI: their base URL + model + key in an environment variable."""

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None,
                 cache_path: str | Path = "outputs/llm_cache.jsonl", temperature: float = 0.2, timeout: int = 120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.cache_path = Path(cache_path)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row["text"]

    def __call__(self, messages: list[dict]) -> str:
        k = hashlib.sha256(json.dumps([self.model, self.temperature, messages]).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        self.cache[k] = text
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "model": self.model, "text": text}, ensure_ascii=False) + "\n")
        return text


@dataclass
class Feedback:
    text: str                 # final text, tags included
    source: str               # "llm" | "template" (fallback) | "template_only"
    draft: str | None         # the LLM's draft, if any
    draft_report: CheckReport | None
    report: CheckReport


def write_feedback(findings: list[Finding], exercise: str, llm: LLMClient | None = None) -> Feedback:
    """LLM draft -> claim check -> template if the draft fails (or no LLM)."""
    if llm is None:
        text = template_feedback(findings)
        return Feedback(text, "template_only", None, None, check(text, findings))
    draft = llm(grounded_prompt(findings, exercise))
    rep = check(draft, findings)
    if rep.ok and rep.n_claims > 0:
        return Feedback(draft, "llm", draft, rep, rep)
    text = template_feedback(findings)
    return Feedback(text, "template", draft, rep, check(text, findings))
