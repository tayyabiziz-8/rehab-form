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
import sys
import time
import urllib.error
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
    if unit == "°":
        return f"{s}°"
    if unit == "shoulder widths" and abs(x) == 1:
        unit = "shoulder width"
    return f"{s} {unit}"


def _when(f: Finding) -> str:
    if f.start_s is None:
        return ""
    if f.window_source == "worst_rep" and f.status == "error" and f.end_s is not None:
        return f" (most clearly in the repetition around {f.start_s:g}–{f.end_s:g} s)"
    if f.end_s is None or f.end_s == f.start_s:
        return f" (around {f.start_s:g} s)"
    return f" (around {f.start_s:g}–{f.end_s:g} s)"


def template_feedback(findings: list[Finding], max_errors: int = 3) -> str:
    """Deterministic, always passes the claim checker."""
    all_findings = findings
    findings = patient_findings(findings)
    errors = [f for f in findings if f.status == "error"][:max_errors]
    oks = [f for f in findings if f.status == "ok"]
    sents = []
    for f in errors:
        advice = f.advice[0].upper() + f.advice[1:]
        if f.quotable and f.value is not None and f.typical is not None and f.value < 0 \
                and f.measure_label.endswith("above your shoulders"):
            what = f.measure_label.replace("height of ", "").replace(" above your shoulders", "")
            sents.append(f"{advice}: {what} stayed {_fmt(abs(f.value), f.unit)} below your shoulders{_when(f)}, "
                         f"while in correct repetitions they reach about {_fmt(f.typical, f.unit)} above. [{f.id}]")
        elif f.quotable and f.value is not None and f.typical is not None:
            sents.append(f"{advice}: the {f.measure_label} reached {_fmt(f.value, f.unit)}{_when(f)}, "
                         f"while correct repetitions typically reach about {_fmt(f.typical, f.unit)}. [{f.id}]")
        else:
            sents.append(f"{advice}{_when(f)}. [{f.id}]")
    if oks:
        f = oks[0]
        sents.append(f"Good: {f.advice}. [{f.id}]")
    if not findings:
        return "No automatic form checks are validated for this exercise yet." if not all_findings \
            else "Keep it up!"
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


def baseline_prompt(bp: np.ndarray, exercise: str, fps: float = 10.0, every_s: float = 0.5,
                    max_chars: int = 9000) -> list[dict]:
    """The pose sequence alone (upper body, rounded pixels), no measurements.

    Compact table: one header, one row per sample. Sampled every 0.5 s, or
    more sparsely for long recordings so the prompt stays under ``max_chars``
    (about 3,000 tokens; free API tiers cap a single request at ~8,000)."""
    names = list(UPPER)
    T = len(bp)
    every = every_s
    while True:
        step = max(1, int(round(fps * every)))
        rows = []
        for t in range(0, T, step):
            vals = [round(t / fps, 1)]
            for j in UPPER.values():
                vals += [int(round(v)) if np.isfinite(v) else None for v in bp[t, j, :2]]
            rows.append(vals)
        table = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
        if len(table) <= max_chars or every >= 4:
            break
        every *= 1.5
    header = ["t_s"] + [f"{n}_{a}" for n in names for a in ("x", "y")]
    user = (f"Exercise: {EXERCISES.get(exercise, exercise)}\n"
            f"Landmarks every {step / fps:g} s (x, y pixels, y grows downwards; person's left = l_).\n"
            f"Columns: {json.dumps(header)}\n{table}")
    return [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------
class LLMClient:
    """e.g. Ollama: base_url http://localhost:11434/v1, model qwen2.5:7b, no key.
    Groq / OpenRouter / OpenAI: their base URL + model + key in an environment variable."""

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None,
                 cache_path: str | Path = "outputs/llm_cache.jsonl", temperature: float = 0.2, timeout: int = 120,
                 max_tokens: int = 2048, reasoning_effort: str | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.max_tokens = max_tokens   # also limits what a free tier counts as 'requested' per call
        self.reasoning_effort = reasoning_effort   # e.g. "low" for gpt-oss: fewer hidden tokens, fewer empty replies
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.cache_path = Path(cache_path)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row["text"]

    def _post(self, body: bytes) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "rehabform/0.1",
                   **({"Authorization": f"Bearer {self.key}"} if self.key else {})}
        for attempt in range(12):
            req = urllib.request.Request(self.url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code == 429:
                    wait = _retry_seconds(e.headers.get("retry-after"), detail)
                    per_day = "per day" in detail.lower() or "(tpd)" in detail.lower() or "(rpd)" in detail.lower()
                    # the daily allowance is a rolling window: short waits free up, long ones mean "come back later"
                    if wait > 900:
                        raise RateLimited(f"{self.model}: {'daily ' if per_day else ''}limit reached "
                                          f"(wait {wait / 60:.0f} min). "
                                          f"Finished calls are saved; run the same command again later. "
                                          f"Server said: {_short(detail)}") from None
                    print(f"  [rate limit] waiting {wait:.0f} s ({_short(detail)})", file=sys.stderr, flush=True)
                    time.sleep(wait + 1)
                    continue
                if e.code in (500, 502, 503) and attempt < 5:
                    time.sleep(10 * (attempt + 1))
                    continue
                if e.code == 413:
                    raise RuntimeError(f"prompt too large for this API tier: {_short(detail)}") from None
                raise RuntimeError(f"HTTP {e.code} from {self.url}: {_short(detail)}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:   # e.g. after the laptop slept
                if attempt < 5:
                    print(f"  [network] {e}; retrying", file=sys.stderr, flush=True)
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        raise RateLimited(f"{self.model}: still rate limited after 12 tries; run the same command again later")

    def __call__(self, messages: list[dict]) -> str:
        k = hashlib.sha256(json.dumps([self.model, self.temperature, messages]).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        body = json.dumps(payload).encode()
        text = self._post(body)["choices"][0]["message"]["content"] or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models
        if not text:
            raise RuntimeError("empty reply (reasoning may have used up max_tokens)")
        self.cache[k] = text
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "model": self.model, "text": text}, ensure_ascii=False) + "\n")
        return text

    def list_models(self) -> list[str]:
        req = urllib.request.Request(self.url.replace("/chat/completions", "/models"), headers={
            "User-Agent": "rehabform/0.1", **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return sorted(m["id"] for m in json.loads(r.read().decode("utf-8"))["data"])


def load_env_file(path: str | Path = ".env") -> None:
    """KEY=VALUE lines into os.environ (existing variables win). Keep .env out of git."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class RateLimited(RuntimeError):
    """The API refuses more calls for now (e.g. a daily token limit). Stop and resume later."""


def _short(detail: str) -> str:
    try:
        return json.loads(detail)["error"]["message"][:200]
    except Exception:
        return detail[:200]


def _retry_seconds(header: str | None, detail: str) -> float:
    """Groq sends 'retry-after' in seconds and 'Please try again in 1m23.5s' in the message."""
    try:
        if header:
            return float(header)
    except ValueError:
        pass
    m = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", detail)
    if m and any(m.groups()):
        h, mi, se = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + se
    return 20.0


@dataclass
class Feedback:
    text: str                 # final text, tags included
    source: str               # "llm" | "template" (fallback) | "template_only"
    draft: str | None         # the LLM's draft, if any
    draft_report: CheckReport | None
    report: CheckReport


NOT_SEEN_TEXT = ("We could not see you clearly enough to check this exercise. Please record again with your "
                 "whole upper body and hips in view, facing the camera.")


def patient_findings(findings: list[Finding]) -> list[Finding]:
    """Borderline findings go to the clinician view only; the patient hears about clear errors and passes."""
    return [f for f in findings if f.status != "borderline"]


def write_feedback(findings: list[Finding], exercise: str, llm: LLMClient | None = None) -> Feedback:
    """LLM draft -> claim check -> template if the draft fails (or no LLM)."""
    findings = patient_findings(findings)
    if llm is None or not findings:
        text = template_feedback(findings)
        return Feedback(text, "template_only", None, None, check(text, findings))
    draft = llm(grounded_prompt(findings, exercise))
    rep = check(draft, findings)
    if rep.ok and rep.n_claims > 0:
        return Feedback(draft, "llm", draft, rep, rep)
    text = template_feedback(findings)
    return Feedback(text, "template", draft, rep, check(text, findings))


def split_for_app(text: str, findings: list[Finding]) -> dict:
    """Feedback for an app: the plain text, and each sentence with the findings it cites
    (so the clinician view can link a sentence to its measurement and time window)."""
    from .claims import split_sentences
    sents = []
    for s in split_sentences(text):
        cites = [c.strip() for m in re.findall(r"\[\s*(F\d+(?:\s*,\s*F\d+)*)\s*\]", s) for c in m.split(",")]
        sents.append({"text": strip_tags(s), "cites": cites})
    return {"text": strip_tags(text), "sentences": sents}
