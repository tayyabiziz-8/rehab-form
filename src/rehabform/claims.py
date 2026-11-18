"""Claim checker: is every sentence of the feedback backed by a finding?

Rules (a sentence fails on the first that applies; all reasons are listed)
  * cites a finding id that does not exist                    -> unknown_finding
  * contains a number, body part or side but cites nothing    -> untagged_claim
  * a number that is not one of the cited findings' numbers   -> unsupported_number
    (numbers of non-quotable findings are never allowed)
  * a body part the cited findings are not about              -> unsupported_body_part
  * 'left' / 'right' when the cited findings have no such side -> unsupported_side
  * corrective wording while every cited finding passed       -> contradicts_finding
  * medical claims (pain, injury, diagnosis, ...)             -> out_of_scope

Sentences with no number, body part or side and no citation (e.g. "Keep it
up!") are allowed as 'generic'.

``error_categories`` is a separate, approximate keyword classifier used only
to compare grounded feedback with an ungrounded baseline that has no citations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .findings import Finding

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                     "ninety": 90, "half": 0.5})

# body-part words -> which finding body parts may back them
BODY_TERMS = {
    "arm": {"BothArms", "LeftArm", "RightArm"}, "arms": {"BothArms", "LeftArm", "RightArm"},
    "elbow": {"BothArms", "LeftArm", "RightArm"}, "elbows": {"BothArms", "LeftArm", "RightArm"},
    "hand": {"BothArms", "LeftArm", "RightArm"}, "hands": {"BothArms", "LeftArm", "RightArm"},
    "wrist": {"BothArms", "LeftArm", "RightArm"}, "wrists": {"BothArms", "LeftArm", "RightArm"},
    "shoulder": {"BothArms", "LeftArm", "RightArm", "Spine"}, "shoulders": {"BothArms", "LeftArm", "RightArm", "Spine"},
    "trunk": {"Spine"}, "torso": {"Spine"}, "spine": {"Spine"}, "back": {"Spine"}, "waist": {"Spine"},
    "hip": {"Spine"}, "hips": {"Spine"}, "upper body": {"Spine"}, "body": {"Spine"},
    "head": set(), "neck": set(), "knee": set(), "knees": set(), "leg": set(), "legs": set(),
    "foot": set(), "feet": set(), "ankle": set(), "ankles": set(),
}
SIDE_WORDS = ("left", "right")
OUT_OF_SCOPE = ("pain", "hurt", "injur", "diagnos", "damage", "tear", "surgery", "medicat", "disease",
                "arthritis", "inflamm")
CORRECTIVE = ("try to", "should", "need to", "not enough", "too little", "too much", "further", "higher",
              "more", "increase", "make sure", "remember to")


@dataclass
class SentenceCheck:
    text: str
    cites: list[str]
    ok: bool
    kind: str                     # "grounded" | "generic" | "rejected"
    reasons: list[str] = field(default_factory=list)


@dataclass
class CheckReport:
    sentences: list[SentenceCheck]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.sentences)

    @property
    def n_claims(self) -> int:
        return sum(s.kind != "generic" for s in self.sentences)

    @property
    def n_unsupported(self) -> int:
        return sum(not s.ok for s in self.sentences)

    def reasons(self) -> list[str]:
        return [r for s in self.sentences for r in s.reasons]


_TAG = re.compile(r"\[\s*(F\d+(?:\s*,\s*F\d+)*)\s*\]")


def split_sentences(text: str) -> list[str]:
    # keep a trailing [F1] tag with the sentence it follows ("... higher. [F1]" or "... higher [F1].")
    text = re.sub(r"([.!?])\s*(\[\s*F\d+(?:\s*,\s*F\d+)*\s*\])", r" \2\1", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _numbers(text: str) -> list[float]:
    t = _TAG.sub(" ", text)
    t = re.sub(r"(\d)\s*[–-]\s*(\d)", r"\1 \2", t)          # 12–14 s -> two numbers
    nums = [float(x) for x in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", t)]
    for w in re.findall(r"[a-z]+", t.lower()):
        if w in NUMBER_WORDS and w not in ("one",):             # "one" is too often a pronoun
            nums.append(float(NUMBER_WORDS[w]))
    return nums


def _body_terms(text: str) -> list[str]:
    t = " " + _TAG.sub(" ", text.lower()) + " "
    found = []
    if " upper body" in t:
        found.append("upper body")
        t = t.replace(" upper body", " ")
    for term in BODY_TERMS:
        if term != "upper body" and re.search(rf"\b{term}\b", t):
            found.append(term)
    return found


def _matches(x: float, allowed: list[float]) -> bool:
    return any(abs(x - a) <= (0.5 if float(a).is_integer() else 0.05) for a in allowed)


def check(text: str, findings: list[Finding]) -> CheckReport:
    by_id = {f.id: f for f in findings}
    out = []
    for sent in split_sentences(text):
        cites = [c.strip() for m in _TAG.findall(sent) for c in m.split(",")]
        body = _TAG.sub(" ", sent)
        low = body.lower()
        nums, terms = _numbers(body), _body_terms(body)
        sides = [w for w in SIDE_WORDS if re.search(rf"\b{w}\b", low)]
        reasons = []
        if any(w in low for w in OUT_OF_SCOPE):
            reasons.append("out_of_scope")
        unknown = [c for c in cites if c not in by_id]
        if unknown:
            reasons.append(f"unknown_finding:{','.join(unknown)}")
        cited = [by_id[c] for c in cites if c in by_id]
        if not cites:
            if nums or terms or sides:
                reasons.append("untagged_claim")
            out.append(SentenceCheck(sent, cites, not reasons, "generic" if not reasons else "rejected", reasons))
            continue
        allowed = [a for f in cited for a in f.allowed_numbers()]
        for x in nums:
            if not _matches(x, allowed):
                reasons.append(f"unsupported_number:{x:g}")
        parts = {f.body_part for f in cited}
        for term in terms:
            if not (BODY_TERMS[term] & parts):
                reasons.append(f"unsupported_body_part:{term}")
        cited_sides = {f.side for f in cited if f.side} | {
            {"LeftArm": "left", "RightArm": "right"}[f.body_part] for f in cited if f.body_part in ("LeftArm", "RightArm")}
        for s in sides:
            if s not in cited_sides:
                reasons.append(f"unsupported_side:{s}")
        if cited and all(f.status == "ok" for f in cited) and any(c in low for c in CORRECTIVE):
            reasons.append("contradicts_finding")
        out.append(SentenceCheck(sent, cites, not reasons, "grounded" if not reasons else "rejected", reasons))
    return CheckReport(out)


# ---------------------------------------------------------------------------
# approximate error categories (for the ungrounded-baseline comparison only)
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS = {
    "arms_height": r"\b(raise|lift|higher|height|up)\b.*\b(arm|arms|hand|hands|wrist|wrists|elbows?)\b|"
                   r"\b(arm|arms|hand|hands|wrists?|elbows?)\b.*\b(higher|up|raise[ds]?|lift(ed)?|height)\b",
    "trunk_tilt": r"\b(tilt|lean|bend|side[- ]?bend|stretch)\w*\b.*\b(side|sideways|trunk|torso|body|waist|left|right|over)\b|"
                  r"\b(trunk|torso|upper body|body)\b.*\b(tilt|lean|bend)\w*",
    "trunk_rotation": r"\b(rotat|twist|turn)\w*",
    "arms_open": r"\b(open|spread|wider|apart|outspread)\b",
}
DETECTOR_CATEGORY = {
    "CTK_arms_not_raised": "arms_height", "RTK_arms_not_raised": "arms_height",
    "ELK_body_not_tilted": "trunk_tilt", "RTK_torso_not_rotated": "trunk_rotation", "CTK_error3": "arms_open",
}


def error_categories(text: str) -> set[str]:
    """Categories of error the text asks the person to correct (corrective sentences only)."""
    cats = set()
    for sent in split_sentences(text):
        low = _TAG.sub(" ", sent).lower()
        if not any(c in low for c in CORRECTIVE + ("raise", "lift", "bend", "turn", "keep")):
            continue
        if re.search(r"\b(good|great|well done|nice|excellent)\b", low) and not re.search(r"\b(but|however)\b", low):
            continue
        for cat, pat in CATEGORY_PATTERNS.items():
            if re.search(pat, low):
                cats.add(cat)
    return cats
