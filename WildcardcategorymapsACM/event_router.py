#!/usr/bin/env python3
# event_router.py
# Single-file, modular UMD Event Router: normalization, spell-fix, category routing, entities, day parsing.

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Protocol
from difflib import get_close_matches

# ===================== Constants & Shared Regex =====================

# Leet-speak character mapping
LEET_MAP = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s"
})

# Category fallback synonyms for broad queries
CATEGORY_SYNONYMS = {
    "sports": {"sport", "sports", "game", "games", "match", "matches", "athletics", "mbb", "wbb"},
    "careers": {"career", "careers", "job", "jobs", "internship", "internships", "resume", "handshake"},
    "academics": {"class", "classes", "lecture", "seminar", "colloquium", "thesis", "academic", "academics"},
    "arts_culture": {"art", "arts", "culture", "theater", "theatre", "concert", "clarice", "dance", "film", "screening"},
}

# Day of week variants (abbreviations + full names)
DAY_VARIANTS = {
    "sunday": {"sun", "sunday"},
    "monday": {"mon", "monday"},
    "tuesday": {"tue", "tues", "tuesday"},
    "wednesday": {"wed", "weds", "wednesday"},
    "thursday": {"thu", "thur", "thurs", "thursday"},
    "friday": {"fri", "friday"},
    "saturday": {"sat", "saturday"},
}

# Stopwords to filter out from entity extraction
STOPWORDS = {
    "show", "me", "any", "on", "for", "the", "a", "an", "to", "at",
    "in", "of", "watch", "see", "give", "all", "events"
}

# Shared regexes
NON_USEFUL_CHARS = re.compile(r"[^a-z0-9\s:/&-]")
MULTI_SPACE = re.compile(r"\s+")
BOOL_SPLIT = re.compile(r"\b(?:or|and)\b", flags=re.IGNORECASE)
DELIMS = re.compile(r"[|,\/]+")
QUOTED = re.compile(r'"([^"]+)"|\'([^\']+)\'')
QUOTED_STRIP = re.compile(r'["\']([^"\']+)["\']')

# ===================== Types =====================

class RetrieverFn(Protocol):
    def __call__(self, category: str, entities: List[str], day: Optional[str]) -> List[Dict]:
        ...

# ===================== Router (NLP layer) =====================

class EventRouter:
    """
    Language-understanding layer:
    - Normalization (leet-speak, punctuation, case)
    - Spell correction with category-aware vocabulary
    - Category classification (wildcards) + synonym fallback
    - Entity extraction (quoted phrases, boolean splits, stopword filtering)
    - Day detection (full names + abbreviations)
    """

    def __init__(self, category_map: Dict[str, List[str]]):
        self.category_map = category_map
        self._compile_patterns()
        self._build_vocabulary()

    def _compile_patterns(self):
        self.compiled_patterns = {
            cat: [self._wildcard_to_regex(p) for p in patterns]
            for cat, patterns in self.category_map.items()
        }

    def _build_vocabulary(self):
        self.category_vocab = {
            cat: sorted(
                {self._strip_wildcards(p) for p in patterns if self._strip_wildcards(p)},
                key=len, reverse=True
            )
            for cat, patterns in self.category_map.items()
        }
        words = {
            word
            for phrases in self.category_vocab.values()
            for phrase in phrases
            for word in phrase.split()
        }
        words |= set(self.category_map.keys())  # include category names
        self.global_vocab = sorted(words, key=len, reverse=True)

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = text.translate(LEET_MAP)
        text = NON_USEFUL_CHARS.sub(" ", text)
        text = MULTI_SPACE.sub(" ", text).strip()
        return text

    @staticmethod
    def _wildcard_to_regex(pattern: str):
        escaped = re.sub(r"([.^$*+?{}\[\]|()\\])", r"\\\1", pattern)
        regex_pattern = escaped.replace("%", ".*").replace("_", ".")
        return re.compile(regex_pattern, re.IGNORECASE)

    @staticmethod
    def _strip_wildcards(pattern: str) -> str:
        return MULTI_SPACE.sub(" ", pattern.replace("%", " ").replace("_", " ").strip())

    def correct_spelling(self, text: str, cutoff: float = 0.78) -> str:
        tokens = text.split()
        vocab_set = set(self.global_vocab)
        out = []
        for t in tokens:
            alpha = re.sub(r"[^a-z]", "", t)
            cand = t if t.isalpha() else (alpha if alpha else t)
            if cand in vocab_set or len(cand) < 3:
                out.append(cand or t)
                continue
            threshold = 0.72 if len(cand) <= 5 else cutoff
            m = get_close_matches(cand, self.global_vocab, n=1, cutoff=threshold)
            out.append(m[0] if m else cand)
        return " ".join(out)

    def classify_category(self, text: str) -> List[Tuple[str, int, List[str]]]:
        hits: List[Tuple[str, int, List[str]]] = []
        for category, regex_list in self.compiled_patterns.items():
            matched = [r.pattern for r in regex_list if r.search(text)]
            if matched:
                hits.append((category, len(matched), matched))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def _fallback_category(self, text: str) -> Optional[str]:
        tokens = set(text.split())
        for category, keywords in CATEGORY_SYNONYMS.items():
            if tokens & keywords:
                return category
        return None

    @staticmethod
    def extract_day_filter(text: str) -> Optional[str]:
        for day, variants in DAY_VARIANTS.items():
            for v in variants:
                if re.search(rf"\b{re.escape(v)}\b", text):
                    return day
        return None

    @staticmethod
    def parse_boolean_groups(text: str) -> List[str]:
        quoted = QUOTED.findall(text)
        quoted_phrases = ["".join(q).strip() for q in quoted if any(q)]
        text = QUOTED_STRIP.sub(" ", text)
        clean = DELIMS.sub(" ", text)
        parts = BOOL_SPLIT.split(clean)
        parts = [p.strip() for p in parts if p.strip()]
        parts = [" ".join([w for w in p.split() if w not in STOPWORDS]).strip() for p in parts]
        parts = [p for p in parts if p]
        return quoted_phrases + parts

    def align_to_vocabulary(self, candidates: List[str], vocab_phrases: List[str]) -> List[str]:
        aligned, seen = [], set()
        for cand in candidates:
            m = get_close_matches(cand, vocab_phrases, n=1, cutoff=0.7)
            if m:
                if m[0] not in seen:
                    seen.add(m[0]); aligned.append(m[0])
                continue
            vocab_words = {w for phrase in vocab_phrases for w in phrase.split()}
            kept = [w for w in cand.split() if w in vocab_words]
            if kept:
                phrase = " ".join(kept)
                if phrase not in seen:
                    seen.add(phrase); aligned.append(phrase)
        return aligned

    def parse_query(self, query: str) -> Dict:
        normalized = self._normalize(query)
        corrected = self.correct_spelling(normalized)
        day = self.extract_day_filter(corrected)
        category_scores = self.classify_category(corrected)
        top_category = category_scores[0][0] if category_scores else self._fallback_category(corrected)

        entities: List[str] = []
        if top_category:
            candidates = self.parse_boolean_groups(corrected)
            vocab = self.category_vocab.get(top_category, [])
            entities = self.align_to_vocabulary(candidates, vocab)

        return {
            "normalized": normalized,
            "corrected": corrected,
            "category": top_category,
            "day": day,
            "entities": entities,
            "category_scores": category_scores,
        }

# ===================== Default Retriever (in-memory) =====================

class EventRetriever:
    """
    Default in-memory retriever. Swap this out with your DB/search implementation.
    """

    def __init__(self, events: List[Dict]):
        self.events = events

    def retrieve(self, category: str, entities: List[str], day: Optional[str] = None) -> List[Dict]:
        results = [e for e in self.events if e.get("category") == category]

        if entities:
            wanted = {x.lower().strip() for x in entities}
            category_keywords = {
                "sports", "sport", "careers", "career",
                "academics", "academic", "arts", "culture",
                "arts_culture", "arts culture"
            }
            if not (wanted <= category_keywords):
                if category == "sports":
                    results = [
                        e for e in results
                        if any(t.lower() in wanted for t in e.get("tags", []))
                    ]
                else:
                    results = [
                        e for e in results
                        if any(any(k in t.lower() for k in wanted) for t in e.get("tags", []))
                        or any(k in e.get("title", "").lower() for k in wanted)
                        or any(k in e.get("venue", "").lower() for k in wanted)
                    ]

        if day:
            results = [e for e in results if e.get("start", "").lower().startswith(day)]

        return results

# ===================== Orchestrator =====================

class EventRouterSystem:
    """
    Orchestration layer combining router (intent parsing) + retriever (data fetch).
    Accepts either a custom retriever function or the default in-memory retriever.
    """

    def __init__(
        self,
        category_map: Dict[str, List[str]],
        events: Optional[List[Dict]] = None,
        retriever: Optional[RetrieverFn] = None
    ):
        self.router = EventRouter(category_map)
        if retriever:
            self._custom_retriever: Optional[RetrieverFn] = retriever
            self.retriever = None
        else:
            self.retriever = EventRetriever(events or [])
            self._custom_retriever = None

    def query(self, user_input: str) -> Dict:
        parsed = self.router.parse_query(user_input)

        if parsed["category"]:
            if self._custom_retriever:
                results = self._custom_retriever(parsed["category"], parsed["entities"], parsed["day"])
            else:
                results = self.retriever.retrieve(parsed["category"], parsed["entities"], parsed["day"])
        else:
            results = []

        parsed["results"] = results
        parsed["intent_tuple"] = (parsed["category"], tuple(parsed["entities"]), parsed["day"])
        return parsed

# ===================== Public API =====================

__all__ = [
    "EventRouter",
    "EventRetriever",
    "EventRouterSystem",
    "LEET_MAP",
    "CATEGORY_SYNONYMS",
    "DAY_VARIANTS",
    "STOPWORDS",
]
