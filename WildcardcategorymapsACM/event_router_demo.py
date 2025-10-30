#!/usr/bin/env python3
# event_router.py
# Modular UMD Event Router: wildcard matching, typo-correction, category routing

import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Callable
from difflib import get_close_matches


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


class EventRouter:
    """
    Core event routing engine that handles:
    - Human-like text normalization (leet-speak, junk removal)
    - Smart spell correction with category awareness
    - Category classification with fallback synonyms
    - Entity extraction with quoted phrase support
    - Day filtering with abbreviations
    """
    
    def __init__(self, category_map: Dict[str, List[str]]):
        """
        Initialize the router with category definitions.
        
        Args:
            category_map: Dict mapping category names to lists of wildcard patterns
                         (e.g., {"sports": ["%football%", "%basketball%"]})
        """
        self.category_map = category_map
        self._compile_patterns()
        self._build_vocabulary()
    
    def _compile_patterns(self):
        """Compile wildcard patterns into regex for efficient matching."""
        self.compiled_patterns = {
            cat: [self._wildcard_to_regex(p) for p in patterns]
            for cat, patterns in self.category_map.items()
        }
    
    def _build_vocabulary(self):
        """Build vocabulary from category patterns + category names for spell correction."""
        self.category_vocab = {
            cat: sorted(
                {self._strip_wildcards(p) for p in patterns if self._strip_wildcards(p)},
                key=len,
                reverse=True
            )
            for cat, patterns in self.category_map.items()
        }
        
        # Build global vocabulary including category names
        words = {
            word
            for phrases in self.category_vocab.values()
            for phrase in phrases
            for word in phrase.split()
        }
        words |= set(self.category_map.keys())  # Include category names (e.g., "sports")
        self.global_vocab = sorted(words, key=len, reverse=True)
    
    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text like a human would understand it:
        - Unicode normalization
        - Leet-speak conversion (f0otbal1l -> football)
        - Remove noisy punctuation
        - Lowercase and whitespace cleanup
        """
        text = unicodedata.normalize("NFKC", text).lower()
        text = text.translate(LEET_MAP)  # Fix leet-speak
        text = re.sub(r"[^a-z0-9\s:/&-]", " ", text)  # Keep only useful chars
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def _wildcard_to_regex(pattern: str) -> re.Pattern:
        """Convert SQL-style wildcard pattern to compiled regex."""
        escaped = re.sub(r"([.^$*+?{}\[\]|()\\])", r"\\\1", pattern)
        regex_pattern = escaped.replace("%", ".*").replace("_", ".")
        return re.compile(regex_pattern, re.IGNORECASE)
    
    @staticmethod
    def _strip_wildcards(pattern: str) -> str:
        """Remove wildcards and clean up pattern for vocabulary."""
        return re.sub(r"\s+", " ", pattern.replace("%", " ").replace("_", " ").strip())
    
    def correct_spelling(self, text: str, cutoff: float = 0.78) -> str:
        """
        Apply smart fuzzy spell correction:
        - Handles mixed alphanumeric tokens
        - Lower cutoff for short words (more lenient)
        - Uses vocabulary built from patterns + category names
        
        Args:
            text: Input text
            cutoff: Similarity threshold for fuzzy matching (0-1)
        
        Returns:
            Text with corrected spellings
        """
        tokens = text.split()
        vocab_set = set(self.global_vocab)
        corrected = []
        
        for token in tokens:
            # Extract alphabetic portion for matching
            alpha = re.sub(r"[^a-z]", "", token)
            candidate = token if token.isalpha() else (alpha if alpha else token)
            
            # Skip if already in vocab or too short
            if candidate in vocab_set or len(candidate) < 3:
                corrected.append(candidate or token)
                continue
            
            # Use lower cutoff for shorter words (more lenient)
            threshold = 0.72 if len(candidate) <= 5 else cutoff
            matches = get_close_matches(candidate, self.global_vocab, n=1, cutoff=threshold)
            corrected.append(matches[0] if matches else candidate)
        
        return " ".join(corrected)
    
    def classify_category(self, text: str) -> List[Tuple[str, int, List[str]]]:
        """
        Classify text into categories based on pattern matching.
        
        Args:
            text: Normalized input text
        
        Returns:
            List of (category, score, matched_patterns) sorted by score descending
        """
        hits = []
        for category, regex_list in self.compiled_patterns.items():
            matched_patterns = [r.pattern for r in regex_list if r.search(text)]
            if matched_patterns:
                hits.append((category, len(matched_patterns), matched_patterns))
        
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits
    
    def _fallback_category(self, text: str) -> Optional[str]:
        """
        Fallback category detection using synonyms for broad queries.
        Used when wildcard matching finds nothing.
        
        Args:
            text: Normalized text
        
        Returns:
            Category name or None
        """
        tokens = set(text.split())
        for category, keywords in CATEGORY_SYNONYMS.items():
            if tokens & keywords:
                return category
        return None
    
    @staticmethod
    def extract_day_filter(text: str) -> Optional[str]:
        """
        Extract day-of-week filter with abbreviation support.
        Handles: thu, thurs, thursday, fri, weds, etc.
        
        Returns:
            Canonical day name (lowercase) or None
        """
        for day, variants in DAY_VARIANTS.items():
            for variant in variants:
                if re.search(rf"\b{re.escape(variant)}\b", text, re.IGNORECASE):
                    return day
        return None
    
    @staticmethod
    def parse_boolean_groups(text: str) -> List[str]:
        """
        Parse text into entity groups with smart handling:
        - Preserves quoted phrases ("info session")
        - Splits on boolean operators (or, and)
        - Filters out stopwords
        - Removes single-word stopword fragments
        
        Returns:
            List of entity phrase candidates
        """
        # Extract and preserve quoted phrases
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        quoted_phrases = ["".join(q).strip() for q in quoted if any(q)]
        
        # Remove quotes from main text
        text = re.sub(r'["\']([^"\']+)["\']', " ", text)
        
        # Replace delimiters with spaces
        clean = re.sub(r"[|,\/]+", " ", text)
        
        # Split on boolean keywords
        parts = re.split(r"\b(?:or|and)\b", clean, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        
        # Filter out stopwords from each part
        parts = [
            " ".join([w for w in p.split() if w not in STOPWORDS]).strip()
            for p in parts
        ]
        parts = [p for p in parts if p]  # Remove empty strings
        
        return quoted_phrases + parts
    
    def align_to_vocabulary(self, candidates: List[str], vocab_phrases: List[str]) -> List[str]:
        """
        Align candidate phrases to known vocabulary phrases.
        
        Args:
            candidates: List of candidate phrases from query
            vocab_phrases: Vocabulary phrases for the category
        
        Returns:
            List of aligned phrases
        """
        aligned = []
        seen = set()
        
        for candidate in candidates:
            # Try phrase-level fuzzy matching
            matches = get_close_matches(candidate, vocab_phrases, n=1, cutoff=0.7)
            if matches:
                if matches[0] not in seen:
                    seen.add(matches[0])
                    aligned.append(matches[0])
                continue
            
            # Fallback: keep words contained in vocabulary
            vocab_words = {w for phrase in vocab_phrases for w in phrase.split()}
            kept_words = [w for w in candidate.split() if w in vocab_words]
            if kept_words:
                phrase = " ".join(kept_words)
                if phrase not in seen:
                    seen.add(phrase)
                    aligned.append(phrase)
        
        return aligned
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse user query into structured intent.
        
        Args:
            query: Raw user query string
        
        Returns:
            Dict containing:
                - normalized: Normalized text
                - corrected: Spell-corrected text
                - category: Top matching category or None
                - day: Day filter or None
                - entities: List of extracted entity phrases
                - category_scores: List of (category, score, patterns) tuples
        """
        normalized = self._normalize(query)
        corrected = self.correct_spelling(normalized)
        day = self.extract_day_filter(corrected)
        category_scores = self.classify_category(corrected)
        
        # Use wildcard match or fallback to synonym matching
        top_category = (
            category_scores[0][0] if category_scores 
            else self._fallback_category(corrected)
        )
        
        # Extract entities relevant to the category
        entities = []
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


class EventRetriever:
    """
    Handles event retrieval and filtering based on parsed query intent.
    Designed to be easily swapped with a database-backed implementation.
    """
    
    def __init__(self, events: List[Dict]):
        """
        Initialize with event data.
        
        Args:
            events: List of event dicts, each containing:
                    - id, title, start, venue, tags, category
        """
        self.events = events
    
    def retrieve(
        self, 
        category: str, 
        entities: List[str], 
        day: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve events matching the given criteria.
        Enhanced with title/venue substring matching for non-sports.
        
        Args:
            category: Event category
            entities: List of entity phrases to match
            day: Optional day filter (e.g., "thursday")
        
        Returns:
            List of matching event dicts
        """
        # Filter by category
        results = [e for e in self.events if e["category"] == category]
        
        # Filter by entities (if specific entities are provided)
        if entities:
            wanted = {x.lower().strip() for x in entities}
            
            # If entity is just the category name itself, treat as "show all"
            # e.g., "show me sports" should return all sports events
            category_keywords = {"sports", "sport", "careers", "career", 
                               "academics", "academic", "arts", "culture",
                               "arts_culture", "arts culture"}
            
            # Check if entities are only category-level keywords
            if wanted <= category_keywords:
                # Don't filter by entities - show all events in category
                pass
            elif category == "sports":
                # Exact tag matching for sports
                results = [
                    e for e in results 
                    if any(tag.lower() in wanted for tag in e.get("tags", []))
                ]
            else:
                # Substring matching for tags, title, and venue
                results = [
                    e for e in results
                    if any(any(keyword in tag.lower() for keyword in wanted) 
                           for tag in e.get("tags", []))
                    or any(keyword in e["title"].lower() for keyword in wanted)
                    or any(keyword in e.get("venue", "").lower() for keyword in wanted)
                ]
        
        # Filter by day
        if day:
            results = [e for e in results if e["start"].lower().startswith(day)]
        
        return results


class EventRouterSystem:
    """
    Complete event routing system combining router and retriever.
    Main interface for integrating into larger applications.
    Supports pluggable retriever functions.
    """
    
    def __init__(
        self, 
        category_map: Dict[str, List[str]], 
        events: Optional[List[Dict]] = None,
        retriever: Optional[Callable] = None
    ):
        """
        Initialize the complete system.
        
        Args:
            category_map: Category definitions with wildcard patterns
            events: Event data (optional if using custom retriever)
            retriever: Optional custom retriever function(category, entities, day) -> List[Dict]
        """
        self.router = EventRouter(category_map)
        
        if retriever:
            # Use custom retriever function (e.g., database query)
            self._custom_retriever = retriever
            self.retriever = None
        else:
            # Use default EventRetriever with provided events
            self.retriever = EventRetriever(events or [])
            self._custom_retriever = None
    
    def query(self, user_input: str) -> Dict:
        """
        Process user query and return results with machine-friendly summary.
        
        Args:
            user_input: Raw user query string
        
        Returns:
            Dict containing:
                - All fields from router.parse_query()
                - results: List of matching event dicts
                - intent_tuple: (category, entities_tuple, day) for logging/analytics
        """
        parsed = self.router.parse_query(user_input)
        
        # Retrieve matching events using appropriate retriever
        if parsed["category"]:
            if self._custom_retriever:
                results = self._custom_retriever(
                    parsed["category"],
                    parsed["entities"],
                    parsed["day"]
                )
            else:
                results = self.retriever.retrieve(
                    parsed["category"],
                    parsed["entities"],
                    parsed["day"]
                )
        else:
            results = []
        
        parsed["results"] = results
        
        # Add machine-friendly summary for logging/analytics
        parsed["intent_tuple"] = (
            parsed["category"],
            tuple(parsed["entities"]),
            parsed["day"]
        )
        
        return parsed


# ==================== DEMO DATA ====================

DEMO_CATEGORY_MAP = {
    "sports": [
        "%football%", "%basketball%", "%soccer%", "%volleyball%", "%lacrosse%",
        "%baseball%", "%softball%", "%track%", "%cross country%", "%tennis%",
        "%golf%", "%field hockey%", "%swim%", "%wrestling%",
        "%terps%", "%xfinity center%", "%secu stadium%",
        "%recwell%", "%intramural%", "%club sports%", "%terpville%", "%tailgate%"
    ],
    "careers": [
        "%career fair%", "%career & internship fair%", "%internship%", "%resume%",
        "%cover letter%", "%interview%", "%networking%", "%employer%",
        "%handshake%", "%career center%", "%info session%"
    ],
    "academics": [
        "%lecture%", "%seminar%", "%workshop%", "%colloquium%", "%thesis%",
        "%dissertation%", "%guest speaker%", "%department%", "%office hours%",
        "%tutoring%"
    ],
    "arts_culture": [
        "%theatre%", "%theater%", "%dance%", "%music%", "%recital%", "%concert%",
        "%choir%", "%orchestra%", "%clarice%", "%art exhibit%", "%film%",
        "%screening%", "%gallery%"
    ],
}

DEMO_EVENTS = [
    # Sports
    {"id": "s1", "title": "Maryland Football vs. Example U", "start": "Thursday 7:00 PM", "venue": "SECU Stadium", "tags": ["football"], "category": "sports"},
    {"id": "s2", "title": "Maryland Wrestling Dual vs. Sample State", "start": "Thursday 6:00 PM", "venue": "Xfinity Pavilion", "tags": ["wrestling"], "category": "sports"},
    {"id": "s3", "title": "Maryland MBB vs. Demo College", "start": "Friday 8:00 PM", "venue": "Xfinity Center", "tags": ["basketball"], "category": "sports"},
    {"id": "s4", "title": "Club Soccer Scrimmage", "start": "Tuesday 5:30 PM", "venue": "RecWell Fields", "tags": ["soccer", "club sports"], "category": "sports"},
    
    # Careers
    {"id": "c1", "title": "Resume Lab (Career Center)", "start": "Thursday 2:00 PM", "venue": "Hornbake South", "tags": ["resume", "career center", "workshop"], "category": "careers"},
    {"id": "c2", "title": "Employer Info Session: Data Analytics @BigCo", "start": "Wednesday 5:00 PM", "venue": "Virtual / Handshake", "tags": ["info session", "employer", "handshake"], "category": "careers"},
    {"id": "c3", "title": "Career & Internship Fair (STEM)", "start": "Monday 11:00 AM", "venue": "Xfinity Center", "tags": ["career fair"], "category": "careers"},
    
    # Academics
    {"id": "a1", "title": "CS Colloquium: Secure ML Systems", "start": "Thursday 3:30 PM", "venue": "Iribe 0318", "tags": ["colloquium", "ml", "security"], "category": "academics"},
    
    # Arts & Culture
    {"id": "ac1", "title": "MFA Dance Concert", "start": "Thursday 7:30 PM", "venue": "The Clarice", "tags": ["dance", "concert", "clarice"], "category": "arts_culture"},
]


# ==================== CLI INTERFACE ====================

def print_results(result: Dict):
    """Pretty print query results."""
    print("\n--- Parsed Intent ---")
    print(f"Original    : {result.get('_original', 'N/A')}")
    print(f"Normalized  : {result['normalized']}")
    print(f"Spell-fixed : {result['corrected']}")
    print(f"Category    : {result['category'] or '(none)'}")
    print(f"Day Filter  : {result['day'] or '(none)'}")
    print(f"Entities    : {', '.join(result['entities']) if result['entities'] else '(none)'}")
    print(f"Intent Tuple: {result['intent_tuple']}")
    
    if result["category_scores"]:
        scores = ", ".join([f"{c}:{s}" for c, s, _ in result["category_scores"]])
        print(f"Category scores: {scores}")
    
    print("\n--- Results ---")
    if not result["results"]:
        print("No matching events found.")
        return
    
    for event in result["results"]:
        print(f"- {event['title']} | {event['start']} | {event['venue']} | tags: {', '.join(event['tags'])}")


def run_cli():
    """Run interactive CLI demo with sanity check examples."""
    system = EventRouterSystem(DEMO_CATEGORY_MAP, DEMO_EVENTS)
    
    print("""
UMD Event Router (Enhanced Demo)
- Leet-speak normalization (f0otbal1l → football)
- Smart spell correction with category awareness
- Quoted phrase support ("info session")
- Day abbreviations (thu, weds, fri)
- Fallback category detection for broad queries

Sanity Check Examples:
  shOw mE f0otbal1l games on thursday
  i want to watch sports
  info session and resume on weds
  eVENts aT c1arice
    
""")
    
    while True:
        try:
            query = input("\nQuery (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        
        if not query:
            continue
        
        if query.lower() in {"exit", "quit", ":q"}:
            print("Bye.")
            break
        
        result = system.query(query)
        result["_original"] = query  # Store original for display
        print_results(result)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single query mode
        system = EventRouterSystem(DEMO_CATEGORY_MAP, DEMO_EVENTS)
        query = " ".join(sys.argv[1:])
        result = system.query(query)
        result["_original"] = query
        print_results(result)
    else:
        # Interactive mode
        run_cli()