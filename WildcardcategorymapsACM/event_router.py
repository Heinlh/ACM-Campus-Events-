#!/usr/bin/env python3
"""
UMD Event Router - Function-based API for AI Agent Tool Calls

This module provides a suite of functions for parsing natural language event queries
and retrieving matching events. Designed to be called by AI agents helping UMD students
find campus events.

Key capabilities:
- Normalize messy user input (leet-speak, typos, punctuation)
- Spell correction with category-aware vocabulary
- Category classification (sports, careers, academics, arts & culture)
- Entity extraction (event types, venues, keywords)
- Day-of-week detection with abbreviation support
- Event retrieval with flexible matching

Main entry point: parse_and_retrieve_events()
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from difflib import get_close_matches

# ===================== Constants =====================

# Leet-speak character mapping for normalization
LEET_MAP = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s"
})

# Category fallback synonyms for broad queries like "show me sports"
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

# ===================== Helper Functions =====================

def normalize_text(text: str) -> str:
    """
    Normalize user input to clean, searchable text.
    
    This function handles messy human input by:
    - Converting Unicode to standard form (NFKC normalization)
    - Converting to lowercase
    - Translating leet-speak (f0otbal1l → football, basktba11 → basketball)
    - Removing noisy punctuation while preserving useful characters
    - Collapsing multiple spaces
    
    Args:
        text: Raw user input string (e.g., "shOw mE f0otbal1l games!!!")
    
    Returns:
        Normalized string (e.g., "show me football games")
    
    Example:
        >>> normalize_text("shOw mE f0otbal1l games on thursday!!!")
        'show me football games on thursday'
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(LEET_MAP)  # Fix leet-speak
    text = re.sub(r"[^a-z0-9\s:/&-]", " ", text)  # Keep only useful chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocabulary_from_patterns(category_map: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Build vocabulary dictionaries from category wildcard patterns.
    
    This creates two vocabularies:
    1. Category-specific vocab: Maps each category to its relevant phrases
    2. Global vocab: All words across all categories (for spell correction)
    
    The vocabulary is built by stripping wildcards from patterns like "%football%"
    to get "football", then breaking phrases into individual words.
    
    Args:
        category_map: Dict mapping category names to lists of wildcard patterns
                     Example: {"sports": ["%football%", "%basketball%"]}
    
    Returns:
        Tuple of (category_vocab, global_vocab):
        - category_vocab: Dict[str, List[str]] - category to sorted phrase list
        - global_vocab: List[str] - all unique words sorted by length (longest first)
    
    Example:
        >>> patterns = {"sports": ["%football%", "%basketball game%"]}
        >>> cat_vocab, global_vocab = build_vocabulary_from_patterns(patterns)
        >>> cat_vocab["sports"]
        ['basketball game', 'football']
        >>> "basketball" in global_vocab
        True
    """
    def strip_wildcards(pattern: str) -> str:
        """Remove SQL wildcards and clean up."""
        return re.sub(r"\s+", " ", pattern.replace("%", " ").replace("_", " ").strip())
    
    # Build category-specific vocabularies
    category_vocab = {
        cat: sorted(
            {strip_wildcards(p) for p in patterns if strip_wildcards(p)},
            key=len,
            reverse=True
        )
        for cat, patterns in category_map.items()
    }
    
    # Build global vocabulary from all phrases + category names
    words = {
        word
        for phrases in category_vocab.values()
        for phrase in phrases
        for word in phrase.split()
    }
    words |= set(category_map.keys())  # Include category names
    global_vocab = sorted(words, key=len, reverse=True)
    
    return category_vocab, global_vocab


def correct_spelling(text: str, global_vocab: List[str], cutoff: float = 0.78) -> str:
    """
    Apply fuzzy spell correction using vocabulary-based matching.
    
    This function intelligently corrects typos by:
    - Comparing each word against known vocabulary
    - Using lower similarity thresholds for short words (more lenient)
    - Handling mixed alphanumeric tokens
    - Preserving words already in vocabulary
    
    Args:
        text: Input text (should be normalized first)
        global_vocab: List of known vocabulary words
        cutoff: Similarity threshold for matching (0-1). Default 0.78 for longer words
    
    Returns:
        Text with corrected spellings
    
    Example:
        >>> vocab = ["football", "basketball", "game"]
        >>> correct_spelling("footbal gme", vocab)
        'football game'
        >>> correct_spelling("basktball", vocab)
        'basketball'
    """
    tokens = text.split()
    vocab_set = set(global_vocab)
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
        matches = get_close_matches(candidate, global_vocab, n=1, cutoff=threshold)
        corrected.append(matches[0] if matches else candidate)
    
    return " ".join(corrected)


def compile_category_patterns(category_map: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """
    Compile SQL-style wildcard patterns into regex patterns for efficient matching.
    
    Converts patterns like "%football%" into regex patterns like ".*football.*".
    This allows flexible matching:
    - "football game" matches "%football%"
    - "info session" matches "%info session%"
    
    Args:
        category_map: Dict mapping category names to wildcard pattern lists
    
    Returns:
        Dict mapping category names to lists of compiled regex patterns
    
    Example:
        >>> patterns = {"sports": ["%football%", "%basketball%"]}
        >>> compiled = compile_category_patterns(patterns)
        >>> compiled["sports"][0].search("football game")
        <re.Match object...>
    """
    def wildcard_to_regex(pattern: str) -> re.Pattern:
        """Convert SQL wildcard pattern to compiled regex."""
        escaped = re.sub(r"([.^$*+?{}\[\]|()\\])", r"\\\1", pattern)
        regex_pattern = escaped.replace("%", ".*").replace("_", ".")
        return re.compile(regex_pattern, re.IGNORECASE)
    
    return {
        cat: [wildcard_to_regex(p) for p in patterns]
        for cat, patterns in category_map.items()
    }


def classify_category(text: str, compiled_patterns: Dict[str, List[re.Pattern]]) -> List[Tuple[str, int, List[str]]]:
    """
    Classify text into event categories using pattern matching.
    
    Tests the input text against all category patterns and returns matches
    sorted by relevance (number of patterns matched).
    
    Args:
        text: Normalized and spell-corrected text
        compiled_patterns: Dict of category to compiled regex patterns
    
    Returns:
        List of (category, score, matched_patterns) tuples, sorted by score descending.
        Score = number of patterns that matched
    
    Example:
        >>> # Assuming compiled patterns for sports include "%football%"
        >>> classify_category("football game", compiled_patterns)
        [('sports', 1, ['.*football.*'])]
    """
    hits = []
    for category, regex_list in compiled_patterns.items():
        matched_patterns = [r.pattern for r in regex_list if r.search(text)]
        if matched_patterns:
            hits.append((category, len(matched_patterns), matched_patterns))
    
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def detect_category_fallback(text: str) -> Optional[str]:
    """
    Fallback category detection using keyword synonyms for broad queries.
    
    Used when wildcard pattern matching finds nothing. Handles queries like:
    - "show me sports" → sports
    - "any career events?" → careers
    - "i want to see arts" → arts_culture
    
    Args:
        text: Normalized text
    
    Returns:
        Category name if found, None otherwise
    
    Example:
        >>> detect_category_fallback("show me sports")
        'sports'
        >>> detect_category_fallback("i want career help")
        'careers'
    """
    tokens = set(text.split())
    for category, keywords in CATEGORY_SYNONYMS.items():
        if tokens & keywords:
            return category
    return None


def extract_day_filter(text: str) -> Optional[str]:
    """
    Extract day-of-week filter from text with abbreviation support.
    
    Recognizes both full names and common abbreviations:
    - thursday, thu, thur, thurs
    - wednesday, wed, weds
    - friday, fri
    
    Args:
        text: Text to search for day names
    
    Returns:
        Canonical day name (lowercase) or None if no day found
    
    Example:
        >>> extract_day_filter("events on thu")
        'thursday'
        >>> extract_day_filter("weds afternoon")
        'wednesday'
        >>> extract_day_filter("next week")
        None
    """
    for day, variants in DAY_VARIANTS.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", text, re.IGNORECASE):
                return day
    return None


def parse_entity_groups(text: str) -> List[str]:
    """
    Extract entity phrases from text with smart handling.
    
    This function:
    - Preserves quoted phrases ("info session" stays together)
    - Splits on boolean operators (or, and)
    - Filters out stopwords
    - Handles delimiters (|, /, ,)
    
    Args:
        text: Text to parse for entities
    
    Returns:
        List of entity phrase candidates
    
    Example:
        >>> parse_entity_groups('"info session" and resume')
        ['info session', 'resume']
        >>> parse_entity_groups("football or basketball games")
        ['football', 'basketball']
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


def align_entities_to_vocabulary(candidates: List[str], vocab_phrases: List[str]) -> List[str]:
    """
    Align candidate entity phrases to known vocabulary phrases.
    
    This function refines extracted entities by:
    - Fuzzy matching against known phrases
    - Word-level filtering when phrase matching fails
    - Deduplication
    
    Args:
        candidates: List of candidate entity phrases from query
        vocab_phrases: Vocabulary phrases for the detected category
    
    Returns:
        List of aligned entity phrases
    
    Example:
        >>> vocab = ["football", "basketball game", "info session"]
        >>> align_entities_to_vocabulary(["footbal", "info sesion"], vocab)
        ['football', 'info session']
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


def parse_event_query(query: str, category_map: Dict[str, List[str]]) -> Dict:
    """
    Parse a natural language event query into structured intent.
    
    This is the main NLP function that processes user queries through the full pipeline:
    1. Normalize text (leet-speak, punctuation, case)
    2. Correct spelling
    3. Classify category
    4. Extract day filter
    5. Extract and align entities
    
    Args:
        query: Raw user query string (e.g., "shOw mE f0otbal1l games on thursday")
        category_map: Category definitions with wildcard patterns
    
    Returns:
        Dict containing:
            - normalized: Normalized text
            - corrected: Spell-corrected text
            - category: Top matching category or None
            - day: Day filter or None
            - entities: List of extracted entity phrases
            - category_scores: List of (category, score, patterns) tuples
    
    Example:
        >>> category_map = {"sports": ["%football%", "%basketball%"]}
        >>> result = parse_event_query("shOw mE f0otbal1l on thursday", category_map)
        >>> result["category"]
        'sports'
        >>> result["entities"]
        ['football']
        >>> result["day"]
        'thursday'
    """
    # Build vocabularies
    category_vocab, global_vocab = build_vocabulary_from_patterns(category_map)
    
    # Normalize and correct
    normalized = normalize_text(query)
    corrected = correct_spelling(normalized, global_vocab)
    
    # Extract day filter
    day = extract_day_filter(corrected)
    
    # Classify category
    compiled_patterns = compile_category_patterns(category_map)
    category_scores = classify_category(corrected, compiled_patterns)
    
    # Use wildcard match or fallback to synonym matching
    top_category = (
        category_scores[0][0] if category_scores 
        else detect_category_fallback(corrected)
    )
    
    # Extract entities relevant to the category
    entities = []
    if top_category:
        candidates = parse_entity_groups(corrected)
        vocab = category_vocab.get(top_category, [])
        entities = align_entities_to_vocabulary(candidates, vocab)
    
    return {
        "normalized": normalized,
        "corrected": corrected,
        "category": top_category,
        "day": day,
        "entities": entities,
        "category_scores": category_scores,
    }


def retrieve_matching_events(
    events: List[Dict],
    category: str,
    entities: List[str],
    day: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve events matching the parsed query intent.
    
    Filters events by:
    1. Category (exact match)
    2. Entities (tag/title/venue matching with different strategies per category)
    3. Day (start time matching)
    
    Matching strategies:
    - Sports: Strict tag matching (reduces false positives)
    - Other categories: Flexible substring matching in tags, title, and venue
    - Broad queries (entities are just category keywords): Return all events in category
    
    Args:
        events: List of event dicts with fields: id, title, start, venue, tags, category
        category: Event category to filter by
        entities: List of entity phrases to match against
        day: Optional day filter (e.g., "thursday")
    
    Returns:
        List of matching event dicts
    
    Example:
        >>> events = [
        ...     {"id": "1", "title": "Football Game", "start": "thursday 7pm",
        ...      "venue": "Stadium", "tags": ["football"], "category": "sports"}
        ... ]
        >>> retrieve_matching_events(events, "sports", ["football"], "thursday")
        [{'id': '1', 'title': 'Football Game', ...}]
    """
    # Filter by category
    results = [e for e in events if e.get("category") == category]
    
    # Filter by entities (if specific entities are provided)
    if entities:
        wanted = {x.lower().strip() for x in entities}
        
        # Category-level keywords don't filter - they mean "show all in this category"
        category_keywords = {
            "sports", "sport", "careers", "career",
            "academics", "academic", "arts", "culture",
            "arts_culture", "arts culture"
        }
        
        if not (wanted <= category_keywords):
            if category == "sports":
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
                    or any(keyword in e.get("title", "").lower() for keyword in wanted)
                    or any(keyword in e.get("venue", "").lower() for keyword in wanted)
                ]
    
    # Filter by day
    if day:
        results = [e for e in results if e.get("start", "").lower().startswith(day)]
    
    return results


def parse_and_retrieve_events(
    query: str,
    category_map: Dict[str, List[str]],
    events: List[Dict]
) -> Dict:
    """
    Main entry point: Parse natural language query and retrieve matching events.
    
    This function combines the full pipeline:
    1. Parse query into structured intent (category, entities, day)
    2. Retrieve events matching that intent
    3. Return comprehensive results with metadata
    
    This is the primary function an AI agent should call when a student asks
    about campus events.
    
    Args:
        query: Raw user query (e.g., "shOw mE f0otbal1l games on thursday")
        category_map: Category definitions with wildcard patterns
                     Example: {"sports": ["%football%", "%basketball%"]}
        events: List of event dicts with fields: id, title, start, venue, tags, category
    
    Returns:
        Dict containing:
            - All fields from parse_event_query()
            - results: List of matching event dicts
            - intent_tuple: (category, entities_tuple, day) for logging/analytics
    
    Example:
        >>> category_map = {"sports": ["%football%"]}
        >>> events = [{"id": "1", "title": "Football", "start": "thursday 7pm",
        ...            "venue": "Stadium", "tags": ["football"], "category": "sports"}]
        >>> result = parse_and_retrieve_events("football on thursday", category_map, events)
        >>> result["category"]
        'sports'
        >>> len(result["results"])
        1
        >>> result["intent_tuple"]
        ('sports', ('football',), 'thursday')
    """
    # Parse the query
    parsed = parse_event_query(query, category_map)
    
    # Retrieve matching events
    if parsed["category"]:
        results = retrieve_matching_events(
            events,
            parsed["category"],
            parsed["entities"],
            parsed["day"]
        )
    else:
        results = []
    
    # Add results and machine-friendly summary
    parsed["results"] = results
    parsed["intent_tuple"] = (
        parsed["category"],
        tuple(parsed["entities"]),
        parsed["day"]
    )
    
    return parsed


def format_result_as_json(result: Dict, pretty: bool = True) -> str:
    """
    Format the query result as JSON string.
    
    Args:
        result: Result dict from parse_and_retrieve_events
        pretty: If True, format with indentation for readability
    
    Returns:
        JSON string representation of the result
    """
    # Create a serializable version of the result
    json_result = {
        "query": result.get("_original", ""),
        "normalized": result["normalized"],
        "corrected": result["corrected"],
        "category": result["category"],
        "day": result["day"],
        "entities": result["entities"],
        "intent_tuple": {
            "category": result["intent_tuple"][0],
            "entities": list(result["intent_tuple"][1]),
            "day": result["intent_tuple"][2]
        },
        "event_count": len(result["results"]),
        "events": result["results"]
    }
    
    if pretty:
        return json.dumps(json_result, indent=2, ensure_ascii=False)
    else:
        return json.dumps(json_result, ensure_ascii=False)


# ===================== Public API =====================

__all__ = [
    "parse_and_retrieve_events",
    "parse_event_query",
    "retrieve_matching_events",
    "normalize_text",
    "correct_spelling",
    "extract_day_filter",
    "CATEGORY_SYNONYMS",
    "DAY_VARIANTS",
    "STOPWORDS",
]

