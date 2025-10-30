Here’s a drop-in README you can paste into your repo.

---

# UMD Event Router

**One-liner:** Turn messy, human requests like “shOw mE f0otbal1l games on thursday” into a clean, structured search, then return the matching events.

---

## For non-technical stakeholders

### What it does

* **Understands human phrasing.** It cleans up typos, slang, and “leet” characters (e.g., `f0otbal1l` → `football`).
* **Figures out what you mean.** It identifies the **category** (sports, careers, academics, arts & culture), the **specifics** you asked for (like *football* or *info session*), and the **day** (Thu, Fri, etc.).
* **Finds matching events.** It filters event data by those signals and returns the best matches.

### Why it matters

* Users can type naturally (“i want to watch sports”), and the system still finds relevant events.
* It’s **modular**: the language understanding is separate from where events are stored. We can swap the data source without touching the brain.
* It’s **transparent**: Each query is converted into a simple **intent tuple** `(category, (entities…), day)` that you can log and analyze.

### Real examples

* “**i want to watch sports**” → show all sports events (broad query fallback)
* “**‘info session’ and resume on weds**” → career events matching those phrases on Wednesday
* “**events at clarice on fri**” → arts & culture events at The Clarice on Friday

---

## For developers

### Architecture (single-file module: `event_router.py`)

* **Router (NLP layer):** Normalizes text, corrects likely typos, classifies category via wildcard patterns (with a synonym fallback), extracts entities (supports quoted phrases), and detects day names/abbreviations.
* **Retriever (data layer):** Default in-memory filter. Replace with your DB/search retriever using the same `(category, entities, day)` inputs.
* **System (orchestrator):** Wires the Router and Retriever together and returns a uniform response object including `intent_tuple` and `results`.

```
User Query
   ↓
[Router] normalize → correct spelling → find category → extract entities → detect day
   ↓
intent = (category, entities, day)
   ↓
[Retriever] query backend with intent
   ↓
Results + intent returned to caller
```

### Public API

```python
from event_router import EventRouterSystem
```

### Minimal usage

```python
from event_router import EventRouterSystem

CATEGORY_MAP = {
    "sports": ["%football%", "%basketball%", "%soccer%"],
    "careers": ["%resume%", "%info session%", "%internship%"],
    "academics": ["%lecture%", "%seminar%", "%workshop%"],
    "arts_culture": ["%concert%", "%clarice%", "%film%"],
}

# Option A: Use your own data source
def my_db_retriever(category, entities, day):
    # translate (category, entities, day) into a SQL/ES query
    # return list[dict]: {"id","title","start","venue","tags","category"}
    return []

system = EventRouterSystem(category_map=CATEGORY_MAP, retriever=my_db_retriever)

resp = system.query("shOw mE f0otbal1l games on thursday")
# resp["intent_tuple"] == ("sports", ("football",), "thursday")
# resp["results"]      == list of event dicts
```

### Inputs & Outputs

**Input:** `user_input: str` (free-form text)

**Output (dict):**

* `normalized`: cleaned text (lowercased, punctuation/leet fixed)
* `corrected`: softly spell-corrected text based on vocab
* `category`: top category or `None`
* `day`: canonical day (e.g., `"thursday"`) or `None`
* `entities`: list of aligned phrases (e.g., `["football"]`, `["info session","resume"]`)
* `category_scores`: diagnostic data for category matching
* `results`: list of event dicts
* `intent_tuple`: `(category, tuple(entities), day)`

### Data contract (events)

The retriever expects **event objects** with:

```python
{
  "id": str,
  "title": str,
  "start": str,     # starts with canonical day e.g., "thursday 7:00 PM"
  "venue": str,
  "tags": list[str],
  "category": str,  # one of your CATEGORY_MAP keys
}
```

### Configuration

**Category patterns (`CATEGORY_MAP`)**

* SQL-style wildcards: `"%football%"`, `"%info session%"`, etc.
* These generate both the **category classifier** and the **spell-correction vocabulary**.

**Synonyms / Days / Stopwords**

* Built-ins handle synonyms (`game`→sports), day abbreviations (`thu`, `weds`), and filler words (“show me any…”).
* You can extend `CATEGORY_SYNONYMS`, `DAY_VARIANTS`, and `STOPWORDS` if your users speak differently.

### Key implementation details

* **Normalization:** Unicode NFKC → lower → leet map → junk removal → space collapse.
* **Spell correction:** `difflib.get_close_matches` against a vocab built from wildcard phrases **and category names**. Lenient threshold for short tokens.
* **Category routing:** Regex from SQL wildcards; if no hits, synonym fallback.
* **Entity extraction:** Quoted phrase preservation + boolean split on `and/or`, minus stopwords; then aligned to category vocab via fuzzy/word-level match.
* **Retriever semantics:**

  * Sports → strict tag matching (reduces false positives).
  * Non-sports → tags/title/venue substring matching (more flexible).
  * Day filter → `event["start"].lower().startswith(day)`.

### Swapping the data source

Use a **custom retriever**:

```python
def retriever(category, entities, day):
    # Build WHERE clause or ES query pieces
    # category: str
    # entities: list[str]
    # day: Optional[str]
    return run_query(category, entities, day)  # list[dict]

system = EventRouterSystem(category_map=CATEGORY_MAP, retriever=retriever)
```

Or use the default in-memory retriever:

```python
from event_router import EventRetriever, EventRouterSystem

events = [...]  # your list of event dicts
system = EventRouterSystem(category_map=CATEGORY_MAP, events=events)
```

### Extending behavior

* **Multi-category responses:** run retrieval for the top N categories and return a `{category: results}` map.
* **Negations:** detect “not / except” segments and exclude those terms during entity filtering.
* **Telemetries:** log `intent_tuple`, correction deltas, and category scores for continuous improvement.

### Testing (suggested)

Create intent snapshots to lock behavior:

```python
import pytest
from event_router import EventRouter

def test_parse_football_thursday():
    router = EventRouter({"sports": ["%football%"]})
    out = router.parse_query("shOw mE f0otbal1l games on thursday")
    assert out["category"] == "sports"
    assert "football" in out["entities"]
    assert out["day"] == "thursday"
```

### Performance notes

* Works fast for typical CLI/API usage; most cost is regex + `difflib` on modest vocabularies.
* If your `CATEGORY_MAP` grows large, consider caching compiled regexes (already done) and trimming vocab to high-value tokens.

---

## FAQ

**Q: What if a user types only “sports”?**
A: Synonym fallback selects the `sports` category and shows all sports events (no entity filter).

**Q: Will it over-correct words?**
A: It corrects only when similarity is high, with extra caution on short tokens.

**Q: How do we support new terms (e.g., “volleyball”)?**
A: Add a wildcard pattern (e.g., `"%volleyball%"`) to the `sports` list. That feeds both classification and spell-correction vocabulary—no code changes needed.

