"""
Entity Extractor

Extracts relevant entities from user queries and agent responses.
Uses LLM for semantic extraction with fallback regex patterns.
"""

import re
from typing import Optional
from graph_state import ExtractedEntities

# Known patterns for fast extraction
METRIC_PATTERNS = [
    "price", "cost", "rate", "rating", "score", "review", "amenities",
    "facilities", "cleanliness", "location", "wifi", "parking", "breakfast",
    "service", "staff", "noise", "room", "bed", "bathroom", "pool", "gym"
]

LOCATION_PATTERNS = [
    r"london", r"paris", r"rome", r"tokyo", r"new york", r"dubai",
    r"johor bahru", r"kuala lumpur", r"singapore", r"penang", r"langkawi",
    r"melaka", r"kota kinabalu", r"ipoh", r"george town"
]


def extract_entities_regex(text: str) -> ExtractedEntities:
    """Fast regex-based extraction for common patterns."""
    text_lower = text.lower()

    entities = ExtractedEntities()

    # Extract metrics
    for metric in METRIC_PATTERNS:
        if metric in text_lower:
            entities.metrics.append(metric)

    # Extract locations
    for loc_pattern in LOCATION_PATTERNS:
        if re.search(loc_pattern, text_lower):
            # Capitalize properly
            match = re.search(loc_pattern, text_lower)
            if match:
                entities.locations.append(match.group().title())

    # Extract hotel IDs (BKG_xxx or ABB_xxx)
    hotel_ids = re.findall(r'(BKG_\d+|ABB_\d+)', text, re.IGNORECASE)
    entities.hotels.extend([h.upper() for h in hotel_ids])

    # Extract competitor mentions
    if any(word in text_lower for word in ["competitor", "nearby", "similar", "compare"]):
        entities.topics.append("competitor_analysis")

    return entities


def extract_entities_llm(text: str, llm) -> ExtractedEntities:
    """
    LLM-based entity extraction for richer semantic understanding.

    Args:
        text: Text to extract from
        llm: LLM instance (LLMWithFallback)
    """
    prompt = f"""Extract entities from this hotel-related text. Return ONLY a JSON object.

Text: "{text}"

Extract:
- hotels: Hotel names or IDs mentioned (e.g., "Renaissance Johor Bahru", "BKG_123")
- metrics: Business metrics (price, rating, reviews, amenities, cleanliness, etc.)
- competitors: Competitor names or "competitors" if mentioned generically
- locations: Cities or areas
- topics: Specific topics (wifi, noise, parking, breakfast, etc.)

Return JSON only, no explanation:
{{"hotels": [], "metrics": [], "competitors": [], "locations": [], "topics": []}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON from response
        import json

        # Handle markdown code blocks
        if "```" in content:
            content = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            content = content.group(1) if content else "{}"

        data = json.loads(content)
        return ExtractedEntities.from_dict(data)

    except Exception as e:
        print(f"[EntityExtractor] LLM extraction failed: {e}, falling back to regex")
        return extract_entities_regex(text)


def extract_entities(
        text: str,
        llm: Optional[object] = None,
        use_llm: bool = True
) -> ExtractedEntities:
    """
    Main extraction function. Uses LLM if available, otherwise regex.

    Args:
        text: Text to extract entities from
        llm: Optional LLM instance
        use_llm: Whether to use LLM (set False for speed)
    """
    # Always run regex for fast common patterns
    regex_entities = extract_entities_regex(text)

    # Optionally enhance with LLM
    if use_llm and llm is not None:
        llm_entities = extract_entities_llm(text, llm)
        return regex_entities.merge(llm_entities)

    return regex_entities


# Testing
if __name__ == "__main__":
    test_texts = [
        "How does my hotel's wifi compare to competitors?",
        "What are guests saying about cleanliness at Malmaison London?",
        "Compare my price to BKG_12345 and nearby hotels in London",
        "Show me the rating trend",
    ]

    for text in test_texts:
        entities = extract_entities(text, use_llm=False)
        print(f"\nText: {text}")
        print(f"Entities: {entities.to_dict()}")