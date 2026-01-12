"""
Memory Manager

Implements hybrid memory strategy:
- Recent turns: Keep last N turns verbatim
- Summary: Compress older turns into summary
- Entities: Persist extracted entities
"""

from typing import Optional
from graph_state import (
    AgentState, ExtractedEntities, ConversationTurn,
    MAX_RECENT_TURNS, SUMMARY_TRIGGER
)


def compress_turns_to_summary(
        turns: list[dict],
        existing_summary: str,
        llm
) -> str:
    """
    Compress conversation turns into a summary.

    Args:
        turns: Turns to compress
        existing_summary: Previous summary to build upon
        llm: LLM instance
    """
    if not turns:
        return existing_summary

    # Format turns for summarization
    turns_text = "\n".join([
        f"{t['role'].upper()}: {t['content'][:500]}"  # Truncate long responses
        for t in turns
    ])

    prompt = f"""Summarize this conversation segment concisely, preserving key facts and decisions.

Previous context: {existing_summary or 'None'}

New conversation:
{turns_text}

Write a brief summary (2-3 sentences) capturing:
- Main topics discussed
- Key findings or insights
- Any decisions or conclusions

Summary:"""

    try:
        response = llm.invoke(prompt)
        new_summary = response.content.strip()

        # Combine with existing summary if present
        if existing_summary:
            return f"{existing_summary} {new_summary}"
        return new_summary

    except Exception as e:
        print(f"[MemoryManager] Summary generation failed: {e}")
        # Fallback: simple concatenation
        return existing_summary or "Previous discussion covered hotel analysis."


def update_memory(
        state: AgentState,
        new_turn: ConversationTurn,
        llm: Optional[object] = None
) -> AgentState:
    """
    Update state memory with new turn.

    Implements hybrid strategy:
    1. Add new turn to recent_turns
    2. If recent_turns exceeds MAX, compress oldest to summary
    3. Update entities
    """
    recent = state.get("recent_turns", []).copy()
    summary = state.get("summary", "")
    turn_count = state.get("turn_count", 0) + 1

    # Add new turn
    recent.append(new_turn.to_dict())

    # Check if we need to compress
    if len(recent) > MAX_RECENT_TURNS and llm is not None:
        # Take oldest turns to compress
        to_compress = recent[:-MAX_RECENT_TURNS]
        recent = recent[-MAX_RECENT_TURNS:]

        # Compress to summary
        summary = compress_turns_to_summary(to_compress, summary, llm)
        print(f"[MemoryManager] Compressed {len(to_compress)} turns to summary")

    # Update state
    return {
        **state,
        "recent_turns": recent,
        "summary": summary,
        "turn_count": turn_count,
    }


def get_context_for_agent(state: AgentState) -> str:
    """
    Build context string for agent from state.

    Returns formatted context including:
    - Summary of older conversation
    - Extracted entities
    - Recent turns (for agent to reference)
    """
    parts = []

    # Add summary if exists
    summary = state.get("summary", "")
    if summary:
        parts.append(f"[Previous Context]\n{summary}")

    # Add entities
    entities_dict = state.get("entities", {})
    if entities_dict:
        entities = ExtractedEntities.from_dict(entities_dict)
        entity_context = entities.to_context_string()
        if entity_context != "No prior context.":
            parts.append(f"\n[Key Entities]\n{entity_context}")

    # Add recent turns
    recent = state.get("recent_turns", [])
    if recent:
        # Only include last 2 for immediate context (rest is for full history)
        recent_formatted = "\n".join([
            f"{t['role'].upper()}: {t['content'][:300]}..."
            if len(t['content']) > 300 else f"{t['role'].upper()}: {t['content']}"
            for t in recent[-2:]
        ])
        parts.append(f"\n[Recent Exchange]\n{recent_formatted}")

    return "\n".join(parts) if parts else ""


def merge_entities(state: AgentState, new_entities: ExtractedEntities) -> dict:
    """Merge new entities with existing state entities."""
    existing = ExtractedEntities.from_dict(state.get("entities", {}))
    merged = existing.merge(new_entities)
    return merged.to_dict()


# Testing
if __name__ == "__main__":
    # Test memory update
    state = AgentState(
        query="",
        selected_agent="",
        response="",
        recent_turns=[],
        summary="",
        entities={},
        hotel_id="BKG_123",
        hotel_name="Test Hotel",
        city="Test City",
        turn_count=0
    )

    # Simulate adding turns
    for i in range(6):
        turn = ConversationTurn(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn {i} content here",
            agent_used="review_analyst" if i % 2 == 1 else None
        )
        state = update_memory(state, turn, llm=None)
        print(f"After turn {i}: {len(state['recent_turns'])} recent turns")

    print(f"\nFinal recent turns: {state['recent_turns']}")
    print(f"Summary: {state['summary']}")