"""
LangGraph State Schema

Defines the conversation state with:
- Entity extraction (hotels, metrics, competitors)
- Hybrid memory (recent turns verbatim + older summarized)
"""

from typing import TypedDict, Annotated, Optional
from dataclasses import dataclass, field
from langgraph.graph.message import add_messages


@dataclass
class ExtractedEntities:
    """Entities mentioned in conversation."""
    hotels: list[str] = field(default_factory=list)  # Hotel names/IDs mentioned
    metrics: list[str] = field(default_factory=list)  # price, rating, amenities, etc.
    competitors: list[str] = field(default_factory=list)  # Competitor names/IDs
    locations: list[str] = field(default_factory=list)  # Cities, areas
    topics: list[str] = field(default_factory=list)  # wifi, cleanliness, noise, etc.

    def merge(self, other: "ExtractedEntities") -> "ExtractedEntities":
        """Merge entities, deduplicating."""
        return ExtractedEntities(
            hotels=list(dict.fromkeys(self.hotels + other.hotels)),
            metrics=list(dict.fromkeys(self.metrics + other.metrics)),
            competitors=list(dict.fromkeys(self.competitors + other.competitors)),
            locations=list(dict.fromkeys(self.locations + other.locations)),
            topics=list(dict.fromkeys(self.topics + other.topics)),
        )

    def to_context_string(self) -> str:
        """Format entities as context for LLM."""
        parts = []
        if self.hotels:
            parts.append(f"Hotels discussed: {', '.join(self.hotels)}")
        if self.metrics:
            parts.append(f"Metrics of interest: {', '.join(self.metrics)}")
        if self.competitors:
            parts.append(f"Competitors mentioned: {', '.join(self.competitors)}")
        if self.locations:
            parts.append(f"Locations: {', '.join(self.locations)}")
        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")
        return "\n".join(parts) if parts else "No prior context."

    def to_dict(self) -> dict:
        return {
            "hotels": self.hotels,
            "metrics": self.metrics,
            "competitors": self.competitors,
            "locations": self.locations,
            "topics": self.topics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedEntities":
        return cls(
            hotels=data.get("hotels", []),
            metrics=data.get("metrics", []),
            competitors=data.get("competitors", []),
            locations=data.get("locations", []),
            topics=data.get("topics", []),
        )


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    role: str  # "user" or "assistant"
    content: str
    agent_used: Optional[str] = None  # Which agent handled this

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "agent_used": self.agent_used}


class AgentState(TypedDict):
    """
    LangGraph state for the multi-agent system.

    Hybrid memory strategy:
    - recent_turns: Last N turns verbatim (default 4)
    - summary: Compressed summary of older turns
    - entities: Extracted entities across all turns
    
    Multi-agent collaboration:
    - agent_queue: Sequential list of agents to execute
    - intermediate_results: Results from each agent in chain
    - agents_executed: Track which agents have run
    """
    # Current query
    query: str

    # Routing decision
    selected_agent: str
    
    # Multi-agent workflow support
    agent_queue: list[str]  # Queue of agents to execute sequentially
    intermediate_results: list[dict]  # Results from each agent in chain
    agents_executed: list[str]  # Track which agents have run

    # Agent response
    response: str

    # Hybrid Memory
    recent_turns: list[dict]  # Last N turns (verbatim)
    summary: str  # Compressed older context
    entities: dict  # ExtractedEntities as dict

    # Hotel context (fixed for session)
    hotel_id: str
    hotel_name: str
    city: str

    # Control
    turn_count: int


# Configuration
MAX_RECENT_TURNS = 4  # Keep last 4 turns verbatim
SUMMARY_TRIGGER = 6  # Summarize when total turns exceed this