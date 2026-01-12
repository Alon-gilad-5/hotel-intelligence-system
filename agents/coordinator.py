"""
LangGraph Coordinator

Wraps existing BaseAgent architecture with LangGraph for:
- Stateful conversation management
- Entity extraction
- Hybrid memory (recent + summary)
"""

import sys
sys.path.insert(0, '/mnt/project')

from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from graph_state import AgentState, ExtractedEntities, ConversationTurn
from entity_extractor import extract_entities
from memory_manager import update_memory, get_context_for_agent, merge_entities

# Import existing agents
from base_agent import LLMWithFallback
from review_analyst import ReviewAnalystAgent
from competitor_analyst import CompetitorAnalystAgent
from market_intel import MarketIntelAgent
from benchmark_agent import BenchmarkAgent


class LangGraphCoordinator:
    """
    LangGraph-based coordinator with stateful memory.

    Wraps existing agents as graph nodes.
    """

    ROUTING_PROMPT = """Route this query to the appropriate agent.

    Available agents:
    - review_analyst: Guest feedback, sentiment, complaints (wifi, noise, cleanliness)
    - competitor_analyst: Finding competitors, nearby hotels, similarity
    - market_intel: External factors (weather, events, Google Maps)
    - benchmark_agent: Comparing metrics (price, rating), rankings
    
    Context from conversation:
    {context}
    
    Current Query: {query}
    
    Respond with ONLY the agent name (one of: review_analyst, competitor_analyst, market_intel, benchmark_agent):"""

    def __init__(self, hotel_id: str, hotel_name: str, city: str):
        self.hotel_id = hotel_id
        self.hotel_name = hotel_name
        self.city = city

        # Shared LLM
        self.llm = LLMWithFallback()

        # Initialize specialist agents
        self.agents = {
            "review_analyst": ReviewAnalystAgent(hotel_id, hotel_name, city),
            "competitor_analyst": CompetitorAnalystAgent(hotel_id, hotel_name, city),
            "market_intel": MarketIntelAgent(hotel_id, hotel_name, city),
            "benchmark_agent": BenchmarkAgent(hotel_id, hotel_name, city),
        }

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Define the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("review_analyst", self._make_agent_node("review_analyst"))
        workflow.add_node("competitor_analyst", self._make_agent_node("competitor_analyst"))
        workflow.add_node("market_intel", self._make_agent_node("market_intel"))
        workflow.add_node("benchmark_agent", self._make_agent_node("benchmark_agent"))
        workflow.add_node("update_memory", self._update_memory_node)

        # Define edges
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "route")

        # Conditional routing based on selected_agent
        workflow.add_conditional_edges(
            "route",
            lambda state: state["selected_agent"],
            {
                "review_analyst": "review_analyst",
                "competitor_analyst": "competitor_analyst",
                "market_intel": "market_intel",
                "benchmark_agent": "benchmark_agent",
            }
        )

        # All agents go to memory update, then end
        for agent_name in self.agents.keys():
            workflow.add_edge(agent_name, "update_memory")

        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def _extract_entities_node(self, state: AgentState) -> AgentState:
        """Node: Extract entities from current query."""
        query = state["query"]

        # Extract entities (use LLM for richer extraction)
        new_entities = extract_entities(query, llm=self.llm, use_llm=True)

        # Merge with existing
        merged = merge_entities(state, new_entities)

        print(f"[Graph] Extracted entities: {new_entities.to_dict()}")

        return {**state, "entities": merged}

    def _route_node(self, state: AgentState) -> AgentState:
        """Node: Route to appropriate agent."""
        query = state["query"]
        context = get_context_for_agent(state)

        prompt = self.ROUTING_PROMPT.format(context=context, query=query)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        agent_name = response.content.strip().lower()

        # Validate
        if agent_name not in self.agents:
            print(f"[Graph] Invalid agent '{agent_name}', defaulting to review_analyst")
            agent_name = "review_analyst"

        print(f"[Graph] Routing to: {agent_name}")

        return {**state, "selected_agent": agent_name}

    def _make_agent_node(self, agent_name: str):
        """Factory: Create a node function for an agent."""

        def agent_node(state: AgentState) -> AgentState:
            """Execute the specialist agent."""
            agent = self.agents[agent_name]
            query = state["query"]

            # Inject context into query for agent awareness
            context = get_context_for_agent(state)
            if context:
                enhanced_query = f"""[Conversation Context]
{context}

[Current Question]
{query}"""
            else:
                enhanced_query = query

            # Run agent
            response = agent.run(enhanced_query)

            # Extract entities from response too
            response_entities = extract_entities(response, use_llm=False)  # Fast regex only
            merged = merge_entities(state, response_entities)

            return {**state, "response": response, "entities": merged}

        return agent_node

    def _update_memory_node(self, state: AgentState) -> AgentState:
        """Node: Update hybrid memory with new exchange."""

        # Add user turn
        user_turn = ConversationTurn(role="user", content=state["query"])
        state = update_memory(state, user_turn, llm=self.llm)

        # Add assistant turn
        assistant_turn = ConversationTurn(
            role="assistant",
            content=state["response"],
            agent_used=state["selected_agent"]
        )
        state = update_memory(state, assistant_turn, llm=self.llm)

        return state

    def get_initial_state(self) -> AgentState:
        """Get fresh initial state."""
        return AgentState(
            query="",
            selected_agent="",
            response="",
            recent_turns=[],
            summary="",
            entities={},
            hotel_id=self.hotel_id,
            hotel_name=self.hotel_name,
            city=self.city,
            turn_count=0
        )

    def run(self, query: str, state: AgentState = None) -> tuple[str, AgentState]:
        """
        Run a query through the graph.

        Args:
            query: User query
            state: Existing state (for multi-turn). If None, starts fresh.

        Returns:
            Tuple of (response, updated_state)
        """
        if state is None:
            state = self.get_initial_state()

        # Set current query
        state["query"] = query

        # Run graph
        final_state = self.graph.invoke(state)

        return final_state["response"], final_state


def run_chat():
    """Interactive chat loop with LangGraph coordinator."""
    print("=" * 50)
    print("HOTEL INTELLIGENCE SYSTEM")
    print("LangGraph Architecture with Hybrid Memory")
    print("=" * 50)

    # Default hotel context
    HOTEL_ID = "BKG_12345"
    HOTEL_NAME = "Renaissance Johor Bahru Hotel"
    CITY = "Johor Bahru"

    coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
    state = coordinator.get_initial_state()

    print(f"\nContext: {HOTEL_NAME} in {CITY}")
    print("Type 'q' to quit, 'state' to see current state.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            break

        if query.lower() == "state":
            print(f"\n--- Current State ---")
            print(f"Turn count: {state.get('turn_count', 0)}")
            print(f"Recent turns: {len(state.get('recent_turns', []))}")
            print(f"Summary: {state.get('summary', 'None')[:200]}...")
            print(f"Entities: {state.get('entities', {})}")
            print("---\n")
            continue

        if query.lower() == "context":
            context = get_context_for_agent(state)
            print(f"\n--- Context for Agent ---\n{context}\n---\n")
            continue

        try:
            response, state = coordinator.run(query, state)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            import traceback
            print(f"\nError: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    run_chat()