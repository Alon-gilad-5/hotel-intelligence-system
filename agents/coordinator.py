"""
LangGraph Coordinator

Wraps existing BaseAgent architecture with LangGraph for:
- Stateful conversation management
- Entity extraction
- Hybrid memory (recent + summary)
- Multi-agent collaboration for complex queries
"""

import sys
import os

# Add project root and agents folder to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

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
    LangGraph-based coordinator with stateful memory and multi-agent collaboration.

    Supports sequential execution of multiple agents for complex queries.
    """

    ROUTING_PROMPT = """Route this query to the appropriate agent(s).

    SINGLE-AGENT PATTERNS:
    - Guest feedback/reviews for MY hotel (no comparison) → review_analyst
    - "What do guests say about X?" (MY hotel only) → review_analyst
    - Compare reviews vs neighbors / competitor review analysis → competitor_analyst (uses NLP tool)
    - "How do my reviews compare?" / "What are my weaknesses?" → competitor_analyst
    - Feature impact / "Why is my rating X?" / "How to improve rating?" → benchmark_agent (uses LR tool)
    - Simple ranking "Rank hotels by X" → benchmark_agent
    - Events, weather, external factors → market_intel

    MULTI-AGENT PATTERNS (rare):
    - "Full competitive analysis" → competitor_analyst, benchmark_agent
    - "Compare everything with competitors" → competitor_analyst, benchmark_agent

    CRITICAL ROUTING RULES:
    1. Questions about MY hotel reviews only (no comparison) → review_analyst
    2. Questions COMPARING with neighbors/competitors → competitor_analyst (NLP tool)
    3. Questions about FEATURE IMPACT or WHY rating differs → benchmark_agent (LR tool)
    4. Simple metric rankings → benchmark_agent (rank_by_metric)
    5. If unsure, use SINGLE agent

    Available agents:
    - review_analyst: Guest feedback for THIS hotel only (RAG search)
    - competitor_analyst: Compare reviews vs neighbors, find similar hotels (NLP tool)
    - benchmark_agent: Feature impact analysis, rankings (LR tool)
    - market_intel: External factors (weather, events, Google Maps)

    Context from conversation:
    {context}

    Current Query: {query}

    Respond with agent name(s):"""

    ROUTING_EXAMPLES = """
    Query: "What do guests say about wifi?"
    → review_analyst (single hotel, no comparison)

    Query: "How do my reviews compare to neighbors?"
    → competitor_analyst (comparison, uses NLP tool)

    Query: "What are my weaknesses vs competition?"
    → competitor_analyst (comparison, uses NLP tool)

    Query: "Why is my rating lower than competitors?"
    → benchmark_agent (feature impact, uses LR tool)

    Query: "How can I improve my rating?"
    → benchmark_agent (feature impact, uses LR tool)

    Query: "Rank hotels by price"
    → benchmark_agent (simple ranking)

    Query: "Are there events this weekend?"
    → market_intel

    Query: "Give me a full competitive analysis"
    → competitor_analyst, benchmark_agent (multi-agent)
    """

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
        """Build the LangGraph workflow with multi-agent support."""

        # Define the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("check_queue", self._check_queue_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)
        workflow.add_node("update_memory", self._update_memory_node)

        # Define edges
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "route")
        workflow.add_edge("route", "execute_agent")
        workflow.add_edge("execute_agent", "check_queue")
        
        # Conditional: more agents in queue → execute next, else → aggregate
        workflow.add_conditional_edges(
            "check_queue",
            self._should_continue,
            {
                "continue": "execute_agent",
                "aggregate": "aggregate_results",
            }
        )
        
        workflow.add_edge("aggregate_results", "update_memory")
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
        """Node: Route to appropriate agent(s) - supports multi-agent chains."""
        query = state["query"]
        context = get_context_for_agent(state)

        prompt = self.ROUTING_PROMPT.format(context=context, query=query)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw_response = response.content.strip().lower()
        
        # Parse potentially multiple agents
        agent_names = [a.strip() for a in raw_response.replace("\n", ",").split(",")]
        agent_names = [a for a in agent_names if a in self.agents]
        
        # Fallback if nothing valid
        if not agent_names:
            print(f"[Graph] Invalid routing '{raw_response}', defaulting to review_analyst")
            agent_names = ["review_analyst"]

        # First agent goes to selected_agent, rest go to queue
        selected = agent_names[0]
        queue = agent_names[1:] if len(agent_names) > 1 else []
        
        if queue:
            print(f"[Graph] Multi-agent workflow: {' → '.join(agent_names)}")
        else:
            print(f"[Graph] Routing to: {selected}")

        return {
            **state, 
            "selected_agent": selected,
            "agent_queue": queue,
            "intermediate_results": [],
            "agents_executed": []
        }

    def _execute_agent_node(self, state: AgentState) -> AgentState:
        """Node: Execute the currently selected agent with support for long-running Databricks jobs."""
        import time
        import threading
        import sys
        
        agent_name = state["selected_agent"]
        agent = self.agents[agent_name]
        query = state["query"]
        
        # Databricks job timeout configuration (in seconds)
        # NLP and LR tools take ~15 minutes each
        DATABRICKS_TIMEOUTS = {
            "competitor_analyst": 1080,   # 18 min (NLP tool takes ~15 min + buffer)
            "benchmark_agent": 1080,      # 18 min (LR tool takes ~15 min + buffer)
        }
        
        # Check if this agent might use Databricks tools (long-running)
        is_databricks_job = agent_name in DATABRICKS_TIMEOUTS
        timeout = DATABRICKS_TIMEOUTS.get(agent_name, 300)  # Default 5 min
        
        # Build context including results from previous agents in chain
        context_parts = []
        
        # Add conversation context
        conv_context = get_context_for_agent(state)
        if conv_context:
            context_parts.append(f"[Conversation Context]\n{conv_context}")
        
        # Add intermediate results from previous agents in this chain
        intermediate = state.get("intermediate_results", [])
        if intermediate:
            context_parts.append("[Results from Previous Agents]")
            for result in intermediate:
                context_parts.append(f"\n--- {result['agent']} ---\n{result['response'][:1500]}")
        
        # Build enhanced query
        if context_parts:
            enhanced_query = "\n\n".join(context_parts) + f"\n\n[Current Question]\n{query}"
        else:
            enhanced_query = query

        # Inform user about long-running jobs
        if is_databricks_job:
            print(f"[{agent_name}] Starting Databricks analysis (takes ~15 minutes)...")
            print(f"[{agent_name}] Timeout set to {timeout // 60} minutes")
        else:
            print(f"[{agent_name}] Executing...")
        
        # Run agent with progress indicator for long jobs
        start_time = time.time()
        response = None
        error_msg = None
        
        try:
            if is_databricks_job:
                # Progress indicator in background
                stop_progress = threading.Event()
                
                def show_progress():
                    while not stop_progress.is_set():
                        elapsed = int(time.time() - start_time)
                        mins, secs = divmod(elapsed, 60)
                        sys.stdout.write(f"\r[{agent_name}] Running... {mins}m {secs}s ")
                        sys.stdout.flush()
                        stop_progress.wait(timeout=5)  # Update every 5 seconds
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                
                progress_thread = threading.Thread(target=show_progress, daemon=True)
                progress_thread.start()
                
                try:
                    response = agent.run(enhanced_query)
                finally:
                    stop_progress.set()
                    progress_thread.join(timeout=1)
            else:
                response = agent.run(enhanced_query)
                
        except TimeoutError as e:
            error_msg = f"Databricks job timed out after {timeout // 60} minutes. The analysis may still be running in the background."
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
        
        elapsed = time.time() - start_time
        print(f"[{agent_name}] Completed in {elapsed:.1f}s")
        
        # Handle errors gracefully
        if error_msg:
            response = f"⚠️ {error_msg}\n\nPlease try again or use a simpler query."
            print(f"[{agent_name}] Error: {error_msg}")

        # Extract entities from response
        response_entities = extract_entities(response, use_llm=False)
        merged = merge_entities(state, response_entities)

        # Store this agent's result
        new_intermediate = intermediate + [{
            "agent": agent_name,
            "response": response,
            "elapsed_seconds": elapsed
        }]
        
        # Track executed agents
        executed = state.get("agents_executed", []) + [agent_name]

        return {
            **state, 
            "response": response, 
            "entities": merged,
            "intermediate_results": new_intermediate,
            "agents_executed": executed
        }

    def _check_queue_node(self, state: AgentState) -> AgentState:
        """Node: Check if more agents need to run and prepare next."""
        queue = state.get("agent_queue", [])
        
        if queue:
            # Pop next agent from queue
            next_agent = queue[0]
            remaining_queue = queue[1:]
            print(f"[Graph] Next in queue: {next_agent} (remaining: {remaining_queue})")
            return {**state, "selected_agent": next_agent, "agent_queue": remaining_queue}
        
        return state

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue to next agent or aggregate."""
        # Check if the currently selected agent has been executed yet
        # If not, we need to continue and execute it
        selected = state.get("selected_agent", "")
        executed = state.get("agents_executed", [])
        
        # Continue if there's a selected agent that hasn't been executed yet
        if selected and selected not in executed:
            return "continue"
        return "aggregate"

    def _aggregate_results_node(self, state: AgentState) -> AgentState:
        """Node: Aggregate results from multi-agent chain into final response."""
        intermediate = state.get("intermediate_results", [])
        executed = state.get("agents_executed", [])
        
        # If only one agent ran, use its response directly
        if len(intermediate) <= 1:
            return state
        
        # Multiple agents ran - synthesize results
        print(f"[Graph] Aggregating results from {len(intermediate)} agents: {executed}")
        
        # Build synthesis prompt
        results_text = ""
        for result in intermediate:
            results_text += f"\n\n=== {result['agent'].upper()} RESULTS ===\n{result['response']}"
        
        synthesis_prompt = f"""You are synthesizing results from multiple specialist agents to answer this query:

Query: {state['query']}

{results_text}

Provide a unified, coherent response that:
1. Combines insights from all agents
2. Directly answers the user's question
3. Highlights any comparisons or patterns
4. Is well-organized and easy to read

Synthesized Response:"""

        # Use LLM to synthesize
        synthesis = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        
        final_response = f"[Multi-Agent Analysis: {' → '.join(executed)}]\n\n{synthesis.content}"
        
        return {**state, "response": final_response}

    def _update_memory_node(self, state: AgentState) -> AgentState:
        """Node: Update hybrid memory with new exchange."""

        # Add user turn
        user_turn = ConversationTurn(role="user", content=state["query"])
        state = update_memory(state, user_turn, llm=self.llm)

        # Add assistant turn (include which agents were used)
        executed = state.get("agents_executed", [state["selected_agent"]])
        assistant_turn = ConversationTurn(
            role="assistant",
            content=state["response"],
            agent_used=", ".join(executed)
        )
        state = update_memory(state, assistant_turn, llm=self.llm)

        return state

    def get_initial_state(self) -> AgentState:
        """Get fresh initial state."""
        return AgentState(
            query="",
            selected_agent="",
            agent_queue=[],
            intermediate_results=[],
            agents_executed=[],
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
    print("+ Multi-Agent Collaboration")
    print("=" * 50)

    # Default hotel context
    HOTEL_ID = "BKG_177691"
    HOTEL_NAME = "Malmaison London"
    CITY = "London"

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
            print(f"Last agents used: {state.get('agents_executed', [])}")
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
