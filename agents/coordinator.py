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

    # Valid agent IDs for routing
    VALID_AGENTS = {"review_analyst", "competitor_analyst", "benchmark_agent", "market_intel"}

    ROUTING_PROMPT = """Route this query to the appropriate agent(s).

ROUTING RULES:
- Guest feedback/reviews for MY hotel only → review_analyst
- Compare with neighbors/competitors → competitor_analyst
- Feature impact / "Why is my rating X?" / rankings → benchmark_agent
- Events, weather, attractions, external factors → market_intel
- "Full competitive analysis" → competitor_analyst, benchmark_agent

Available: review_analyst, competitor_analyst, benchmark_agent, market_intel

Context: {context}
Query: {query}

RESPOND WITH ONLY agent names separated by commas. NO explanation. Examples:
- review_analyst
- competitor_analyst, benchmark_agent
- market_intel

Your answer:"""

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

    def _extract_agent_names(self, raw_response: str) -> list:
        """
        Robustly extract agent names from LLM response.
        Handles verbose explanations by scanning for valid agent IDs.
        """
        text = raw_response.lower()
        found = []
        
        # First try: simple comma/newline split (ideal case)
        candidates = [a.strip() for a in text.replace("\n", ",").split(",")]
        for c in candidates:
            if c in self.VALID_AGENTS:
                found.append(c)
        
        if found:
            return found
        
        # Second try: scan text for any valid agent name mentions
        # This handles verbose responses like "I would route to benchmark_agent"
        for agent in self.VALID_AGENTS:
            if agent in text:
                found.append(agent)
        
        # Preserve order of appearance in text
        if found:
            positions = [(text.index(a), a) for a in found]
            positions.sort()
            return [a for _, a in positions]
        
        return []

    def _route_node(self, state: AgentState) -> AgentState:
        """Node: Route to appropriate agent(s) - supports multi-agent chains."""
        query = state["query"]
        context = get_context_for_agent(state)
        
        # Truncate context to avoid token overflow
        if len(context) > 500:
            context = context[:500] + "..."

        prompt = self.ROUTING_PROMPT.format(context=context, query=query)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw_response = response.content.strip()
        
        # Robust extraction of agent names
        agent_names = self._extract_agent_names(raw_response)
        
        # Fallback if nothing valid
        if not agent_names:
            print(f"[Graph] Invalid routing '{raw_response[:100]}...', defaulting to review_analyst")
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
        
        # Token budget limits for context
        MAX_CONV_CONTEXT = 800
        MAX_INTERMEDIATE_RESULT = 1000
        MAX_ENHANCED_QUERY = 3000
        
        # Build context including results from previous agents in chain
        context_parts = []
        
        # Add conversation context (truncated)
        conv_context = get_context_for_agent(state)
        if conv_context:
            if len(conv_context) > MAX_CONV_CONTEXT:
                conv_context = conv_context[:MAX_CONV_CONTEXT] + "..."
            context_parts.append(f"[Conversation Context]\n{conv_context}")
        
        # Add intermediate results from previous agents in this chain (truncated)
        intermediate = state.get("intermediate_results", [])
        if intermediate:
            context_parts.append("[Results from Previous Agents]")
            for result in intermediate:
                resp = result['response'][:MAX_INTERMEDIATE_RESULT]
                if len(result['response']) > MAX_INTERMEDIATE_RESULT:
                    resp += "..."
                context_parts.append(f"\n--- {result['agent']} ---\n{resp}")
        
        # Build enhanced query
        if context_parts:
            enhanced_query = "\n\n".join(context_parts) + f"\n\n[Current Question]\n{query}"
        else:
            enhanced_query = query
        
        # Final truncation to keep total under budget
        if len(enhanced_query) > MAX_ENHANCED_QUERY:
            enhanced_query = enhanced_query[:MAX_ENHANCED_QUERY] + "\n... [context truncated]"

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

        # Append chart URLs if benchmark_agent ran (LLM tends to skip them)
        if agent_name == "benchmark_agent" and not error_msg:
            try:
                from databricks_tools import get_chart_urls, clear_chart_urls
                chart_urls = get_chart_urls()
                if chart_urls:
                    response += "\n\n**📈 Analysis Charts:**\n" + "\n".join(chart_urls)
                    clear_chart_urls()
            except ImportError:
                pass  # databricks_tools not available

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
