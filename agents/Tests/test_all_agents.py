"""
Test All Agents

Quick verification that all specialist agents initialize and respond correctly.
Tests anti-hallucination prompts and output validation.
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


def test_decorator(func):
    """Decorator to run tests with error handling."""
    def wrapper():
        print(f"\n{'='*60}")
        print(f"TEST: {func.__name__.replace('_', ' ').title()}")
        print('='*60)
        try:
            func()
            print(f"[PASS] {func.__name__}")
            return True
        except Exception as e:
            print(f"[FAIL] {func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    return wrapper


@test_decorator
def test_review_analyst_init():
    """Test ReviewAnalystAgent initialization and anti-hallucination prompt."""
    from review_analyst import ReviewAnalystAgent
    
    agent = ReviewAnalystAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    prompt = agent.get_system_prompt()
    
    # Check anti-hallucination elements
    assert "STRICT RULES" in prompt, "Missing STRICT RULES section"
    assert "NO HALLUCINATIONS" in prompt, "Missing NO HALLUCINATIONS"
    assert "ONLY state facts" in prompt or "ONLY" in prompt, "Missing fact-only rule"
    
    print(f"   [OK] Agent initialized with {len(agent.get_tools())} tools")
    print(f"   [OK] Anti-hallucination prompt present")
    print(f"   [OK] validate_output={agent.validate_output}")


@test_decorator
def test_competitor_analyst_init():
    """Test CompetitorAnalystAgent initialization and anti-hallucination prompt."""
    from competitor_analyst import CompetitorAnalystAgent
    
    agent = CompetitorAnalystAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    prompt = agent.get_system_prompt()
    
    # Check anti-hallucination elements
    assert "STRICT RULES" in prompt, "Missing STRICT RULES section"
    assert "NO HALLUCINATIONS" in prompt, "Missing NO HALLUCINATIONS"
    assert "ONLY list competitors" in prompt, "Missing competitors-only rule"
    
    print(f"   [OK] Agent initialized with {len(agent.get_tools())} tools")
    print(f"   [OK] Anti-hallucination prompt present")
    print(f"   [OK] validate_output={agent.validate_output}")


@test_decorator
def test_market_intel_init():
    """Test MarketIntelAgent initialization and anti-hallucination prompt."""
    from market_intel import MarketIntelAgent
    
    agent = MarketIntelAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    prompt = agent.get_system_prompt()
    
    # Check anti-hallucination elements
    assert "STRICT RULES" in prompt, "Missing STRICT RULES section"
    assert "NO HALLUCINATIONS" in prompt, "Missing NO HALLUCINATIONS"
    assert "ONLY report information" in prompt, "Missing info-only rule"
    
    print(f"   [OK] Agent initialized with {len(agent.get_tools())} tools")
    print(f"   [OK] Anti-hallucination prompt present")
    print(f"   [OK] validate_output={agent.validate_output}")


@test_decorator
def test_benchmark_agent_init():
    """Test BenchmarkAgent initialization and anti-hallucination prompt."""
    from benchmark_agent import BenchmarkAgent
    
    agent = BenchmarkAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    prompt = agent.get_system_prompt()
    
    # Check anti-hallucination elements
    assert "STRICT RULES" in prompt, "Missing STRICT RULES section"
    assert "NO HALLUCINATIONS" in prompt, "Missing NO HALLUCINATIONS"
    assert "ONLY report numbers" in prompt, "Missing numbers-only rule"
    
    print(f"   [OK] Agent initialized with {len(agent.get_tools())} tools")
    print(f"   [OK] Anti-hallucination prompt present")
    print(f"   [OK] validate_output={agent.validate_output}")


@test_decorator
def test_output_validator():
    """Test OutputValidator utility."""
    from utils.output_validator import validate_response, OutputValidator, ConfidenceLevel
    
    # Test 1: Valid response with quotes
    tool_outputs = [
        "Review 1: The wifi was excellent and fast.",
        "Review 2: Great breakfast buffet."
    ]
    response = 'From the reviews: "The wifi was excellent and fast." Guests also mentioned great breakfast.'
    
    result = validate_response(response, tool_outputs)
    print(f"   [OK] Validation result: risk={result.hallucination_risk:.0%}, valid={result.is_valid}")
    
    # Test 2: Potentially hallucinated response
    bad_response = "Many guests complain about the slow wifi and noisy rooms."
    bad_result = validate_response(bad_response, tool_outputs)
    print(f"   [OK] Bad response risk: {bad_result.hallucination_risk:.0%}")
    assert bad_result.hallucination_risk > result.hallucination_risk, "Bad response should have higher risk"
    
    # Test 3: Check warnings
    print(f"   [OK] Warnings detected: {len(bad_result.warnings)}")
    for w in bad_result.warnings[:2]:
        print(f"       - {w[:60]}...")


@test_decorator  
def test_validation_in_base_agent():
    """Test that BaseAgent has validation integrated."""
    from base_agent import BaseAgent, VALIDATION_AVAILABLE
    
    print(f"   [OK] VALIDATION_AVAILABLE = {VALIDATION_AVAILABLE}")
    
    # Check that agent collects tool outputs
    from review_analyst import ReviewAnalystAgent
    agent = ReviewAnalystAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London",
        validate_output=True,
        strict_validation=False
    )
    
    assert hasattr(agent, '_tool_outputs'), "Agent missing _tool_outputs attribute"
    assert hasattr(agent, 'validate_output'), "Agent missing validate_output attribute"
    assert agent.validate_output == True, "Validation should be enabled"
    
    print(f"   [OK] BaseAgent has validation hooks")


@test_decorator
def test_competitor_analyst_tools():
    """Test CompetitorAnalystAgent tool execution (without LLM)."""
    from competitor_analyst import CompetitorAnalystAgent
    
    agent = CompetitorAnalystAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    # Test geo search (uses RAG - may return no results if DB empty)
    result = agent.find_competitors_geo(city="London", k=3)
    print(f"   [OK] find_competitors_geo: {result[:80]}...")
    
    # Test ML search (currently returns empty, falls back to geo)
    result2 = agent.find_competitors_similar(k=3)
    print(f"   [OK] find_competitors_similar: {result2[:80]}...")


@test_decorator
def test_benchmark_agent_tools():
    """Test BenchmarkAgent tool execution (without LLM)."""
    from benchmark_agent import BenchmarkAgent
    
    agent = BenchmarkAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    # Test get my hotel data
    result = agent.get_my_hotel_data()
    print(f"   [OK] get_my_hotel_data: {result[:80]}...")
    
    # Test compare metric
    result2 = agent.compare_metric(metric="rating", k=3)
    print(f"   [OK] compare_metric(rating): {result2[:80]}...")


@test_decorator
def test_multi_agent_state():
    """Test that AgentState has multi-agent fields."""
    from graph_state import AgentState
    
    # Check that new fields exist in TypedDict annotations
    annotations = AgentState.__annotations__
    
    assert "agent_queue" in annotations, "Missing agent_queue field"
    assert "intermediate_results" in annotations, "Missing intermediate_results field"
    assert "agents_executed" in annotations, "Missing agents_executed field"
    
    print(f"   [OK] agent_queue field present")
    print(f"   [OK] intermediate_results field present")
    print(f"   [OK] agents_executed field present")


@test_decorator
def test_coordinator_multi_agent_graph():
    """Test that LangGraphCoordinator has multi-agent workflow nodes."""
    from coordinator import LangGraphCoordinator
    
    coordinator = LangGraphCoordinator(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    # Check that coordinator has the right methods
    assert hasattr(coordinator, '_execute_agent_node'), "Missing _execute_agent_node"
    assert hasattr(coordinator, '_check_queue_node'), "Missing _check_queue_node"
    assert hasattr(coordinator, '_aggregate_results_node'), "Missing _aggregate_results_node"
    assert hasattr(coordinator, '_should_continue'), "Missing _should_continue"
    
    print(f"   [OK] _execute_agent_node method present")
    print(f"   [OK] _check_queue_node method present")
    print(f"   [OK] _aggregate_results_node method present")
    print(f"   [OK] _should_continue method present")
    
    # Test initial state has multi-agent fields
    state = coordinator.get_initial_state()
    assert "agent_queue" in state, "Initial state missing agent_queue"
    assert "intermediate_results" in state, "Initial state missing intermediate_results"
    assert "agents_executed" in state, "Initial state missing agents_executed"
    assert state["agent_queue"] == [], "agent_queue should be empty list"
    
    print(f"   [OK] Initial state has multi-agent fields")
    
    # Test routing prompt mentions multi-agent
    assert "MULTI-AGENT" in coordinator.ROUTING_PROMPT, "Routing prompt should mention MULTI-AGENT"
    assert "comma-separated" in coordinator.ROUTING_PROMPT.lower(), "Routing prompt should mention comma-separated"
    
    print(f"   [OK] Routing prompt supports multi-agent")


@test_decorator
def test_should_continue_logic():
    """Test the _should_continue conditional logic."""
    from coordinator import LangGraphCoordinator
    
    coordinator = LangGraphCoordinator(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    # Test with no selected agent - should aggregate
    state_empty = {"selected_agent": "", "agents_executed": []}
    result = coordinator._should_continue(state_empty)
    assert result == "aggregate", f"No selected agent should return 'aggregate', got '{result}'"
    print(f"   [OK] No selected agent -> 'aggregate'")
    
    # Test with selected agent already executed - should aggregate
    state_executed = {"selected_agent": "review_analyst", "agents_executed": ["review_analyst"]}
    result = coordinator._should_continue(state_executed)
    assert result == "aggregate", f"Agent already executed should return 'aggregate', got '{result}'"
    print(f"   [OK] Agent already executed -> 'aggregate'")
    
    # Test with selected agent NOT yet executed - should continue
    state_pending = {"selected_agent": "benchmark_agent", "agents_executed": ["review_analyst"]}
    result = coordinator._should_continue(state_pending)
    assert result == "continue", f"Agent pending execution should return 'continue', got '{result}'"
    print(f"   [OK] Agent pending execution -> 'continue'")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("ALL AGENTS TEST SUITE")
    print("="*60)
    
    tests = [
        test_review_analyst_init,
        test_competitor_analyst_init,
        test_market_intel_init,
        test_benchmark_agent_init,
        test_output_validator,
        test_validation_in_base_agent,
        test_competitor_analyst_tools,
        test_benchmark_agent_tools,
        test_multi_agent_state,
        test_coordinator_multi_agent_graph,
        test_should_continue_logic,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    failed = len(results) - passed
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"[PASS] Passed:  {passed}")
    print(f"[FAIL] Failed:  {failed}")
    print("="*60)
    
    if failed == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[WARN] {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
