"""
User Testing Framework

Interactive testing framework for evaluating the Hotel Intelligence System.
Supports:
- Predefined test scenarios
- Interactive testing mode
- Feedback collection
- Metrics tracking
- Report generation
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

# Add paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))


# ===========================================
# DATA CLASSES
# ===========================================

@dataclass
class UserFeedback:
    """Feedback from a single test interaction."""
    query: str
    response: str
    elapsed_seconds: float
    agent_used: str
    
    # User ratings (1-5 scale)
    relevance_rating: Optional[int] = None  # Did it answer the question?
    accuracy_rating: Optional[int] = None  # Was the information correct?
    actionability_rating: Optional[int] = None  # Were insights useful/actionable?
    speed_rating: Optional[int] = None  # Was response time acceptable?
    overall_rating: Optional[int] = None  # Overall satisfaction
    
    # Additional feedback
    would_use_again: Optional[bool] = None
    comments: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TestScenario:
    """A predefined test scenario."""
    id: str
    name: str
    description: str
    query: str
    expected_agent: str
    category: str  # review_analysis, competitor, benchmark, market_intel
    difficulty: str  # easy, medium, hard


@dataclass  
class TestSession:
    """A complete testing session."""
    session_id: str
    tester_name: str
    tester_role: str  # property_owner, analyst, manager, other
    hotel_id: str
    hotel_name: str
    city: str
    feedbacks: list = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


# ===========================================
# TEST SCENARIOS
# ===========================================

PREDEFINED_SCENARIOS = [
    # Review Analysis Scenarios
    TestScenario(
        id="RA001",
        name="WiFi Feedback Analysis",
        description="Analyze what guests are saying about WiFi quality",
        query="What are guests saying about wifi quality?",
        expected_agent="review_analyst",
        category="review_analysis",
        difficulty="easy"
    ),
    TestScenario(
        id="RA002",
        name="Cleanliness Reviews",
        description="Find feedback about room cleanliness",
        query="Show me reviews mentioning cleanliness or hygiene",
        expected_agent="review_analyst",
        category="review_analysis",
        difficulty="easy"
    ),
    TestScenario(
        id="RA003",
        name="Breakfast Feedback",
        description="Analyze breakfast service feedback",
        query="What do guests think about the breakfast?",
        expected_agent="review_analyst",
        category="review_analysis",
        difficulty="easy"
    ),
    TestScenario(
        id="RA004",
        name="Staff Service Analysis",
        description="Analyze feedback about staff and service quality",
        query="How do guests rate our staff and service?",
        expected_agent="review_analyst",
        category="review_analysis",
        difficulty="medium"
    ),
    
    # Competitor Analysis Scenarios
    TestScenario(
        id="CA001",
        name="Competitor Comparison",
        description="Compare reviews with competitors",
        query="How do my reviews compare to competitors?",
        expected_agent="competitor_analyst",
        category="competitor",
        difficulty="medium"
    ),
    TestScenario(
        id="CA002",
        name="Weakness Analysis",
        description="Find weaknesses compared to neighbors",
        query="What are my weaknesses compared to similar properties?",
        expected_agent="competitor_analyst",
        category="competitor",
        difficulty="medium"
    ),
    TestScenario(
        id="CA003",
        name="Strength Analysis",
        description="Identify competitive advantages",
        query="What are my strengths compared to competitors?",
        expected_agent="competitor_analyst",
        category="competitor",
        difficulty="medium"
    ),
    
    # Benchmark Scenarios
    TestScenario(
        id="BM001",
        name="Rating Comparison",
        description="Compare rating with other hotels",
        query="How does my rating compare to other hotels in the city?",
        expected_agent="benchmark_agent",
        category="benchmark",
        difficulty="easy"
    ),
    TestScenario(
        id="BM002",
        name="Feature Impact Analysis",
        description="Understand what affects rating",
        query="Why is my rating lower than competitors? What can I improve?",
        expected_agent="benchmark_agent",
        category="benchmark",
        difficulty="hard"
    ),
    TestScenario(
        id="BM003",
        name="Market Ranking",
        description="See market position",
        query="Rank hotels by rating in my city",
        expected_agent="benchmark_agent",
        category="benchmark",
        difficulty="easy"
    ),
    
    # Market Intel Scenarios
    TestScenario(
        id="MI001",
        name="Event Search",
        description="Find upcoming events in the area",
        query="Are there any major events happening this weekend?",
        expected_agent="market_intel",
        category="market_intel",
        difficulty="easy"
    ),
    TestScenario(
        id="MI002",
        name="Weather Forecast",
        description="Get weather information",
        query="What's the weather forecast for next week?",
        expected_agent="market_intel",
        category="market_intel",
        difficulty="easy"
    ),
    TestScenario(
        id="MI003",
        name="Local Attractions",
        description="Find nearby attractions",
        query="What attractions are near my hotel?",
        expected_agent="market_intel",
        category="market_intel",
        difficulty="medium"
    ),
    
    # Multi-Agent Scenarios
    TestScenario(
        id="MA001",
        name="Full Competitive Analysis",
        description="Comprehensive comparison with competitors",
        query="Give me a full competitive analysis",
        expected_agent="competitor_analyst,benchmark_agent",
        category="multi_agent",
        difficulty="hard"
    ),
    TestScenario(
        id="MA002",
        name="Improvement Recommendations",
        description="Get actionable improvement suggestions",
        query="What should I improve to increase my rating?",
        expected_agent="benchmark_agent",
        category="multi_agent",
        difficulty="hard"
    ),
]


# ===========================================
# USER TESTING FRAMEWORK
# ===========================================

class UserTestingFramework:
    """
    Framework for conducting user testing sessions.
    """
    
    def __init__(self, hotel_id: str, hotel_name: str, city: str):
        from coordinator import LangGraphCoordinator
        
        self.hotel_id = hotel_id
        self.hotel_name = hotel_name
        self.city = city
        
        print("\n" + "="*60)
        print("USER TESTING FRAMEWORK")
        print("="*60)
        print(f"Hotel: {hotel_name} ({hotel_id})")
        print(f"City: {city}")
        
        print("\nInitializing system...")
        self.coordinator = LangGraphCoordinator(hotel_id, hotel_name, city)
        self.state = self.coordinator.get_initial_state()
        
        print("[OK] System ready for testing")
        
        # Current session
        self.current_session: Optional[TestSession] = None
        self.all_sessions: list[TestSession] = []
    
    def start_session(self, tester_name: str, tester_role: str) -> TestSession:
        """Start a new testing session."""
        session_id = f"TS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = TestSession(
            session_id=session_id,
            tester_name=tester_name,
            tester_role=tester_role,
            hotel_id=self.hotel_id,
            hotel_name=self.hotel_name,
            city=self.city
        )
        
        # Reset coordinator state for new session
        self.state = self.coordinator.get_initial_state()
        
        print(f"\n[OK] Session started: {session_id}")
        print(f"   Tester: {tester_name} ({tester_role})")
        
        return self.current_session
    
    def run_query(self, query: str) -> tuple[str, float, str]:
        """
        Run a query and return response, time, and agent used.
        """
        start_time = time.time()
        
        response, self.state = self.coordinator.run(query, self.state)
        
        elapsed = time.time() - start_time
        agents_used = self.state.get("agents_executed", [])
        agent_str = ", ".join(agents_used) if agents_used else "unknown"
        
        return response, elapsed, agent_str
    
    def collect_feedback(self, query: str, response: str, elapsed: float, 
                        agent_used: str) -> UserFeedback:
        """
        Collect feedback from user for a single interaction.
        Returns UserFeedback object.
        """
        feedback = UserFeedback(
            query=query,
            response=response,
            elapsed_seconds=elapsed,
            agent_used=agent_used
        )
        
        print("\n" + "-"*40)
        print("FEEDBACK COLLECTION")
        print("-"*40)
        print("\nPlease rate the following (1-5, or press Enter to skip):")
        
        # Collect ratings
        try:
            feedback.relevance_rating = self._get_rating("Relevance (Did it answer your question?)")
            feedback.accuracy_rating = self._get_rating("Accuracy (Was the information correct?)")
            feedback.actionability_rating = self._get_rating("Actionability (Were insights useful?)")
            feedback.speed_rating = self._get_rating(f"Speed ({elapsed:.1f}s - Was this acceptable?)")
            feedback.overall_rating = self._get_rating("Overall Satisfaction")
            
            # Would use again
            use_again = input("\nWould you use this system again? (y/n): ").strip().lower()
            feedback.would_use_again = use_again == 'y'
            
            # Comments
            feedback.comments = input("\nAny additional comments? (or press Enter to skip): ").strip()
            
        except (KeyboardInterrupt, EOFError):
            print("\n[WARN] Feedback collection cancelled")
        
        return feedback
    
    def _get_rating(self, prompt: str) -> Optional[int]:
        """Get a 1-5 rating from user."""
        try:
            response = input(f"  {prompt} (1-5): ").strip()
            if response == "":
                return None
            rating = int(response)
            if 1 <= rating <= 5:
                return rating
            else:
                print("    (Invalid rating, skipping)")
                return None
        except (ValueError, KeyboardInterrupt):
            return None
    
    def run_scenario(self, scenario: TestScenario, collect_feedback: bool = True) -> UserFeedback:
        """
        Run a predefined test scenario.
        """
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.name} ({scenario.id})")
        print(f"Category: {scenario.category} | Difficulty: {scenario.difficulty}")
        print("="*60)
        print(f"\nDescription: {scenario.description}")
        print(f"Query: {scenario.query}")
        print(f"Expected Agent: {scenario.expected_agent}")
        
        input("\nPress Enter to run this scenario...")
        
        # Run query
        print("\n[Running query...]")
        response, elapsed, agent_used = self.run_query(scenario.query)
        
        # Display response
        print(f"\n[Response] (Agent: {agent_used}, Time: {elapsed:.2f}s)")
        print("-"*40)
        print(response[:2000])
        if len(response) > 2000:
            print("...[truncated]")
        print("-"*40)
        
        # Check if correct agent was used
        expected_agents = [a.strip() for a in scenario.expected_agent.split(",")]
        actual_agents = [a.strip() for a in agent_used.split(",")]
        routing_correct = any(a in expected_agents for a in actual_agents)
        
        print(f"\n✓ Routing: {'CORRECT' if routing_correct else 'DIFFERENT'}")
        print(f"  Expected: {scenario.expected_agent}")
        print(f"  Actual: {agent_used}")
        
        # Collect feedback
        feedback = UserFeedback(
            query=scenario.query,
            response=response,
            elapsed_seconds=elapsed,
            agent_used=agent_used
        )
        
        if collect_feedback:
            feedback = self.collect_feedback(scenario.query, response, elapsed, agent_used)
        
        # Add to session
        if self.current_session:
            self.current_session.feedbacks.append(feedback.to_dict())
        
        return feedback
    
    def run_interactive_mode(self):
        """
        Run interactive testing mode - user enters custom queries.
        """
        print("\n" + "="*60)
        print("INTERACTIVE TESTING MODE")
        print("="*60)
        print("Enter your queries to test the system.")
        print("Commands: 'quit' to exit, 'feedback' to provide feedback")
        print("="*60)
        
        last_response = None
        last_elapsed = None
        last_agent = None
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query.lower() == 'feedback' and last_response:
                    feedback = self.collect_feedback(
                        self.state.get("query", ""),
                        last_response,
                        last_elapsed,
                        last_agent
                    )
                    if self.current_session:
                        self.current_session.feedbacks.append(feedback.to_dict())
                    print("[OK] Feedback recorded")
                    continue
                
                if not query:
                    continue
                
                # Run query
                response, elapsed, agent_used = self.run_query(query)
                
                print(f"\nAgent ({agent_used}, {elapsed:.2f}s): {response[:1500]}")
                if len(response) > 1500:
                    print("...[truncated]")
                
                last_response = response
                last_elapsed = elapsed
                last_agent = agent_used
                
            except KeyboardInterrupt:
                print("\n\n[WARN] Testing interrupted")
                break
    
    def run_scenario_suite(self, categories: list = None, collect_feedback: bool = True) -> list[UserFeedback]:
        """
        Run a suite of test scenarios.
        
        Args:
            categories: List of categories to test (None = all)
            collect_feedback: Whether to collect feedback after each scenario
        """
        scenarios = PREDEFINED_SCENARIOS
        
        if categories:
            scenarios = [s for s in scenarios if s.category in categories]
        
        print("\n" + "="*60)
        print("SCENARIO TEST SUITE")
        print("="*60)
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Categories: {list(set(s.category for s in scenarios))}")
        
        feedbacks = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n\n[{i}/{len(scenarios)}] ", end="")
            
            try:
                feedback = self.run_scenario(scenario, collect_feedback)
                feedbacks.append(feedback)
            except KeyboardInterrupt:
                print("\n[WARN] Suite interrupted")
                break
        
        return feedbacks
    
    def end_session(self) -> TestSession:
        """End the current testing session."""
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()
            self.all_sessions.append(self.current_session)
            
            print(f"\n[OK] Session ended: {self.current_session.session_id}")
            print(f"   Total interactions: {len(self.current_session.feedbacks)}")
            
            return self.current_session
        return None
    
    def generate_session_report(self, session: TestSession = None) -> str:
        """Generate a report for a testing session."""
        session = session or self.current_session
        
        if not session:
            return "No session to report"
        
        report = []
        report.append("=" * 60)
        report.append("USER TESTING SESSION REPORT")
        report.append("=" * 60)
        report.append(f"Session ID: {session.session_id}")
        report.append(f"Tester: {session.tester_name} ({session.tester_role})")
        report.append(f"Hotel: {session.hotel_name} ({session.hotel_id})")
        report.append(f"Start: {session.start_time}")
        report.append(f"End: {session.end_time or 'Ongoing'}")
        report.append(f"Total Interactions: {len(session.feedbacks)}")
        report.append("")
        
        # Calculate averages
        ratings = {
            "relevance": [],
            "accuracy": [],
            "actionability": [],
            "speed": [],
            "overall": []
        }
        
        would_use_again = []
        response_times = []
        
        for fb in session.feedbacks:
            if fb.get("relevance_rating"):
                ratings["relevance"].append(fb["relevance_rating"])
            if fb.get("accuracy_rating"):
                ratings["accuracy"].append(fb["accuracy_rating"])
            if fb.get("actionability_rating"):
                ratings["actionability"].append(fb["actionability_rating"])
            if fb.get("speed_rating"):
                ratings["speed"].append(fb["speed_rating"])
            if fb.get("overall_rating"):
                ratings["overall"].append(fb["overall_rating"])
            if fb.get("would_use_again") is not None:
                would_use_again.append(fb["would_use_again"])
            if fb.get("elapsed_seconds"):
                response_times.append(fb["elapsed_seconds"])
        
        report.append("--- AVERAGE RATINGS (1-5) ---")
        for metric, values in ratings.items():
            if values:
                avg = sum(values) / len(values)
                report.append(f"  {metric.title()}: {avg:.2f} (n={len(values)})")
            else:
                report.append(f"  {metric.title()}: N/A")
        
        report.append("")
        report.append("--- OTHER METRICS ---")
        
        if would_use_again:
            yes_count = sum(1 for w in would_use_again if w)
            report.append(f"  Would Use Again: {yes_count}/{len(would_use_again)} ({100*yes_count/len(would_use_again):.0f}%)")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            report.append(f"  Avg Response Time: {avg_time:.2f}s")
            report.append(f"  Min/Max Time: {min(response_times):.2f}s / {max(response_times):.2f}s")
        
        # Comments
        comments = [fb.get("comments") for fb in session.feedbacks if fb.get("comments")]
        if comments:
            report.append("")
            report.append("--- USER COMMENTS ---")
            for i, comment in enumerate(comments, 1):
                report.append(f"  [{i}] {comment}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_session(self, session: TestSession = None, filepath: str = None):
        """Save session data to JSON file."""
        session = session or self.current_session
        
        if not session:
            print("No session to save")
            return None
        
        if filepath is None:
            filepath = os.path.join(
                _PROJECT_ROOT, 
                f"user_testing_session_{session.session_id}.json"
            )
        
        with open(filepath, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
        
        print(f"[OK] Session saved to: {filepath}")
        return filepath


# ===========================================
# QUICK AUTOMATED TESTING (No User Input)
# ===========================================

def run_automated_test(hotel_id: str = "BKG_177691", 
                       hotel_name: str = "Malmaison London",
                       city: str = "London",
                       categories: list = None):
    """
    Run automated testing without user input.
    Good for CI/CD or quick validation.
    """
    framework = UserTestingFramework(hotel_id, hotel_name, city)
    framework.start_session("Automated Tester", "automated")
    
    scenarios = PREDEFINED_SCENARIOS
    if categories:
        scenarios = [s for s in scenarios if s.category in categories]
    
    results = []
    
    print(f"\n[TEST] Running automated test on {len(scenarios)} scenarios...")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario.name}...")
        
        try:
            response, elapsed, agent_used = framework.run_query(scenario.query)
            
            # Check routing
            expected_agents = [a.strip() for a in scenario.expected_agent.split(",")]
            actual_agents = [a.strip() for a in agent_used.split(",")]
            routing_correct = any(a in expected_agents for a in actual_agents)
            
            # Explicit failure markers - only these count as failures
            FAILURE_MARKERS = [
                "agent execution failed",
                "tool call validation failed", 
                "unknown tool",
                "failed to call a function",
            ]
            response_lower = response.lower()
            has_failure = any(marker in response_lower for marker in FAILURE_MARKERS)
            
            results.append({
                "scenario_id": scenario.id,
                "scenario_name": scenario.name,
                "query": scenario.query,
                "elapsed_seconds": elapsed,
                "agent_used": agent_used,
                "expected_agent": scenario.expected_agent,
                "routing_correct": routing_correct,
                "response_length": len(response),
                "success": len(response) > 50 and not has_failure
            })
            
            status = "[OK]" if routing_correct else "[WARN]"
            print(f"   {status} {agent_used} ({elapsed:.2f}s)")
            
        except Exception as e:
            results.append({
                "scenario_id": scenario.id,
                "scenario_name": scenario.name,
                "query": scenario.query,
                "error": str(e),
                "success": False
            })
            print(f"   [ERROR] Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("AUTOMATED TEST SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r.get("success", False))
    routing_correct = sum(1 for r in results if r.get("routing_correct", False))
    avg_time = sum(r.get("elapsed_seconds", 0) for r in results) / len(results)
    
    print(f"Success Rate: {successful}/{len(results)} ({100*successful/len(results):.0f}%)")
    print(f"Routing Accuracy: {routing_correct}/{len(results)} ({100*routing_correct/len(results):.0f}%)")
    print(f"Average Response Time: {avg_time:.2f}s")
    
    # Save results
    filepath = os.path.join(_PROJECT_ROOT, "automated_test_results.json")
    with open(filepath, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "hotel_id": hotel_id,
            "hotel_name": hotel_name,
            "city": city,
            "summary": {
                "total_tests": len(results),
                "successful": successful,
                "routing_correct": routing_correct,
                "avg_response_time": avg_time
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    return results


# ===========================================
# MAIN
# ===========================================

def main():
    """Main entry point for user testing."""
    print("\n" + "="*60)
    print("HOTEL INTELLIGENCE SYSTEM - USER TESTING")
    print("="*60)
    print("\nModes:")
    print("  1. Interactive Testing (custom queries)")
    print("  2. Scenario Suite (predefined tests)")
    print("  3. Automated Testing (no user input)")
    print("  4. Quick Single Query")
    
    mode = input("\nSelect mode (1-4): ").strip()
    
    # Get hotel context
    hotel_id = input("Hotel ID (default: BKG_177691): ").strip() or "BKG_177691"
    hotel_name = input("Hotel Name (default: Malmaison London): ").strip() or "Malmaison London"
    city = input("City (default: London): ").strip() or "London"
    
    if mode == "1":
        # Interactive mode
        framework = UserTestingFramework(hotel_id, hotel_name, city)
        
        tester_name = input("Your name: ").strip() or "Anonymous"
        tester_role = input("Your role (owner/analyst/manager/other): ").strip() or "other"
        
        framework.start_session(tester_name, tester_role)
        framework.run_interactive_mode()
        
        session = framework.end_session()
        print(framework.generate_session_report(session))
        framework.save_session(session)
        
    elif mode == "2":
        # Scenario suite
        framework = UserTestingFramework(hotel_id, hotel_name, city)
        
        tester_name = input("Your name: ").strip() or "Anonymous"
        tester_role = input("Your role (owner/analyst/manager/other): ").strip() or "other"
        
        print("\nCategories: review_analysis, competitor, benchmark, market_intel, multi_agent")
        categories_input = input("Categories to test (comma-separated, or 'all'): ").strip()
        categories = None if categories_input == 'all' else [c.strip() for c in categories_input.split(",")]
        
        framework.start_session(tester_name, tester_role)
        framework.run_scenario_suite(categories=categories, collect_feedback=True)
        
        session = framework.end_session()
        print(framework.generate_session_report(session))
        framework.save_session(session)
        
    elif mode == "3":
        # Automated testing
        run_automated_test(hotel_id, hotel_name, city)
        
    elif mode == "4":
        # Quick single query
        framework = UserTestingFramework(hotel_id, hotel_name, city)
        query = input("Enter your query: ").strip()
        
        response, elapsed, agent = framework.run_query(query)
        print(f"\n[Agent: {agent}, Time: {elapsed:.2f}s]")
        print(response)
    
    else:
        print("Invalid mode")


if __name__ == "__main__":
    main()
