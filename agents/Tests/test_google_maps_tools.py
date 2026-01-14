"""
Tests for Google Maps Scraping Tools

Tests the separation of concerns between:
- scrape_google_maps_reviews (Review Analyst - extracts review text)
- scrape_google_maps_business (Market Intel - extracts business metadata)
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
def test_imports():
    """Verify all imports work correctly."""
    from agents.utils.google_maps_scraper import (
        scrape_google_maps_reviews,
        scrape_google_maps_business,
        format_reviews_for_agent,
        format_business_for_agent
    )
    print("   [OK] All imports successful")


@test_decorator
def test_reviews_tool_returns_correct_structure():
    """Verify Review Analyst tool returns expected keys."""
    from agents.utils.google_maps_scraper import scrape_google_maps_reviews
    
    # This test doesn't actually scrape - it checks the function signature
    # and that it returns the right structure even on error
    result = scrape_google_maps_reviews("Test Hotel That Does Not Exist 12345")
    
    # Check required keys exist
    required_keys = ["hotel_name", "total_reviews", "average_rating", "reviews", 
                     "source", "scraped_at", "success", "warnings"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    assert result["source"] == "google_maps"
    assert isinstance(result["reviews"], list)
    print(f"   [OK] All required keys present: {required_keys}")


@test_decorator
def test_business_tool_returns_correct_structure():
    """Verify Market Intel tool returns expected keys."""
    from agents.utils.google_maps_scraper import scrape_google_maps_business
    
    result = scrape_google_maps_business("Test Hotel That Does Not Exist 12345")
    
    # Check required keys exist
    required_keys = ["hotel_name", "rating", "review_count", "price_level",
                     "category", "address", "coordinates", "phone", "website",
                     "hours", "source", "scraped_at", "success", "warnings"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    assert result["source"] == "google_maps"
    assert "reviews" not in result, "Business tool should NOT include reviews list"
    print(f"   [OK] All required keys present (no 'reviews' key)")


@test_decorator
def test_business_tool_excludes_review_text():
    """Verify Market Intel tool does NOT include review text."""
    from agents.utils.google_maps_scraper import scrape_google_maps_business
    
    result = scrape_google_maps_business("Malmaison London")
    
    # Should have review_count but NOT reviews list
    assert "review_count" in result, "Should have review_count"
    assert "reviews" not in result, "Should NOT have reviews list"
    
    print(f"   [OK] Business tool correctly excludes review text")
    print(f"   [OK] review_count present: {result.get('review_count', 'N/A')}")


@test_decorator
def test_type_coercion():
    """Verify integer parameters are coerced from strings."""
    from agents.utils.google_maps_scraper import scrape_google_maps_reviews
    
    # This should not raise an error even if max_reviews is a string
    result = scrape_google_maps_reviews("Test Hotel", max_reviews="5")
    
    assert "reviews" in result
    print("   [OK] max_reviews='5' (string) handled correctly")


@test_decorator
def test_format_reviews_for_agent():
    """Verify the reviews formatter produces readable output."""
    from agents.utils.google_maps_scraper import format_reviews_for_agent
    
    # Mock result
    mock_result = {
        "hotel_name": "Test Hotel",
        "total_reviews": 100,
        "average_rating": 4.5,
        "reviews": [
            {"text": "Great hotel!", "rating": 5, "date": "2 days ago", "reviewer_name": "John"},
            {"text": "Nice location", "rating": 4, "date": "1 week ago", "reviewer_name": "Jane"}
        ],
        "source": "google_maps",
        "scraped_at": "2025-01-13T12:00:00Z",
        "success": True,
        "warnings": []
    }
    
    output = format_reviews_for_agent(mock_result)
    
    assert "Test Hotel" in output
    assert "Great hotel!" in output
    assert "Nice location" in output
    assert "John" in output
    print(f"   [OK] Formatted output contains expected content")
    print(f"   Preview: {output[:100]}...")


@test_decorator
def test_format_business_for_agent():
    """Verify the business formatter produces readable output."""
    from agents.utils.google_maps_scraper import format_business_for_agent
    
    # Mock result
    mock_result = {
        "hotel_name": "Test Hotel",
        "rating": 4.5,
        "review_count": 500,
        "price_level": "£££",
        "category": "4-star hotel",
        "address": "123 Test Street",
        "coordinates": {"lat": 51.5, "lng": -0.1},
        "phone": "+44 123 456 7890",
        "website": "https://test.com",
        "hours": {"raw": "Open 24 hours"},
        "source": "google_maps",
        "scraped_at": "2025-01-13T12:00:00Z",
        "success": True,
        "warnings": []
    }
    
    output = format_business_for_agent(mock_result)
    
    assert "Test Hotel" in output
    assert "4.5" in output
    assert "500" in output
    assert "123 Test Street" in output
    assert "reviews" not in output.lower() or "review count" in output.lower()
    print(f"   [OK] Formatted output contains business metadata")
    print(f"   Preview: {output[:100]}...")


@test_decorator
def test_review_analyst_uses_new_tool():
    """Verify Review Analyst agent uses the refactored tool."""
    from review_analyst import ReviewAnalystAgent
    
    agent = ReviewAnalystAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    tools = agent.get_tools()
    tool_names = [t.__name__ for t in tools]
    
    assert "scrape_google_maps_reviews" in tool_names
    print(f"   [OK] Review Analyst has scrape_google_maps_reviews tool")


@test_decorator
def test_market_intel_uses_new_tool():
    """Verify Market Intel agent uses the refactored tool."""
    from market_intel import MarketIntelAgent
    
    agent = MarketIntelAgent(
        hotel_id="BKG_177691",
        hotel_name="Malmaison London",
        city="London"
    )
    
    tools = agent.get_tools()
    tool_names = [t.__name__ for t in tools]
    
    assert "scrape_google_maps_business" in tool_names
    assert "scrape_google_maps_reviews" not in tool_names, "Market Intel should NOT have reviews tool"
    print(f"   [OK] Market Intel has scrape_google_maps_business tool")
    print(f"   [OK] Market Intel does NOT have reviews tool")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("GOOGLE MAPS TOOLS TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_reviews_tool_returns_correct_structure,
        test_business_tool_returns_correct_structure,
        test_business_tool_excludes_review_text,
        test_type_coercion,
        test_format_reviews_for_agent,
        test_format_business_for_agent,
        test_review_analyst_uses_new_tool,
        test_market_intel_uses_new_tool,
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
