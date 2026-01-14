"""
Google Maps Scraper Module

Provides two purpose-specific tools for scraping Google Maps data:
- scrape_google_maps_reviews: For Review Analyst (extracts review text)
- scrape_google_maps_business: For Market Intel (extracts business metadata)

Both tools use a shared base scraper to avoid code duplication.
"""

import urllib.parse
import random
import logging
from datetime import datetime, timezone
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def _scrape_google_maps_raw(query: str, extract_reviews: bool = True, max_reviews: int = 10) -> dict:
    """
    Base scraper that extracts all data from a Google Maps place page.
    
    This is a private helper function. Use the public tools instead:
    - scrape_google_maps_reviews() for Review Analyst
    - scrape_google_maps_business() for Market Intel
    
    Args:
        query: Hotel name and location to search
        extract_reviews: Whether to extract individual review text
        max_reviews: Maximum number of reviews to extract (if extract_reviews=True)
    
    Returns:
        dict with all available data from the Google Maps place page
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "error": "Playwright not installed. Run: pip install playwright && playwright install chromium",
            "success": False
        }
    
    result = {
        "success": False,
        "name": "",
        "rating": None,
        "review_count": 0,
        "reviews": [],
        "address": "",
        "phone": "",
        "website": "",
        "category": "",
        "price_level": "",
        "hours": {},
        "coordinates": {"lat": None, "lng": None},
        "nearby_pois": [],
        "popular_times": {},
        "photos_count": 0,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": "google_maps",
        "warnings": []
    }
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.google.com/maps/search/{encoded_query}"
    
    logger.info(f"Scraping Google Maps: {query}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=random.choice(USER_AGENTS),
                locale="en-US"
            )
            page = context.new_page()
            
            # Navigate to search results
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Handle cookie consent (EU regions)
            _handle_cookie_consent(page)
            
            # Random delay to appear more human
            page.wait_for_timeout(random.randint(2000, 3500))
            
            # Click first result to get to place page
            try:
                first_result = page.locator('a[href*="maps/place"]').first
                if first_result.is_visible(timeout=3000):
                    first_result.click()
                    page.wait_for_timeout(random.randint(2000, 3000))
            except Exception as e:
                logger.debug(f"Could not click first result: {e}")
            
            # Extract basic info
            result["name"] = _extract_name(page)
            result["rating"] = _extract_rating(page)
            result["review_count"] = _extract_review_count(page)
            result["address"] = _extract_address(page)
            result["phone"] = _extract_phone(page)
            result["website"] = _extract_website(page)
            result["category"] = _extract_category(page)
            result["price_level"] = _extract_price_level(page)
            result["hours"] = _extract_hours(page)
            
            # Extract coordinates from URL
            result["coordinates"] = _extract_coordinates(page.url)
            
            # Extract nearby POIs (from "Explore nearby" section)
            result["nearby_pois"] = _extract_nearby_pois(page)
            
            # Extract reviews if requested
            if extract_reviews:
                result["reviews"] = _extract_reviews(page, max_reviews)
            
            result["success"] = True
            browser.close()
            
    except Exception as e:
        result["error"] = str(e)
        result["warnings"].append(f"Scraping error: {e}")
        logger.error(f"Google Maps scraping error: {e}")
    
    return result


def _handle_cookie_consent(page) -> None:
    """Handle cookie consent dialogs."""
    try:
        # Try various button texts
        for text in ["Reject all", "Accept all", "I agree"]:
            try:
                btn = page.get_by_role("button", name=text)
                if btn.is_visible(timeout=1500):
                    btn.click()
                    page.wait_for_timeout(500)
                    return
            except:
                continue
    except:
        pass


def _extract_name(page) -> str:
    """Extract place name."""
    try:
        # Try multiple selectors
        for selector in ['h1.DUwDvf', 'h1[data-attrid="title"]', 'h1']:
            elem = page.locator(selector).first
            if elem.is_visible(timeout=1000):
                return elem.inner_text().strip()
    except:
        pass
    return ""


def _extract_rating(page) -> Optional[float]:
    """Extract average rating."""
    try:
        for selector in ['div.fontDisplayLarge', 'span.ceNzKf', 'span[aria-hidden="true"]']:
            elem = page.locator(selector).first
            if elem.is_visible(timeout=1000):
                text = elem.inner_text().strip()
                # Parse rating (format: "4.3" or "4,3")
                rating_text = text.replace(",", ".")
                try:
                    return float(rating_text)
                except ValueError:
                    continue
    except:
        pass
    return None


def _extract_review_count(page) -> int:
    """Extract total review count."""
    try:
        # Look for text like "(1,247)" or "1,247 reviews"
        for selector in ['span.fontBodyMedium', 'button[aria-label*="reviews"]']:
            elements = page.locator(selector).all()
            for elem in elements:
                try:
                    text = elem.inner_text().strip()
                    # Extract number from text like "(1,247)" or "1,247 reviews"
                    import re
                    match = re.search(r'[\d,]+', text.replace(".", ""))
                    if match:
                        count = int(match.group().replace(",", ""))
                        if count > 0:
                            return count
                except:
                    continue
    except:
        pass
    return 0


def _extract_address(page) -> str:
    """Extract address."""
    try:
        # Address is often in a data-item-id="address" button
        elem = page.locator('button[data-item-id="address"]').first
        if elem.is_visible(timeout=1000):
            return elem.inner_text().strip()
    except:
        pass
    return ""


def _extract_phone(page) -> str:
    """Extract phone number."""
    try:
        elem = page.locator('button[data-item-id^="phone"]').first
        if elem.is_visible(timeout=1000):
            return elem.inner_text().strip()
    except:
        pass
    return ""


def _extract_website(page) -> str:
    """Extract website URL."""
    try:
        elem = page.locator('a[data-item-id="authority"]').first
        if elem.is_visible(timeout=1000):
            return elem.get_attribute("href") or ""
    except:
        pass
    return ""


def _extract_category(page) -> str:
    """Extract business category."""
    try:
        # Category is usually a button with jsaction="pane.rating.category"
        elem = page.locator('button[jsaction*="category"]').first
        if elem.is_visible(timeout=1000):
            return elem.inner_text().strip()
    except:
        pass
    return ""


def _extract_price_level(page) -> str:
    """Extract price level (e.g., ££, $$$)."""
    try:
        # Price level often appears near rating
        text = page.locator('span[aria-label*="Price"]').first.inner_text()
        return text.strip()
    except:
        pass
    return ""


def _extract_hours(page) -> dict:
    """Extract business hours."""
    hours = {}
    try:
        # Try to find and click hours section to expand
        hours_btn = page.locator('button[data-item-id*="oh"]').first
        if hours_btn.is_visible(timeout=1000):
            # Extract text which shows hours
            text = hours_btn.inner_text()
            # Simple extraction - actual parsing would be more complex
            hours["raw"] = text.strip()
    except:
        pass
    return hours


def _extract_coordinates(url: str) -> dict:
    """Extract coordinates from Google Maps URL."""
    coords = {"lat": None, "lng": None}
    try:
        import re
        # URL format: .../@51.5204,-0.0987,17z/...
        match = re.search(r'@(-?\d+\.?\d*),(-?\d+\.?\d*)', url)
        if match:
            coords["lat"] = float(match.group(1))
            coords["lng"] = float(match.group(2))
    except:
        pass
    return coords


def _extract_nearby_pois(page) -> list:
    """Extract nearby points of interest."""
    pois = []
    try:
        # This is complex - simplified implementation
        # In reality, would need to scroll and parse the "Explore nearby" section
        pass
    except:
        pass
    return pois


def _extract_reviews(page, max_reviews: int = 10) -> list:
    """Extract individual reviews with text, rating, date, and reviewer."""
    reviews = []
    
    try:
        # Click Reviews tab
        for selector in ['button[aria-label*="Reviews"]', 'button:has-text("Reviews")', 'tab:has-text("Reviews")']:
            try:
                tab = page.locator(selector).first
                if tab.is_visible(timeout=2000):
                    tab.click()
                    page.wait_for_timeout(random.randint(2000, 3000))
                    break
            except:
                continue
        
        # Scroll to load more reviews
        for _ in range(min(3, max_reviews // 5)):
            try:
                page.mouse.wheel(0, 500)
                page.wait_for_timeout(random.randint(800, 1200))
            except:
                break
        
        # Extract reviews - try multiple selectors
        review_containers = page.locator('div[data-review-id]').all()
        
        if not review_containers:
            # Fallback to older selector
            review_containers = page.locator('div.jftiEf').all()
        
        for container in review_containers[:max_reviews]:
            review = {
                "text": "",
                "rating": None,
                "date": "",
                "reviewer_name": ""
            }
            
            try:
                # Extract review text
                text_elem = container.locator('span.wiI7pd').first
                if text_elem.is_visible(timeout=500):
                    review["text"] = text_elem.inner_text().strip()
                else:
                    # Try alternative selector
                    text_elem = container.locator('div.MyEned').first
                    if text_elem.is_visible(timeout=500):
                        review["text"] = text_elem.inner_text().strip()
                
                # Extract rating (from aria-label like "5 stars")
                try:
                    stars_elem = container.locator('span[aria-label*="star"]').first
                    aria = stars_elem.get_attribute("aria-label")
                    if aria:
                        import re
                        match = re.search(r'(\d)', aria)
                        if match:
                            review["rating"] = int(match.group(1))
                except:
                    pass
                
                # Extract date
                try:
                    date_elem = container.locator('span.rsqaWe').first
                    if date_elem.is_visible(timeout=500):
                        review["date"] = date_elem.inner_text().strip()
                except:
                    pass
                
                # Extract reviewer name
                try:
                    name_elem = container.locator('div.d4r55').first
                    if name_elem.is_visible(timeout=500):
                        review["reviewer_name"] = name_elem.inner_text().strip()
                except:
                    pass
                
                # Only add if we got some text
                if review["text"] and len(review["text"]) > 10:
                    reviews.append(review)
                    
            except Exception as e:
                logger.debug(f"Error extracting review: {e}")
                continue
        
    except Exception as e:
        logger.debug(f"Error extracting reviews: {e}")
    
    return reviews


# =============================================================================
# PUBLIC TOOLS
# =============================================================================

def scrape_google_maps_reviews(query: str, max_reviews: int = 10) -> dict:
    """
    Scrape Google Maps reviews for sentiment analysis.
    
    Purpose: Extract guest review TEXT for the Review Analyst agent.
    
    Args:
        query: Hotel name and location (e.g., "Malmaison London")
        max_reviews: Maximum number of reviews to extract (default: 10)
    
    Returns:
        dict with keys:
            - hotel_name: str
            - total_reviews: int
            - average_rating: float
            - reviews: list[dict] with text, rating, date, reviewer_name
            - source: "google_maps"
            - scraped_at: str (ISO timestamp)
            - success: bool
            - warnings: list[str]
    
    Example:
        >>> result = scrape_google_maps_reviews("Malmaison London")
        >>> print(result["reviews"][0]["text"])
        "Great location but wifi was unreliable..."
    """
    max_reviews = int(max_reviews)  # Coerce in case LLM passes string
    
    raw = _scrape_google_maps_raw(query, extract_reviews=True, max_reviews=max_reviews)
    
    return {
        "hotel_name": raw.get("name", query),
        "total_reviews": raw.get("review_count", 0),
        "average_rating": raw.get("rating"),
        "reviews": raw.get("reviews", []),
        "source": "google_maps",
        "scraped_at": raw.get("scraped_at"),
        "success": raw.get("success", False),
        "warnings": raw.get("warnings", [])
    }


def scrape_google_maps_business(query: str, include_nearby: bool = True) -> dict:
    """
    Scrape Google Maps business data for market intelligence.
    
    Purpose: Extract business metadata, ratings, and local context 
    for the Market Intel agent. Does NOT include review text.
    
    Args:
        query: Hotel name and location (e.g., "Malmaison London")
        include_nearby: Whether to extract nearby POIs (default: True)
    
    Returns:
        dict with keys:
            - hotel_name: str
            - rating: float
            - review_count: int (count only, not the reviews themselves)
            - price_level: str (e.g., "£££")
            - category: str (e.g., "4-star hotel")
            - address: str
            - coordinates: dict with lat, lng
            - phone: str
            - website: str
            - hours: dict
            - nearby_pois: list[dict] (if include_nearby=True)
            - source: "google_maps"
            - scraped_at: str (ISO timestamp)
            - success: bool
            - warnings: list[str]
    
    Example:
        >>> result = scrape_google_maps_business("Malmaison London")
        >>> print(result["rating"], result["review_count"])
        4.3, 1247
    """
    # Don't extract reviews - we only need business metadata
    raw = _scrape_google_maps_raw(query, extract_reviews=False)
    
    result = {
        "hotel_name": raw.get("name", query),
        "rating": raw.get("rating"),
        "review_count": raw.get("review_count", 0),
        "price_level": raw.get("price_level", ""),
        "category": raw.get("category", ""),
        "address": raw.get("address", ""),
        "coordinates": raw.get("coordinates", {"lat": None, "lng": None}),
        "phone": raw.get("phone", ""),
        "website": raw.get("website", ""),
        "hours": raw.get("hours", {}),
        "source": "google_maps",
        "scraped_at": raw.get("scraped_at"),
        "success": raw.get("success", False),
        "warnings": raw.get("warnings", [])
    }
    
    if include_nearby:
        result["nearby_pois"] = raw.get("nearby_pois", [])
    
    return result


def format_reviews_for_agent(result: dict) -> str:
    """
    Format scrape_google_maps_reviews result as a string for agent consumption.
    
    Args:
        result: Output from scrape_google_maps_reviews()
    
    Returns:
        Formatted string suitable for agent tool output
    """
    if not result.get("success"):
        warnings = result.get("warnings", [])
        error_msg = warnings[0] if warnings else "Unknown error"
        return f"Google Maps scraping failed: {error_msg}"
    
    output = f"=== Google Maps Reviews for {result['hotel_name']} ===\n"
    output += f"Average Rating: {result['average_rating']}\n"
    output += f"Total Reviews: {result['total_reviews']}\n\n"
    
    reviews = result.get("reviews", [])
    if not reviews:
        output += "No review text could be extracted.\n"
        return output
    
    output += f"--- Reviews ({len(reviews)} extracted) ---\n\n"
    
    for i, review in enumerate(reviews, 1):
        output += f"[{i}] "
        if review.get("rating"):
            output += f"({'*' * review['rating']}) "
        if review.get("reviewer_name"):
            output += f"{review['reviewer_name']}"
        if review.get("date"):
            output += f" - {review['date']}"
        output += f"\n{review['text']}\n\n"
    
    return output


def format_business_for_agent(result: dict) -> str:
    """
    Format scrape_google_maps_business result as a string for agent consumption.
    
    Args:
        result: Output from scrape_google_maps_business()
    
    Returns:
        Formatted string suitable for agent tool output
    """
    if not result.get("success"):
        warnings = result.get("warnings", [])
        error_msg = warnings[0] if warnings else "Unknown error"
        return f"Google Maps scraping failed: {error_msg}"
    
    output = f"=== Google Maps Business Data for {result['hotel_name']} ===\n\n"
    
    if result.get("rating"):
        output += f"Rating: {result['rating']}/5\n"
    if result.get("review_count"):
        output += f"Review Count: {result['review_count']}\n"
    if result.get("category"):
        output += f"Category: {result['category']}\n"
    if result.get("price_level"):
        output += f"Price Level: {result['price_level']}\n"
    
    output += "\n--- Contact ---\n"
    if result.get("address"):
        output += f"Address: {result['address']}\n"
    if result.get("phone"):
        output += f"Phone: {result['phone']}\n"
    if result.get("website"):
        output += f"Website: {result['website']}\n"
    
    coords = result.get("coordinates", {})
    if coords.get("lat") and coords.get("lng"):
        output += f"Coordinates: {coords['lat']}, {coords['lng']}\n"
    
    hours = result.get("hours", {})
    if hours.get("raw"):
        output += f"\n--- Hours ---\n{hours['raw']}\n"
    
    pois = result.get("nearby_pois", [])
    if pois:
        output += f"\n--- Nearby POIs ({len(pois)}) ---\n"
        for poi in pois[:5]:
            output += f"• {poi.get('name', 'Unknown')} ({poi.get('type', '')}) - {poi.get('distance', '')}\n"
    
    return output


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing Google Maps scraper...")
    
    print("\n1. Testing scrape_google_maps_reviews:")
    reviews_result = scrape_google_maps_reviews("Malmaison London", max_reviews=3)
    print(format_reviews_for_agent(reviews_result))
    
    print("\n2. Testing scrape_google_maps_business:")
    business_result = scrape_google_maps_business("Malmaison London")
    print(format_business_for_agent(business_result))
