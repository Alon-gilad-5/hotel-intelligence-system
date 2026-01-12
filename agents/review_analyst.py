"""
Review Analyst Agent

Analyzes guest reviews: sentiment, topics, complaints, praise.
Can search internal DB (RAG) OR scrape live Google Maps data.
"""

import urllib.parse
from typing import Optional
from agents.base_agent import BaseAgent


class ReviewAnalystAgent(BaseAgent):
    """Specialist agent for review analysis."""

    def get_system_prompt(self) -> str:
        return f"""You are a Review Analyst for {self.hotel_name} in {self.city}.

    Your job is to analyze guest feedback.
    You have access to a hierarchy of data sources.

    STRATEGY (Follow this order strictly):
    1. **Internal DB**: Check `search_booking_reviews` AND `search_airbnb_reviews`.
    2. **Specialized Scrapers**: 
       - IF internal DB fails, you MUST try `scrape_google_maps_reviews`.
       - IF that fails, you MUST try `scrape_tripadvisor_reviews`.
    3. **General Web Search**: Use `search_web_free` ONLY if all specialized scrapers fail.

    CRITICAL INSTRUCTIONS:
    - If you use ANY external tool, READ the text returned yourself.
    - DO NOT use 'analyze_sentiment_topics' on external text.
    - If one scraper fails (e.g., Google Maps), DO NOT STOP. Try the next one (TripAdvisor).

    Hotel context:
    - Hotel ID: {self.hotel_id}
    - Hotel Name: {self.hotel_name}
    - City: {self.city}
    """

    def get_tools(self) -> list:
        return [
            self.search_booking_reviews,
            self.search_airbnb_reviews,
            self.scrape_google_maps_reviews,
            self.scrape_tripadvisor_reviews,
            self.search_web_free,
            self.analyze_sentiment_topics,
        ]

    def search_booking_reviews(self, query: str, k: int = 5) -> str:
        """
        Search internal Booking.com reviews.
        """
        # Filter by hotel_id to prevent seeing other hotels' data
        docs = self.search_rag(
            query,
            namespace="booking_reviews",
            k=k,
            filter_dict={"hotel_id": self.hotel_id}
        )

        if not docs:
            return "No Booking.com reviews found in internal database."

        output = f"=== Booking.com Reviews ({len(docs)} found) ===\n\n"
        for i, doc in enumerate(docs, 1):
            output += f"[{i}] {doc.page_content}\n\n"

        return output

    def search_airbnb_reviews(self, query: str, k: int = 5) -> str:
        """
        Search internal Airbnb reviews.
        """
        # Filter by hotel_id
        docs = self.search_rag(
            query,
            namespace="airbnb_reviews",
            k=k,
            filter_dict={"hotel_id": self.hotel_id}
        )

        if not docs:
            return "No Airbnb reviews found in internal database."

        output = f"=== Airbnb Reviews ({len(docs)} found) ===\n\n"
        for i, doc in enumerate(docs, 1):
            output += f"[{i}] {doc.page_content}\n\n"

        return output

    def analyze_sentiment_topics(self, topic: str) -> str:
        """
        ONLY for analyzing INTERNAL database reviews.
        DO NOT use this for data you just scraped from Google Maps.
        """
        # Filter by hotel_id for both sources
        booking_docs = self.search_rag(
            topic,
            namespace="booking_reviews",
            k=5,
            filter_dict={"hotel_id": self.hotel_id}
        )
        airbnb_docs = self.search_rag(
            topic,
            namespace="airbnb_reviews",
            k=5,
            filter_dict={"hotel_id": self.hotel_id}
        )

        all_reviews = []
        for doc in booking_docs + airbnb_docs:
            all_reviews.append(doc.page_content)

        if not all_reviews:
            return f"No reviews found mentioning '{topic}' in internal database. Suggest using scrape_google_maps_reviews."

        # Use LLM to analyze sentiment of internal data
        analysis_prompt = f"""Analyze the sentiment of these reviews regarding "{topic}".

Reviews:
{chr(10).join(all_reviews[:10])}

Provide:
1. Overall sentiment (Positive/Negative/Mixed)
2. Key positive points
3. Key negative points
4. Actionable recommendations
"""

        response = self.llm.invoke(analysis_prompt)
        return response.content

    def scrape_google_maps_reviews(self, query: Optional[str] = None) -> str:
        """
        Scrape LIVE Google Maps reviews using Playwright.
        Use this if internal database searches fail.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Error: Playwright not installed. Run: pip install playwright && playwright install chromium"

        search_query = query or self.hotel_name
        # Build the Google Maps URL
        url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query)}"
        print(f"[ReviewAnalyst] 🌍 Scraping Google Maps: {search_query}...")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # 1. Navigate to Search Results
                page.goto(url, wait_until="domcontentloaded", timeout=60000)

                # Reject Cookies (common in EU regions)
                try:
                    page.get_by_role("button", name="Reject all").click(timeout=3000)
                except:
                    pass

                # 2. Click the first result (usually the hotel/place)
                try:
                    page.locator("a[href*='/maps/place/']").first.click(timeout=5000)
                    page.wait_for_timeout(3000)
                except:
                    # We might already be on the place page
                    pass

                # 3. Click "Reviews" Tab (Robust Logic)
                clicked_reviews = False
                # Try finding the tab by text name
                for name in ["Reviews", "Reviews", "Opinions"]:
                    try:
                        page.get_by_role("tab", name=name).click(timeout=2000)
                        clicked_reviews = True
                        break
                    except:
                        continue

                if not clicked_reviews:
                    # Fallback: try aria-label or button
                    try:
                        page.locator('button[aria-label*="Reviews"]').click(timeout=2000)
                    except:
                        pass

                page.wait_for_timeout(3000)

                # 4. Extract Review Text
                reviews = []

                # Try the new Google Maps class for review body
                elements = page.locator('div[class*="fontBodyMedium"]').all()

                if not elements:
                    # Fallback to the older known class
                    elements = page.locator('span.wiI7pd').all()

                for i, elem in enumerate(elements[:8]):
                    try:
                        text = elem.inner_text().strip()
                        if len(text) > 20:
                            reviews.append(text)
                    except:
                        pass

                browser.close()

                if not reviews:
                    return "Found the place on Maps, but could not extract review text (CSS selectors might have changed)."

                output = f"=== LIVE Google Maps Reviews for {search_query} ===\n\n"
                for i, r in enumerate(reviews, 1):
                    output += f"[{i}] {r}\n\n"

                return output

        except Exception as e:
            return f"Playwright error: {e}"


    def scrape_tripadvisor_reviews(self, query: Optional[str] = None) -> str:
        """
        Scrape LIVE TripAdvisor reviews using Playwright.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Error: Playwright not installed."

        search_query = query or self.hotel_name
        print(f"[ReviewAnalyst] 🦉 Scraping TripAdvisor: {search_query}...")

        try:
            with sync_playwright() as p:
                # Launch browser (headless=True is faster, but False is less suspicious to anti-bots)
                browser = p.chromium.launch(headless=True)

                # Create context with a real user agent to avoid immediate blocking
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()

                # 1. Search via DuckDuckGo to find the direct TripAdvisor link
                # (Bypassing TripAdvisor's internal search bar which is often buggy for bots)
                ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote('site:tripadvisor.com ' + search_query)}"
                page.goto(ddg_url, wait_until="domcontentloaded", timeout=30000)

                # 2. Click the first TripAdvisor result
                try:
                    # Look for a link that contains "tripadvisor" and "Hotel_Review" or similar
                    page.locator("a[href*='tripadvisor']").first.click(timeout=10000)
                    page.wait_for_timeout(5000)  # Wait for page load
                except:
                    browser.close()
                    return "Could not find TripAdvisor link via search."

                # 3. Handle Cookie Banners (Common on TripAdvisor)
                try:
                    # Look for generic "Accept" or "I Agree" buttons
                    page.get_by_role("button", name="Accept").click(timeout=2000)
                except:
                    pass

                # 4. Extract Reviews
                reviews = []

                # TripAdvisor selectors change often. We look for the "q" (quote) class or generic review containers
                # Strategy: Get all spans/divs that look like review text
                try:
                    # Common TripAdvisor review text class (often starts with 'Q' or is inside a 'q' tag)
                    # We grab text that is reasonably long to filter out menu items
                    elements = page.locator('div[data-test-target="review-title"]').all()

                    # If titles found, grab the body text usually next to it
                    if elements:
                        # Grab the review bodies (often span with class 'QvCXh' or generic)
                        body_elements = page.locator('span[class*="QvCXh"]').all()  # Common dynamic class
                        if not body_elements:
                            body_elements = page.locator('div[data-test-target="review-body"]').all()

                        for i, elem in enumerate(body_elements[:5]):
                            text = elem.inner_text().strip()
                            if len(text) > 20:
                                reviews.append(text)
                    else:
                        # Fallback: Just grab large blocks of text
                        divs = page.locator("div").all()
                        for div in divs:
                            text = div.inner_text()
                            if len(text) > 100 and len(text) < 1000 and "wrote a review" not in text:
                                reviews.append(text)
                                if len(reviews) >= 5: break
                except:
                    pass

                browser.close()

                if not reviews:
                    return "Reached TripAdvisor, but could not extract reviews (Anti-bot or CSS change)."

                output = f"=== LIVE TripAdvisor Reviews for {search_query} ===\n\n"
                for i, r in enumerate(reviews, 1):
                    output += f"[{i}] {r}\n\n"

                return output

        except Exception as e:
            return f"TripAdvisor Scraping Error: {e}"

    def search_web_free(self, query: str) -> str:
        """
        Fallback: Search the web using the 'ddgs' library.
        Bypasses CAPTCHAs and handles the new package name.
        """
        # 1. Robust Import
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return "Error: 'ddgs' package not installed."

        print(f"[ReviewAnalyst] 🔍 Searching Web (DDGS)...")

        # 2. Strategy: Try specific query first, then broad
        # We search for "wifi" specifically, not just "wifi signal" to get more hits
        clean_query = query.replace(self.hotel_name, "").replace("signal", "").strip()

        queries_to_try = [
            # High precision: Site specific
            f"{self.hotel_name} wifi reviews site:tripadvisor.com",
            # Medium precision: General review sites
            f"{self.hotel_name} {clean_query} reviews",
            # Broad: Just the hotel and the topic
            f"{self.hotel_name} {clean_query}"
        ]

        results = []

        try:
            ddgs = DDGS()
            for search_term in queries_to_try:
                print(f"   - Trying: {search_term}")
                # Fetch up to 5 results per query
                # Note: 'keywords' arg is standard, but some versions use just the first arg
                current_results = ddgs.text(keywords=search_term, max_results=5)

                if current_results:
                    results.extend(current_results)
                    # If we found good results (more than 2), stop trying broader queries
                    if len(results) >= 3:
                        break

            if not results:
                return "Performed web search but found no results."

            # 3. Format output
            # Deduplicate by link to avoid repeating the same result
            seen_links = set()
            unique_results = []

            for r in results:
                link = r.get('href')
                if link not in seen_links:
                    seen_links.add(link)
                    unique_results.append(r)

            output = f"=== Web Search Results ({len(unique_results)} found) ===\n\n"
            for i, r in enumerate(unique_results[:7], 1):  # Limit to top 7
                output += f"[{i}] Title: {r.get('title')}\n"
                output += f"    Snippet: {r.get('body')}\n"
                output += f"    Link: {r.get('href')}\n\n"

            return output

        except Exception as e:
            return f"Web Search Error: {e}"