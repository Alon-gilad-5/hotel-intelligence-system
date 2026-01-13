"""
Review Analyst Agent

Analyzes guest reviews: sentiment, topics, complaints, praise.
Can search internal DB (RAG) OR scrape live Google Maps data.
"""

import urllib.parse
from typing import Optional
from agents.base_agent import BaseAgent
from agents.utils.bright_data import search_google_serp, format_serp_results


class ReviewAnalystAgent(BaseAgent):
    """Specialist agent for review analysis."""

    def get_system_prompt(self) -> str:
        return f"""You are a Review Analyst for {self.hotel_name} in {self.city}.

STRICT RULES - NO HALLUCINATIONS:
1. ONLY state facts that appear EXACTLY in tool outputs.
2. ALWAYS quote the exact text from tool results when making claims.
3. If the topic is NOT in any tool output, say: "No information found about [topic]."
4. NEVER make up or assume guest opinions.

RESPONSE FORMAT:
- Quote exact text: "From [source]: '[exact quote]'"
- Include URL when available
- If nothing found: "I searched [X] sources but found no reviews about [topic]."

TOOL ORDER:
1. search_booking_reviews - YOUR hotel's internal reviews
2. search_airbnb_reviews - YOUR hotel's internal reviews
3. search_competitor_reviews - Use when you have competitor hotel IDs (from previous agent results)
4. search_web_free - web search (use if internal DB has no relevant info)

MULTI-AGENT CONTEXT:
If you receive "[Results from Previous Agents]" with competitor hotel IDs (like BKG_123 or ABB_456),
use search_competitor_reviews to get reviews for EACH competitor hotel ID mentioned.
This is critical for comparison queries.

Stop after finding relevant data. Do not call unnecessary tools.

Hotel: {self.hotel_name} (ID: {self.hotel_id}) in {self.city}
"""

    def get_tools(self) -> list:
        return [
            self.search_booking_reviews,
            self.search_airbnb_reviews,
            self.search_competitor_reviews,  # NEW: Search reviews for any hotel by ID
            self.scrape_google_maps_reviews,
            self.scrape_tripadvisor_reviews,
            self.search_web_google,  # Bright Data SERP API (preferred)
            self.search_web_free,    # DuckDuckGo fallback (free)
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

    def search_competitor_reviews(self, hotel_id: str, query: str, k: int = 3) -> str:
        """
        Search reviews for a SPECIFIC hotel by ID (competitor or any hotel).
        Use this when you have competitor hotel IDs from previous agent results.
        
        Args:
            hotel_id: The hotel ID to search (e.g., "BKG_123456" or "ABB_789")
            query: Topic to search for in reviews (e.g., "cleanliness", "wifi")
            k: Number of reviews to return
        """
        # Determine namespace from hotel_id prefix
        if hotel_id.startswith("BKG_"):
            namespace = "booking_reviews"
        elif hotel_id.startswith("ABB_"):
            namespace = "airbnb_reviews"
        else:
            # Try both
            namespace = "booking_reviews"
        
        docs = self.search_rag(
            query,
            namespace=namespace,
            k=k,
            filter_dict={"hotel_id": hotel_id}
        )
        
        # If no results in booking, try airbnb
        if not docs and not hotel_id.startswith("BKG_"):
            docs = self.search_rag(
                query,
                namespace="airbnb_reviews",
                k=k,
                filter_dict={"hotel_id": hotel_id}
            )

        if not docs:
            return f"No reviews found for hotel {hotel_id} about '{query}'."

        output = f"=== Reviews for {hotel_id} about '{query}' ({len(docs)} found) ===\n\n"
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
        Note: Always searches for hotel name to find the place, topic is used for context only.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Error: Playwright not installed. Run: pip install playwright && playwright install chromium"

        # Always search for hotel name to find the correct place
        search_query = self.hotel_name
        topic = query  # Keep track of what we're looking for
        # Build the Google Maps URL
        url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query)}"
        print(f"[ReviewAnalyst] Scraping Google Maps: {search_query} (looking for: {topic})...")

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
        Note: Always searches for hotel name only (not the specific query) to find the hotel page.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Error: Playwright not installed."

        # IMPORTANT: Always search for hotel name only to find the main page
        # The specific topic (wifi, etc.) is used to filter results later
        search_query = self.hotel_name
        topic_filter = query.lower() if query else None
        print(f"[ReviewAnalyst] Scraping TripAdvisor: {search_query} (topic: {topic_filter})...")

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
                # Search for hotel name + "reviews" to find review pages
                ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote('site:tripadvisor.com ' + search_query + ' reviews')}"
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

    def search_web_google(self, query: str) -> str:
        """
        Search Google using Bright Data's SERP API (PREFERRED).
        Returns real Google search results with review snippets.
        Use this before search_web_free for better results.
        """
        print(f"[ReviewAnalyst] Searching Google (Bright Data SERP)...")
        
        # Extract topic keywords to prioritize relevant results
        topic_words = []
        important_keywords = ['wifi', 'internet', 'signal', 'connection', 'speed', 'noise', 
                             'clean', 'breakfast', 'staff', 'service', 'parking', 'pool',
                             'location', 'bed', 'room', 'bathroom', 'air conditioning', 'ac']
        query_lower = query.lower()
        for word in important_keywords:
            if word in query_lower:
                topic_words.append(word)
        
        topic_str = ' '.join(topic_words) if topic_words else 'review'
        
        # Build search query - hotel name + topic + review context
        search_query = f"{self.hotel_name} {topic_str} review"
        print(f"   - Query: {search_query}")
        
        # Call Bright Data SERP API
        result = search_google_serp(search_query, num_results=10)
        
        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            print(f"   - Bright Data error: {error_msg}")
            return f"Google search failed: {error_msg}. Try search_web_free as fallback."
        
        if not result["results"]:
            return "Google search returned no results. Try search_web_free as fallback."
        
        # Format results, prioritizing those with topic keywords
        output = format_serp_results(result["results"], topic_keywords=topic_words)
        return output

    def search_web_free(self, query: str) -> str:
        """
        Fallback: Search the web using the 'ddgs' library.
        Searches multiple review sites for guest feedback.
        """
        # 1. Robust Import
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return "Error: 'ddgs' package not installed."

        print(f"[ReviewAnalyst] Searching Web (DDGS)...")

        # 2. Extract topic keywords from query
        # Keep important words like "wifi", "signal", "internet", etc.
        topic_words = []
        important_keywords = ['wifi', 'internet', 'signal', 'connection', 'speed', 'noise', 
                             'clean', 'breakfast', 'staff', 'service', 'parking', 'pool',
                             'location', 'bed', 'room', 'bathroom', 'air conditioning', 'ac']
        query_lower = query.lower()
        for word in important_keywords:
            if word in query_lower:
                topic_words.append(word)
        
        topic_str = ' '.join(topic_words) if topic_words else 'review'

        queries_to_try = [
            # TripAdvisor specific (most likely to have detailed reviews)
            f'"{self.hotel_name}" {topic_str} review site:tripadvisor.com',
            # Google reviews / general
            f'"{self.hotel_name}" {topic_str} guest review',
            # Booking.com reviews
            f'"{self.hotel_name}" {topic_str} site:booking.com',
            # Broad search with quotes for exact hotel match
            f'"{self.hotel_name}" {topic_str}',
        ]

        results = []

        try:
            ddgs = DDGS()
            for search_term in queries_to_try:
                print(f"   - Trying: {search_term}")
                # Fetch up to 8 results per query for better coverage
                try:
                    current_results = list(ddgs.text(search_term, max_results=8))
                    if current_results:
                        results.extend(current_results)
                except Exception as e:
                    print(f"   - Search failed: {e}")
                    continue

            if not results:
                return "Performed web search but found no results."

            # 3. Format output - prioritize results with topic keywords in snippet
            seen_links = set()
            relevant_results = []
            other_results = []

            for r in results:
                link = r.get('href', '')
                body = r.get('body', '').lower()
                
                if link in seen_links:
                    continue
                seen_links.add(link)
                
                # Prioritize results that mention the topic in the snippet
                has_topic = any(word in body for word in topic_words)
                if has_topic:
                    relevant_results.append(r)
                else:
                    other_results.append(r)

            # Combine: relevant first, then others
            unique_results = relevant_results[:5] + other_results[:3]

            output = f"=== Web Search Results ({len(unique_results)} found) ===\n\n"
            for i, r in enumerate(unique_results[:8], 1):
                output += f"[{i}] Title: {r.get('title')}\n"
                output += f"    Snippet: {r.get('body')}\n"
                output += f"    Link: {r.get('href')}\n\n"

            return output

        except Exception as e:
            return f"Web Search Error: {e}"