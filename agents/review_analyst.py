"""
Review Analyst Agent

Analyzes guest reviews: sentiment, topics, complaints, praise.
Can search internal DB (RAG) OR scrape live Google Maps data.
"""

import urllib.parse
from typing import Optional
from agents.base_agent import BaseAgent
from agents.utils.bright_data import search_google_serp, format_serp_results
from agents.utils.google_maps_scraper import (
    scrape_google_maps_reviews as _scrape_gmaps_reviews,
    format_reviews_for_agent
)


class ReviewAnalystAgent(BaseAgent):
    """Specialist agent for review analysis."""
    
    def __init__(self, hotel_id: str, hotel_name: str, city: str):
        """Initialize with strict validation enabled by default."""
        super().__init__(
            hotel_id=hotel_id,
            hotel_name=hotel_name,
            city=city,
            validate_output=True,
            strict_validation=True  # Enable strict mode to prevent hallucinations
        )

    # Topic keyword mappings for better filtering
    TOPIC_KEYWORDS = {
        'cleanliness': ['clean', 'dirty', 'spotless', 'hygiene', 'tidy', 'sanitary', 'dust', 'stain', 'housekeeping'],
        'wifi': ['wifi', 'wi-fi', 'internet', 'connection', 'signal', 'speed', 'broadband', 'network'],
        'breakfast': ['breakfast', 'morning meal', 'buffet', 'food', 'dining'],
        'staff': ['staff', 'service', 'reception', 'helpful', 'friendly', 'rude', 'attentive'],
        'location': ['location', 'situated', 'area', 'neighbourhood', 'neighborhood', 'central', 'convenient'],
        'room': ['room', 'bed', 'comfortable', 'spacious', 'small', 'cozy', 'furniture'],
        'bathroom': ['bathroom', 'shower', 'toilet', 'bath', 'towel', 'water'],
        'noise': ['noise', 'quiet', 'loud', 'peaceful', 'noisy', 'sound'],
        'parking': ['parking', 'car', 'garage', 'vehicle'],
        'pool': ['pool', 'swimming', 'swim'],
        'ac': ['air conditioning', 'ac', 'a/c', 'cooling', 'heating', 'temperature'],
    }

    def _filter_reviews_by_topic(self, reviews: list, topic: str) -> tuple:
        """
        Filter reviews to only those mentioning the topic.
        
        Returns:
            (relevant_reviews, other_reviews)
        """
        if not topic:
            return reviews, []
        
        topic_lower = topic.lower()
        
        # Get keywords for this topic
        keywords = []
        for key, words in self.TOPIC_KEYWORDS.items():
            if key in topic_lower or any(w in topic_lower for w in words):
                keywords.extend(words)
        
        # Also add the topic itself as a keyword
        keywords.extend(topic_lower.split())
        keywords = list(set(keywords))  # Deduplicate
        
        relevant = []
        other = []
        
        for review in reviews:
            review_lower = review.lower() if isinstance(review, str) else str(review).lower()
            if any(kw in review_lower for kw in keywords):
                relevant.append(review)
            else:
                other.append(review)
        
        return relevant, other

    def get_system_prompt(self) -> str:
        return f"""You are a friendly Review Analyst helping the owner of {self.hotel_name} in {self.city}.

CRITICAL ACCURACY RULES:
1. Only cite reviews that actually exist in tool outputs - NEVER invent or infer.
2. If no reviews mention a topic, say so honestly: "I found no reviews mentioning [topic]."
3. DISTINGUISH between actual reviews vs web search snippets:
   - Actual reviews: Quote the text directly
   - Web snippets: Say "Web search found listings mentioning X, but no actual review text"
4. NEVER use phrases like "some guests", "most guests", "overall guests" unless you have actual review counts
5. If you only have web search snippets (not reviews), say: "I found no actual reviews about this topic in the database or on the web."

YOUR ROLE:
Help the property owner understand what guests are saying about their property.
Summarize feedback in a way that's actionable and easy to understand.

TOOLS (use in this order):
1. search_booking_reviews / search_airbnb_reviews - Search your property's reviews
2. search_competitor_reviews - Search a competitor's reviews (need their hotel ID)
3. search_web_google / search_web_free - Web search for additional info

HOW TO RESPOND:
Write like you're summarizing guest feedback for a busy property owner.

1. **Be EXPLICIT about data sources**:
   - "From 5 guest reviews: [quote]"
   - "Web search found listings but no review text"
   - "No reviews mention this topic"

2. **Use ACTUAL quotes** - Don't paraphrase unless you cite the source:
   - ✅ "One guest wrote: 'WiFi was slow'"
   - ❌ "Some guests complained about WiFi" (without quotes)

3. **Count precisely** - If you say "multiple", "several", "many":
   - ✅ "3 guests mentioned WiFi issues"
   - ❌ "Several guests mentioned WiFi" (when you found 1)

4. **Admit data limitations**:
   - If you only have snippets: "Web search found property listings mentioning WiFi, but no actual guest reviews"
   - If you have no data: "I couldn't find any reviews about [topic]. Consider asking guests for feedback."

5. **Suggest next steps** - Only when you truly have insufficient data

Property: {self.hotel_name} (ID: {self.hotel_id}) in {self.city}
"""

    def get_tools(self) -> list:
        import os
        
        tools = [
            self.search_booking_reviews,
            self.search_airbnb_reviews,
            self.search_competitor_reviews,
            self.search_web_google,      # BrightData API - works everywhere
            self.search_web_free,        # DuckDuckGo - works everywhere
            self.analyze_sentiment_topics,
        ]
        
        # Playwright tools only work locally (no browser on Databricks)
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
            tools.append(self.scrape_google_maps_reviews)
            tools.append(self.scrape_tripadvisor_reviews)
        
        return tools

    def search_booking_reviews(self, query: str, k: int = 10) -> str:
        """
        Search internal Booking.com reviews and filter by topic relevance.
        """
        k = int(k)  # Coerce in case LLM passes string
        # Filter by hotel_id to prevent seeing other hotels' data
        docs = self.search_rag(
            query,
            namespace="booking_reviews",
            k=k,
            filter_dict={"hotel_id": self.hotel_id}
        )

        if not docs:
            return "No Booking.com reviews found in internal database."

        # Filter by topic relevance
        reviews = [doc.page_content for doc in docs]
        relevant, other = self._filter_reviews_by_topic(reviews, query)

        output = f"=== Booking.com Reviews ({len(docs)} found) ===\n"
        output += f"(Searched for: '{query}')\n\n"

        if relevant:
            output += f"**RELEVANT REVIEWS ({len(relevant)} mentioning '{query}'):**\n\n"
            for i, review in enumerate(relevant[:8], 1):
                output += f"[{i}] {review}\n\n"
        else:
            output += f"**No reviews specifically mention '{query}'.**\n\n"

        if other and len(relevant) < 3:
            output += f"\n**OTHER REVIEWS (general, not specifically about '{query}'):**\n\n"
            for i, review in enumerate(other[:3], 1):
                output += f"[{i}] {review}\n\n"

        return output

    def search_airbnb_reviews(self, query: str, k: int = 10) -> str:
        """
        Search internal Airbnb reviews and filter by topic relevance.
        """
        k = int(k)  # Coerce in case LLM passes string
        # Filter by hotel_id
        docs = self.search_rag(
            query,
            namespace="airbnb_reviews",
            k=k,
            filter_dict={"hotel_id": self.hotel_id}
        )

        if not docs:
            return "No Airbnb reviews found in internal database."

        # Filter by topic relevance
        reviews = [doc.page_content for doc in docs]
        relevant, other = self._filter_reviews_by_topic(reviews, query)

        output = f"=== Airbnb Reviews ({len(docs)} found) ===\n"
        output += f"(Searched for: '{query}')\n\n"

        if relevant:
            output += f"**RELEVANT REVIEWS ({len(relevant)} mentioning '{query}'):**\n\n"
            for i, review in enumerate(relevant[:8], 1):
                output += f"[{i}] {review}\n\n"
        else:
            output += f"**No reviews specifically mention '{query}'.**\n\n"

        if other and len(relevant) < 3:
            output += f"\n**OTHER REVIEWS (general, not specifically about '{query}'):**\n\n"
            for i, review in enumerate(other[:3], 1):
                output += f"[{i}] {review}\n\n"

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
        k = int(k)  # Coerce in case LLM passes string
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

    def scrape_google_maps_reviews(self, query: Optional[str] = None, max_reviews: int = 10) -> str:
        """
        Scrape LIVE Google Maps reviews for sentiment analysis.
        
        Purpose: Extract guest review TEXT for sentiment analysis.
        Use this if internal database searches fail.
        
        Args:
            query: Optional topic to note (hotel name is always used for search)
            max_reviews: Maximum number of reviews to extract (default: 10)
        """
        max_reviews = int(max_reviews)  # Coerce in case LLM passes string
        search_query = self.hotel_name
        topic = query  # Keep track of what we're looking for
        
        print(f"[ReviewAnalyst] Scraping Google Maps reviews: {search_query} (topic: {topic})...")
        
        # Use shared scraper
        result = _scrape_gmaps_reviews(search_query, max_reviews=max_reviews)
        
        # Format for agent output
        return format_reviews_for_agent(result)


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

                # Filter reviews by topic if specified
                if topic_filter:
                    relevant, other = self._filter_reviews_by_topic(reviews, topic_filter)
                    
                    output = f"=== LIVE TripAdvisor Reviews for {search_query} ===\n"
                    output += f"(Filtered for topic: '{topic_filter}')\n\n"
                    
                    if relevant:
                        output += f"**RELEVANT TO '{topic_filter.upper()}' ({len(relevant)} found):**\n\n"
                        for i, r in enumerate(relevant[:10], 1):
                            output += f"[{i}] {r}\n\n"
                    else:
                        output += f"**No reviews specifically mention '{topic_filter}'.**\n\n"
                    
                    if other and not relevant:
                        output += f"\n**OTHER REVIEWS (not mentioning '{topic_filter}'):**\n\n"
                        for i, r in enumerate(other[:3], 1):
                            output += f"[{i}] {r}\n\n"
                else:
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
            output = f"=== Web Search Results ===\n"
            output += f"(Searched for: '{topic_str}')\n\n"
            
            if relevant_results:
                output += f"**RELEVANT TO '{topic_str.upper()}' ({len(relevant_results)} found):**\n\n"
                for i, r in enumerate(relevant_results[:5], 1):
                    output += f"[{i}] Title: {r.get('title')}\n"
                    output += f"    Snippet: {r.get('body')}\n"
                    output += f"    Link: {r.get('href')}\n\n"
            else:
                output += f"**No results specifically mention '{topic_str}'.**\n\n"
            
            if other_results and len(relevant_results) < 3:
                output += f"\n**OTHER RESULTS (not specifically about '{topic_str}'):**\n\n"
                for i, r in enumerate(other_results[:3], 1):
                    output += f"[{i}] Title: {r.get('title')}\n"
                    output += f"    Snippet: {r.get('body')}\n\n"

            return output

        except Exception as e:
            return f"Web Search Error: {e}"