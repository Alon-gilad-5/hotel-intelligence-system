"""
Market Intel Agent

Gathers external market intelligence: events, weather, Google Maps data.
Uses Playwright (free) and BrightData (paid - use sparingly).
"""

import os
import urllib.parse
from typing import Optional
from agents.base_agent import BaseAgent


class MarketIntelAgent(BaseAgent):
    """Specialist agent for market intelligence."""

    def get_system_prompt(self) -> str:
        return f"""You are a Market Intelligence Analyst for {self.hotel_name} in {self.city}.

Your job is to gather external information that affects hotel demand:
- Local events (concerts, conferences, festivals)
- Weather conditions
- Google Maps reviews and ratings
- Local attractions and points of interest

Cost awareness:
- Playwright scraping is FREE - prefer this when possible
- BrightData API costs money - use only when Playwright can't handle it

When answering:
1. Use appropriate tools to gather information
2. Explain how findings might impact the hotel business
3. Provide actionable recommendations

Hotel context:
- Hotel ID: {self.hotel_id}
- Hotel Name: {self.hotel_name}
- City: {self.city}
"""

    def get_tools(self) -> list:
        return [
            self.scrape_google_maps_reviews,
            self.search_events_free,
            self.search_web_brightdata,
        ]

    def scrape_google_maps_reviews(self, query: Optional[str] = None) -> str:
        """
        Scrape Google Maps reviews using Playwright (FREE).

        Args:
            query: Hotel name to search. Defaults to this hotel.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Error: Playwright not installed. Run: pip install playwright && playwright install chromium"

        search_query = query or self.hotel_name
        encoded_query = urllib.parse.quote(search_query)
        url = f"https://www.google.com/maps/search/{encoded_query}"

        print(f"[MarketIntel] Scraping Google Maps: {search_query}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                    locale="en-US"
                )
                page = context.new_page()

                page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Handle cookie consent
                try:
                    accept_btn = page.locator('button:has-text("Accept all")').first
                    if accept_btn.is_visible(timeout=2000):
                        accept_btn.click()
                        page.wait_for_timeout(1000)
                except:
                    pass

                page.wait_for_timeout(3000)

                # Click first result
                try:
                    first_result = page.locator('a[href*="maps/place"]').first
                    if first_result.is_visible(timeout=3000):
                        first_result.click()
                        page.wait_for_timeout(2000)
                except:
                    pass

                # Click Reviews tab
                try:
                    for selector in ['button[aria-label*="Reviews"]', 'button:has-text("Reviews")']:
                        tab = page.locator(selector).first
                        if tab.is_visible(timeout=1000):
                            tab.click()
                            page.wait_for_timeout(2000)
                            break
                except:
                    pass

                # Extract data
                result = {"hotel": search_query, "rating": "", "reviews": []}

                # Get rating
                try:
                    rating_elem = page.locator('div.fontDisplayLarge').first
                    if rating_elem.is_visible(timeout=2000):
                        result["rating"] = rating_elem.inner_text().strip()
                except:
                    pass

                # Get reviews
                try:
                    review_elements = page.locator('span.wiI7pd').all()
                    for elem in review_elements[:8]:
                        text = elem.inner_text().strip()
                        if len(text) > 20:
                            result["reviews"].append(text[:500])
                except:
                    pass

                browser.close()

                # Format output
                output = f"=== Google Maps Data for {result['hotel']} ===\n"
                if result["rating"]:
                    output += f"Rating: {result['rating']}\n"
                output += f"\n--- Reviews ({len(result['reviews'])} found) ---\n\n"

                for i, review in enumerate(result["reviews"], 1):
                    output += f"[{i}] {review}\n\n"

                return output if result["reviews"] else "No reviews found on Google Maps."

        except Exception as e:
            return f"Playwright error: {e}"

    def search_events_free(self, city: Optional[str] = None, date: Optional[str] = None) -> str:
        """
        Search for local events using free web scraping.

        Args:
            city: City to search. Defaults to hotel's city.
            date: Date to search (YYYY-MM-DD). Defaults to upcoming.
        """
        search_city = city or self.city

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Playwright not installed."

        search_query = f"events in {search_city}"
        if date:
            search_query += f" {date}"

        print(f"[MarketIntel] Searching events: {search_query}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # Use DuckDuckGo (no captcha)
                encoded = urllib.parse.quote(search_query)
                page.goto(f"https://duckduckgo.com/?q={encoded}", timeout=15000)
                page.wait_for_timeout(2000)

                # Extract search results
                results = []
                links = page.locator('article').all()

                for link in links[:5]:
                    try:
                        title = link.locator('h2').inner_text()
                        snippet = link.locator('span').first.inner_text()
                        results.append(f"• {title}: {snippet[:150]}")
                    except:
                        continue

                browser.close()

                if results:
                    return f"=== Events in {search_city} ===\n\n" + "\n\n".join(results)
                else:
                    return f"No events found for {search_city}."

        except Exception as e:
            return f"Event search error: {e}"

    def search_web_brightdata(self, query: str) -> str:
        """
        Search web using BrightData API (PAID - use sparingly).
        Only use when Playwright methods fail.

        Args:
            query: Search query
        """
        import asyncio

        api_token = os.getenv("BRIGHTDATA_API_TOKEN")
        if not api_token:
            return "BrightData API token not configured."

        print(f"[MarketIntel] BrightData search (PAID): {query}")

        async def _search():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            import platform

            npx = "npx.cmd" if platform.system() == "Windows" else "npx"
            params = StdioServerParameters(
                command=npx,
                args=["-y", "@brightdata/mcp"],
                env={"API_TOKEN": api_token}
            )

            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()

                        result = await session.call_tool(
                            "search_engine",
                            arguments={"query": query}
                        )

                        text = str(result)
                        if hasattr(result, 'content') and result.content:
                            text = result.content[0].text

                        return text[:2000]

            except Exception as e:
                return f"BrightData error: {e}"

        try:
            return asyncio.run(_search())
        except Exception as e:
            return f"Error: {e}"