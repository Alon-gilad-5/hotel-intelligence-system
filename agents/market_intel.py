"""
Market Intel Agent

Gathers external market intelligence: events, weather, Google Maps data.
Uses Playwright (free) and BrightData (paid - use sparingly).
"""

import os
import urllib.parse
from typing import Optional
from agents.base_agent import BaseAgent
from agents.utils.google_maps_scraper import (
    scrape_google_maps_business as _scrape_gmaps_business,
    format_business_for_agent
)


class MarketIntelAgent(BaseAgent):
    """Specialist agent for market intelligence."""

    def get_system_prompt(self) -> str:
        return f"""You are a Market Intelligence Analyst for {self.hotel_name} in {self.city}.

STRICT RULES - NO HALLUCINATIONS:
1. ONLY report information that appears EXACTLY in tool outputs.
2. NEVER make up events, weather data, or ratings.
3. Quote exact text: "From [source]: '[exact quote]'"
4. If scraping fails or returns no data, say: "Could not retrieve [data type]."

Your job is to gather external information that affects hotel demand:
- Local events (concerts, conferences, festivals)
- Weather conditions
- Google Maps reviews and ratings
- Local attractions and points of interest

RESPONSE FORMAT:
- Quote scraped content directly
- Include URLs when available
- Clearly distinguish between confirmed data and interpretation

Cost awareness:
- Playwright scraping is FREE - prefer this when possible
- BrightData API costs money - use only when Playwright can't handle it

Hotel context:
- Hotel ID: {self.hotel_id}
- Hotel Name: {self.hotel_name}
- City: {self.city}
"""

    def get_tools(self) -> list:
        return [
            self.scrape_google_maps_business,  # Business data (no review text)
            self.search_events_free,
            self.search_web_brightdata,
        ]

    def scrape_google_maps_business(self, query: Optional[str] = None, include_nearby: bool = True) -> str:
        """
        Scrape Google Maps business data for market intelligence.
        
        Purpose: Extract business metadata, ratings, and local context.
        Does NOT include review text (use Review Analyst for that).
        
        Args:
            query: Hotel name to search. Defaults to this hotel.
            include_nearby: Whether to extract nearby POIs (default: True)
        
        Returns:
            Formatted string with rating, review count, address, hours,
            category, contact info, and nearby POIs.
        """
        search_query = query or self.hotel_name
        
        print(f"[MarketIntel] Scraping Google Maps business data: {search_query}")
        
        # Use shared scraper (business metadata only, no review text)
        result = _scrape_gmaps_business(search_query, include_nearby=include_nearby)
        
        # Format for agent output
        return format_business_for_agent(result)

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