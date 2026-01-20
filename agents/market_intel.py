"""
Market Intel Agent (Databricks Compatible) - V4

Gathers external market intelligence: events, weather, Google Maps data.
- BrightData request endpoint with zone configuration (paid)

V4 Changes:
- Uses BrightData /request endpoint with zone
- Reduced requests to control spend
- Structured event output: Event, Description, Date
"""

import os
import time
import urllib.parse
from typing import Optional, Union
from agents.base_agent import BaseAgent

# Simple in-memory cache to reduce BrightData usage
_CACHE = {}
_CACHE_TTL_SEC = 60 * 30  # 30 minutes


class MarketIntelAgent(BaseAgent):
    """Specialist agent for market intelligence."""

    def get_system_prompt(self) -> str:
        return f"""You are a Market Intelligence Analyst for {self.hotel_name} in {self.city}.

STRICT RULES - NO HALLUCINATIONS:
1. ONLY report information that appears EXACTLY in tool outputs.
2. NEVER make up events, weather data, or ratings.
3. When reporting events, present structured data: Event, Description, Date.
4. If scraping fails or returns no data, say: "Could not retrieve [data type]."

Your job is to gather external information that affects hotel demand:
- Local events (concerts, conferences, festivals)
- Weather conditions
- Google Maps reviews and ratings
- Local attractions and points of interest

RESPONSE FORMAT FOR EVENTS:
- Present events with clear structure: Event, Description, Date
- Do NOT just list sources/URLs - extract the actual event information
- If event data is incomplete, note what's missing (e.g., "Date not specified")

Hotel context:
- Hotel ID: {self.hotel_id}
- Hotel Name: {self.hotel_name}
- City: {self.city}
"""

    def get_tools(self) -> list:
        """Return available tools based on environment."""
        tools = [
            self.search_google_maps,
            self.search_events,
            self.search_weather,
            self.search_web_brightdata,
        ]
        return tools

    # ==========================================
    # HELPER: BrightData SERP API
    # ==========================================
    
    def _serp_search(self, query: str, num_results: int = 10) -> dict:
        """Execute a Google SERP search via BrightData request endpoint."""
        import requests
        
        api_token = os.getenv("BRIGHTDATA_API_TOKEN")
        if not api_token:
            return {"error": "BrightData API token not configured"}

        base_url = os.getenv("BRIGHTDATA_BASE_URL", "https://api.brightdata.com/request")
        zone = os.getenv("BRIGHTDATA_ZONE", "serp_api3")
        format_type = os.getenv("BRIGHTDATA_SERP_FORMAT", "json")

        cache_key = f"serp::{query.lower()}::{num_results}::{format_type}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        encoded_query = urllib.parse.quote(query)
        google_url = f"https://www.google.com/search?q={encoded_query}&num={num_results}&hl=en"
        
        try:
            response = requests.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "zone": zone,
                    "url": google_url,
                    "format": format_type
                },
                timeout=30
            )
            
            if response.status_code == 200:
                if format_type.lower() == "json":
                    data = response.json()
                else:
                    data = {"raw": response.text}

                # BrightData /request often wraps payload under "body"
                if isinstance(data, dict) and "body" in data:
                    body = data.get("body", "")
                    if isinstance(body, dict):
                        data = body
                    elif isinstance(body, str):
                        try:
                            import json
                            data = json.loads(body)
                        except Exception:
                            data = {"raw": body}

                if os.getenv("BRIGHTDATA_DEBUG") == "1":
                    if isinstance(data, dict):
                        keys = ", ".join(list(data.keys())[:10])
                        print(f"[MarketIntel][Debug] SERP keys: {keys}")
                    if "raw" in data:
                        print(f"[MarketIntel][Debug] SERP raw length: {len(data['raw'])}")
                self._cache_set(cache_key, data)
                return data
            else:
                return {"error": f"SERP API failed: {response.status_code} - {response.text[:200]}"}
                
        except Exception as e:
            return {"error": f"SERP API error: {str(e)}"}

    def _cache_get(self, key: str):
        item = _CACHE.get(key)
        if not item:
            return None
        value, ts = item
        if time.time() - ts > _CACHE_TTL_SEC:
            _CACHE.pop(key, None)
            return None
        return value

    def _cache_set(self, key: str, value) -> None:
        _CACHE[key] = (value, time.time())

    # ==========================================
    # GOOGLE MAPS
    # ==========================================
    
    def search_google_maps(self, query: Optional[str] = None) -> str:
        """
        Search Google Maps for business data.
        
        Args:
            query: Hotel/business name to search. Defaults to this hotel.
        """
        search_query = query or f"{self.hotel_name} {self.city}"
        return self._google_maps_via_search(search_query)
    
    def _google_maps_brightdata(self, query: str) -> str:
        """Google Maps search via BrightData SERP API."""
        import requests
        
        api_token = os.getenv("BRIGHTDATA_API_TOKEN")
        if not api_token:
            return "BrightData API token not configured."
        
        print(f"[MarketIntel] Google Maps via BrightData: {query}")
        
        try:
            # Use Google Maps SERP endpoint if available; fallback to search
            response = requests.post(
                "https://api.brightdata.com/serp/google_maps",
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "country": "uk",
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_google_maps_result(data, query)
            else:
                # Fallback to regular search
                return self._google_maps_via_search(query)
                
        except Exception as e:
            return f"Google Maps search error: {str(e)}"
    
    def _google_maps_via_search(self, query: str) -> str:
        """Fallback: Get Maps-like data from regular Google search."""
        data = self._serp_search(f"{query} rating reviews address")
        
        if "error" in data:
            return data["error"]
        
        results = [f"=== Google Search Results for '{query}' ===\n"]
        
        # Check for knowledge panel (often has business info)
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            results.append(f"Business: {kg.get('title', 'N/A')}")
            results.append(f"Rating: {kg.get('rating', 'N/A')}")
            results.append(f"Address: {kg.get('address', 'N/A')}")
            results.append(f"Phone: {kg.get('phone', 'N/A')}")
            results.append("")
        
        # Add organic results
        if "organic" in data:
            for item in data["organic"][:3]:
                results.append(f"• {item.get('title', '')}")
                results.append(f"  {item.get('description', item.get('snippet', ''))[:200]}\n")
        
        return "\n".join(results)
    
    def _format_google_maps_result(self, data: dict, query: str) -> str:
        """Format BrightData Google Maps response."""
        results = [f"=== Google Maps Results for '{query}' ===\n"]
        
        if isinstance(data, dict):
            if "results" in data or "local_results" in data:
                items = data.get("results", data.get("local_results", []))
                for item in items[:5]:
                    name = item.get("title", item.get("name", "Unknown"))
                    rating = item.get("rating", "N/A")
                    reviews = item.get("reviews", item.get("review_count", "N/A"))
                    address = item.get("address", "N/A")
                    results.append(f"• {name}")
                    results.append(f"  Rating: {rating} ({reviews} reviews)")
                    results.append(f"  Address: {address}\n")
            else:
                results.append(str(data)[:1500])
        else:
            results.append(str(data)[:1500])
        
        return "\n".join(results)
    
    # ==========================================
    # EVENTS - V4 (Single SERP query, structured output)
    # ==========================================
    
    def search_events(self, city: Optional[str] = None, date: Optional[str] = None) -> str:
        """
        Search for local events using multiple targeted queries.
        
        Args:
            city: City to search. Defaults to hotel's city.
        date: Date/period (e.g., "this week", "January 2025"). Defaults to "this week".
        """
        search_city = city or self.city
        date_str = date or "this week"
        
        cache_key = f"events::{search_city.lower()}::{date_str.lower()}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        result = self._events_brightdata(search_city, date_str)
        self._cache_set(cache_key, result)
        return result
    
    def _events_brightdata(self, city: str, date_str: str) -> str:
        """
        Search events via a single targeted SERP query with actual dates.
        """
        from datetime import datetime, timedelta
        
        # Convert relative dates to actual dates for better search results
        today = datetime.now()
        
        if "this week" in date_str.lower():
            # Get dates for this week (Mon-Sun)
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)
            date_range = f"{start.strftime('%d %B')} to {end.strftime('%d %B %Y')}"
            query = f"events concerts shows {city} {date_range}"
        elif "this month" in date_str.lower():
            month_name = today.strftime("%B %Y")
            query = f"events concerts shows {city} {month_name}"
        elif "next week" in date_str.lower():
            start = today + timedelta(days=(7 - today.weekday()))
            end = start + timedelta(days=6)
            date_range = f"{start.strftime('%d %B')} to {end.strftime('%d %B %Y')}"
            query = f"events concerts shows {city} {date_range}"
        else:
            query = f"events concerts shows {city} {date_str}"
        
        print(f"[MarketIntel] Searching events for {city} ({date_str}) via BrightData SERP")
        print(f"[MarketIntel] Query: {query}")
        
        data = self._serp_search(query)
        events = self._extract_events_from_serp(data)

        if events:
            results = [f"=== Events in {city} ({date_str}) ===\n"]
            for event in events[:10]:
                results.append(f"Event: {event['name']}")
                results.append(f"Description: {event['description']}")
                results.append(f"Date: {event['date']}")
                if event.get("link"):
                    results.append(f"Link: {event['link']}")
                results.append("")
            return "\n".join(results)

        return f"No events found for {city} ({date_str}). Try different dates or check local event websites."
    
    def _extract_events_from_serp(self, data: dict) -> list:
        """Extract event-like items from SERP results into structured fields."""
        import re

        events = []

        if "error" in data:
            return events

        date_pattern = (
            r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
            r"|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b"
            r"|\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b"
            r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
            r"|\bJanuary|February|March|April|May|June|July|August|September|October|November|December\b"
        )

        if isinstance(data, dict):
            # Google Events widget (if present)
            if "events" in data:
                for item in data["events"][:10]:
                    events.append({
                        "name": item.get("title", "Unknown Event"),
                        "description": item.get("description", item.get("venue", "No description available")),
                        "date": item.get("date", "Date not specified"),
                    })

            # Organic results - be lenient, extract what we can
            if "organic" in data:
                # Debug: print first organic result keys
                if os.getenv("BRIGHTDATA_DEBUG") == "1" and data["organic"]:
                    print(f"[MarketIntel][Debug] Organic results count: {len(data['organic'])}")
                    first = data["organic"][0]
                    print(f"[MarketIntel][Debug] First organic keys: {list(first.keys())}")
                    print(f"[MarketIntel][Debug] First organic: {str(first)[:300]}")
                
                # Only skip truly useless pages
                hard_skip = ["home -", "official site", "cookie", "privacy policy", "login"]
                
                for idx, item in enumerate(data["organic"][:10]):
                    title = item.get("title", "").strip()
                    # BrightData uses 'description' not 'snippet'
                    snippet = item.get("description", item.get("snippet", "")).strip()
                    link = item.get("link", "")

                    if not title:
                        if os.getenv("BRIGHTDATA_DEBUG") == "1":
                            print(f"[MarketIntel][Debug] Skipped item {idx}: no title")
                        continue

                    title_lower = title.lower()
                    if any(phrase in title_lower for phrase in hard_skip):
                        if os.getenv("BRIGHTDATA_DEBUG") == "1":
                            print(f"[MarketIntel][Debug] Skipped item {idx}: hard_skip match in '{title[:50]}'")
                        continue

                    # Try to find a date
                    date_match = re.search(date_pattern, f"{title} {snippet}", re.IGNORECASE)
                    found_date = date_match.group(0) if date_match else "See link for dates"

                    # Use description
                    description = snippet[:300] if snippet else "No description available"

                    events.append({
                        "name": title,
                        "description": description,
                        "date": found_date,
                        "link": link,
                    })

        # Deduplicate
        seen = set()
        unique_events = []
        for event in events:
            key = event["name"].lower()[:60]
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        if os.getenv("BRIGHTDATA_DEBUG") == "1":
            print(f"[MarketIntel][Debug] Events before dedup: {len(events)}, after: {len(unique_events)}")

        return unique_events

    # ==========================================
    # WEATHER
    # ==========================================
    
    def search_weather(self, city: Optional[str] = None, days: Union[int, str] = 3) -> str:
        """
        Search for weather forecast.
        
        Args:
            city: City to search. Defaults to hotel's city.
            days: Number of days to forecast (default: 3)
        """
        search_city = city or self.city
        days = int(days)
        
        return self._weather_brightdata(search_city, days)
    
    def _weather_brightdata(self, city: str, days: int = 3) -> str:
        """Get weather via BrightData SERP API."""
        print(f"[MarketIntel] Weather for {city} via BrightData SERP")
        
        # Simple weather query - Google usually shows weather widget
        data = self._serp_search(f"weather {city}")
        
        if "error" in data:
            return data["error"]
        
        results = [f"=== Weather for {city} ===\n"]
        
        if isinstance(data, dict):
            # BrightData weather widget format (primary)
            if "weather" in data:
                w = data["weather"]
                
                # Temperature
                temp = w.get('temperature', w.get('temp', 'N/A'))
                results.append(f"Temperature: {temp}")
                
                # Conditions from subtitles (BrightData format) or description
                subtitles = w.get('subtitles', [])
                if len(subtitles) > 1:
                    results.append(f"Conditions: {subtitles[1]}")
                elif w.get('description'):
                    results.append(f"Conditions: {w['description']}")
                elif w.get('conditions'):
                    results.append(f"Conditions: {w['conditions']}")
                
                # Additional info (humidity, wind, precipitation - BrightData format)
                for info in w.get('additional_info', []):
                    info_type = info.get('type', '')
                    info_text = info.get('text', '')
                    if info_type and info_text:
                        results.append(f"{info_type}: {info_text}")
                
                # Fallback to direct keys (older format)
                if not w.get('additional_info'):
                    if "humidity" in w:
                        results.append(f"Humidity: {w['humidity']}")
                    if "wind" in w:
                        results.append(f"Wind: {w['wind']}")
                
                # Daily forecast
                if w.get('daily_forecast'):
                    results.append(f"\nForecast ({days} days):")
                    for day in w['daily_forecast'][:days]:
                        day_name = day.get('day', '')
                        temps = day.get('temperature', {})
                        high = temps.get('daytime', 'N/A')
                        low = temps.get('nighttime', 'N/A')
                        results.append(f"  {day_name}: High {high}°C, Low {low}°C")
            
            # Google answer_box fallback
            elif "answer_box" in data:
                ab = data["answer_box"]
                if "answer" in ab:
                    results.append(ab["answer"])
                elif "snippet" in ab:
                    results.append(ab["snippet"])
            
            # Knowledge graph fallback
            elif "knowledge_graph" in data:
                kg = data["knowledge_graph"]
                if kg.get("description"):
                    results.append(kg["description"][:300])
            
            # Organic results fallback
            if len(results) == 1 and "organic" in data:
                results.append("From weather websites:")
                for item in data["organic"][:3]:
                    title = item.get("title", "")
                    snippet = item.get("description", item.get("snippet", ""))[:200]
                    if "°" in snippet or "weather" in title.lower():
                        results.append(f"• {title}")
                        results.append(f"  {snippet}\n")
        
        if len(results) == 1:
            results.append("Could not retrieve structured weather data.")
            results.append(f"Check: https://www.google.com/search?q=weather+{city.replace(' ', '+')}")
        
        return "\n".join(results)

    # ==========================================
    # GENERAL WEB SEARCH
    # ==========================================
    
    def search_web_brightdata(self, query: str) -> str:
        """
        General web search using BrightData SERP API.
        
        Args:
            query: Search query
        """
        print(f"[MarketIntel] Web search: {query}")
        
        data = self._serp_search(query)
        
        if "error" in data:
            return data["error"]
        
        results = [f"=== Search Results for '{query}' ===\n"]
        
        # Answer box (direct answer)
        if "answer_box" in data:
            ab = data["answer_box"]
            answer = ab.get("answer", ab.get("snippet", ""))
            if answer:
                results.append(f"Quick Answer: {answer}\n")
        
        # Organic results
        if "organic" in data:
            for item in data["organic"][:5]:
                title = item.get("title", "Unknown")
                snippet = item.get("description", item.get("snippet", ""))[:200]
                link = item.get("link", "")
                results.append(f"• {title}")
                results.append(f"  {snippet}")
                if link:
                    results.append(f"  URL: {link}")
                results.append("")
        
        if len(results) == 1:
            results.append("No results found.")
        
        return "\n".join(results)