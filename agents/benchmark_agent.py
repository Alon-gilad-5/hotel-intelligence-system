"""
Benchmark Agent

Compares hotel metrics against competitors using dynamic free-text queries.
"""

from typing import Optional, List
from agents.base_agent import BaseAgent
from agents.competitor_analyst import find_competitors_ml


class BenchmarkAgent(BaseAgent):
    """Specialist agent for metric benchmarking."""

    # Supported metrics with extraction patterns
    KNOWN_METRICS = {
        "price": ["price", "cost", "rate", "myr", "usd", "$"],
        "rating": ["rating", "score", "stars", "review_score"],
        "amenities": ["amenities", "facilities", "services"],
        "reviews": ["reviews", "feedback", "review count", "number_of_reviews"],
        "location": ["location", "distance", "nearby", "transport"],
        "cleanliness": ["cleanliness", "clean", "hygiene"],
    }

    def get_system_prompt(self) -> str:
        return f"""You are a Benchmark Analyst for {self.hotel_name} in {self.city}.

STRICT RULES - NO HALLUCINATIONS:
1. ONLY report numbers/metrics that appear EXACTLY in tool outputs.
2. NEVER make up prices, ratings, or rankings.
3. If a metric is "N/A" in tool output, report it as "N/A" - don't estimate.
4. Quote exact values: "Rating: 8.5 (from tool output)"

Your job is to compare hotel metrics against competitors.
Users can ask about ANY metric in free text - interpret their intent.

RESPONSE FORMAT:
- Use exact numbers from tool outputs
- Clearly label the source of each metric
- If comparison not possible (missing data), explain what's missing
- Rankings must be based on actual data, not assumptions

Examples of questions you handle:
- "How does my price compare to competitors?"
- "Am I rated higher than nearby hotels?"
- "Do competitors have better amenities?"
- "What's my position in the market?"

Hotel context:
- Hotel ID: {self.hotel_id}
- Hotel Name: {self.hotel_name}
- City: {self.city}
"""

    def get_tools(self) -> list:
        return [
            self.compare_metric,
            self.get_my_hotel_data,
            self.get_competitor_data,
            self.rank_by_metric,
        ]

    def _interpret_metric(self, user_query: str) -> str:
        """Use LLM to interpret what metric the user wants to compare."""
        prompt = f"""Extract the metric the user wants to compare from this query.
Return ONLY the metric name (one of: price, rating, amenities, reviews, location, cleanliness, or the exact term if none match).

Query: "{user_query}"
Metric:"""

        response = self.llm.invoke(prompt)
        return response.content.strip().lower()

    def compare_metric(self, metric: str, k: int = 5) -> str:
        """
        Compare a specific metric between your hotel and competitors.

        Args:
            metric: The metric to compare (e.g., "price", "rating", "amenities")
            k: Number of competitors to compare against
        """
        print(f"[Benchmark] Comparing metric: {metric}")

        # Get own hotel data
        my_data = self._get_hotel_metric(self.hotel_id, metric)

        # Get competitors
        competitors = self._find_competitors(k)

        if not competitors:
            return f"No competitors found to compare {metric}."

        # Get competitor metrics
        comp_data = []
        for comp in competitors:
            data = self._get_hotel_metric(comp["hotel_id"], metric)
            comp_data.append({
                "name": comp.get("name", comp["hotel_id"]),
                "hotel_id": comp["hotel_id"],
                "value": data.get("value"),
                "raw": data.get("raw", "N/A")
            })

        # Build comparison report
        output = f"=== {metric.title()} Comparison ===\n\n"
        output += f"Your Hotel ({self.hotel_name}):\n"
        output += f"  {metric.title()}: {my_data.get('raw', 'N/A')}\n\n"

        output += f"Competitors ({len(comp_data)}):\n"
        for i, comp in enumerate(comp_data, 1):
            output += f"  {i}. {comp['name']}: {comp['raw']}\n"

        # Calculate position if numeric
        if my_data.get("value") is not None:
            values = [c["value"] for c in comp_data if c["value"] is not None]
            if values:
                my_val = my_data["value"]
                higher = sum(1 for v in values if v > my_val)
                output += f"\nYour position: #{higher + 1} out of {len(values) + 1}\n"

        return output

    def get_my_hotel_data(self) -> str:
        """Get all available data for your hotel."""
        # Search in both sources
        for namespace in ["booking_hotels", "airbnb_hotels"]:
            docs = self.search_rag(
                self.hotel_name,
                namespace=namespace,
                k=1
            )
            if docs:
                return f"=== Your Hotel Data ===\n\n{docs[0].page_content}"

        return "Hotel data not found in database."

    def get_competitor_data(self, hotel_id: str) -> str:
        """
        Get data for a specific competitor.

        Args:
            hotel_id: Competitor hotel ID
        """
        if hotel_id.startswith("BKG_"):
            namespace = "booking_hotels"
        elif hotel_id.startswith("ABB_"):
            namespace = "airbnb_hotels"
        else:
            namespace = "booking_hotels"

        docs = self.search_rag(hotel_id, namespace=namespace, k=1)

        if docs:
            return f"=== Competitor Data ===\n\n{docs[0].page_content}"
        return f"No data found for {hotel_id}"

    def rank_by_metric(self, metric: str, k: int = 10) -> str:
        """
        Rank all hotels by a specific metric.

        Args:
            metric: Metric to rank by (e.g., "rating", "price")
            k: Number of hotels to include in ranking
        """
        print(f"[Benchmark] Ranking by: {metric}")

        # Get all hotels
        all_hotels = []

        for namespace in ["booking_hotels", "airbnb_hotels"]:
            docs = self.search_rag(f"hotels {metric}", namespace=namespace, k=k)
            for doc in docs:
                hotel_data = self._get_hotel_metric(
                    doc.metadata.get("hotel_id", "unknown"),
                    metric,
                    doc=doc
                )
                all_hotels.append({
                    "name": doc.metadata.get("title", "Unknown"),
                    "hotel_id": doc.metadata.get("hotel_id"),
                    "value": hotel_data.get("value"),
                    "raw": hotel_data.get("raw", "N/A"),
                    "is_mine": self.hotel_name.lower() in doc.metadata.get("title", "").lower()
                })

        # Sort by value (descending for rating, ascending for price)
        reverse = metric != "price"
        sorted_hotels = sorted(
            [h for h in all_hotels if h["value"] is not None],
            key=lambda x: x["value"],
            reverse=reverse
        )

        output = f"=== Ranking by {metric.title()} ===\n\n"
        for i, hotel in enumerate(sorted_hotels[:k], 1):
            marker = " ← YOUR HOTEL" if hotel["is_mine"] else ""
            output += f"{i}. {hotel['name']}: {hotel['raw']}{marker}\n"

        return output

    def _find_competitors(self, k: int = 5) -> List[dict]:
        """Find competitors using ML or geo fallback."""
        # Try ML first
        ml_results = find_competitors_ml(self.hotel_id, k=k)
        if ml_results:
            return ml_results

        # Fallback to geographic
        competitors = []

        for namespace in ["booking_hotels", "airbnb_hotels"]:
            docs = self.search_rag(f"hotels in {self.city}", namespace=namespace, k=k)
            for doc in docs:
                name = doc.metadata.get("title", "")
                if self.hotel_name.lower() not in name.lower():
                    competitors.append({
                        "hotel_id": doc.metadata.get("hotel_id"),
                        "name": name,
                        "source": "booking" if "booking" in namespace else "airbnb"
                    })

        return competitors[:k]

    def _get_hotel_metric(self, hotel_id: str, metric: str, doc=None) -> dict:
        """
        Extract a specific metric from hotel data, prioritizing Metadata over Regex.
        """
        if doc is None:
            # Fetch the document if not provided
            namespace = "booking_hotels" if hotel_id.startswith("BKG_") else "airbnb_hotels"
            docs = self.search_rag(hotel_id, namespace=namespace, k=1, filter_dict={"hotel_id": hotel_id})
            doc = docs[0] if docs else None

        if doc is None:
            return {"value": None, "raw": "N/A"}

        # 1. Try Structured Metadata First
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        import re

        # Check Rating in Metadata
        if metric == "rating":
            # Check for 'rating' or 'review_score' in metadata
            val = metadata.get("rating") or metadata.get("review_score")
            if val is not None:
                try:
                    return {"value": float(val), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            # Fallback: Regex
            match = re.search(r'(?:Rating|Score|Stars)[:\s]+(\d+\.?\d*)', content, re.IGNORECASE)
            if match:
                return {"value": float(match.group(1)), "raw": match.group(1)}

        # Check Price in Metadata
        elif metric == "price":
            val = metadata.get("price")
            if val is not None:
                try:
                    clean_val = str(val).replace("$", "").replace(",", "")
                    return {"value": float(clean_val), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            # Fallback: Regex
            match = re.search(r'(?:Price|Rate|Cost)[:\s]+[\$MYR\s]*(\d+(?:[.,]\d{2})?)', content, re.IGNORECASE)
            if match:
                return {"value": float(match.group(1).replace(',', '')), "raw": f"${match.group(1)}"}

        # Check Cleanliness in Metadata (Booking.com has category scores)
        elif metric == "cleanliness":
            # Try metadata fields
            val = metadata.get("cleanliness") or metadata.get("cleanliness_score") or metadata.get("clean")
            if val is not None:
                try:
                    return {"value": float(val), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            # Fallback: Regex patterns for cleanliness scores
            patterns = [
                r'Cleanliness[:\s]+(\d+\.?\d*)',
                r'Clean(?:liness)?[:\s]*(\d+\.?\d*)\s*(?:/\s*10)?',
                r'(?:Room|Hotel)\s+Clean(?:liness)?[:\s]+(\d+\.?\d*)',
            ]
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return {"value": float(match.group(1)), "raw": match.group(1)}
            # Try to find cleanliness mentions in text (qualitative)
            if re.search(r'\b(?:very\s+)?clean\b', content, re.IGNORECASE):
                return {"value": None, "raw": "Mentioned positively"}
            if re.search(r'\b(?:not\s+)?(?:clean|dirty)\b', content, re.IGNORECASE):
                return {"value": None, "raw": "Mentioned negatively"}

        # Check Location score
        elif metric == "location":
            val = metadata.get("location") or metadata.get("location_score")
            if val is not None:
                try:
                    return {"value": float(val), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            match = re.search(r'Location[:\s]+(\d+\.?\d*)', content, re.IGNORECASE)
            if match:
                return {"value": float(match.group(1)), "raw": match.group(1)}

        # Check Amenities/Facilities (count-based)
        elif metric == "amenities":
            # Try to count amenities mentioned
            amenity_keywords = ['wifi', 'pool', 'gym', 'spa', 'parking', 'breakfast', 'restaurant', 
                               'bar', 'room service', 'air conditioning', 'laundry', 'concierge']
            found = []
            content_lower = content.lower()
            for kw in amenity_keywords:
                if kw in content_lower:
                    found.append(kw)
            if found:
                return {"value": len(found), "raw": f"{len(found)} amenities: {', '.join(found[:5])}"}

        # Check Review count
        elif metric == "reviews":
            val = metadata.get("review_count") or metadata.get("number_of_reviews")
            if val is not None:
                try:
                    return {"value": int(val), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            match = re.search(r'(\d+)\s*reviews?', content, re.IGNORECASE)
            if match:
                return {"value": int(match.group(1)), "raw": match.group(1)}

        return {"value": None, "raw": "N/A"}