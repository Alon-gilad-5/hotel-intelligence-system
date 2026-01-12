"""
Competitor Analyst Agent

Identifies competitors using geographic proximity and ML-based similarity.
"""

from typing import Optional, List
from agents.base_agent import BaseAgent


# ===========================================
# BLACK-BOX ML INTEGRATION POINT
# ===========================================

def find_competitors_ml(hotel_id: str, k: int = 5) -> List[dict]:
    """
    Find competitors using ML-based similarity.

    THIS IS THE BLACK-BOX INTEGRATION POINT.
    Replace the internals when your team's ML model is ready.

    Expected return format:
    [
        {"hotel_id": "ABB_123", "similarity_score": 0.95, "source": "airbnb"},
        {"hotel_id": "BKG_456", "similarity_score": 0.89, "source": "booking"},
        ...
    ]

    Args:
        hotel_id: The hotel to find competitors for
        k: Number of competitors to return

    Returns:
        List of competitor dicts with hotel_id, similarity_score, source
    """
    # TODO: Replace with actual ML model call
    # Example integration:
    # from ml_model import CompetitorModel
    # model = CompetitorModel.load("path/to/model")
    # return model.predict(hotel_id, k=k)

    # Current placeholder: returns empty list (falls back to geo)
    print(f"[ML Black-Box] Called with hotel_id={hotel_id}, k={k}")
    print("[ML Black-Box] No model integrated yet - returning empty list")
    return []


# ===========================================
# COMPETITOR ANALYST AGENT
# ===========================================

class CompetitorAnalystAgent(BaseAgent):
    """Specialist agent for competitor identification."""

    def get_system_prompt(self) -> str:
        return f"""You are a Competitor Analyst for {self.hotel_name} in {self.city}.

    Your job is to identify and analyze competitors.
    You can find competitors by:
    1. Geographic proximity (same city/area)
    2. ML-based similarity (when available)
    
    When answering:
    1. Use the appropriate tool based on the query
    2. Provide relevant competitor details
    3. Explain why they are competitors
    
    Hotel context:
    - Hotel ID: {self.hotel_id}
    - Hotel Name: {self.hotel_name}
    - City: {self.city}
    """

    def get_tools(self) -> list:
        return [
            self.find_competitors_geo,
            self.find_competitors_similar,
            self.get_competitor_details,
        ]

    def find_competitors_geo(self, city: Optional[str] = None, k: int = 5) -> str:
        """
        Find competitors by geographic location.

        Args:
            city: City to search in (defaults to hotel's city)
            k: Number of competitors to return
        """
        search_city = city or self.city

        # Search both Booking and Airbnb hotels
        booking_docs = self.search_rag(
            f"hotels in {search_city}",
            namespace="booking_hotels",
            k=k
        )

        airbnb_docs = self.search_rag(
            f"properties in {search_city}",
            namespace="airbnb_hotels",
            k=k
        )

        # Filter out own hotel
        competitors = []

        for doc in booking_docs:
            hotel_name = doc.metadata.get("title", "")
            if self.hotel_name.lower() not in hotel_name.lower():
                competitors.append({
                    "name": hotel_name,
                    "hotel_id": doc.metadata.get("hotel_id"),
                    "source": "booking",
                    "rating": doc.metadata.get("rating"),
                    "city": doc.metadata.get("city"),
                })

        for doc in airbnb_docs:
            hotel_name = doc.metadata.get("title", "")
            if self.hotel_name.lower() not in hotel_name.lower():
                competitors.append({
                    "name": hotel_name,
                    "hotel_id": doc.metadata.get("hotel_id"),
                    "source": "airbnb",
                    "rating": doc.metadata.get("rating"),
                    "city": doc.metadata.get("city"),
                })

        if not competitors:
            return f"No competitors found in {search_city}."

        output = f"=== Geographic Competitors in {search_city} ({len(competitors[:k])} found) ===\n\n"
        for i, comp in enumerate(competitors[:k], 1):
            output += f"{i}. {comp['name']}\n"
            output += f"   Source: {comp['source'].title()}\n"
            output += f"   Rating: {comp['rating']}\n"
            output += f"   ID: {comp['hotel_id']}\n\n"

        return output

    def find_competitors_similar(self, k: int = 5) -> str:
        """
        Find competitors using ML-based similarity model.
        Falls back to geographic search if ML model not available.

        Args:
            k: Number of competitors to return
        """
        # Try ML-based approach first
        ml_results = find_competitors_ml(self.hotel_id, k=k)

        if ml_results:
            output = f"=== ML-Based Similar Competitors ({len(ml_results)} found) ===\n\n"
            for i, comp in enumerate(ml_results, 1):
                output += f"{i}. Hotel ID: {comp['hotel_id']}\n"
                output += f"   Similarity: {comp['similarity_score']:.2f}\n"
                output += f"   Source: {comp['source'].title()}\n\n"
            return output

        # Fallback to geographic
        print("[CompetitorAnalyst] ML model not available, falling back to geographic search")
        return self.find_competitors_geo(k=k)

    def get_competitor_details(self, hotel_id: str) -> str:
        """
        Get detailed information about a specific competitor.

        Args:
            hotel_id: The competitor's hotel ID (e.g., "BKG_123" or "ABB_456")
        """
        # Determine source from ID prefix
        if hotel_id.startswith("BKG_"):
            namespace = "booking_hotels"
        elif hotel_id.startswith("ABB_"):
            namespace = "airbnb_hotels"
        else:
            return f"Invalid hotel ID format: {hotel_id}"

        # Search by hotel_id
        docs = self.search_rag(
            hotel_id,
            namespace=namespace,
            k=1,
            filter_dict={"hotel_id": hotel_id}
        )

        if not docs:
            # Fallback: search without filter
            docs = self.search_rag(hotel_id, namespace=namespace, k=1)

        if not docs:
            return f"No details found for hotel ID: {hotel_id}"

        return f"=== Competitor Details ===\n\n{docs[0].page_content}"