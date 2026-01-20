"""
Competitor Analyst Agent (Updated)

Identifies competitors using:
1. NLP Tool via Databricks (NEW - primary for Airbnb)
2. Geographic proximity (fallback)
"""

import os
from typing import Optional, List
from agents.base_agent import BaseAgent

# Import Databricks tools - try multiple paths
DATABRICKS_AVAILABLE = False
_import_error = None

# Only try Databricks imports if we're actually on Databricks
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    try:
        # Try direct import (when running from agents directory)
        from databricks_tools import (
            run_nlp_analysis,
            get_nlp_neighbors,
            get_nlp_topics,
            get_nlp_evidence,
            format_nlp_results,
            is_airbnb_property,
            extract_raw_id
        )
        DATABRICKS_AVAILABLE = True
        print("[CompetitorAnalyst] Databricks tools loaded successfully")
    except ImportError as e1:
        try:
            # Try with agents prefix
            from agents.databricks_tools import (
                run_nlp_analysis,
                get_nlp_neighbors,
                get_nlp_topics,
                get_nlp_evidence,
                format_nlp_results,
                is_airbnb_property,
                extract_raw_id
            )
            DATABRICKS_AVAILABLE = True
            print("[CompetitorAnalyst] Databricks tools loaded (agents prefix)")
        except ImportError as e2:
            _import_error = f"Import errors: {e1}, {e2}"
            print(f"[CompetitorAnalyst] Warning: Databricks tools not available: {_import_error}")
else:
    print("[CompetitorAnalyst] Not on Databricks - using fallback methods")

# Fallback: Define stub functions if not available
if not DATABRICKS_AVAILABLE:
    def is_airbnb_property(hotel_id: str) -> bool:
        return hotel_id.startswith("ABB_")
    
    def extract_raw_id(hotel_id: str) -> str:
        return hotel_id.replace("ABB_", "").replace("BKG_", "")


class CompetitorAnalystAgent(BaseAgent):
    """Specialist agent for competitor identification and review comparison."""

    # Topics covered by NLP tool
    NLP_TOPICS = [
        "cleanliness and hygiene", "maintenance and repairs", "smell and air quality",
        "noise and sound disturbances", "privacy and quietness", "safety and security",
        "bed comfort and sleep quality", "temperature and climate control",
        "space and layout comfort", "kitchen and cooking facilities", "bathroom quality",
        "wifi and internet quality", "appliances and amenities availability",
        "location and neighborhood", "public transportation access", "parking availability",
        "host communication and responsiveness", "check-in and check-out process",
        "accuracy of listing description", "value for money and pricing"
    ]

    def get_system_prompt(self) -> str:
        return f"""You are a Competitor Analyst for {self.hotel_name} in {self.city}.

STRICT RULES - NO HALLUCINATIONS:
1. ONLY list competitors that appear EXACTLY in tool outputs.
2. NEVER make up hotel names, ratings, or similarity scores.
3. Quote exact data: "From [tool]: [exact data]"
4. If no competitors found, say: "No competitors found in the database."

Your job is to:
1. Identify similar properties (competitors/neighbors)
2. Compare review sentiment across topics vs neighbors
3. Identify strengths and weaknesses vs competition

TOOLS AVAILABLE:
- analyze_vs_neighbors: Run full NLP analysis (takes ~5-10 minutes)
- find_competitors_geo: Quick geographic search (fallback)
- get_competitor_details: Get details for specific competitor

RESPONSE FORMAT:
- List competitors with similarity scores
- Show strengths (where you outperform neighbors)
- Show weaknesses (where neighbors outperform you)
- Include evidence quotes when available

Hotel context:
- Hotel ID: {self.hotel_id}
- Hotel Name: {self.hotel_name}
- City: {self.city}
"""

    def get_tools(self) -> list:
        tools = [
            self.find_competitors_geo,
            self.get_competitor_details,
        ]
        
        # Add NLP tool if available
        if DATABRICKS_AVAILABLE and is_airbnb_property(self.hotel_id):
            tools.insert(0, self.analyze_vs_neighbors)
            tools.append(self.get_topic_evidence)
        
        return tools

    def analyze_vs_neighbors(self, include_evidence: bool = False) -> str:
        """
        Run comprehensive NLP analysis comparing your property vs neighbors.
        
        This analyzes reviews across 20 topics and identifies:
        - Similar properties based on features and location
        - Topics where you have MORE negative reviews than neighbors (weaknesses)
        - Topics where you have FEWER negative reviews than neighbors (strengths)
        
        Args:
            include_evidence: Whether to include example review sentences
            
        Note: This analysis takes 5-10 minutes to complete.
        """
        if not DATABRICKS_AVAILABLE:
            return "NLP analysis not available. Use find_competitors_geo instead."
        
        if not is_airbnb_property(self.hotel_id):
            return f"NLP analysis only supports Airbnb properties. Current hotel ({self.hotel_id}) is not supported."
        
        print(f"\n{'='*50}")
        print(f"Analyzing {self.hotel_name} vs neighbors...")
        print(f"This may take 5-10 minutes. Please wait...")
        print(f"{'='*50}\n")
        
        # Run NLP analysis
        result = run_nlp_analysis(self.hotel_id)
        
        if result.get("status") not in ["ok", "success"]:
            return f"Analysis failed: {result.get('error_message', 'Unknown error')}"
        
        # Get results from Delta tables
        raw_id = extract_raw_id(self.hotel_id)
        topics = get_nlp_topics(property_id=raw_id)
        neighbors = get_nlp_neighbors(property_id=raw_id)
        evidence = get_nlp_evidence(run_id=result.get("run_id")) if include_evidence else None
        
        return format_nlp_results(topics, neighbors, include_evidence, evidence)

    def get_topic_evidence(self, topic: str, limit: int = 5) -> str:
        """
        Get evidence sentences for a specific topic.
        
        Args:
            topic: Topic to get evidence for (e.g., "wifi and internet quality")
            limit: Max number of evidence items
        """
        if not DATABRICKS_AVAILABLE:
            return "Evidence retrieval not available."
        
        evidence = get_nlp_evidence(topic=topic, limit=limit)
        
        if not evidence:
            return f"No evidence found for topic: {topic}"
        
        output = [f"=== Evidence for '{topic}' ===\n"]
        for e in evidence:
            text = e.get("sentence_text", "")[:300]
            sentiment = e.get("sentiment_label", "unknown")
            role = e.get("evidence_role", "")
            prop_id = e.get("evidence_property_id", "")
            
            output.append(f"[{sentiment.upper()}] (Property {prop_id}, {role})")
            output.append(f'  "{text}"')
            output.append("")
        
        return "\n".join(output)

    def find_competitors_geo(self, city: Optional[str] = None, k: int = 5) -> str:
        """
        Find competitors by geographic location (fallback method).
        
        Args:
            city: City to search in (defaults to hotel's city)
            k: Number of competitors to return
        """
        k = int(k)
        search_city = city or self.city
        print(f"[Competitor] Searching for competitors in {search_city}")
        
        competitors = []
        
        for namespace in ["booking_hotels", "airbnb_hotels"]:
            docs = self.search_rag(f"hotels in {search_city}", namespace=namespace, k=k)
            for doc in docs:
                name = doc.metadata.get("title", "")
                hotel_id = doc.metadata.get("hotel_id", "")
                
                # Skip own hotel
                if self.hotel_name.lower() in name.lower():
                    continue
                    
                competitors.append({
                    "hotel_id": hotel_id,
                    "name": name,
                    "source": "Booking.com" if "booking" in namespace else "Airbnb"
                })
        
        if not competitors:
            return f"No competitors found in {search_city}."
        
        output = f"=== Competitors in {search_city} ===\n\n"
        for i, comp in enumerate(competitors[:k], 1):
            output += f"{i}. {comp['name']}\n"
            output += f"   ID: {comp['hotel_id']} | Source: {comp['source']}\n\n"
        
        return output

    def get_competitor_details(self, hotel_id: str) -> str:
        """
        Get detailed information about a specific competitor.
        
        Args:
            hotel_id: Competitor's hotel ID
        """
        # Determine namespace
        if hotel_id.startswith("BKG_"):
            namespace = "booking_hotels"
        elif hotel_id.startswith("ABB_"):
            namespace = "airbnb_hotels"
        else:
            namespace = "booking_hotels"
        
        docs = self.search_rag(hotel_id, namespace=namespace, k=1)
        
        if docs:
            return f"=== Competitor Details ===\n\n{docs[0].page_content}"
        
        return f"No details found for competitor {hotel_id}"