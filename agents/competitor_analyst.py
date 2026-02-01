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
    
    # Mapping common user terms to NLP topic names
    TOPIC_ALIASES = {
        "responsive": "host communication and responsiveness",
        "responsiveness": "host communication and responsiveness",
        "response time": "host communication and responsiveness",
        "communication": "host communication and responsiveness",
        "host": "host communication and responsiveness",
        "wifi": "wifi and internet quality",
        "internet": "wifi and internet quality",
        "clean": "cleanliness and hygiene",
        "cleanliness": "cleanliness and hygiene",
        "noise": "noise and sound disturbances",
        "quiet": "privacy and quietness",
        "location": "location and neighborhood",
        "parking": "parking availability",
        "check-in": "check-in and check-out process",
        "check-out": "check-in and check-out process",
        "price": "value for money and pricing",
        "value": "value for money and pricing",
        "bed": "bed comfort and sleep quality",
        "sleep": "bed comfort and sleep quality",
        "kitchen": "kitchen and cooking facilities",
        "bathroom": "bathroom quality",
        "temperature": "temperature and climate control",
        "ac": "temperature and climate control",
        "heating": "temperature and climate control",
        "maintenance": "maintenance and repairs",
        "safety": "safety and security",
        "security": "safety and security",
    }

    def get_system_prompt(self) -> str:
        # Build tool descriptions based on availability (geo tools are fallback-only)
        if DATABRICKS_AVAILABLE and is_airbnb_property(self.hotel_id):
            tool_descriptions = """- analyze_vs_neighbors: Deep NLP analysis comparing reviews across all topics (takes 5-10 min)
- get_topic_evidence: Get specific review evidence for a topic - USE THIS when user asks about a specific topic!"""
        else:
            tool_descriptions = """- find_competitors_geo: Quick search for nearby competitors (fallback only)
- get_competitor_details: Details about a specific competitor (fallback only)"""
        
        # List available topics for the agent
        topics_list = ", ".join([f'"{t}"' for t in self.NLP_TOPICS[:5]]) + "..."
        
        return f"""{self.get_chart_instruction()}You are a friendly Competitor Analyst helping the owner of {self.hotel_name} in {self.city}.

ACCURACY RULES:
1. Only report data from tool outputs - never invent statistics.
2. If tools return no data, honestly say you couldn't find information.
3. ONLY call tools listed below - do NOT attempt to call any other tools.

YOUR ROLE:
Help the property owner understand how they compare to similar properties nearby.
Focus on actionable insights they can use to improve their business.

CRITICAL - TOPIC-SPECIFIC QUESTIONS:
When the user asks about a SPECIFIC topic (e.g., "responsiveness", "wifi", "cleanliness"):
1. First run analyze_vs_neighbors to get comparison data
2. Look for that specific topic in the results (strengths/weaknesses sections)
3. If you need more detail, use get_topic_evidence with the matching topic name
4. **Focus your answer on THAT SPECIFIC TOPIC** - don't give a general overview

Topic name mappings (user term → NLP topic):
- "responsiveness" or "responsive" → "host communication and responsiveness"
- "wifi" or "internet" → "wifi and internet quality"
- "clean" or "cleanliness" → "cleanliness and hygiene"
- "check-in" → "check-in and check-out process"
- "price" or "value" → "value for money and pricing"

Available NLP topics: {topics_list}

AVAILABLE TOOLS (only use these):
{tool_descriptions}

HOW TO RESPOND:
Write conversationally, like you're advising a friend who owns this property.

1. **Answer the SPECIFIC question asked** - If they ask about responsiveness, focus on responsiveness!
2. **Find the relevant topic in results** - Look in strengths/weaknesses for the topic they asked about
3. **Report actual data** - Include negative rates and gaps from the analysis
4. **Give specific recommendations** - What should they actually DO to improve?
5. **Keep it focused** - Don't list all topics if they only asked about one

Property: {self.hotel_name} (ID: {self.hotel_id}) in {self.city}
"""

    def get_tools(self) -> list:
        # Only expose geo tools when NLP tools are unavailable or unsupported.
        if DATABRICKS_AVAILABLE and is_airbnb_property(self.hotel_id):
            return [self.analyze_vs_neighbors, self.get_topic_evidence]
        return [self.find_competitors_geo, self.get_competitor_details]

    def analyze_vs_neighbors(self, include_evidence: bool = False, focus_topic: str = None) -> str:
        """
        Run comprehensive NLP analysis comparing your property vs neighbors.
        
        This analyzes reviews across 20 topics and identifies:
        - Similar properties based on features and location
        - Topics where you have MORE negative reviews than neighbors (weaknesses)
        - Topics where you have FEWER negative reviews than neighbors (strengths)
        
        Args:
            include_evidence: Whether to include example review sentences (analysis takes 5-10 minutes)
            focus_topic: Optional - a specific topic to highlight in results (e.g., "responsiveness", "wifi")
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
            # Fallback to geo methods when NLP tool fails
            print("[CompetitorAnalyst] NLP tool failed; falling back to geo-based competitor search.")
            fallback_header = "Analysis failed; falling back to geo-based competitor search."
            geo_results = self.find_competitors_geo()
            return f"{fallback_header}\n\n{geo_results}\n\nDetails:\n{result.get('error_message', 'Unknown error')}"
        
        # Get results from Delta tables
        raw_id = extract_raw_id(self.hotel_id)
        topics = get_nlp_topics(property_id=raw_id)
        neighbors = get_nlp_neighbors(property_id=raw_id)
        evidence = get_nlp_evidence(run_id=result.get("run_id")) if include_evidence else None
        
        # Extract charts from result (check ui_artifacts.charts)
        charts = result.get("charts", {})
        if not charts:
            ui_artifacts = result.get("ui_artifacts", {})
            if isinstance(ui_artifacts, dict):
                charts = ui_artifacts.get("charts", {})
        
        # Format base results (now includes charts)
        output = format_nlp_results(
            topics, neighbors, include_evidence, evidence,
            charts=charts, property_id=self.hotel_id
        )
        
        # If a focus topic was specified, add specific data for that topic
        if focus_topic:
            focus_topic_lower = focus_topic.lower().strip()
            matched_topic = self.TOPIC_ALIASES.get(focus_topic_lower, focus_topic)
            
            # Find the topic in results
            topic_data = None
            for t in topics:
                if matched_topic.lower() in t.get("topic", "").lower():
                    topic_data = t
                    break
            
            if topic_data:
                output += f"\n\n=== FOCUS: {matched_topic.title()} ===\n"
                output += f"Your negative rate: {topic_data.get('target_negative_rate', 0):.1%}\n"
                output += f"Neighbors' negative rate: {topic_data.get('neighbors_negative_rate', 0):.1%}\n"
                output += f"Gap: {topic_data.get('negative_rate_gap', 0):+.1%}\n"
                output += f"Classification: {topic_data.get('kind', 'neutral').upper()}\n"
            else:
                output += f"\n\n=== FOCUS: {matched_topic.title()} ===\n"
                output += f"No specific data found for '{matched_topic}'. Try get_topic_evidence for review quotes.\n"
        
        return output

    def get_topic_evidence(self, topic: str, limit: int = 5) -> str:
        """
        Get actual review sentences for a specific topic - USE THIS when user asks about a specific topic!
        
        This returns real guest review quotes that mention the topic, helping you understand
        what guests actually say about responsiveness, wifi, cleanliness, etc.
        
        Args:
            topic: The NLP topic name. Must use exact topic names:
                   - "host communication and responsiveness" (for responsiveness questions)
                   - "wifi and internet quality" (for wifi/internet questions)
                   - "cleanliness and hygiene" (for cleanliness questions)
                   - "check-in and check-out process" (for check-in questions)
                   - "value for money and pricing" (for price/value questions)
                   - "noise and sound disturbances" (for noise questions)
                   - "bed comfort and sleep quality" (for bed/sleep questions)
                   - "location and neighborhood" (for location questions)
            limit: Max number of evidence items to return (default 5)
        
        Returns:
            Actual review quotes mentioning the topic with sentiment labels
        """
        if not DATABRICKS_AVAILABLE:
            return "Evidence retrieval not available."
        
        # Try to match user's term to exact topic name
        topic_lower = topic.lower().strip()
        matched_topic = self.TOPIC_ALIASES.get(topic_lower, topic)
        
        # Also check if any NLP topic contains the search term
        if matched_topic == topic:  # No alias found
            for nlp_topic in self.NLP_TOPICS:
                if topic_lower in nlp_topic.lower():
                    matched_topic = nlp_topic
                    break
        
        print(f"[CompetitorAnalyst] Getting evidence for topic: '{matched_topic}'")
        
        evidence = get_nlp_evidence(topic=matched_topic, limit=limit)
        
        if not evidence:
            return f"No evidence found for topic: {matched_topic}. Try one of: {', '.join(self.NLP_TOPICS[:5])}..."
        
        output = [f"=== Guest Reviews about '{matched_topic}' ===\n"]
        for e in evidence:
            text = e.get("sentence_text", "")[:300]
            sentiment = e.get("sentiment_label", "unknown")
            role = e.get("evidence_role", "")
            prop_id = e.get("evidence_property_id", "")
            
            # Make it clear if this is about the user's property or neighbors
            source = "YOUR PROPERTY" if role == "target" else "NEIGHBOR"
            
            output.append(f"[{sentiment.upper()}] ({source} - Property {prop_id})")
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