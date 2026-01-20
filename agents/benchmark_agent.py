"""
Benchmark Agent (Updated)

Compares hotel metrics against competitors using:
1. LR Tool via Databricks (NEW - primary for feature insights)
2. RAG-based ranking (kept for simple rankings)
"""

import os
from typing import Optional, List
from agents.base_agent import BaseAgent

# Import Databricks tools - only on Databricks
DATABRICKS_AVAILABLE = False

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    try:
        from databricks_tools import (
            run_lr_analysis,
            format_lr_insights,
            is_airbnb_property,
            extract_raw_id
        )
        DATABRICKS_AVAILABLE = True
        print("[BenchmarkAgent] Databricks LR tools loaded successfully")
    except ImportError as e:
        try:
            from agents.databricks_tools import (
                run_lr_analysis,
                format_lr_insights,
                is_airbnb_property,
                extract_raw_id
            )
            DATABRICKS_AVAILABLE = True
            print("[BenchmarkAgent] Databricks LR tools loaded (agents prefix)")
        except ImportError as e2:
            print(f"[BenchmarkAgent] Warning: Databricks LR tools not available: {e}, {e2}")
else:
    print("[BenchmarkAgent] Not on Databricks - using RAG-based ranking only")

# Fallback stub
if not DATABRICKS_AVAILABLE:
    def is_airbnb_property(hotel_id: str) -> bool:
        return hotel_id.startswith("ABB_")


class BenchmarkAgent(BaseAgent):
    """Specialist agent for metric benchmarking and feature impact analysis."""

    KNOWN_METRICS = {
        "price": ["price", "cost", "rate", "myr", "usd", "$"],
        "rating": ["rating", "score", "stars", "review_score"],
        "amenities": ["amenities", "facilities", "services"],
        "reviews": ["reviews", "feedback", "review count"],
        "location": ["location", "distance", "nearby"],
        "cleanliness": ["cleanliness", "clean", "hygiene"],
    }

    def get_system_prompt(self) -> str:
        return f"""You are a Benchmark Analyst for {self.hotel_name} in {self.city}.

STRICT RULES - NO HALLUCINATIONS:
1. ONLY report numbers/metrics that appear EXACTLY in tool outputs.
2. NEVER make up prices, ratings, or rankings.
3. If a metric is "N/A", report it as "N/A" - don't estimate.
4. Quote exact values from tool outputs.

Your job is to:
1. Explain WHY your rating differs from competitors (feature impact analysis)
2. Identify opportunities to improve ratings
3. Rank hotels by specific metrics

TOOLS AVAILABLE:
- analyze_feature_impact: ML-based analysis of what drives ratings (recommended for Airbnb)
- rank_by_metric: Simple ranking by price, rating, etc.
- get_my_hotel_data: Your hotel's data
- get_competitor_data: Specific competitor's data

RESPONSE FORMAT:
- Lead with key insights from feature analysis
- Show which features help vs hurt your rating
- Highlight actionable opportunities
- Use exact numbers from tool outputs

Hotel context:
- Hotel ID: {self.hotel_id}
- Hotel Name: {self.hotel_name}
- City: {self.city}
"""

    def get_tools(self) -> list:
        tools = [
            self.rank_by_metric,
            self.get_my_hotel_data,
            self.get_competitor_data,
        ]
        
        # Add LR tool if available and Airbnb property
        if DATABRICKS_AVAILABLE and is_airbnb_property(self.hotel_id):
            tools.insert(0, self.analyze_feature_impact)
        
        return tools

    def analyze_feature_impact(self, top_n: int = 10) -> str:
        """
        Analyze what features impact your rating vs neighbors using ML.
        
        This trains a Linear Regression model on your H3-bucket neighbors
        and identifies:
        - Which features have positive vs negative impact on ratings
        - How your values compare to market averages
        - Opportunities for improvement
        
        Args:
            top_n: Number of top insights to return
            
        Note: This analysis takes ~15 minutes to complete.
        """
        if not DATABRICKS_AVAILABLE:
            return "Feature impact analysis not available. Use rank_by_metric instead."
        
        if not is_airbnb_property(self.hotel_id):
            return f"Feature analysis only supports Airbnb properties. Current hotel ({self.hotel_id}) is not supported."
        
        print(f"\n{'='*50}")
        print(f"Analyzing feature impacts for {self.hotel_name}...")
        print(f"Training model on neighbors. This takes ~15 minutes...")
        print(f"{'='*50}\n")
        
        result = run_lr_analysis(self.hotel_id)
        
        if result.get("status") != "success":
            return f"Analysis failed: {result.get('error_message', result.get('message', 'Unknown error'))}"
        
        return format_lr_insights(result, top_n=int(top_n))

    def rank_by_metric(self, metric: str, k: int = 10) -> str:
        """
        Rank all hotels by a specific metric.
        
        Args:
            metric: Metric to rank by (rating, price, reviews)
            k: Number of hotels to include
        """
        k = int(k)
        print(f"[Benchmark] Ranking by: {metric}")
        
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
        
        # Sort
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

    def get_my_hotel_data(self) -> str:
        """Get all available data for your hotel."""
        for namespace in ["booking_hotels", "airbnb_hotels"]:
            docs = self.search_rag(self.hotel_name, namespace=namespace, k=1)
            if docs:
                return f"=== Your Hotel Data ===\n\n{docs[0].page_content}"
        return "Hotel data not found in database."

    def get_competitor_data(self, hotel_id: str) -> str:
        """Get data for a specific competitor."""
        namespace = "booking_hotels" if hotel_id.startswith("BKG_") else "airbnb_hotels"
        docs = self.search_rag(hotel_id, namespace=namespace, k=1)
        
        if docs:
            return f"=== Competitor Data ===\n\n{docs[0].page_content}"
        return f"No data found for {hotel_id}"

    def _get_hotel_metric(self, hotel_id: str, metric: str, doc=None) -> dict:
        """Extract a specific metric from hotel data."""
        import re
        
        if doc is None:
            namespace = "booking_hotels" if hotel_id.startswith("BKG_") else "airbnb_hotels"
            docs = self.search_rag(hotel_id, namespace=namespace, k=1)
            doc = docs[0] if docs else None
        
        if doc is None:
            return {"value": None, "raw": "N/A"}
        
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        if metric == "rating":
            val = metadata.get("rating") or metadata.get("review_score")
            if val:
                try:
                    return {"value": float(val), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            match = re.search(r'(?:Rating|Score)[:\s]+(\d+\.?\d*)', content, re.IGNORECASE)
            if match:
                return {"value": float(match.group(1)), "raw": match.group(1)}
        
        elif metric == "price":
            val = metadata.get("price")
            if val:
                try:
                    clean = str(val).replace("$", "").replace(",", "")
                    return {"value": float(clean), "raw": str(val)}
                except (ValueError, TypeError):
                    pass
            match = re.search(r'(?:Price|Rate)[:\s]+[\$]?(\d+(?:\.\d{2})?)', content, re.IGNORECASE)
            if match:
                return {"value": float(match.group(1)), "raw": f"${match.group(1)}"}
        
        return {"value": None, "raw": "N/A"}
