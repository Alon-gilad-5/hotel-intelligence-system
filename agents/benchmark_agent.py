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
        # Build tool descriptions based on availability
        tool_descriptions = """- rank_by_metric: Compare your property against others by price, rating, etc.
- get_my_hotel_data: Your property's details
- get_competitor_data: A specific competitor's details"""
        
        if DATABRICKS_AVAILABLE and is_airbnb_property(self.hotel_id):
            tool_descriptions = """- analyze_feature_impact: ML analysis showing which features affect ratings (takes ~15 min)
""" + tool_descriptions
        
        return f"""You are a friendly Benchmark Analyst helping the owner of {self.hotel_name} in {self.city}.

ACCURACY RULES:
1. Only report numbers that appear in tool outputs - never invent statistics.
2. If data is unavailable, say so honestly.
3. ONLY call tools listed below - do NOT attempt to call any other tools.

YOUR ROLE:
Help the property owner understand what features impact their ratings and how to improve.

AVAILABLE TOOLS (only use these):
{tool_descriptions}

HOW TO RESPOND:
Write like you're a data-savvy consultant explaining findings to a client.

1. **Use only your available tools** - Don't reference or try to call tools not listed above
2. **Lead with insights from available data** - Use ranking and hotel data to provide recommendations
3. **Explain in plain English** - Make data understandable
4. **Prioritize actionable items** - Focus on things they can actually change
5. **Be encouraging** - Highlight what they're doing well, not just problems

Property: {self.hotel_name} (ID: {self.hotel_id}) in {self.city}
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
            top_n: Number of top insights to return (analysis takes ~15 minutes)
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
        
        return format_lr_insights(result, top_n=int(top_n), property_id=self.hotel_id)

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
