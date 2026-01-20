"""
Databricks Tools Integration

Wrapper functions for calling NLP and LR notebooks via dbutils.notebook.run().
Provides clean interfaces for agents to use Databricks jobs.

Note: PySpark imports are done lazily inside functions to avoid import errors
when this module is loaded outside of Databricks.
"""

import json
from typing import Optional, Dict, Any, List


# ===========================================
# CONFIGURATION
# ===========================================

# Notebook paths (update if cloned to different location)
NLP_NOTEBOOK_PATH = "/Workspace/Users/alon-gilad@campus.technion.ac.il/(Clone) NLP Tool For Review Analysis"
LR_NOTEBOOK_PATH = "/Workspace/Users/alon-gilad@campus.technion.ac.il/(Alon Clone) linear regression"

# Data paths
DELTA_PATH = "/mnt/lab94290/cluster_19/airbnb_h3_simvector"
NLP_OUTPUT_BASE = "/mnt/lab94290/cluster_19/nlp_outputs"

# FileStore path for charts (accessible via URL)
FILESTORE_CHARTS_PATH = "/FileStore/hotel_intel_charts"

# Timeouts (seconds) - Both tools take ~15 minutes
NLP_TIMEOUT = 1200  # 20 minutes for NLP
LR_TIMEOUT = 1200   # 20 minutes for LR


# ===========================================
# HELPER FUNCTIONS
# ===========================================

def extract_raw_id(hotel_id: str) -> str:
    """
    Strip prefix from hotel ID.
    
    Args:
        hotel_id: ID like 'ABB_40458495' or 'BKG_12345'
        
    Returns:
        Raw numeric ID like '40458495'
    """
    return hotel_id.replace("ABB_", "").replace("BKG_", "")


def get_source(hotel_id: str) -> str:
    """
    Determine data source from prefix.
    
    Args:
        hotel_id: ID with prefix
        
    Returns:
        'airbnb', 'booking', or 'unknown'
    """
    if hotel_id.startswith("ABB_"):
        return "airbnb"
    elif hotel_id.startswith("BKG_"):
        return "booking"
    return "unknown"


def is_airbnb_property(hotel_id: str) -> bool:
    """Check if property is from Airbnb (supported by new tools)."""
    return hotel_id.startswith("ABB_")


def save_chart_to_filestore(b64_data: str, filename: str, property_id: str) -> Optional[str]:
    """
    Save a base64-encoded chart to Databricks FileStore and return its URL.
    
    Args:
        b64_data: Base64-encoded image data
        filename: Name for the file (e.g., 'impact_chart')
        property_id: Property ID for organizing files
    
    The URL format is: https://<workspace>/files/<path>
    """
    import base64
    import os
    
    try:
        # Get dbutils from notebook namespace (Databricks injects it as a global)
        try:
            # Method 1: Get from global namespace (works when module is imported in notebook)
            import sys
            if 'dbutils' in dir(sys.modules.get('__main__', {})):
                dbutils = getattr(sys.modules['__main__'], 'dbutils')
            else:
                # Method 2: Get from IPython namespace (Databricks uses IPython)
                from IPython import get_ipython
                ip = get_ipython()
                if ip and 'dbutils' in ip.user_ns:
                    dbutils = ip.user_ns['dbutils']
                else:
                    print("[Chart] dbutils not available - cannot save to FileStore")
                    return None
        except Exception as e:
            print(f"[Chart] Failed to get dbutils: {e}")
            return None
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(b64_data)
        
        # Create unique path for this property
        file_path = f"{FILESTORE_CHARTS_PATH}/{property_id}/{filename}.png"
        
        # Ensure directory exists
        dir_path = f"{FILESTORE_CHARTS_PATH}/{property_id}"
        try:
            dbutils.fs.mkdirs(dir_path)
        except:
            pass  # Directory might already exist
        
        # Write file to DBFS (FileStore is part of DBFS)
        dbutils.fs.put(file_path, b64_data, overwrite=True)
        
        # For binary files, we need to use a different approach
        # FileStore expects files in /dbfs/ path for direct write
        local_path = f"/dbfs{file_path}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(img_bytes)
        
        # Get workspace URL for constructing the file URL
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
        except:
            workspace_url = "your-databricks-workspace.cloud.databricks.com"
        
        # FileStore URLs follow this pattern
        url_path = file_path.replace("/FileStore/", "files/")
        full_url = f"https://{workspace_url}/{url_path}"
        
        print(f"[Chart] Saved to: {full_url}")
        return full_url
        
    except Exception as e:
        print(f"[Chart] Failed to save chart: {e}")
        return None


# ===========================================
# NLP TOOL WRAPPER
# ===========================================

def run_nlp_analysis(
    hotel_id: str,
    delta_path: str = DELTA_PATH,
    out_base_path: str = NLP_OUTPUT_BASE,
    timeout_seconds: int = NLP_TIMEOUT,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run NLP review analysis comparing property vs neighbors.
    
    This tool:
    - Finds similar properties using H3 geo-buckets + cosine similarity
    - Analyzes review sentiment across 20 topics
    - Identifies strengths and weaknesses vs neighbors
    
    Args:
        hotel_id: Hotel ID (will strip ABB_/BKG_ prefix)
        delta_path: Path to Delta table with property data
        out_base_path: Base path for Delta outputs
        timeout_seconds: Max wait time
        params: Optional dict of NLP parameters
        
    Returns:
        Dict with status, paths, and analysis results
    """
    # Validate source
    if not is_airbnb_property(hotel_id):
        return {
            "status": "error",
            "error_message": f"NLP tool only supports Airbnb properties (ABB_ prefix). Got: {hotel_id}"
        }
    
    raw_id = extract_raw_id(hotel_id)
    params_json = json.dumps(params) if params else ""
    
    print(f"[NLP Tool] Analyzing property {raw_id}...")
    print(f"[NLP Tool] This takes ~15 minutes. Please wait...")
    
    try:
        # Call Databricks notebook (dbutils is available in Databricks runtime)
        result = dbutils.notebook.run(
            NLP_NOTEBOOK_PATH,
            timeout_seconds,
            arguments={
                "pid": raw_id,
                "delta_path": delta_path,
                "out_base_path": out_base_path,
                "params_json": params_json
            }
        )
        
        # Parse result
        result_dict = json.loads(result)
        
        # Handle the bug where success is wrapped in error
        if result_dict.get("status") == "error" and "error_message" in result_dict:
            try:
                inner = json.loads(result_dict["error_message"])
                if inner.get("status") == "ok":
                    result_dict = inner
            except (json.JSONDecodeError, TypeError):
                pass
        
        print(f"[NLP Tool] Analysis complete. Status: {result_dict.get('status')}")
        return result_dict
        
    except Exception as e:
        error_msg = str(e)
        if "TIMEDOUT" in error_msg:
            return {
                "status": "error",
                "error_message": f"Analysis timed out after {timeout_seconds} seconds. Try increasing timeout."
            }
        return {
            "status": "error",
            "error_message": error_msg
        }


def get_nlp_neighbors(run_id: Optional[str] = None, property_id: Optional[str] = None) -> List[Dict]:
    """
    Read neighbors from NLP output Delta table.
    
    Args:
        run_id: Specific run ID to filter
        property_id: Property ID to filter (raw, no prefix)
        
    Returns:
        List of neighbor dicts with similarity scores
    """
    # Lazy import - only works on Databricks
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    
    neighbors_path = f"{NLP_OUTPUT_BASE}/neighbors"
    
    try:
        df = spark.read.format("delta").load(neighbors_path)
        
        if run_id:
            df = df.filter(df.run_id == run_id)
        if property_id:
            df = df.filter(df.property_id == property_id)
            
        # Get most recent run if no filter
        if not run_id and not property_id:
            df = df.orderBy(df.run_utc_ts.desc()).limit(50)
            
        return [row.asDict() for row in df.collect()]
        
    except Exception as e:
        print(f"[NLP Tool] Error reading neighbors: {e}")
        return []


def get_nlp_topics(run_id: Optional[str] = None, property_id: Optional[str] = None, kind: Optional[str] = None) -> List[Dict]:
    """
    Read topic analysis from NLP output Delta table.
    
    Args:
        run_id: Specific run ID to filter
        property_id: Property ID to filter
        kind: 'weakness' or 'strength' to filter
        
    Returns:
        List of topic dicts with negative rate gaps
    """
    # Lazy import - only works on Databricks
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    
    topics_path = f"{NLP_OUTPUT_BASE}/topics"
    
    try:
        df = spark.read.format("delta").load(topics_path)
        
        if run_id:
            df = df.filter(df.run_id == run_id)
        if property_id:
            df = df.filter(df.property_id == property_id)
        if kind:
            df = df.filter(df.kind == kind)
            
        # Get most recent run if no filter
        if not run_id and not property_id:
            df = df.orderBy(df.run_utc_ts.desc()).limit(20)
            
        return [row.asDict() for row in df.collect()]
        
    except Exception as e:
        print(f"[NLP Tool] Error reading topics: {e}")
        return []


def get_nlp_evidence(run_id: Optional[str] = None, topic: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Read evidence sentences from NLP output Delta table.
    
    Args:
        run_id: Specific run ID to filter
        topic: Topic to filter evidence for
        limit: Max number of evidence items
        
    Returns:
        List of evidence dicts with sentences
    """
    # Lazy import - only works on Databricks
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    
    evidence_path = f"{NLP_OUTPUT_BASE}/evidence"
    
    try:
        df = spark.read.format("delta").load(evidence_path)
        
        if run_id:
            df = df.filter(df.run_id == run_id)
        if topic:
            df = df.filter(df.topic == topic)
            
        df = df.limit(limit)
        return [row.asDict() for row in df.collect()]
        
    except Exception as e:
        print(f"[NLP Tool] Error reading evidence: {e}")
        return []


# ===========================================
# LR TOOL WRAPPER
# ===========================================

def run_lr_analysis(
    hotel_id: str,
    timeout_seconds: int = LR_TIMEOUT
) -> Dict[str, Any]:
    """
    Run Linear Regression benchmark analysis.
    
    This tool:
    - Trains LR model on H3-bucket neighbors
    - Identifies feature impacts on ratings
    - Provides actionable insights and opportunities
    
    Args:
        hotel_id: Hotel ID (will strip ABB_/BKG_ prefix)
        timeout_seconds: Max wait time
        
    Returns:
        Dict with status, insights, and chart data
    """
    # Validate source
    if not is_airbnb_property(hotel_id):
        return {
            "status": "error",
            "error_message": f"LR tool only supports Airbnb properties (ABB_ prefix). Got: {hotel_id}"
        }
    
    raw_id = extract_raw_id(hotel_id)
    
    print(f"[LR Tool] Analyzing property {raw_id}...")
    print(f"[LR Tool] Training model on neighbors. This takes ~15 minutes...")
    
    try:
        result = dbutils.notebook.run(
            LR_NOTEBOOK_PATH,
            timeout_seconds,
            arguments={"property_id": raw_id}
        )
        
        result_dict = json.loads(result)
        print(f"[LR Tool] Analysis complete. Status: {result_dict.get('status')}")
        return result_dict
        
    except Exception as e:
        error_msg = str(e)
        if "TIMEDOUT" in error_msg:
            return {
                "status": "error",
                "error_message": f"Analysis timed out after {timeout_seconds} seconds."
            }
        return {
            "status": "error",
            "error_message": error_msg
        }


def format_lr_insights(result: Dict[str, Any], top_n: int = 10, property_id: str = None) -> str:
    """
    Format LR insights for LLM consumption.
    
    Args:
        result: Raw LR tool output
        top_n: Number of top insights to include
        property_id: Property ID for saving charts
    """
    if result.get("status") != "success":
        return f"Analysis failed: {result.get('error_message', 'Unknown error')}"
    
    insights = result.get("llm_context", {}).get("insights", [])
    
    if not insights:
        return "No insights available from analysis."
    
    # Filter to top N by importance
    sorted_insights = sorted(
        [i for i in insights if i.get("importance_pct", 0) > 0],
        key=lambda x: x.get("importance_pct", 0),
        reverse=True
    )[:top_n]
    
    output_lines = ["=== Feature Impact Analysis ===\n"]
    
    for i, insight in enumerate(sorted_insights, 1):
        name = insight.get("name", "Unknown").replace("__imp", "").replace("amen_", "").replace("_", " ").title()
        my_val = insight.get("my_value", "N/A")
        market_avg = insight.get("market_avg", "N/A")
        trend = insight.get("market_trend", "Unknown")
        impact = insight.get("current_impact", 0)
        opportunity = insight.get("opportunity", 0)
        importance = insight.get("importance_pct", 0)
        
        # Format numbers
        if isinstance(market_avg, float):
            market_avg = f"{market_avg:.2f}"
        if isinstance(impact, float):
            impact_str = f"{impact:+.3f}"
        else:
            impact_str = str(impact)
            
        output_lines.append(f"{i}. **{name}** ({importance:.1f}% importance)")
        output_lines.append(f"   Your value: {my_val} | Market avg: {market_avg}")
        output_lines.append(f"   Trend: {trend} | Impact: {impact_str}")
        
        if isinstance(opportunity, float) and opportunity > 0.05:
            output_lines.append(f"   Opportunity: +{opportunity:.3f} potential gain")
        
        output_lines.append("")
    
    # Process and save charts if available
    # Check both 'charts' and 'ui_artifacts.charts' fields (notebook uses ui_artifacts.charts)
    charts = result.get("charts", {})
    if not charts:
        ui_artifacts = result.get("ui_artifacts", {})
        charts = ui_artifacts.get("charts", {}) if isinstance(ui_artifacts, dict) else {}
    
    if charts and property_id:
        raw_id = extract_raw_id(property_id) if property_id else "unknown"
        chart_urls = []
        
        for chart_name, b64_data in charts.items():
            # Handle both direct base64 strings and nested dict with 'data' key
            if isinstance(b64_data, dict):
                b64_data = b64_data.get("data") or b64_data.get("base64") or b64_data.get("image")
            
            if b64_data and isinstance(b64_data, str):
                url = save_chart_to_filestore(b64_data, chart_name, raw_id)
                if url:
                    # Create clickable markdown link
                    display_name = chart_name.replace("_", " ").title()
                    chart_urls.append(f"📊 **[{display_name}]({url})**")
        
        if chart_urls:
            output_lines.append("\n=== Visualization Charts ===")
            output_lines.append("Click the links below to view the analysis charts:\n")
            output_lines.extend(chart_urls)
            output_lines.append("")
    
    return "\n".join(output_lines)


def format_nlp_results(
    topics: List[Dict],
    neighbors: List[Dict],
    include_evidence: bool = False,
    evidence: Optional[List[Dict]] = None
) -> str:
    """
    Format NLP analysis results for LLM consumption.
    
    Args:
        topics: Topic analysis results
        neighbors: Neighbor list
        include_evidence: Whether to include evidence sentences
        evidence: Evidence data if include_evidence is True
        
    Returns:
        Formatted string for agent response
    """
    output_lines = []
    
    # Neighbors section
    if neighbors:
        output_lines.append("=== Similar Properties (Neighbors) ===\n")
        for n in neighbors[:10]:
            sim = n.get("similarity", 0)
            pid = n.get("neighbor_property_id", n.get("property_id", "Unknown"))
            output_lines.append(f"- Property {pid}: {sim:.2%} similarity")
        output_lines.append("")
    
    # Weaknesses
    weaknesses = [t for t in topics if t.get("kind") == "weakness"]
    if weaknesses:
        output_lines.append("=== Weaknesses vs Neighbors ===\n")
        for w in sorted(weaknesses, key=lambda x: abs(x.get("negative_rate_gap", 0)), reverse=True)[:5]:
            topic = w.get("topic", "Unknown")
            gap = w.get("negative_rate_gap", 0)
            your_rate = w.get("target_negative_rate", 0)
            neighbor_rate = w.get("neighbors_negative_rate", 0)
            
            output_lines.append(f"**{topic.title()}**")
            output_lines.append(f"  Your negative rate: {your_rate:.1%} | Neighbors: {neighbor_rate:.1%}")
            output_lines.append(f"  Gap: {gap:+.1%} (you're worse)")
            output_lines.append("")
    
    # Strengths
    strengths = [t for t in topics if t.get("kind") == "strength"]
    if strengths:
        output_lines.append("=== Strengths vs Neighbors ===\n")
        for s in sorted(strengths, key=lambda x: abs(x.get("negative_rate_gap", 0)), reverse=True)[:5]:
            topic = s.get("topic", "Unknown")
            gap = s.get("negative_rate_gap", 0)
            your_rate = s.get("target_negative_rate", 0)
            neighbor_rate = s.get("neighbors_negative_rate", 0)
            
            output_lines.append(f"**{topic.title()}**")
            output_lines.append(f"  Your negative rate: {your_rate:.1%} | Neighbors: {neighbor_rate:.1%}")
            output_lines.append(f"  Gap: {gap:+.1%} (you're better)")
            output_lines.append("")
    
    # Evidence
    if include_evidence and evidence:
        output_lines.append("=== Supporting Evidence ===\n")
        for e in evidence[:5]:
            topic = e.get("topic", "")
            text = e.get("sentence_text", "")[:200]
            sentiment = e.get("sentiment_label", "")
            output_lines.append(f"[{topic}] ({sentiment}): \"{text}...\"")
            output_lines.append("")
    
    return "\n".join(output_lines) if output_lines else "No analysis results available."


# ===========================================
# UNIFIED ANALYSIS FUNCTION
# ===========================================

def analyze_property_comprehensive(
    hotel_id: str,
    include_nlp: bool = True,
    include_lr: bool = True,
    include_evidence: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive analysis using both NLP and LR tools.
    
    Args:
        hotel_id: Hotel ID (with prefix)
        include_nlp: Run NLP review analysis
        include_lr: Run LR benchmark analysis
        include_evidence: Include evidence sentences in NLP results
        
    Returns:
        Dict with combined results from both tools
    """
    results = {
        "hotel_id": hotel_id,
        "raw_id": extract_raw_id(hotel_id),
        "source": get_source(hotel_id),
        "nlp": None,
        "lr": None,
        "formatted_output": ""
    }
    
    if not is_airbnb_property(hotel_id):
        results["error"] = "Only Airbnb properties (ABB_ prefix) are supported by Databricks tools."
        return results
    
    output_parts = []
    
    # Run NLP analysis
    if include_nlp:
        print("\n" + "="*50)
        print("Running NLP Review Analysis...")
        print("="*50)
        
        nlp_result = run_nlp_analysis(hotel_id)
        results["nlp"] = nlp_result
        
        if nlp_result.get("status") == "ok":
            run_id = nlp_result.get("run_id")
            raw_id = extract_raw_id(hotel_id)
            
            topics = get_nlp_topics(property_id=raw_id)
            neighbors = get_nlp_neighbors(property_id=raw_id)
            evidence = get_nlp_evidence(run_id=run_id) if include_evidence else None
            
            output_parts.append(format_nlp_results(topics, neighbors, include_evidence, evidence))
    
    # Run LR analysis
    if include_lr:
        print("\n" + "="*50)
        print("Running LR Benchmark Analysis...")
        print("="*50)
        
        lr_result = run_lr_analysis(hotel_id)
        results["lr"] = lr_result
        
        if lr_result.get("status") == "success":
            output_parts.append(format_lr_insights(lr_result, property_id=hotel_id))
    
    results["formatted_output"] = "\n\n".join(output_parts)
    return results
