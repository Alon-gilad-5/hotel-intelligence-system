"""
Bright Data MCP Integration

Provides web search and scraping functionality using Bright Data's MCP server.
Uses the MCP protocol to communicate with the @brightdata/mcp server.
"""

import os
import asyncio
import subprocess
import json
from typing import Optional


def search_google_serp(
    query: str,
    num_results: int = 10,
    country: str = "my"  # Malaysia by default
) -> dict:
    """
    Search Google using Bright Data's MCP server.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default 10)
        country: Country code for geo-targeting (default "my" for Malaysia)
    
    Returns:
        dict with keys:
            - success: bool
            - results: list of dicts with title, snippet, link
            - error: str (if success is False)
    """
    try:
        # Run the async function
        return asyncio.run(_search_via_mcp(query, num_results))
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "error": f"MCP search error: {str(e)}"
        }


async def _search_via_mcp(query: str, num_results: int = 10) -> dict:
    """
    Internal async function to search via MCP.
    Uses the Bright Data MCP server's web_data_google_search tool.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from dotenv import load_dotenv
    
    # Force reload .env to get fresh token
    load_dotenv(override=True)
    
    api_token = os.getenv("BRIGHTDATA_API_TOKEN")
    browser_auth = os.getenv("BROWSER_AUTH", "")
    
    print(f"[BrightData MCP] Using token: {api_token[:20] if api_token else 'NOT SET'}...")
    
    if not api_token:
        return {
            "success": False,
            "results": [],
            "error": "BRIGHTDATA_API_TOKEN not found in environment"
        }
    
    # Server parameters for @brightdata/mcp
    server_params = StdioServerParameters(
        command="npx.cmd",  # Windows uses npx.cmd
        args=["-y", "@brightdata/mcp"],
        env={
            **os.environ,
            "API_TOKEN": api_token,
            "PRO_MODE": "true",
            "GROUPS": "browser,business",
            "BROWSER_AUTH": browser_auth,
        }
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # List available tools to find the search tool
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"[BrightData MCP] Available tools: {tool_names}")
                
                # Use scrape_as_markdown to scrape Google search results page
                # This uses browser-based scraping which may have different auth
                import urllib.parse
                
                if "scrape_as_markdown" in tool_names:
                    google_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                    print(f"[BrightData MCP] Scraping: {google_url}")
                    result = await session.call_tool("scrape_as_markdown", {"url": google_url})
                elif "search_engine" in tool_names:
                    # Fallback to search_engine tool
                    result = await session.call_tool("search_engine", {"query": query})
                else:
                    return {
                        "success": False,
                        "results": [],
                        "error": f"No search tool found. Available: {tool_names}"
                    }
                
                # Parse results from markdown
                results = []
                if result and result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            markdown_text = content.text
                            
                            # Parse markdown to extract search results
                            # Google search results in markdown have patterns like:
                            # ## [Title](URL)
                            # Snippet text
                            import re
                            
                            # Find result blocks (markdown links followed by text)
                            pattern = r'\[([^\]]+)\]\(([^)]+)\)(?:\s*\n\s*([^\n]+))?'
                            matches = re.findall(pattern, markdown_text)
                            
                            for title, url, snippet in matches:
                                # Filter out navigation links and keep only content
                                if url.startswith('http') and 'tripadvisor' in url.lower() or 'booking' in url.lower() or 'facebook' in url.lower() or 'review' in title.lower():
                                    results.append({
                                        "title": title.strip(),
                                        "snippet": snippet.strip() if snippet else "",
                                        "link": url
                                    })
                            
                            # If no structured results, split by lines and look for content
                            if len(results) < 3:
                                lines = markdown_text.split('\n')
                                current_result = {"title": "", "snippet": "", "link": ""}
                                
                                for line in lines:
                                    line = line.strip()
                                    if len(line) > 30 and not line.startswith('#') and not line.startswith('['):
                                        # Looks like content
                                        if not current_result["snippet"]:
                                            current_result["title"] = line[:100]
                                            current_result["snippet"] = line
                                        else:
                                            current_result["snippet"] += " " + line
                                        
                                        if len(current_result["snippet"]) > 200:
                                            results.append(current_result)
                                            current_result = {"title": "", "snippet": "", "link": ""}
                
                return {
                    "success": True,
                    "results": results[:num_results] if results else [{"title": "Scraped Content", "snippet": markdown_text[:500], "link": ""}],
                    "error": None
                }
                
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "error": f"MCP connection error: {str(e)}"
        }


def format_serp_results(results: list, topic_keywords: list = None) -> str:
    """
    Format SERP results into a readable string for the agent.
    Optionally prioritizes results containing topic keywords.
    
    Args:
        results: List of result dicts with title, snippet, link
        topic_keywords: Optional list of keywords to prioritize
    
    Returns:
        Formatted string of results
    """
    if not results:
        return "No results found."
    
    # Optionally sort by relevance to topic
    if topic_keywords:
        def relevance_score(r):
            text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
            return sum(1 for kw in topic_keywords if kw.lower() in text)
        
        results = sorted(results, key=relevance_score, reverse=True)
    
    output = f"=== Google Search Results ({len(results)} found) ===\n\n"
    
    for i, r in enumerate(results, 1):
        output += f"[{i}] {r.get('title', 'No title')}\n"
        output += f"    Snippet: {r.get('snippet', 'No snippet')}\n"
        output += f"    URL: {r.get('link', 'No link')}\n\n"
    
    return output
