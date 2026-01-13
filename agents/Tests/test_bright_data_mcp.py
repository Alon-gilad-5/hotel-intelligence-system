"""Test Bright Data MCP integration."""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

from utils.bright_data import search_google_serp

print('Testing Bright Data MCP integration...')
print(f'API Token: {os.getenv("BRIGHTDATA_API_TOKEN", "NOT SET")[:20]}...')

result = search_google_serp('Renaissance Johor Bahru Hotel wifi review')

print(f'\nSuccess: {result["success"]}')

if result['success']:
    print(f'Results: {len(result["results"])}')
    for i, r in enumerate(result['results'][:3], 1):
        snippet = r["snippet"][:150] if r["snippet"] else "N/A"
        print(f'\n[{i}] {r.get("title", "No title")}')
        print(f'    {snippet}...')
else:
    print(f'Error: {result["error"]}')
