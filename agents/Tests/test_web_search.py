"""Test web search to see what results we're getting."""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

ddgs = DDGS()

# Try different queries - the user found wifi info on TripAdvisor
queries = [
    'Renaissance Johor Bahru Hotel wifi signal review',
    'Renaissance Johor Bahru Hotel wifi tripadvisor',
    '"Renaissance Johor Bahru Hotel" wifi',
]

for q in queries:
    print(f'\n{"="*60}')
    print(f'Query: {q}')
    print("="*60)
    try:
        results = list(ddgs.text(q, max_results=5))
        for i, r in enumerate(results, 1):
            title = r.get('title', 'N/A')
            body = r.get('body', 'N/A')
            href = r.get('href', 'N/A')
            print(f'\n[{i}] {title}')
            print(f'    Snippet: {body}')
            print(f'    URL: {href}')
    except Exception as e:
        print(f'Error: {e}')

print("\n" + "="*60)
print("DONE")
