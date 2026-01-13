# Hotel Intelligence Multi-Agent System

A sophisticated multi-agent system for hotel intelligence and competitive analysis, built with LangGraph and powered by LLMs. This system provides comprehensive insights into hotel performance, competitor analysis, market intelligence, and guest reviews through specialized AI agents working in orchestration.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Specialist Agents](#specialist-agents)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Ingestion](#data-ingestion)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Overview

The Hotel Intelligence Multi-Agent System is designed to help hotel managers and analysts gain deep insights into their property's performance by:
- Analyzing guest reviews and sentiment from multiple sources
- Identifying and analyzing competitors
- Gathering market intelligence (events, weather, external factors)
- Benchmarking metrics against competitors
- Maintaining conversational context with hybrid memory

The system uses **LangGraph** for stateful workflow orchestration and supports multiple data sources including Booking.com, Airbnb, Google Maps, TripAdvisor, and general web search.

## System Architecture

![System Architecture](images/Untitled%20diagram-2026-01-05-211127.png)

The system follows a **coordinator-specialist pattern** with the following components:

### LangGraph Coordinator

The coordinator manages the conversation flow and routes queries to appropriate specialist agents:

1. **Extract Entities**: Identifies hotels, metrics, competitors, locations, and topics from user queries
2. **Route**: Uses an LLM to intelligently route queries to the appropriate specialist agent
3. **Execute Agent**: Runs the selected specialist agent with contextual information
4. **Update Memory**: Maintains hybrid memory (recent turns + compressed summary + entities)

### Hybrid Memory System

The system implements a sophisticated memory management strategy:

- **Recent Turns**: Keeps the last 4 conversation turns verbatim for immediate context
- **Summary**: Compresses older turns into a running summary to prevent token bloat
- **Entities**: Persists extracted entities (hotels, metrics, competitors, locations, topics) across the conversation

![Detailed System Diagram](images/Gemini_Generated_Image_dmyusfdmyusfdmyu.png)

This diagram shows the complete workflow including:
- Query routing to specialist agents
- Integration with the shared LLM service (Gemini with Llama-3 fallback)
- Vector similarity search using Pinecone
- Various data scrapers (Google Maps, TripAdvisor, Web Search)
- Memory management and context flow

## Specialist Agents

![Specialist Agents Overview](images/Gemini_Generated_Image_gt90rmgt90rmgt90.png)

The system includes four specialized agents, each with distinct responsibilities:

### 1. Review Analyst Agent

**Purpose**: Analyze guest feedback, sentiment, and specific complaints/praise

**Capabilities**:
- Searches internal database (Booking.com and Airbnb reviews)
- Scrapes live Google Maps reviews using Playwright
- Scrapes live TripAdvisor reviews using Playwright
- **Google Search via Bright Data MCP** (premium, real Google results)
- Falls back to DuckDuckGo web search when needed
- Analyzes sentiment for specific topics (wifi, cleanliness, noise, etc.)
- **Anti-hallucination safeguards**: Only quotes verbatim from tool outputs

**Tools**:
- `search_booking_reviews`: Query internal Booking.com review database
- `search_airbnb_reviews`: Query internal Airbnb review database
- `scrape_google_maps_reviews`: Live scraping of Google Maps reviews (FREE)
- `scrape_tripadvisor_reviews`: Live scraping of TripAdvisor reviews (FREE)
- `search_web_google`: **NEW** - Google search via Bright Data MCP (PAID)
- `search_web_free`: DuckDuckGo web search fallback (FREE)
- `analyze_sentiment_topics`: LLM-powered sentiment analysis

**Anti-Hallucination Design**:
The Review Analyst uses a strict citation-required prompt that:
- Only allows claims that appear VERBATIM in tool outputs
- Requires quoting exact text with source attribution
- Explicitly reports when topics are NOT found in any source
- Distinguishes between amenity listings and actual guest feedback

**Example Queries**:
- "What are guests saying about wifi quality?"
- "Show me recent complaints about cleanliness"
- "Analyze sentiment about breakfast service"

**Location**: `agents/review_analyst.py`

### 2. Competitor Analyst Agent

**Purpose**: Identify and analyze direct competitors based on geography and similarity

**Capabilities**:
- Finds competitors by geographic proximity (same city/area)
- Uses ML-based similarity matching (black-box integration point)
- Retrieves detailed competitor information
- Filters out own hotel from results

**Tools**:
- `find_competitors_geo`: Geographic competitor search across Booking.com and Airbnb data
- `find_competitors_similar`: ML-based similarity search (with geo fallback)
- `get_competitor_details`: Detailed information about specific competitors

**ML Integration Point**:
The `find_competitors_ml()` function in `competitor_analyst.py` serves as a black-box integration point for custom ML models. It expects:

```python
[
    {"hotel_id": "ABB_123", "similarity_score": 0.95, "source": "airbnb"},
    {"hotel_id": "BKG_456", "similarity_score": 0.89, "source": "booking"}
]
```

**Example Queries**:
- "Who are my main competitors?"
- "Find similar hotels in the area"
- "Show me details for hotel BKG_12345"

**Location**: `agents/competitor_analyst.py`

### 3. Market Intel Agent

**Purpose**: Gather external market intelligence affecting hotel demand

**Capabilities**:
- Scrapes Google Maps for ratings and reviews
- Searches for local events (concerts, conferences, festivals)
- Gathers weather and external factors
- Identifies local attractions and points of interest

**Tools**:
- `scrape_google_maps_reviews`: Live Google Maps data extraction (FREE)
- `search_events_free`: Playwright-based event search (FREE)
- `search_web_brightdata`: BrightData API for complex scraping (PAID - use sparingly)

**Cost Awareness**:
- Prioritizes free Playwright-based scraping
- Only uses paid BrightData API when necessary

**Example Queries**:
- "Are there any major events happening in the city this week?"
- "What's the weather forecast for next week?"
- "Show me Google Maps rating for my hotel"

**Location**: `agents/market_intel.py`

### 4. Benchmark Agent

**Purpose**: Compare hotel metrics against competitors

**Capabilities**:
- Compares any metric in free-text queries (price, rating, amenities, etc.)
- Ranks hotels by specific metrics
- Provides actionable insights and market position
- Extracts metrics from both metadata and content

**Tools**:
- `compare_metric`: Compare specific metrics (price, rating, amenities, cleanliness)
- `get_my_hotel_data`: Retrieve all data for your hotel
- `get_competitor_data`: Retrieve data for specific competitors
- `rank_by_metric`: Rank all hotels by a metric to see market position

**Supported Metrics**:
- Price/Cost/Rate
- Rating/Score/Stars
- Amenities/Facilities
- Review Count
- Location/Distance
- Cleanliness/Hygiene

**Example Queries**:
- "How does my price compare to competitors?"
- "Am I rated higher than nearby hotels?"
- "Rank hotels by rating in my city"
- "Do competitors have better amenities?"

**Location**: `agents/benchmark_agent.py`

## Features

### Core Features

1. **Intelligent Query Routing**: Automatically routes user queries to the most appropriate specialist agent
2. **Multi-Agent Collaboration**: Complex queries trigger sequential execution of multiple agents with result aggregation
3. **Stateful Conversations**: Maintains context across multiple turns with hybrid memory
4. **Entity Extraction**: Automatically identifies and tracks hotels, metrics, competitors, locations, and topics
5. **Multi-Source Data**: Integrates Booking.com, Airbnb, Google Maps, TripAdvisor, and web search
6. **LLM Fallback**: Automatic failover from Gemini to Groq/Llama-3 on quota errors
7. **Tool Calling**: All agents support function calling for data retrieval and analysis
8. **Real-Time Scraping**: Live data extraction from Google Maps and TripAdvisor using Playwright

### Multi-Agent Collaboration

For complex queries that require data from multiple sources, the system automatically triggers a multi-agent workflow:

**Example Query:** "How clean are rooms compared to competitors?"

**Workflow:**
```
competitor_analyst → review_analyst → benchmark_agent → aggregate_results
```

| Step | Agent | Action |
|------|-------|--------|
| 1 | `competitor_analyst` | Finds 5 London competitors with ratings |
| 2 | `review_analyst` | Searches cleanliness reviews for each competitor |
| 3 | `benchmark_agent` | Compares cleanliness metrics across hotels |
| 4 | `aggregate` | LLM synthesizes findings into coherent response |

**Multi-Agent Routing Patterns:**
| Query Pattern | Agents Triggered |
|---------------|------------------|
| "Compare [topic] with competitors" | competitor_analyst → review_analyst → benchmark_agent |
| "Compare metrics with competitors" | competitor_analyst → benchmark_agent |
| "Market analysis with comparison" | market_intel → benchmark_agent |

The coordinator maintains an `agent_queue` in state and executes agents sequentially, passing intermediate results to each subsequent agent.

### Technical Features

- **Vector Search**: Pinecone-based RAG for fast similarity search across hotel and review data
- **Embeddings**: Uses BAAI/bge-m3 for high-quality embeddings (1024 dimensions)
- **Spark Processing**: Efficient data processing with PySpark
- **Anti-Bot Handling**: Robust scraping with user-agent spoofing and cookie handling
- **Namespace Isolation**: Separate Pinecone namespaces for booking_hotels, booking_reviews, airbnb_hotels, airbnb_reviews

## Installation

### Prerequisites

- Python 3.9+
- Node.js (for BrightData MCP, optional)
- Chromium browser (for Playwright)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/hotel-intelligence-system.git
cd hotel-intelligence-system
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Playwright

```bash
pip install playwright
playwright install chromium
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional (for fallback LLM)
GROQ_API_KEY=your_groq_api_key

# Optional (for Bright Data MCP - Google search & scraping)
BRIGHTDATA_API_TOKEN=your_brightdata_token
BROWSER_AUTH=your_browser_auth_if_needed
```

**Getting Bright Data API Token:**
1. Go to https://brightdata.com/cp/
2. Navigate to Settings → API tokens
3. Create a new token with permissions for SERP API and Web Scraper
4. Copy the token to your `.env` file

### Step 5: Install Additional Tools (Optional)

For web search functionality:
```bash
pip install ddgs
# or
pip install duckduckgo-search
```

For Bright Data integration:
```bash
# MCP server (requires Node.js)
npm install -g @brightdata/mcp

# LangChain integration
pip install langchain-brightdata

# MCP Python client
pip install mcp
```

## Usage

### Interactive Chat Mode

Run the coordinator in interactive chat mode:

```bash
cd agents
python coordinator.py
```

This starts an interactive session where you can ask questions about your hotel:

```
================================================
HOTEL INTELLIGENCE SYSTEM
LangGraph Architecture with Hybrid Memory
================================================

Context: Malmaison London in London
Type 'q' to quit, 'state' to see current state.

You: What are guests saying about wifi quality?
[Graph] Extracted entities: {'topics': ['wifi'], 'metrics': ['quality']}
[Graph] Routing to: review_analyst
[ReviewAnalyst] Tool: search_booking_reviews
   >>> Tool Output: Found 5 reviews mentioning wifi...

Agent: Based on the reviews, guests have mixed feedback about wifi...

You: How clean are rooms compared to competitors?
[Graph] Multi-agent workflow: competitor_analyst → review_analyst → benchmark_agent
[competitor_analyst] Executing...
   >>> Tool Output: === Geographic Competitors in London (5 found) ===
   1. Bedford Hotel (Rating: 8.0)
   2. Grand Hotel Bellevue London (Rating: 8.2)
   ...
[review_analyst] Executing...
   >>> Tool Output: Reviews for cleanliness...
[benchmark_agent] Executing...
   >>> Tool Output: === Cleanliness Comparison ===
[Graph] Aggregating results from 3 agents...

Agent: Based on my analysis of 5 London competitors...
```

### Programmatic Usage

```python
from agents.coordinator import LangGraphCoordinator

# Initialize coordinator
coordinator = LangGraphCoordinator(
    hotel_id="BKG_177691",
    hotel_name="Malmaison London",
    city="London"
)

# Get initial state
state = coordinator.get_initial_state()

# Run queries
response, state = coordinator.run("What are guests saying about cleanliness?", state)
print(response)

# Continue conversation with maintained context
response, state = coordinator.run("How does that compare to competitors?", state)
print(response)
```

### Special Commands

- `state`: View current conversation state (turn count, recent turns, summary, entities)
- `context`: View formatted context that would be passed to agents
- `q` / `quit` / `exit`: Exit the chat session

### Running Tests

Test scripts are located in `agents/Tests/`:

```bash
# Test all agents (11 tests) - initialization, tools, validation, multi-agent
python agents/Tests/test_all_agents.py

# Test LangGraph integration (9 tests) - state, memory, routing
python agents/Tests/LangGraph_Test

# Check RAG database contents
python agents/Tests/check_rag.py

# Test web search tools
python agents/Tests/test_web_search.py

# Test Bright Data MCP integration
python agents/Tests/test_bright_data_mcp.py
```

**Test Coverage:**
| Test Suite | Tests | What's Tested |
|------------|-------|---------------|
| `test_all_agents.py` | 11 | Agent init, anti-hallucination prompts, tools, validation, multi-agent state |
| `LangGraph_Test` | 9 | Graph state, entity extraction, memory, routing, compression |

## Data Ingestion

The system includes a multi-source data ingestion pipeline that processes hotel and review data into Pinecone.

### Running Ingestion

```bash
python ingestion.py
```

### Ingestion Process

1. **Load Data**: Reads Booking.com and Airbnb parquet files using PySpark
2. **Process Hotels**: Extracts hotel metadata (name, location, rating, facilities)
3. **Process Reviews**: Extracts and structures guest reviews
4. **Generate Embeddings**: Uses BAAI/bge-m3 to create 1024-dim embeddings
5. **Upload to Pinecone**: Stores in separate namespaces:
   - `booking_hotels`: Booking.com hotel data
   - `booking_reviews`: Booking.com reviews
   - `airbnb_hotels`: Airbnb property data
   - `airbnb_reviews`: Airbnb reviews

### Data Format

**Booking.com Hotels**:
- ID: `BKG_<hotel_id>`
- Fields: title, description, city, country, review_score, facilities

**Airbnb Properties**:
- ID: `ABB_<property_id>`
- Fields: name, description, location, country, ratings, amenities, price, category

**Reviews** (both sources):
- ID: `<hotel_id>_R<index>`
- Fields: review text, reviewer name, hotel context

### Customizing Ingestion

Edit `ingestion.py` to modify:
- Sample size: `sample_size=500` parameter (how many hotels to ingest)
- City filter: `city_filter="London"` (focus on a specific city for competitor analysis)
- Clear existing: `clear_existing=True` (wipe Pinecone before re-ingesting)
- Data paths: `booking_path` and `airbnb_path`
- Index name: `index_name="booking-agent"`

**City-Focused Ingestion Example:**
```python
run_ingestion(
    booking_path="data/sampled_booking_data.parquet",
    airbnb_path="data/sampled_airbnb_data.parquet",
    index_name="booking-agent",
    sample_size=500,
    city_filter="London",  # Only ingest London hotels
    clear_existing=True    # Clear old data first
)
```

**Available Cities in Dataset:**
| City | Hotels | Notes |
|------|--------|-------|
| London | 332 | Best for testing |
| Rome | 315 | |
| Paris | 290 | |
| Tokyo | 181 | |
| Kuala Lumpur | 120 | |

## Project Structure

```
Agent_Pipeline_Testing/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Base class with LLM fallback and RAG utilities
│   ├── coordinator.py          # LangGraph coordinator (main entry point)
│   ├── graph_state.py          # State schema and memory configuration
│   ├── entity_extractor.py     # Entity extraction (LLM + regex)
│   ├── memory_manager.py       # Hybrid memory management
│   ├── review_analyst.py       # Review analysis agent
│   ├── competitor_analyst.py   # Competitor identification agent
│   ├── market_intel.py         # Market intelligence agent
│   ├── benchmark_agent.py      # Benchmarking agent
│   │
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   ├── bright_data.py      # Bright Data MCP integration for Google search
│   │   └── output_validator.py # Structured output validation to catch hallucinations
│   │
│   └── Tests/                  # Test scripts
│       ├── __init__.py
│       ├── test_all_agents.py  # All agents test suite (11 tests)
│       ├── LangGraph_Test      # LangGraph integration tests (9 tests)
│       ├── check_rag.py        # RAG database verification
│       ├── test_web_search.py  # Web search tool testing
│       └── test_bright_data_mcp.py  # Bright Data MCP testing
│
├── data/
│   ├── sampled_booking_data.parquet    # Booking.com dataset
│   └── sampled_airbnb_data.parquet     # Airbnb dataset
│
├── images/
│   ├── Untitled diagram-2026-01-05-211127.png           # Architecture diagram
│   ├── Gemini_Generated_Image_dmyusfdmyusfdmyu.png     # System flow diagram
│   └── Gemini_Generated_Image_gt90rmgt90rmgt90.png     # Agent capabilities
│
├── ingestion.py               # Data ingestion pipeline
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not in repo)
├── mcp.json                   # MCP configuration for BrightData
└── README.md                  # This file
```

## Configuration

### Memory Configuration

Edit `agents/graph_state.py`:

```python
MAX_RECENT_TURNS = 4        # Keep last N turns verbatim
SUMMARY_TRIGGER = 6         # Summarize when total turns exceed this
```

### LLM Configuration

Edit `agents/base_agent.py`:

```python
# Primary model
model="gemini-2.0-flash"

# Fallback model
FALLBACK_MODEL = "llama-3.3-70b-versatile"
```

### Embedding Configuration

```python
INDEX_NAME = "booking-agent"          # Pinecone index name
EMBEDDING_MODEL = "BAAI/bge-m3"       # HuggingFace embedding model
```

### Pinecone Configuration

The system automatically creates a Pinecone index with:
- Dimension: 1024 (for BAAI/bge-m3)
- Metric: Cosine similarity
- Cloud: AWS
- Region: us-east-1 (Serverless)

## Workflow Explanation

### Example: Multi-Turn Conversation

```
User: "What are guests saying about wifi?"

1. Extract Entities Node:
   - Extracts: topics=['wifi'], metrics=[]
   - Merges with existing entities

2. Route Node:
   - Context: Previous entities + recent turns
   - LLM decides: route to review_analyst

3. Review Analyst Agent:
   - Receives enhanced query with context
   - Calls search_booking_reviews(query="wifi")
   - Calls search_airbnb_reviews(query="wifi")
   - If needed: scrape_google_maps_reviews()
   - Analyzes and responds

4. Update Memory Node:
   - Adds user turn to recent_turns
   - Adds assistant turn with agent_used="review_analyst"
   - Compresses older turns to summary if needed
   - Returns updated state

User: "How does that compare to competitors?"

1. Extract Entities Node:
   - Extracts: topics=['competitor_analysis']
   - Merges with existing: topics=['wifi', 'competitor_analysis']

2. Route Node:
   - Context now includes: "User previously asked about wifi"
   - LLM understands "that" refers to wifi
   - Routes to: benchmark_agent

3. Benchmark Agent:
   - Receives context: previous discussion about wifi
   - Interprets "that" as "wifi quality metric"
   - Compares wifi amenities/reviews across competitors
   - Responds with comparative analysis
```

This demonstrates:
- **Entity persistence** across turns
- **Contextual routing** based on conversation history
- **Implicit reference resolution** ("that" = wifi)
- **Agent specialization** working together

## API Keys and Costs

### Free Services
- **Gemini Flash**: Free tier available (quota limits)
- **Groq/Llama-3**: Free tier available
- **HuggingFace Embeddings**: Free (runs locally)
- **Playwright Scraping**: Free (Google Maps, TripAdvisor, DuckDuckGo)

### Paid Services
- **Pinecone**: Free tier (1 index, limited storage), paid plans available
- **Bright Data MCP**: Paid scraping service with multiple tools:
  - SERP API: ~$2.70 per 1,000 searches
  - Web Scraper: ~$3-5 per 1,000 pages
  - Browser zones for anti-bot bypass

### Recommendations
- Start with free tier for all services
- Monitor Gemini quota usage
- Set up Groq API key for automatic fallback
- Only use BrightData for complex scraping needs

## Advanced Features

### Bright Data MCP Integration

The system integrates with Bright Data's MCP (Model Context Protocol) server for premium web scraping capabilities:

**Available MCP Tools:**
- `scrape_as_markdown`: Scrape any webpage as markdown
- `search_engine`: Google search results
- `web_data_google_maps_reviews`: Google Maps reviews
- `web_data_booking_hotel_listings`: Booking.com data

**Configuration** (`mcp.json`):
```json
{
  "mcpServers": {
    "brightdata": {
      "command": "npx.cmd",
      "args": ["-y", "@brightdata/mcp"],
      "env": {
        "API_TOKEN": "${BRIGHTDATA_API_TOKEN}",
        "PRO_MODE": "true",
        "GROUPS": "browser,business"
      }
    }
  }
}
```

**Usage in Code** (`agents/utils/bright_data.py`):
```python
from agents.utils.bright_data import search_google_serp

result = search_google_serp("hotel wifi reviews")
if result["success"]:
    for r in result["results"]:
        print(f"{r['title']}: {r['snippet']}")
```

### Anti-Hallucination Design

**All agents** implement strict safeguards against LLM hallucinations through:

**1. Citation-Required Prompts (All Agents):**

Each specialist agent has anti-hallucination rules in their system prompt:

| Agent | Key Rules |
|-------|-----------|
| **Review Analyst** | Quote exact text, cite sources, report when topics not found |
| **Competitor Analyst** | Only list competitors from tool outputs, never make up names/ratings |
| **Market Intel** | Quote scraped content directly, say "Could not retrieve" on failures |
| **Benchmark Agent** | Report exact numbers, N/A stays N/A, rankings from actual data only |

**2. Structured Output Validation (`agents/utils/output_validator.py`):**

The system includes automated validation that checks agent responses against tool outputs:

```python
from agents.utils.output_validator import validate_response

# Validate a response
result = validate_response(
    response="Guests say wifi is slow...",
    tool_outputs=["Review 1: Great location", "Review 2: Nice breakfast"],
    strict=False
)

print(f"Hallucination Risk: {result.hallucination_risk:.0%}")
print(f"Warnings: {result.warnings}")
print(f"Valid: {result.is_valid}")
```

**Validation Features:**
- **Pattern Detection**: Catches phrases like "guests say...", "many reviewers mention..." without supporting data
- **Quote Verification**: Validates that quoted text exists in tool outputs
- **Numeric Claim Checking**: Verifies ratings/prices against actual tool data
- **Risk Scoring**: 0-1 hallucination risk score with configurable thresholds

**3. Agent-Level Validation:**

```python
# Enable validation when creating agents
agent = ReviewAnalystAgent(
    hotel_id="BKG_177691",
    hotel_name="Malmaison London",
    city="London",
    validate_output=True,      # Enable validation (default: True)
    strict_validation=False    # Strict mode appends warnings to high-risk responses
)

# Get validation result alongside response
response, validation = agent.run("What do guests say about wifi?", return_validation=True)
print(f"Risk: {validation.hallucination_risk:.0%}")
```

**4. Example Good Response:**
```
From TripAdvisor: "Warm service but poor wifi" 
(https://www.tripadvisor.com/ShowUserReviews-...)

I found 1 review mentioning wifi. The guest reported poor wifi quality.
No other sources mentioned wifi signal strength.
```

### Custom ML Model Integration

Replace the black-box function in `agents/competitor_analyst.py`:

```python
def find_competitors_ml(hotel_id: str, k: int = 5) -> List[dict]:
    # Your ML model integration here
    from your_ml_module import CompetitorModel

    model = CompetitorModel.load("path/to/model")
    results = model.predict(hotel_id, k=k)

    return [
        {
            "hotel_id": result.hotel_id,
            "similarity_score": result.score,
            "source": result.source
        }
        for result in results
    ]
```

### Adding New Specialist Agents

1. Create new agent class inheriting from `BaseAgent`
2. Implement `get_system_prompt()` and `get_tools()`
3. Add agent to coordinator's `self.agents` dict
4. Update routing in `_build_graph()`

Example:

```python
# agents/pricing_agent.py
class PricingAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "You are a pricing analyst..."

    def get_tools(self) -> list:
        return [self.analyze_pricing_trends, self.suggest_price]
```

## Troubleshooting

### Common Issues

**1. Playwright browser not found**
```bash
playwright install chromium
```

**2. Pinecone connection error**
- Verify `PINECONE_API_KEY` in `.env`
- Check index name matches in code

**3. Gemini quota exceeded**
- System auto-falls back to Groq
- Ensure `GROQ_API_KEY` is set

**4. Scraping fails (403/CAPTCHA)**
- Google/TripAdvisor may block requests
- Try different user-agent strings
- Consider adding delays between requests

**5. Empty search results**
- Verify data was ingested (`python ingestion.py`)
- Check namespace names match
- Ensure hotel_id filter is correct

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow existing code structure and naming conventions
- Add docstrings to all functions and classes
- Update README.md for new features
- Test with both Gemini and Groq fallback
- Ensure scrapers handle errors gracefully

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangGraph** for stateful agent orchestration
- **LangChain** for LLM abstractions and tool calling
- **Pinecone** for vector database
- **HuggingFace** for embeddings
- **Google Gemini** and **Groq** for LLM services
- **Playwright** for web scraping capabilities

## Contact

For questions or support, please open an issue on GitHub.

---

Built with LangGraph + LangChain + Pinecone
