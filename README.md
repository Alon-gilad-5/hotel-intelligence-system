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

![System Architecture](images/Architecture.PNG)

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

## Specialist Agents

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
- Databricks workspace (recommended) OR local development environment
- Node.js (for BrightData MCP, optional - only for local development)
- Chromium browser (for Playwright - only for local development)

## Running on Databricks (Recommended)

The system is designed to run on **Databricks**, which provides:
- Built-in Spark for data processing
- Long-running job support (15+ minute ML analyses)
- Native integration with Azure Blob Storage
- Better resource management for ML workloads

### Step 1: Connect Repository to Databricks

1. **Open Databricks Workspace** → Navigate to **Repos** in the sidebar

2. **Add Repository**:
   - Click **Add Repo** or **Create Repo**
   - Select **Git** as the source
   - Enter my repository URL: `https://github.com/Alon-gilad-5/hotel-intelligence-system.git`
   
   ![Integration Step 1](images/Integration_step1.png)

3. **Configure Repository**:
   - Choose a **Repo Name** (e.g., `hotel-intelligence-system`)
   - Select **Branch**: `main` (or your default branch)
   - Click **Create Repo**
   
   ![Integration Step 2](images/Integration_step2.png)

4. **Verify Repository**:
   - The repo URL is `https://github.com/Alon-gilad-5/hotel-intelligence-system`
   - You should see the project structure with `agents/`, `tests/`, `docs/`, etc.
   
   ![Integration Step 3](images/Integration_step3.png)

### Step 2: Create or Select a Cluster

1. **Navigate to Compute** → **Create Cluster** (or use existing)
2. **Cluster Configuration**:
   - **Cluster Mode**: Standard (single user) or Shared
   - **Databricks Runtime**: 13.3 LTS or later (Python 3.11)
   - **Node Type**: Standard (e.g., `i3.xlarge` or larger for ML workloads)
   - **Workers**: 1-2 workers recommended for development

### Step 3: Install Dependencies on Cluster

**Option A: Using Cluster Libraries (Recommended - Faster)**

1. Go to your cluster → **Libraries** tab
2. Click **Install New**
3. Install from **PyPI**:
   ```
   langchain==0.3.14
   langgraph==0.2.60
   langchain-google-genai==2.0.7
   langchain-huggingface==0.1.2
   langchain-pinecone==0.2.0
   langchain-groq==0.2.2
   gradio==3.50.2
   sentence-transformers
   spark-nlp
   nltk
   duckduckgo-search
   python-dotenv
   ```

**Option B: Using the Provided Notebook (Automatic)**

The Databricks notebook provided in the project already contains an installation cell at the beginning. If you didn't install dependencies via cluster libraries (Option A), simply run the notebook - it will automatically install all required packages in the first cell:

```python
# 1. Install UI and Embedding models
%pip install gradio==3.50.2 sentence-transformers --quiet

# 2. Install Utilities
%pip install spark-nlp nltk duckduckgo-search python-dotenv

# 3. Install LangChain Framework & Graph
%pip install langchain==0.3.14 langgraph==0.2.60

# 4. Install LangChain Integrations
%pip install langchain-google-genai==2.0.7 langchain-huggingface==0.1.2 langchain-pinecone==0.2.0 langchain-groq==0.2.2

# 5. Apply changes
dbutils.library.restartPython()
```

**Note**: The notebook will handle dependency installation automatically if needed. Option A (cluster libraries) is faster as dependencies are pre-installed and shared across all notebooks on the cluster.

### Step 4: Configure Environment Variables

**On Databricks, environment variables are set at the cluster level:**

1. **Go to your Cluster** → **Configuration** tab
2. **Click "Edit"** to modify cluster settings
   
   ![Environment Variables Step 1](images/Env_Var_step1.png)

3. **Scroll to Advanced Options** → Expand **Environment Variables**
   
   ![Environment Variables Step 2](images/Env_Var_step2.png)

4. **Add the following variables** (replace with your actual API keys):
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   GOOGLE_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   BRIGHTDATA_API_TOKEN=your_brightdata_token
   DATABRICKS_NLP_NOTEBOOK_PATH=/Workspace/Users/you@example.com/NLP Tool For Review Analysis
   DATABRICKS_LR_NOTEBOOK_PATH=/Workspace/Users/you@example.com/Linear Regression Model
   ```
   
   ![Environment Variables Step 3](images/Env_Var_step3.png)

5. **Save the cluster configuration** - The cluster will restart to apply the new environment variables

**Required Environment Variables:**
- `PINECONE_API_KEY` - Your Pinecone API key (required for vector database)
- `GOOGLE_API_KEY` - Your Google Gemini API key (required for LLM)
- `GROQ_API_KEY` - Your Groq API key (required for fallback LLM)
- `BRIGHTDATA_API_TOKEN` - Your Bright Data API token (optional, for web scraping)
- `DATABRICKS_NLP_NOTEBOOK_PATH` - Full path to the NLP notebook (optional, overrides default)
- `DATABRICKS_LR_NOTEBOOK_PATH` - Full path to the LR notebook (optional, overrides default)

**Alternative: Using Databricks Secrets (More Secure)**

1. **Create Secret Scope**:
   - Go to **Settings** → **User Settings** → **Access Tokens**
   - Or use Databricks CLI: `databricks secrets create-scope --scope hotel-intel`

2. **Add Secrets**:
   ```bash
   databricks secrets put --scope hotel-intel --key PINECONE_API_KEY
   databricks secrets put --scope hotel-intel --key GOOGLE_API_KEY
   databricks secrets put --scope hotel-intel --key GROQ_API_KEY
   databricks secrets put --scope hotel-intel --key BRIGHTDATA_API_TOKEN
   ```

3. **Access in Code** (already implemented in `agents/config_databricks.py`):
   ```python
   from agents.config_databricks import get_secret
   
   api_key = get_secret("PINECONE_API_KEY")  # Automatically uses secrets or env vars
   ```

### Step 5: Import Required Databricks Tool Notebooks

The NLP and LR tools depend on Databricks notebooks that must exist in your workspace.

**For Project Checkers on Workspace "lab94290w3":**

If you are on the same workspace as us (`lab94290w3`), the two notebooks are already shared on the workspace. The system will automatically detect and use the shared notebooks. No additional setup is required.

**For Other Workspaces:**

1. **Import the notebooks from the repo**:
   - `agents/Databricks_Notebooks_Tools/NLP Tool For Review Analysis.ipynb`
   - `agents/Databricks_Notebooks_Tools/Linear Regression Model.ipynb`

2. **Place them in the same workspace as your main notebook**.

3. **If your notebook paths differ, set these env vars**:
   ```
   DATABRICKS_NLP_NOTEBOOK_PATH=/Workspace/Users/<you>/Databricks_Notebooks_Tools/NLP Tool For Review Analysis
   DATABRICKS_LR_NOTEBOOK_PATH=/Workspace/Users/<you>/Databricks_Notebooks_Tools/Linear Regression Model
   ```

Without these notebooks, the NLP and LR tools will fail.

Note: For `CompetitorAnalystAgent`, the geo-based competitor search is fallback-only and is used only when the NLP tool is unavailable or fails.

### Step 5.5: Backup Notebook (Optional - For Testing and Troubleshooting)

In addition to the main project, we've created a backup notebook for convenience and unit testing of the tools. This notebook allows you to:

- Test the NLP and Linear Regression tools independently
- Run the tools with custom questions and properties
- Get faster answers without using the full UI interface
- Visualize tool outputs and understand the tool flow

**Location**: `Databricks_Backup_Notebook/Agent Tools Demonstration.ipynb`

**When to Use**:
- If the main interface notebook (`UI Hotel Intelligence System.ipynb`) doesn't work or encounters issues
- For unit testing and debugging individual tools
- To understand the tool flow and visualizations without the full system
- For faster iteration when testing specific tool functionality

**How to Use**:
1. Import the notebook from the repo: `Databricks_Backup_Notebook/Agent Tools Demonstration.ipynb`
2. Follow the instructions inside the notebook to provide a property and question
3. Run the notebook cells step by step to receive LLM-generated insights

**Note**: This backup notebook is optional and provided solely for those who wish to see the overall tool flow and visualizations independently. The main interface notebook should be used for the full system experience.

### Step 6: Use the Interface Notebook

1. **Open the existing interface notebook**:
   - Located at: `agents/Datsabricks_Interface_Notebook/UI Hotel Intelligence System.ipynb`
   - This notebook contains the Gradio interface ready to use

2. **No configuration needed** - The notebook is pre-configured and ready to run.

3. **Simply run the notebook cells** in order to start the interface.

**Note**: The interface notebook automatically sets up import paths using the current user's email, so no manual configuration is needed.

### Step 6: Run the Notebook

1. **Attach the notebook to your cluster** (with environment variables configured)
2. **Run all cells** or run cells individually
3. **Test with a simple query**:
   ```
   "What are my guests saying about my location?"
   ```

### Troubleshooting Databricks Setup

**Issue: ModuleNotFoundError**
- **Solution**: Ensure all dependencies are installed on the cluster
- Check cluster libraries or run `%pip install <package>` in a notebook cell

**Issue: Environment variables not found**
- **Solution**: 
  - Verify variables are set in cluster configuration
  - Or use Databricks secrets (more secure)
  - Check `agents/config_databricks.py` is using `get_secret()` correctly

**Issue: Import errors (agents module not found)**
- **Solution**: Verify repo path is correct in notebook:
  ```python
  sys.path.insert(0, "/Workspace/Users/<your-email>/hotel-intelligence-system")
  ```

**Issue: Long-running jobs timeout**
- **Solution**: The coordinator already handles this with extended timeouts (15+ minutes for ML analyses)
- Check `agents/coordinator.py` for `DATABRICKS_TIMEOUTS` configuration

## Local Development (Alternative)

If you prefer to run locally instead of Databricks:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Alon-gilad-5/hotel-intelligence-system.git
cd hotel-intelligence-system
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Playwright (for local scraping)

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

# Optional (for Bright Data - Google search & scraping)
BRIGHTDATA_API_TOKEN=your_brightdata_token
BROWSER_AUTH=your_browser_auth_if_needed
```

**Getting API Keys:**
- **Gemini API**: https://makersuite.google.com/app/apikey
- **Pinecone API**: https://app.pinecone.io/
- **Groq API**: https://console.groq.com/
- **Bright Data API**: https://brightdata.com/cp/ → Settings → API tokens

### Step 5: Install Additional Tools (Optional)

For web search functionality:
```bash
pip install ddgs
# or
pip install duckduckgo-search
```

For Bright Data integration (local only):
```bash
# MCP server (requires Node.js)
npm install -g @brightdata/mcp

# LangChain integration
pip install langchain-brightdata

# MCP Python client
pip install mcp
```

## Usage

### Running on Databricks (Recommended)

After completing the setup steps above:

1. **Open your Databricks notebook** (`Hotel_Intelligence_Agent.ipynb`)
2. **Attach to cluster** (with environment variables configured)
3. **Run all cells** to initialize the system
4. **Use the interface**:
   - **Native Databricks**: Use the dropdown widget to select query types or enter custom queries
   - **Streamlit**: Use the web interface at the displayed URL

**Example Queries to Try:**
- "What are my guests saying about my location?"
- "What features should I improve to increase my rating?"
- "How do I compare to my competitors?"
- "Are there any events happening in my city this week?"

### Example Conversations

Here are real examples of conversations with the agent:

**Example 1: Location Feedback**
![Conversation Example 1](images/Conv1.PNG)

**Example 2: Feature Improvement Analysis**
![Conversation Example 2](images/Conv2.PNG)

**Example 3: Competitive Analysis**
![Conversation Example 3](images/Conv3.PNG)

### Local Development (Alternative)

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

Test scripts are located in `tests/`:

```bash
# Test all agents (11 tests) - initialization, tools, validation, multi-agent
python tests/test_all_agents.py

# Test coordinator on Databricks
python tests/test_coordinator_databricks.py

# Test web search tools
python tests/test_web_search.py

# Test Google Maps tools
python tests/test_google_maps_tools.py

# Test tool recovery mechanisms
python tests/test_tool_recovery_databricks.py

# User Testing Framework (interactive or automated)
python tests/user_testing.py
```

**Test Coverage:**
| Test Suite | Tests | What's Tested |
|------------|-------|---------------|
| `test_all_agents.py` | 11 | Agent init, anti-hallucination prompts, tools, validation, multi-agent state |
| `test_coordinator_databricks.py` | 1 | Coordinator execution on Databricks |
| `test_web_search.py` | Multiple | Web search tool functionality |
| `test_google_maps_tools.py` | Multiple | Google Maps scraping tools |
| `user_testing.py` | 15 scenarios | User satisfaction, feedback collection, scenario testing |

**Note**: Debug scripts and development utilities are in `scripts/` (not pushed to git).

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
hotel-intelligence-system/
│
├── agents/                    # Core agent code
│   ├── __init__.py
│   ├── base_agent.py         # Base class with LLM fallback and RAG utilities
│   ├── coordinator.py        # LangGraph coordinator (main entry point)
│   ├── graph_state.py        # State schema and memory configuration
│   ├── entity_extractor.py   # Entity extraction (LLM + regex)
│   ├── memory_manager.py     # Hybrid memory management
│   ├── review_analyst.py     # Review analysis agent
│   ├── competitor_analyst.py # Competitor identification agent
│   ├── market_intel.py       # Market intelligence agent
│   ├── benchmark_agent.py    # Benchmarking agent
│   ├── config_databricks.py  # Databricks configuration and secrets
│   ├── databricks_tools.py   # Databricks-specific tools (NLP, LR analysis)
│   │
│   ├── utils/                # Utility modules
│   │   ├── __init__.py
│   │   ├── bright_data.py    # Bright Data SERP API integration
│   │   ├── google_maps_scraper.py  # Google Maps scraping utilities
│   │   └── output_validator.py    # Structured output validation
│   │
│   └── [Interface Files]     # Databricks interface options
│       ├── databricks_native_interface.py  # Native Databricks widgets
│       ├── streamlit_databricks.py          # Streamlit interface
│       └── html_interactive_interface.py    # HTML-based interface
│
├── tests/                     # Test suite (pushed to git)
│   ├── __init__.py
│   ├── test_all_agents.py    # Comprehensive agent tests
│   ├── test_coordinator_databricks.py  # Databricks coordinator tests
│   ├── test_web_search.py    # Web search tool tests
│   ├── test_google_maps_tools.py  # Google Maps tool tests
│   ├── test_tool_recovery_databricks.py  # Tool recovery tests
│   └── user_testing.py        # User testing framework
│
├── scripts/                   # Development utilities (gitignored)
│   ├── ingestion/            # Data ingestion helper scripts
│   ├── debug/                # Debug and diagnostic scripts
│   └── databricks/           # Databricks-specific utility scripts
│
├── docs/                      # Documentation
│   ├── CHART_FIX_SUMMARY.md
│   ├── EVALUATION_REPORT.md
│   ├── SYSTEM_EVALUATION_REPORT.md
│   └── TOKEN_OPTIMIZATION_EXAMPLE.md
│
├── data/                      # Data files (gitignored - large datasets)
│   ├── sampled_booking_data.parquet
│   └── sampled_airbnb_data.parquet
│
├── images/                    # Architecture diagrams and screenshots
│   ├── Architecture.PNG      # System architecture diagram
│   ├── Conv1.PNG             # Example conversation 1
│   ├── Conv2.PNG             # Example conversation 2
│   ├── Conv3.PNG             # Example conversation 3
│   ├── Integration_step1.png # Git repo integration step 1
│   ├── Integration_step2.png # Git repo integration step 2
│   ├── Integration_step3.png # Git repo integration step 3
│   ├── Env_Var_step1.png     # Environment variables setup step 1
│   ├── Env_Var_step2.png     # Environment variables setup step 2
│   └── Env_Var_step3.png     # Environment variables setup step 3
│
├── ingestion.py              # Main data ingestion pipeline
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in repo)
├── mcp.json                  # MCP configuration for BrightData
├── automated_test_results.json  # Test results (for README)
├── baseline_comparison_results.json  # Comparison results (for README)
└── README.md                 # This file
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

**1. Databricks: ModuleNotFoundError**
- **Solution**: Install dependencies on cluster (see Step 3 in Databricks setup)
- Run `%pip install <package>` in a notebook cell if needed
- Verify cluster libraries are installed and cluster is restarted

**2. Databricks: Environment variables not found**
- **Solution**: 
  - Check cluster configuration → Advanced Options → Environment Variables
  - Or use Databricks secrets (more secure): `dbutils.secrets.get("scope", "key")`
  - Verify `agents/config_databricks.py` is using `get_secret()` correctly

**3. Databricks: Import errors (agents module not found)**
- **Solution**: Verify repo path in notebook:
  ```python
  REPO_PATH = "/Workspace/Users/<your-email>/hotel-intelligence-system"
  sys.path.insert(0, REPO_PATH)
  sys.path.insert(0, f"{REPO_PATH}/agents")
  ```

**4. Databricks: Long-running jobs timeout**
- **Solution**: The coordinator handles this automatically (15+ min timeouts for ML analyses)
- Check `agents/coordinator.py` for `DATABRICKS_TIMEOUTS` configuration
- Ensure cluster has sufficient resources

**5. Pinecone connection error**
- Verify `PINECONE_API_KEY` is set (cluster env vars or secrets)
- Check index name matches in code (`airbnb-index` by default)
- Verify network connectivity from Databricks to Pinecone

**6. Gemini quota exceeded**
- System auto-falls back to Groq/Llama-3
- Ensure `GROQ_API_KEY` is set in cluster environment variables
- Check fallback is working: Look for `[LLM] WARNING: Gemini quota hit!` messages

**7. Empty search results**
- Verify data was ingested (`python ingestion.py` or use ingestion scripts)
- Check namespace names match (`airbnb_reviews`, `booking_reviews`, etc.)
- Ensure hotel_id filter is correct (format: `ABB_40458495` or `BKG_177691`)

**8. Local: Playwright browser not found**
```bash
playwright install chromium
```

**9. Local: Scraping fails (403/CAPTCHA)**
- Google/TripAdvisor may block requests
- Try different user-agent strings
- Consider adding delays between requests
- Note: Playwright scraping is disabled on Databricks (use Bright Data APIs instead)

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
