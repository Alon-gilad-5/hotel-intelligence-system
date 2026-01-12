"""
Base Agent Class

Shared functionality for all specialist agents.
Includes LLM fallback: Gemini (Primary) → Groq/Llama-3 (Fallback)
"""

import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

load_dotenv()

# Configuration
INDEX_NAME = "booking-agent"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Fallback Configuration (Groq is excellent for tool calling)
FALLBACK_MODEL = "llama-3.3-70b-versatile"

class LLMWithFallback:
    """
    Wrapper that automatically falls back to Groq on Gemini quota errors.
    Both models support Tool Calling.
    """

    def __init__(self):
        self._primary = None
        self._fallback = None
        self._using_fallback = False
        self._init_primary()

    def _init_primary(self):
        """Initialize Gemini."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._primary = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_retries=1,
            )
        except Exception as e:
            print(f"[LLM] Could not init Gemini: {e}")
            self._using_fallback = True

    def _init_fallback(self):
        """Initialize Groq (Llama-3)."""
        if self._fallback is None:
            try:
                from langchain_groq import ChatGroq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in .env")

                self._fallback = ChatGroq(
                    model=FALLBACK_MODEL,
                    temperature=0,
                    max_retries=2,
                    api_key=api_key
                )
                print(f"[LLM] Fallback initialized: {FALLBACK_MODEL}")
            except ImportError:
                print("Error: langchain-groq not installed. Run: pip install langchain-groq")
                raise
            except Exception as e:
                print(f"[LLM] Failed to init fallback: {e}")
                raise

    def invoke(self, messages):
        """Invoke with automatic failover."""
        # 1. Try Fallback if already flagged
        if self._using_fallback:
            self._init_fallback()
            return self._fallback.invoke(messages)

        # 2. Try Primary (Gemini)
        try:
            return self._primary.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["quota", "429", "resource", "exhausted", "overloaded"]):
                print(f"[LLM] ⚠️ Gemini quota hit! Switching to {FALLBACK_MODEL}...")
                self._using_fallback = True
                self._init_fallback()
                return self._fallback.invoke(messages)  # Retry immediately with fallback

            # If it's a different error, raise it
            raise e

    def bind_tools(self, tools):
        """
        Bind tools to BOTH models.
        This is the wrapper that returns a runnable.
        """
        # We need to return an object that, when invoked, checks the fallback state
        # and routes to the correct bound model.
        return BoundLLMWithFallback(self, tools)


class BoundLLMWithFallback:
    """Helper class to handle tool binding for the fallback wrapper."""
    def __init__(self, wrapper, tools):
        self.wrapper = wrapper
        self.tools = tools

        # Pre-bind tools to primary
        if self.wrapper._primary:
            self.primary_bound = self.wrapper._primary.bind_tools(tools)
        else:
            self.primary_bound = None

        # Fallback bound is lazy-loaded
        self.fallback_bound = None

    def invoke(self, input):
        # 1. Use Fallback if active
        if self.wrapper._using_fallback:
            return self._get_fallback_bound().invoke(input)

        # 2. Try Primary
        try:
            return self.primary_bound.invoke(input)
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["quota", "429", "resource", "exhausted"]):
                print(f"[LLM] ⚠️ Gemini quota hit (during tool call)! Switching to {FALLBACK_MODEL}...")
                self.wrapper._using_fallback = True
                return self._get_fallback_bound().invoke(input)
            raise e

    def _get_fallback_bound(self):
        """Lazy load and bind fallback model."""
        if not self.fallback_bound:
            self.wrapper._init_fallback()
            self.fallback_bound = self.wrapper._fallback.bind_tools(self.tools)
        return self.fallback_bound


class BaseAgent(ABC):
    """Base class for all specialist agents."""

    def __init__(self, hotel_id: str, hotel_name: str, city: str):
        self.hotel_id = hotel_id
        self.hotel_name = hotel_name
        self.city = city

        # LLM with automatic fallback
        self.llm = LLMWithFallback()

        # Embeddings (lazy load)
        self._embeddings = None

    @property
    def embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None:
            # print(f"[{self.__class__.__name__}] Loading embeddings...")
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    def get_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """Get Pinecone vectorstore for a namespace."""
        return PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=self.embeddings,
            namespace=namespace
        )

    def search_rag(self, query: str, namespace: str, k: int = 5, filter_dict: dict = None) -> list:
        """Search RAG for relevant documents."""
        vectorstore = self.get_vectorstore(namespace)
        try:
            results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
            return results
        except Exception as e:
            print(f"[RAG] Search error in {namespace}: {e}")
            return []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_tools(self) -> list:
        pass

    def run(self, query: str) -> str:
        """Execute the agent with multi-turn tool execution loop."""
        tools = self.get_tools()

        # Bind tools (works for both Gemini AND Groq)
        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
        else:
            llm_with_tools = self.llm

        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=query)
        ]

        # Max turns to prevent infinite loops
        MAX_ITERATIONS = 5
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Invoke LLM
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # Check if the model wants to stop (no tools called)
            if not response.tool_calls:
                return response.content

            # Handle Tool Calls
            tool_map = {t.__name__: t for t in tools}

            for tool_call in response.tool_calls:
                fn_name = tool_call["name"]
                args = tool_call["args"]
                print(f"[{self.__class__.__name__}] 🛠️ Tool: {fn_name}")

                if fn_name in tool_map:
                    try:
                        result = tool_map[fn_name](**args)
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                else:
                    result = f"Unknown tool: {fn_name}"

                print(f"   >>> Tool Output ({fn_name}): {str(result)[:100]}...")

                # Append tool result to history so the model sees it
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        return "Agent stopped after max iterations."