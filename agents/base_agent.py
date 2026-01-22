"""
Base Agent Class

Shared functionality for all specialist agents.
Includes LLM fallback: Gemini (Primary) → Groq/Llama-3 (Fallback)
Includes output validation to catch hallucinations.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

# Handle imports whether run from project root or agents directory
try:
    from agents.config_databricks import get_secret, is_databricks
except ModuleNotFoundError:
    from config_databricks import get_secret, is_databricks
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    
# Import validator (optional - graceful fallback if not available)
try:
    from agents.utils.output_validator import validate_response, OutputValidator
    VALIDATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from utils.output_validator import validate_response, OutputValidator
        VALIDATION_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        VALIDATION_AVAILABLE = False

# Configuration
INDEX_NAME = "booking-agent"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Fallback Configuration (Groq is excellent for tool calling)
FALLBACK_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL_2 = "llama-3.1-8b-instant"  # Secondary fallback when Llama hits rate limit

# Token budget limits for Groq free tier (6000 TPM)
MAX_TOOL_OUTPUT_CHARS = 2000  # Truncate tool outputs to ~500 tokens
MAX_CONTEXT_CHARS = 1500  # Truncate conversation context

class LLMWithFallback:
    """
    Wrapper that automatically falls back to Groq on Gemini quota errors.
    Supports multiple fallback levels: Gemini -> Llama -> Mixtral
    """

    def __init__(self):
        self._primary = None
        self._fallback = None
        self._fallback_2 = None
        self._using_fallback = True
        self._using_fallback_2 = False
        # self._init_primary()

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
                    raise ValueError("GROQ_API_KEY not found. Set it in .env (local) or cluster environment variables (Databricks).")
                self._fallback = ChatGroq(
                    model=FALLBACK_MODEL,
                    temperature=0,
                    max_retries=1,
                    api_key=api_key
                )
                print(f"[LLM] Fallback initialized: {FALLBACK_MODEL}")
            except ImportError:
                print("Error: langchain-groq not installed. Run: pip install langchain-groq")
                raise
            except Exception as e:
                print(f"[LLM] Failed to init fallback: {e}")
                raise

    def _init_fallback_2(self):
        """Initialize secondary fallback (Mixtral)."""
        if self._fallback_2 is None:
            try:
                from langchain_groq import ChatGroq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found.")
                self._fallback_2 = ChatGroq(
                    model=FALLBACK_MODEL_2,
                    temperature=0,
                    max_retries=1,
                    api_key=api_key
                )
                print(f"[LLM] Secondary fallback initialized: {FALLBACK_MODEL_2}")
            except Exception as e:
                print(f"[LLM] Failed to init secondary fallback: {e}")
                raise

    def _get_current_model(self):
        """Get the currently active model."""
        if self._using_fallback_2:
            self._init_fallback_2()
            return self._fallback_2
        elif self._using_fallback:
            self._init_fallback()
            return self._fallback
        return self._primary

    def invoke(self, messages):
        """Invoke with automatic failover through multiple fallback levels."""
        model = self._get_current_model()
        
        try:
            return model.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["quota", "429", "resource", "exhausted", "overloaded", "rate_limit"]):
                # If using primary, switch to fallback 1
                if not self._using_fallback:
                    print(f"[LLM] WARNING: Gemini quota hit! Switching to {FALLBACK_MODEL}...")
                    self._using_fallback = True
                    self._init_fallback()
                    return self._fallback.invoke(messages)
                # If using fallback 1, switch to fallback 2
                elif not self._using_fallback_2:
                    print(f"[LLM] WARNING: {FALLBACK_MODEL} rate limit! Switching to {FALLBACK_MODEL_2}...")
                    self._using_fallback_2 = True
                    self._init_fallback_2()
                    return self._fallback_2.invoke(messages)
                # All fallbacks exhausted
                else:
                    print("[LLM] ERROR: All models rate limited!")
                    raise e
            raise e

    def bind_tools(self, tools):
        """
        Bind tools to models.
        This is the wrapper that returns a runnable.
        """
        return BoundLLMWithFallback(self, tools)


class BoundLLMWithFallback:
    """Helper class to handle tool binding for the fallback wrapper with multiple fallback levels."""
    def __init__(self, wrapper, tools):
        self.wrapper = wrapper
        self.tools = tools

        # Pre-bind tools to primary
        if self.wrapper._primary:
            self.primary_bound = self.wrapper._primary.bind_tools(tools)
        else:
            self.primary_bound = None

        # Fallback bounds are lazy-loaded
        self.fallback_bound = None
        self.fallback_2_bound = None

    def invoke(self, input):
        # 1. Use secondary fallback if active
        if self.wrapper._using_fallback_2:
            return self._get_fallback_2_bound().invoke(input)
        
        # 2. Use primary fallback if active
        if self.wrapper._using_fallback:
            try:
                return self._get_fallback_bound().invoke(input)
            except Exception as e:
                error_str = str(e).lower()
                if any(x in error_str for x in ["quota", "429", "resource", "exhausted", "rate_limit"]):
                    print(f"[LLM] WARNING: {FALLBACK_MODEL} rate limit (during tool call)! Switching to {FALLBACK_MODEL_2}...")
                    self.wrapper._using_fallback_2 = True
                    return self._get_fallback_2_bound().invoke(input)
                raise e

        # 3. Try Primary
        try:
            return self.primary_bound.invoke(input)
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["quota", "429", "resource", "exhausted", "rate_limit"]):
                print(f"[LLM] WARNING: Gemini quota hit (during tool call)! Switching to {FALLBACK_MODEL}...")
                self.wrapper._using_fallback = True
                return self._get_fallback_bound().invoke(input)
            raise e

    def _get_fallback_bound(self):
        """Lazy load and bind fallback model."""
        if not self.fallback_bound:
            self.wrapper._init_fallback()
            self.fallback_bound = self.wrapper._fallback.bind_tools(self.tools)
        return self.fallback_bound

    def _get_fallback_2_bound(self):
        """Lazy load and bind secondary fallback model."""
        if not self.fallback_2_bound:
            self.wrapper._init_fallback_2()
            self.fallback_2_bound = self.wrapper._fallback_2.bind_tools(self.tools)
        return self.fallback_2_bound


class BaseAgent(ABC):
    """Base class for all specialist agents."""

    def __init__(
        self, 
        hotel_id: str, 
        hotel_name: str, 
        city: str,
        validate_output: bool = True,
        strict_validation: bool = False
    ):
        self.hotel_id = hotel_id
        self.hotel_name = hotel_name
        self.city = city

        # LLM with automatic fallback
        self.llm = LLMWithFallback()

        # Embeddings (lazy load)
        self._embeddings = None
        
        # Validation settings
        self.validate_output = validate_output and VALIDATION_AVAILABLE
        self.strict_validation = strict_validation
        self._tool_outputs: list = []  # Collect tool outputs for validation

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
        k = int(k)  # Coerce in case LLM passes string
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

    def _parse_malformed_tool_calls(self, content: str, tool_map: dict) -> list:
        """
        Parse malformed tool calls from Groq/Llama that output:
        <function=tool_name {"arg": "value"}</function>
        or <function=tool_name>{"arg": "value"}</function>
        
        Returns list of parsed tool calls: [{"name": str, "args": dict, "id": str}]
        """
        import re
        import json
        import uuid
        
        parsed = []
        
        # Pattern 1: <function=tool_name {"args"}</function>
        # Pattern 2: <function=tool_name>{"args"}</function>
        patterns = [
            r'<function=(\w+)\s*(\{[^}]+\})\s*(?:</function>|<function>)?',
            r'<function=(\w+)>\s*(\{[^}]+\})\s*</function>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                fn_name = match[0]
                args_str = match[1]
                
                # Only process if tool exists
                if fn_name not in tool_map:
                    continue
                
                try:
                    args = json.loads(args_str)
                    # Normalize args: convert string numbers to int where needed
                    for key, val in args.items():
                        if isinstance(val, str) and val.isdigit():
                            args[key] = int(val)
                    
                    parsed.append({
                        "name": fn_name,
                        "args": args,
                        "id": f"manual_{uuid.uuid4().hex[:8]}"
                    })
                except json.JSONDecodeError:
                    print(f"[ToolRecovery] Failed to parse args for {fn_name}: {args_str[:50]}")
                    continue
        
        return parsed

    def _coerce_tool_args(self, args: dict) -> dict:
        """
        Coerce tool arguments to correct types.
        Prevents schema validation errors like 'expected integer, but got string'.
        """
        if not args:
            return args
        
        coerced = {}
        for key, val in args.items():
            if isinstance(val, str):
                # Try to convert numeric strings to int
                if val.isdigit():
                    coerced[key] = int(val)
                # Try to convert float strings
                elif val.replace('.', '', 1).isdigit() and val.count('.') == 1:
                    coerced[key] = float(val)
                # Try to convert boolean strings
                elif val.lower() in ('true', 'false'):
                    coerced[key] = val.lower() == 'true'
                else:
                    coerced[key] = val
            else:
                coerced[key] = val
        
        return coerced

    def run(self, query: str, return_validation: bool = False) -> str:
        """
        Execute the agent with multi-turn tool execution loop.
        
        Args:
            query: User query to process
            return_validation: If True and validation enabled, returns (response, validation_result)
            
        Returns:
            Response string, or tuple of (response, validation) if return_validation=True
        """
        tools = self.get_tools()
        
        # Reset tool outputs for this run
        self._tool_outputs = []

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
        MAX_ITERATIONS = 8
        iteration = 0
        
        # Build tool map once
        tool_map = {t.__name__: t for t in tools} if tools else {}

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Invoke LLM - with error recovery for malformed tool calls
            tool_calls = []
            response = None
            
            try:
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                # Get tool calls - either from proper response or parse malformed ones
                tool_calls = response.tool_calls if response.tool_calls else []
                
                # If no proper tool calls, try to recover from malformed format in content
                if not tool_calls and response.content and "<function=" in response.content:
                    tool_calls = self._parse_malformed_tool_calls(response.content, tool_map)
                    if tool_calls:
                        print(f"[ToolRecovery] Recovered {len(tool_calls)} tool call(s) from response content")
                        
            except Exception as e:
                error_str = str(e)
                # Check if this is a "tool_use_failed" error with failed_generation (Groq API)
                if "tool_use_failed" in error_str and "failed_generation" in error_str:
                    print(f"[ToolRecovery] Caught Groq API tool_use_failed error, attempting recovery...")
                    
                    # Extract the failed_generation content from error
                    import re
                    match = re.search(r"'failed_generation':\s*'([^']+)'", error_str)
                    if match:
                        failed_content = match.group(1)
                        tool_calls = self._parse_malformed_tool_calls(failed_content, tool_map)
                        
                        if tool_calls:
                            print(f"[ToolRecovery] Recovered {len(tool_calls)} tool call(s) from API error")
                        else:
                            # Could not recover, raise the error
                            raise RuntimeError(f"Tool call failed and could not recover: {error_str[:200]}")
                    else:
                        raise RuntimeError(f"Tool call failed: {error_str[:200]}")
                else:
                    # Not a tool_use_failed error, re-raise
                    raise e

            # Check if the model wants to stop (no tools called)
            if not tool_calls:
                final_response = response.content if response else "No response generated."
                return self._finalize_response(final_response, return_validation)

            # Track if we recovered from API error (no proper response object)
            recovered_from_api_error = response is None

            # Handle Tool Calls
            for tool_call in tool_calls:
                fn_name = tool_call["name"]
                args = tool_call["args"]
                
                # Coerce numeric string args to integers (prevents schema validation errors)
                args = self._coerce_tool_args(args)
                
                print(f"[{self.__class__.__name__}] TOOL: {fn_name}")

                if fn_name in tool_map:
                    try:
                        result = tool_map[fn_name](**args)
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                else:
                    result = f"Unknown tool: {fn_name}"

                result_str = str(result)
                print(f"   >>> Tool Output ({fn_name}): {result_str[:100]}...")
                
                # Collect full tool output for validation
                self._tool_outputs.append(result_str)
                
                # Truncate tool output to avoid token overflow on Groq
                if len(result_str) > MAX_TOOL_OUTPUT_CHARS:
                    truncated = result_str[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated {len(result_str) - MAX_TOOL_OUTPUT_CHARS} chars]"
                else:
                    truncated = result_str

                # Append tool result to history so the model sees it
                # Use HumanMessage if recovered from API error (no proper tool_call chain)
                if recovered_from_api_error:
                    messages.append(HumanMessage(content=f"Tool '{fn_name}' returned:\n{truncated}"))
                else:
                    messages.append(ToolMessage(content=truncated, tool_call_id=tool_call["id"]))

        return self._finalize_response("Agent stopped after max iterations.", return_validation)
    
    def _finalize_response(self, response: str, return_validation: bool = False):
        """
        Finalize response with optional validation.
        
        Args:
            response: The agent's final response
            return_validation: Whether to return validation result
            
        Returns:
            Response string or (response, validation_result) tuple
        """
        validation_result = None
        
        if self.validate_output and self._tool_outputs:
            validation_result = validate_response(
                response, 
                self._tool_outputs,
                strict=self.strict_validation
            )
            
            # Log validation warnings
            if validation_result.warnings:
                print(f"[{self.__class__.__name__}] VALIDATION WARNINGS:")
                for w in validation_result.warnings[:3]:
                    print(f"   ⚠ {w}")
                print(f"   Hallucination Risk: {validation_result.hallucination_risk:.0%}")
            
            # In strict mode, reject response if hallucination risk is too high
            if self.strict_validation and validation_result.hallucination_risk > 0.3:
                # Replace response with safe admission of insufficient data
                response = f"""I searched for information but couldn't find reliable data to answer your question accurately.

**What I searched:**
- Internal review database
- Web search engines

**What I found:**
- Limited or no specific information about your query

**Recommendations:**
1. Try asking about a different aspect of your property
2. Check if there are reviews in your database
3. Consider gathering more guest feedback

To avoid providing inaccurate information, I'm being conservative here. If you'd like, I can share the search snippets I found, but they don't contain enough detail for a confident analysis.

[Validation: High hallucination risk ({validation_result.hallucination_risk:.0%}) - response blocked for accuracy]
"""
        
        if return_validation and validation_result:
            return response, validation_result
        return response