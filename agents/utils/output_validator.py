"""
Output Validator

Validates agent responses against tool outputs to catch hallucinations.
Uses Pydantic for structured validation and LLM-based claim verification.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import re


class ConfidenceLevel(str, Enum):
    """Confidence level for claims."""
    HIGH = "high"       # Directly quoted from tool output
    MEDIUM = "medium"   # Paraphrased from tool output  
    LOW = "low"         # Inferred/interpreted
    UNSUPPORTED = "unsupported"  # No supporting evidence


class Claim(BaseModel):
    """A single claim made in the response."""
    text: str = Field(..., description="The claim text")
    source: Optional[str] = Field(None, description="Source tool/data that supports this")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.UNSUPPORTED)
    supporting_quote: Optional[str] = Field(None, description="Exact quote from tool output")


class ValidatedResponse(BaseModel):
    """Structured response with validation metadata."""
    original_response: str = Field(..., description="Original agent response")
    claims: List[Claim] = Field(default_factory=list)
    hallucination_risk: float = Field(0.0, ge=0.0, le=1.0, description="0-1 risk score")
    warnings: List[str] = Field(default_factory=list)
    tool_outputs_used: List[str] = Field(default_factory=list)
    is_valid: bool = Field(True)
    
    @validator('hallucination_risk')
    def round_risk(cls, v):
        return round(v, 2)


class OutputValidator:
    """
    Validates agent outputs against tool results to detect hallucinations.
    
    Usage:
        validator = OutputValidator()
        result = validator.validate(response, tool_outputs)
    """
    
    # Patterns that suggest hallucination risk
    HALLUCINATION_PATTERNS = [
        r"(?:guests?|reviewers?|visitors?)\s+(?:say|said|mention|mentioned|report|reported|complain|complained)",
        r"(?:many|most|several|some|few)\s+(?:guests?|reviewers?|people)",
        r"(?:commonly|frequently|often|usually|typically)\s+(?:mentioned|reported|complained)",
        r"according to (?:reviews?|guests?|feedback)",
        r"(?:overall|generally|in general),?\s+(?:guests?|reviewers?)",
    ]
    
    # Patterns that suggest grounded claims
    GROUNDED_PATTERNS = [
        r'"[^"]{10,}"',  # Direct quotes
        r"from (?:booking|airbnb|tripadvisor|google)",  # Source attribution
        r"search results? (?:show|indicate)",
        r"(?:the )?(?:tool|search|data|database) (?:returned|found|shows)",
        r"no (?:information|data|results?) found",
    ]
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, fails validation on any unsupported claims
        """
        self.strict_mode = strict_mode
    
    def validate(
        self, 
        response: str, 
        tool_outputs: List[str],
        llm = None
    ) -> ValidatedResponse:
        """
        Validate agent response against tool outputs.
        
        Args:
            response: The agent's final response
            tool_outputs: List of tool output strings collected during execution
            llm: Optional LLM for deep claim extraction (if None, uses regex)
            
        Returns:
            ValidatedResponse with validation metadata
        """
        warnings = []
        claims = []
        
        # Combine all tool outputs for reference
        all_tool_text = "\n".join(tool_outputs).lower()
        response_lower = response.lower()
        
        # 1. Check for hallucination patterns without supporting data
        risk_score = 0.0
        pattern_matches = 0
        
        for pattern in self.HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            if matches:
                pattern_matches += len(matches)
                # Check if the surrounding context is in tool outputs
                for match in matches:
                    # Get surrounding context (50 chars before and after)
                    idx = response_lower.find(match)
                    context_start = max(0, idx - 50)
                    context_end = min(len(response_lower), idx + len(match) + 50)
                    context = response_lower[context_start:context_end]
                    
                    # Check if context appears in tool outputs
                    if not self._fuzzy_match(context, all_tool_text):
                        warnings.append(f"Potential hallucination: '{match}' not found in tool outputs")
                        risk_score += 0.15
        
        # 2. Check for grounded patterns (reduces risk)
        grounded_matches = 0
        for pattern in self.GROUNDED_PATTERNS:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            grounded_matches += len(matches)
        
        # Reduce risk for grounded claims
        if grounded_matches > 0:
            risk_score = max(0, risk_score - (grounded_matches * 0.05))
        
        # 3. Extract direct quotes and verify them
        quotes = re.findall(r'"([^"]{10,})"', response)
        for quote in quotes:
            quote_lower = quote.lower()
            if self._fuzzy_match(quote_lower, all_tool_text, threshold=0.7):
                claims.append(Claim(
                    text=quote,
                    confidence=ConfidenceLevel.HIGH,
                    supporting_quote=quote
                ))
            else:
                warnings.append(f"Quote not found in tool outputs: '{quote[:50]}...'")
                risk_score += 0.2
                claims.append(Claim(
                    text=quote,
                    confidence=ConfidenceLevel.UNSUPPORTED
                ))
        
        # 4. Check for specific factual claims (numbers, ratings, prices)
        number_claims = re.findall(
            r'(?:rated?|score|rating|price|cost|stars?)[:\s]+(\d+\.?\d*)',
            response_lower
        )
        for num in number_claims:
            if num in all_tool_text:
                claims.append(Claim(
                    text=f"Numeric claim: {num}",
                    confidence=ConfidenceLevel.HIGH,
                    supporting_quote=num
                ))
            else:
                warnings.append(f"Numeric claim '{num}' not found in tool outputs")
                risk_score += 0.1
        
        # 5. Check for "No information found" honesty
        if "no information found" in response_lower or "not found" in response_lower:
            if not tool_outputs or all(
                "no" in t.lower() or "not found" in t.lower() or len(t.strip()) < 50 
                for t in tool_outputs
            ):
                # Agent correctly admits no data
                risk_score = max(0, risk_score - 0.1)
        
        # Cap risk score
        risk_score = min(1.0, risk_score)
        
        # Determine validity
        is_valid = risk_score < 0.5 if not self.strict_mode else risk_score < 0.2
        
        return ValidatedResponse(
            original_response=response,
            claims=claims,
            hallucination_risk=risk_score,
            warnings=warnings,
            tool_outputs_used=[t[:100] + "..." if len(t) > 100 else t for t in tool_outputs],
            is_valid=is_valid
        )
    
    def _fuzzy_match(self, needle: str, haystack: str, threshold: float = 0.5) -> bool:
        """
        Check if needle approximately appears in haystack.
        Uses word overlap for fuzzy matching.
        """
        if not needle or not haystack:
            return False
            
        # Exact substring match
        if needle in haystack:
            return True
        
        # Word overlap match
        needle_words = set(re.findall(r'\w+', needle.lower()))
        haystack_words = set(re.findall(r'\w+', haystack.lower()))
        
        if not needle_words:
            return False
            
        overlap = len(needle_words & haystack_words) / len(needle_words)
        return overlap >= threshold
    
    def format_validation_report(self, result: ValidatedResponse) -> str:
        """Format validation result as human-readable report."""
        report = []
        report.append("=" * 50)
        report.append("VALIDATION REPORT")
        report.append("=" * 50)
        
        status = "✓ VALID" if result.is_valid else "✗ POTENTIALLY HALLUCINATED"
        report.append(f"Status: {status}")
        report.append(f"Hallucination Risk: {result.hallucination_risk:.0%}")
        
        if result.warnings:
            report.append(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings[:5]:  # Limit to 5
                report.append(f"  ⚠ {w}")
        
        if result.claims:
            high_conf = [c for c in result.claims if c.confidence == ConfidenceLevel.HIGH]
            report.append(f"\nVerified Claims: {len(high_conf)}/{len(result.claims)}")
        
        report.append("=" * 50)
        return "\n".join(report)


def validate_response(
    response: str, 
    tool_outputs: List[str],
    strict: bool = False
) -> ValidatedResponse:
    """
    Convenience function to validate a response.
    
    Args:
        response: Agent's response text
        tool_outputs: List of tool output strings
        strict: Whether to use strict validation mode
        
    Returns:
        ValidatedResponse object
    """
    validator = OutputValidator(strict_mode=strict)
    return validator.validate(response, tool_outputs)
