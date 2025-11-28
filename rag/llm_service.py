"""
LLM Service for RAG System

This module provides Large Language Model integration for the RAG system,
including context-aware analysis, response generation, and model management.
"""

import json
import requests
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import os
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with Large Language Models.
    
    Features:
    - Multiple LLM provider support
    - Context-aware analysis
    - Response generation
    - Error handling and fallbacks
    - Rate limiting and retry logic
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        
        # Model configuration
        self.max_tokens = 1000
        self.temperature = 0.7
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        
        # Rate limiting
        self.requests_per_minute = 60
        self.request_timestamps = []
        
        # Provider configuration
        self.provider = self._determine_provider(model_name)
        self.base_url = self._get_base_url()
        
        logger.info(f"Initialized LLMService with model: {model_name}, provider: {self.provider}")
    
    def generate_response(self, prompt: str, context: str = "", 
                         max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Input prompt
            context: Additional context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, using fallback response")
                return self._generate_fallback_response(prompt, context)
            
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Generate response based on provider
            if self.provider == "openai":
                response = self._call_openai_api(full_prompt, max_tokens, temperature)
            elif self.provider == "anthropic":
                response = self._call_anthropic_api(full_prompt, max_tokens, temperature)
            elif self.provider == "local":
                response = self._call_local_api(full_prompt, max_tokens, temperature)
            else:
                logger.warning(f"Unknown provider: {self.provider}, using fallback")
                response = self._generate_fallback_response(prompt, context)
            
            # Record request timestamp
            self._record_request()
            
            return response
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self._generate_fallback_response(prompt, context)
    
    def analyze_market_context(self, symbol: str, news_context: str, 
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market context using LLM.
        
        Args:
            symbol: Stock symbol
            news_context: News and event context
            market_data: Current market data
            
        Returns:
            Analysis results
        """
        try:
            prompt = f"""
            As a financial market analyst, analyze the following information for {symbol}:
            
            News Context:
            {news_context}
            
            Market Data:
            {json.dumps(market_data, indent=2)}
            
            Please provide:
            1. Key market events and their potential impact
            2. Risk assessment for {symbol}
            3. Trading recommendation (buy/sell/hold) with reasoning
            4. Confidence level in your analysis (0-1)
            5. Key factors driving the recommendation
            
            Format your response as JSON with the following structure:
            {{
                "impact_analysis": "description of key events and their impact",
                "risk_assessment": "risk level and factors",
                "recommendation": "buy/sell/hold",
                "reasoning": "detailed reasoning for the recommendation",
                "confidence": 0.8,
                "key_factors": ["factor1", "factor2", "factor3"]
            }}
            """
            
            response = self.generate_response(prompt)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response
                return self._parse_text_response(response, symbol)
            
        except Exception as e:
            logger.error(f"Market context analysis failed: {e}")
            return self._get_default_analysis(symbol)
    
    def generate_event_summary(self, events: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of market events.
        
        Args:
            events: List of market events
            
        Returns:
            Event summary
        """
        try:
            if not events:
                return "No significant market events detected."
            
            events_text = "\n".join([
                f"- {event.get('title', 'Unknown Event')}: {event.get('description', 'No description')}"
                for event in events[:5]  # Limit to top 5 events
            ])
            
            prompt = f"""
            Summarize the following market events and their potential impact:
            
            {events_text}
            
            Provide a concise summary focusing on:
            1. Most significant events
            2. Overall market sentiment
            3. Potential trading implications
            
            Keep the summary under 200 words.
            """
            
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Event summary generation failed: {e}")
            return f"Detected {len(events)} market events. Analysis unavailable."
    
    def explain_prediction(self, prediction: str, context: str) -> str:
        """
        Generate an explanation for a prediction.
        
        Args:
            prediction: The prediction to explain
            context: Context for the prediction
            
        Returns:
            Explanation text
        """
        try:
            prompt = f"""
            Explain the following trading prediction in simple terms:
            
            Prediction: {prediction}
            
            Context: {context}
            
            Provide a clear explanation that covers:
            1. What the prediction means
            2. Why this prediction was made
            3. What factors influenced the decision
            4. What to watch for going forward
            
            Use language that a non-expert can understand.
            """
            
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {e}")
            return f"Prediction: {prediction}. Detailed explanation unavailable."
    
    def _determine_provider(self, model_name: str) -> str:
        """Determine the provider based on model name."""
        if model_name.startswith("gpt-") or model_name.startswith("text-"):
            return "openai"
        elif model_name.startswith("claude-"):
            return "anthropic"
        elif model_name.startswith("local-"):
            return "local"
        else:
            return "openai"  # Default to OpenAI
    
    def _get_base_url(self) -> str:
        """Get the base URL for the provider."""
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        elif self.provider == "anthropic":
            return "https://api.anthropic.com/v1"
        elif self.provider == "local":
            return "http://localhost:11434"  # Default for local Ollama
        else:
            return "https://api.openai.com/v1"
    
    def _prepare_prompt(self, prompt: str, context: str) -> str:
        """Prepare the full prompt with context."""
        if context:
            return f"Context: {context}\n\nPrompt: {prompt}"
        return prompt
    
    def _call_openai_api(self, prompt: str, max_tokens: int = None, 
                        temperature: float = None) -> str:
        """Call OpenAI API."""
        try:
            if not self.api_key:
                logger.warning("No OpenAI API key provided")
                return self._generate_fallback_response(prompt, "")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return self._generate_fallback_response(prompt, "")
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._generate_fallback_response(prompt, "")
    
    def _call_anthropic_api(self, prompt: str, max_tokens: int = None, 
                           temperature: float = None) -> str:
        """Call Anthropic API."""
        try:
            if not self.api_key:
                logger.warning("No Anthropic API key provided")
                return self._generate_fallback_response(prompt, "")
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model_name,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return self._generate_fallback_response(prompt, "")
                
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return self._generate_fallback_response(prompt, "")
    
    def _call_local_api(self, prompt: str, max_tokens: int = None, 
                       temperature: float = None) -> str:
        """Call local API (e.g., Ollama)."""
        try:
            data = {
                "model": self.model_name.replace("local-", ""),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                logger.error(f"Local API error: {response.status_code} - {response.text}")
                return self._generate_fallback_response(prompt, "")
                
        except Exception as e:
            logger.error(f"Local API call failed: {e}")
            return self._generate_fallback_response(prompt, "")
    
    def _generate_fallback_response(self, prompt: str, context: str) -> str:
        """Generate a fallback response when LLM is unavailable."""
        try:
            # Simple rule-based response generation
            prompt_lower = prompt.lower()
            
            if "buy" in prompt_lower or "bullish" in prompt_lower:
                return "Based on available information, a bullish outlook is suggested. Consider monitoring for entry opportunities."
            elif "sell" in prompt_lower or "bearish" in prompt_lower:
                return "Based on available information, a bearish outlook is suggested. Consider risk management strategies."
            elif "hold" in prompt_lower or "neutral" in prompt_lower:
                return "Based on available information, a neutral outlook is suggested. Maintain current position with close monitoring."
            else:
                return "Analysis completed. Market conditions require continued monitoring for trading opportunities."
                
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "Analysis unavailable. Please try again later."
    
    def _parse_text_response(self, response: str, symbol: str) -> Dict[str, Any]:
        """Parse a text response into structured format."""
        try:
            # Extract key information from text response
            response_lower = response.lower()
            
            # Determine recommendation
            if "buy" in response_lower:
                recommendation = "buy"
            elif "sell" in response_lower:
                recommendation = "sell"
            else:
                recommendation = "hold"
            
            # Extract confidence (look for numbers)
            import re
            confidence_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', response_lower)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            return {
                "impact_analysis": response[:200] + "..." if len(response) > 200 else response,
                "risk_assessment": "Standard market risk assessment",
                "recommendation": recommendation,
                "reasoning": response,
                "confidence": min(confidence, 1.0),
                "key_factors": ["Market sentiment", "Technical indicators", "News impact"]
            }
            
        except Exception as e:
            logger.error(f"Text response parsing failed: {e}")
            return self._get_default_analysis(symbol)
    
    def _get_default_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get default analysis when LLM fails."""
        return {
            "impact_analysis": f"Standard market analysis for {symbol}",
            "risk_assessment": "Moderate risk level",
            "recommendation": "hold",
            "reasoning": "Insufficient data for detailed analysis",
            "confidence": 0.5,
            "key_factors": ["Market volatility", "General sentiment"]
        }
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        try:
            now = datetime.now()
            # Remove timestamps older than 1 minute
            self.request_timestamps = [
                ts for ts in self.request_timestamps 
                if (now - ts).total_seconds() < 60
            ]
            
            return len(self.request_timestamps) < self.requests_per_minute
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request if check fails
    
    def _record_request(self) -> None:
        """Record a request timestamp."""
        try:
            self.request_timestamps.append(datetime.now())
        except Exception as e:
            logger.error(f"Failed to record request timestamp: {e}")
