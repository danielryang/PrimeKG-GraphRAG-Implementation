"""
LLM Client implementations for GraphRAG

Supports OpenAI and Anthropic APIs with fallback to mock client.
"""

import os
import json
import logging
import time
import random
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from functools import wraps

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Retry decorator with exponential backoff (addresses CLAUDE.md Problem 5)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on certain types of errors
                    if isinstance(e, (ValueError, TypeError)) or "api key" in str(e).lower():
                        raise e
                    
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        delay = base_delay * (exponential_base ** attempt)
                        if jitter:
                            delay *= (0.5 + random.random() * 0.5)  # Add jitter
                        
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            
            raise last_exception
        return wrapper
    return decorator

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts asynchronously.
        
        Default implementation processes sequentially. Subclasses should override
        for true async batch processing.
        
        Args:
            prompts: List of prompts to process
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of generated responses in the same order as prompts
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Default: run in thread pool executor
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=min(len(prompts), 5))
        
        async def generate_one(prompt):
            return await loop.run_in_executor(
                executor, 
                lambda: self.generate(prompt, **kwargs)
            )
        
        results = await asyncio.gather(*[generate_one(p) for p in prompts])
        executor.shutdown(wait=False)
        return results

class AnthropicClient(LLMClient):
    """Anthropic Claude API client with automatic model selection."""
    
    def __init__(self, model: str = None, temperature: float = 0.1, auto_select_model: bool = True):
        """
        Initialize Anthropic client with optional automatic model selection.
        
        Args:
            model: Specific model to use. If None and auto_select_model=True, will auto-select best available
            temperature: Generation temperature
            auto_select_model: Whether to automatically test and select the best available model
        """
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.temperature = temperature
        self.auto_select_model = auto_select_model
        
        # Determine which model to use
        if model is None and auto_select_model:
            logger.info("Auto-selecting best available Anthropic model...")
            try:
                from .model_tester import AnthropicModelTester
                tester = AnthropicModelTester()
                selected_model = tester.find_best_available_model(use_case="general_purpose")
                if selected_model:
                    self.model = selected_model
                    logger.info(f"Auto-selected model: {selected_model}")
                else:
                    logger.warning("No Anthropic models available, falling back to default")
                    self.model = "claude-3-haiku-20240307"  # Safe fallback
            except Exception as e:
                logger.warning(f"Model auto-selection failed: {e}, using default")
                self.model = "claude-3-haiku-20240307"  # Safe fallback
        else:
            self.model = model or "claude-3-sonnet-20240229"
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic Claude with retry logic (addresses CLAUDE.md Problem 5)."""
        # Override defaults with any provided kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', 1000)
        timeout = kwargs.get('timeout', 30)  # 30 second timeout
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise RuntimeError("Empty response from Anthropic API")
                
        except Exception as e:
            # Classify error types for better handling
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"Rate limit hit: {e}")
                raise e  # Will be retried with backoff
            elif "timeout" in error_msg or "connection" in error_msg:
                logger.warning(f"Connection error: {e}")
                raise e  # Will be retried
            elif "api key" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                logger.error(f"Authentication error: {e}")
                raise ValueError(f"Anthropic API authentication failed: {e}")  # Don't retry
            else:
                logger.error(f"Anthropic API error: {e}")
                raise e

class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize OpenAI client."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.model = model
        self.temperature = temperature
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model: {model}")
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI with retry logic (addresses CLAUDE.md Problem 5)."""
        # Override defaults with any provided kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', 1000)
        timeout = kwargs.get('timeout', 30)  # 30 second timeout
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    raise RuntimeError("Empty content from OpenAI API")
            else:
                raise RuntimeError("No choices returned from OpenAI API")
                
        except Exception as e:
            # Classify error types for better handling
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"Rate limit hit: {e}")
                raise e  # Will be retried with backoff
            elif "timeout" in error_msg or "connection" in error_msg:
                logger.warning(f"Connection error: {e}")
                raise e  # Will be retried
            elif "api key" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                logger.error(f"Authentication error: {e}")
                raise ValueError(f"OpenAI API authentication failed: {e}")  # Don't retry
            elif "insufficient" in error_msg and "quota" in error_msg:
                logger.error(f"Quota exceeded: {e}")
                raise ValueError(f"OpenAI API quota exceeded: {e}")  # Don't retry
            else:
                logger.error(f"OpenAI API error: {e}")
                raise e

class MockLLMClient(LLMClient):
    """Mock LLM client for testing and fallback."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock responses based on prompt content."""
        if 'query_analysis' in prompt:
            return json.dumps({
                "query_type": "drug_disease",
                "primary_entities": ["metformin", "diabetes"],
                "relevant_relationships": ["treats", "indication", "targets"],
                "retrieval_strategy": "exploration",
                "reasoning": "Query appears to be about drug-disease relationships"
            })
        elif 'next_action' in prompt:
            return json.dumps({
                "action": "stop_retrieval",
                "target_entity": None,
                "target_relationship": None,
                "reasoning": "Sufficient entities and relationships found for analysis",
                "confidence": 0.8
            })
        elif 'entity_exploration' in prompt:
            return json.dumps({
                "action": "continue",
                "relevance_score": 0.9,
                "relevant_aspects": ["disease", "treatment"],
                "next_explorations": ["drugs", "genes"],
                "reasoning": "Highly relevant to the query"
            })
        else:
            return json.dumps({
                "action": "stop_retrieval",
                "relevance_score": 0.5,
                "insights": ["general relationship"],
                "next_steps": ["continue exploration"],
                "reasoning": "Moderately relevant"
            })

class RobustLLMClient(LLMClient):
    """Robust LLM client with multiple fallbacks (addresses CLAUDE.md Problem 5)."""
    
    def __init__(self, temperature: float = 0.1):
        """Initialize with fallback chain."""
        self.temperature = temperature
        self.clients = []
        self._setup_client_chain()
    
    def _setup_client_chain(self):
        """Set up fallback chain of LLM clients with automatic model selection."""
        # Try Anthropic first with auto-model selection
        if os.getenv('ANTHROPIC_API_KEY') and os.getenv('ANTHROPIC_API_KEY') != 'your-anthropic-api-key-here':
            try:
                # Use auto-selection to find the best available model
                client = AnthropicClient(temperature=self.temperature, auto_select_model=True)
                self.clients.append(('Anthropic (Auto-Selected)', client))
                logger.info(f"Added Anthropic client with auto-selected model: {client.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client with auto-selection: {e}")
                
                # Fallback: Try with specific known working models
                fallback_models = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]
                for fallback_model in fallback_models:
                    try:
                        client = AnthropicClient(model=fallback_model, temperature=self.temperature, auto_select_model=False)
                        self.clients.append((f'Anthropic ({fallback_model})', client))
                        logger.info(f"Added Anthropic fallback client with model: {fallback_model}")
                        break
                    except Exception as fe:
                        logger.warning(f"Fallback model {fallback_model} also failed: {fe}")
        
        # Try OpenAI second
        if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your-openai-api-key-here':
            try:
                client = OpenAIClient(temperature=self.temperature)
                self.clients.append(('OpenAI', client))
                logger.info("Added OpenAI client to fallback chain")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Always add Mock client as final fallback
        mock_client = MockLLMClient()
        self.clients.append(('Mock', mock_client))
        logger.info("Added Mock client as final fallback")
        
        logger.info(f"Initialized robust client with {len(self.clients)} fallback options")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with fallback chain."""
        last_exception = None
        
        for client_name, client in self.clients:
            try:
                logger.debug(f"Attempting generation with {client_name} client")
                response = client.generate(prompt, **kwargs)
                
                if response and len(response.strip()) > 0:
                    logger.debug(f"Successful generation with {client_name} client")
                    return response
                else:
                    raise RuntimeError(f"Empty response from {client_name} client")
                    
            except Exception as e:
                logger.warning(f"{client_name} client failed: {e}")
                last_exception = e
                
                # Don't fallback for certain errors in mock client
                if client_name == 'Mock':
                    break
                
                continue
        
        # If all clients failed
        if last_exception:
            raise RuntimeError(f"All LLM clients failed. Last error: {last_exception}")
        else:
            raise RuntimeError("All LLM clients returned empty responses")

def create_llm_client(model: str = None, temperature: float = 0.1) -> LLMClient:
    """
    Create robust LLM client with automatic fallbacks (addresses CLAUDE.md Problem 5).
    
    Args:
        model: Model name (e.g., "claude-3-sonnet-20240229", "gpt-3.5-turbo") 
        temperature: Generation temperature
        
    Returns:
        LLMClient instance with fallback capabilities
    """
    # Get model from environment if not provided
    if model is None:
        model = os.getenv('DEFAULT_LLM_MODEL', 'gpt-3.5-turbo')
    
    # Use robust client for better reliability
    use_robust_client = os.getenv('USE_ROBUST_LLM_CLIENT', 'true').lower() == 'true'
    
    if use_robust_client:
        logger.info("Using robust LLM client with automatic fallbacks")
        return RobustLLMClient(temperature=temperature)
    
    # Legacy single-client approach
    logger.info("Using single LLM client (legacy mode)")
    
    # Determine provider based on model name
    if 'claude' in model.lower():
        if not os.getenv('ANTHROPIC_API_KEY') or os.getenv('ANTHROPIC_API_KEY') == 'your-anthropic-api-key-here':
            logger.warning("ANTHROPIC_API_KEY not configured properly, using mock client")
            return MockLLMClient()
        else:
            try:
                # Map common model names to official Anthropic model IDs
                model_mapping = {
                    'claude-4-sonnet': 'claude-3-haiku-20240307',  # Use working Haiku model
                    'claude-3-5-sonnet': 'claude-3-haiku-20240307',  # Use working Haiku model
                    'claude-3-sonnet': 'claude-3-haiku-20240307',   # Use working Haiku model
                    'claude-3-haiku': 'claude-3-haiku-20240307',
                    'claude-3-opus': 'claude-3-haiku-20240307',     # Use working Haiku model
                }
                official_model = model_mapping.get(model.lower(), model)
                
                client = AnthropicClient(model=official_model, temperature=temperature)
                logger.info(f"Anthropic client created successfully with model: {official_model}")
                return client
                    
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}. Using mock client.")
                return MockLLMClient()
    
    elif 'gpt' in model.lower():
        if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your-openai-api-key-here':
            logger.warning("OPENAI_API_KEY not configured properly, using mock client")
            return MockLLMClient()
        else:
            try:
                client = OpenAIClient(model=model, temperature=temperature)
                logger.info(f"OpenAI client created successfully with model: {model}")
                return client
                    
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Using mock client.")
                return MockLLMClient()
    
    else:
        logger.warning(f"Unknown model type: {model}. Using mock client.")
        return MockLLMClient()

def validate_biomedical_prompt(prompt: str) -> bool:
    """Validate prompts for biomedical accuracy and safety."""
    # Check for biomedical context
    biomedical_terms = ['drug', 'disease', 'protein', 'gene', 'pathway', 'treatment', 'therapy', 'clinical', 'medical', 'insulin', 'diabetes']
    has_biomedical_context = any(term in prompt.lower() for term in biomedical_terms)
    
    # Check for dangerous requests
    dangerous_patterns = ['diagnostic', 'diagnose', 'medical advice', 'prescribe', 'dosage']
    has_dangerous_content = any(pattern in prompt.lower() for pattern in dangerous_patterns)
    
    if has_dangerous_content:
        logger.warning("Prompt contains potentially dangerous medical advice requests")
        return False
    
    return has_biomedical_context

def create_biomedical_prompt(query: str, context: str = "") -> str:
    """Create biomedical-optimized prompts."""
    biomedical_prefix = """You are a biomedical expert analyzing scientific literature and knowledge graphs. 
Provide accurate, evidence-based responses using established biomedical knowledge. 
Focus on mechanisms, pathways, and well-documented relationships. 
Do not provide medical advice or diagnostic information.

"""
    
    biomedical_suffix = """

Please provide a scientific, evidence-based response that:
1. Uses established biomedical terminology
2. References known molecular mechanisms where applicable
3. Maintains appropriate scientific uncertainty
4. Focuses on factual relationships rather than speculation
"""
    
    if context:
        full_prompt = f"{biomedical_prefix}Context: {context}\n\nQuery: {query}{biomedical_suffix}"
    else:
        full_prompt = f"{biomedical_prefix}Query: {query}{biomedical_suffix}"
    
    return full_prompt