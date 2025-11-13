#!/usr/bin/env python3
"""
Anthropic Model Tester and Auto-Selection System

This module automatically tests available Anthropic models and selects the best working one.
It handles model availability changes and provides robust fallback mechanisms.

FEATURES:
- Tests multiple Anthropic model variants automatically
- Caches working models to avoid repeated API calls
- Provides detailed logging of model availability
- Integrates seamlessly with existing LLM client system
- Handles rate limits and API errors gracefully

USAGE:
    from src.model_tester import AnthropicModelTester
    
    tester = AnthropicModelTester()
    best_model = tester.find_best_available_model()
    print(f"Using model: {best_model}")
"""

import os
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelTestResult:
    """Result of testing a specific model."""
    model_id: str
    available: bool
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    test_timestamp: Optional[float] = None
    response_quality_score: Optional[float] = None

@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a model."""
    model_id: str
    max_tokens: int
    cost_per_1k_tokens: float
    speed_tier: str  # 'fast', 'medium', 'slow'
    capability_tier: str  # 'basic', 'advanced', 'premium'
    recommended_use_cases: List[str]

class AnthropicModelTester:
    """
    Comprehensive Anthropic model tester and selector.
    
    Tests all available Anthropic models and selects the best working one
    based on availability, performance, and use case requirements.
    """
    
    # Comprehensive list of Anthropic models to test (ordered by preference)
    ANTHROPIC_MODELS = [
        # Claude 3.5 Sonnet (Latest and most capable)
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        
        # Claude 3 Sonnet (Balanced performance)
        "claude-3-sonnet-20240229",
        
        # Claude 3 Haiku (Fast and efficient)
        "claude-3-haiku-20240307",
        
        # Claude 3 Opus (Most capable but slower)
        "claude-3-opus-20240229",
        
        # Legacy models (fallback)
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]
    
    # Model capabilities database
    MODEL_CAPABILITIES = {
        "claude-3-5-sonnet-20241022": ModelCapabilities(
            model_id="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            cost_per_1k_tokens=3.00,
            speed_tier="medium",
            capability_tier="premium",
            recommended_use_cases=["complex_reasoning", "code_generation", "analysis"]
        ),
        "claude-3-5-sonnet-20240620": ModelCapabilities(
            model_id="claude-3-5-sonnet-20240620",
            max_tokens=8192,
            cost_per_1k_tokens=3.00,
            speed_tier="medium",
            capability_tier="premium",
            recommended_use_cases=["complex_reasoning", "code_generation", "analysis"]
        ),
        "claude-3-sonnet-20240229": ModelCapabilities(
            model_id="claude-3-sonnet-20240229",
            max_tokens=4096,
            cost_per_1k_tokens=3.00,
            speed_tier="medium",
            capability_tier="advanced",
            recommended_use_cases=["general_purpose", "reasoning", "writing"]
        ),
        "claude-3-haiku-20240307": ModelCapabilities(
            model_id="claude-3-haiku-20240307",
            max_tokens=4096,
            cost_per_1k_tokens=0.25,
            speed_tier="fast",
            capability_tier="basic",
            recommended_use_cases=["quick_responses", "simple_tasks", "high_volume"]
        ),
        "claude-3-opus-20240229": ModelCapabilities(
            model_id="claude-3-opus-20240229",
            max_tokens=4096,
            cost_per_1k_tokens=15.00,
            speed_tier="slow",
            capability_tier="premium",
            recommended_use_cases=["complex_analysis", "research", "high_quality_output"]
        ),
    }
    
    def __init__(self, cache_duration_hours: float = 24):
        """
        Initialize the model tester.
        
        Args:
            cache_duration_hours: How long to cache model test results
        """
        self.cache_duration_hours = cache_duration_hours
        self.cache_file = Path("data/cache/model_test_cache.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Anthropic client if possible
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        
        if self.api_key and self.api_key != 'your-anthropic-api-key-here':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized for model testing")
            except ImportError:
                logger.warning("anthropic package not installed - model testing disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        else:
            logger.warning("ANTHROPIC_API_KEY not configured - model testing disabled")
    
    def load_cache(self) -> Dict[str, ModelTestResult]:
        """Load cached model test results."""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Convert back to ModelTestResult objects
            cached_results = {}
            current_time = time.time()
            
            for model_id, result_dict in cache_data.items():
                result = ModelTestResult(**result_dict)
                
                # Check if cache is still valid
                if result.test_timestamp:
                    age_hours = (current_time - result.test_timestamp) / 3600
                    if age_hours < self.cache_duration_hours:
                        cached_results[model_id] = result
                    else:
                        logger.debug(f"Cache expired for model {model_id} (age: {age_hours:.1f}h)")
                        
            logger.info(f"Loaded {len(cached_results)} cached model test results")
            return cached_results
            
        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")
            return {}
    
    def save_cache(self, results: Dict[str, ModelTestResult]) -> None:
        """Save model test results to cache."""
        try:
            cache_data = {
                model_id: asdict(result) 
                for model_id, result in results.items()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.debug(f"Saved {len(results)} model test results to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    def test_model(self, model_id: str) -> ModelTestResult:
        """
        Test a specific model for availability and basic functionality.
        
        Args:
            model_id: The Anthropic model ID to test
            
        Returns:
            ModelTestResult with test outcome
        """
        logger.info(f"Testing model: {model_id}")
        
        if not self.client:
            return ModelTestResult(
                model_id=model_id,
                available=False,
                error_message="No Anthropic client available",
                test_timestamp=time.time()
            )
        
        # Simple test prompt
        test_prompt = "Hello! Please respond with exactly: 'Model test successful'"
        expected_keywords = ["model", "test", "successful"]
        
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=model_id,
                max_tokens=100,
                temperature=0.1,
                messages=[{"role": "user", "content": test_prompt}]
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            if response.content and len(response.content) > 0:
                response_text = response.content[0].text.lower()
                
                # Calculate quality score based on how well it followed instructions
                quality_score = sum(1 for keyword in expected_keywords if keyword in response_text) / len(expected_keywords)
                
                logger.info(f"[OK] Model {model_id} test successful ({response_time_ms:.0f}ms, quality: {quality_score:.2f})")
                
                return ModelTestResult(
                    model_id=model_id,
                    available=True,
                    response_time_ms=response_time_ms,
                    test_timestamp=time.time(),
                    response_quality_score=quality_score
                )
            else:
                logger.warning(f"[FAIL] Model {model_id} returned empty response")
                return ModelTestResult(
                    model_id=model_id,
                    available=False,
                    error_message="Empty response",
                    test_timestamp=time.time()
                )
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[FAIL] Model {model_id} test failed: {error_msg}")
            
            # Check if it's a model not found error
            if "404" in error_msg or "not found" in error_msg.lower():
                error_type = "Model not found"
            elif "403" in error_msg or "unauthorized" in error_msg.lower():
                error_type = "Access denied"
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                error_type = "Rate limited"
            else:
                error_type = "API error"
            
            return ModelTestResult(
                model_id=model_id,
                available=False,
                error_message=f"{error_type}: {error_msg}",
                test_timestamp=time.time()
            )
    
    def test_all_models(self, use_cache: bool = True) -> Dict[str, ModelTestResult]:
        """
        Test all available Anthropic models.
        
        Args:
            use_cache: Whether to use cached results for recently tested models
            
        Returns:
            Dictionary mapping model IDs to test results
        """
        logger.info("Starting comprehensive Anthropic model testing...")
        
        # Load cached results if requested
        results = self.load_cache() if use_cache else {}
        
        models_to_test = []
        for model_id in self.ANTHROPIC_MODELS:
            if model_id not in results:
                models_to_test.append(model_id)
            else:
                logger.debug(f"Using cached result for {model_id}")
        
        logger.info(f"Testing {len(models_to_test)} models (using {len(results)} cached results)")
        
        # Test models that aren't cached
        for i, model_id in enumerate(models_to_test, 1):
            logger.info(f"Testing model {i}/{len(models_to_test)}: {model_id}")
            
            try:
                result = self.test_model(model_id)
                results[model_id] = result
                
                # Small delay to avoid rate limiting
                if i < len(models_to_test):
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Model testing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error testing {model_id}: {e}")
                results[model_id] = ModelTestResult(
                    model_id=model_id,
                    available=False,
                    error_message=f"Unexpected error: {e}",
                    test_timestamp=time.time()
                )
        
        # Save updated cache
        if use_cache:
            self.save_cache(results)
        
        return results
    
    def find_best_available_model(self, 
                                use_case: str = "general_purpose",
                                prefer_speed: bool = False,
                                prefer_capability: bool = True) -> Optional[str]:
        """
        Find the best available Anthropic model based on criteria.
        
        Args:
            use_case: Intended use case for the model
            prefer_speed: Whether to prioritize faster models
            prefer_capability: Whether to prioritize more capable models
            
        Returns:
            Model ID of the best available model, or None if none available
        """
        logger.info(f"Finding best available model for use case: {use_case}")
        
        # Test all models
        results = self.test_all_models()
        
        # Filter to available models
        available_models = {
            model_id: result for model_id, result in results.items()
            if result.available
        }
        
        if not available_models:
            logger.error("No Anthropic models are currently available!")
            return None
        
        logger.info(f"Found {len(available_models)} available models")
        
        # Score models based on criteria
        model_scores = {}
        
        for model_id, result in available_models.items():
            score = 0
            
            # Base score from model order preference (earlier = higher score)
            try:
                position_score = (len(self.ANTHROPIC_MODELS) - self.ANTHROPIC_MODELS.index(model_id)) * 10
                score += position_score
            except ValueError:
                pass  # Model not in our preference list
            
            # Capability score
            capabilities = self.MODEL_CAPABILITIES.get(model_id)
            if capabilities:
                if use_case in capabilities.recommended_use_cases:
                    score += 20
                
                if prefer_capability:
                    capability_scores = {"premium": 15, "advanced": 10, "basic": 5}
                    score += capability_scores.get(capabilities.capability_tier, 0)
                
                if prefer_speed:
                    speed_scores = {"fast": 15, "medium": 10, "slow": 5}
                    score += speed_scores.get(capabilities.speed_tier, 0)
            
            # Performance score from test results
            if result.response_time_ms:
                # Faster is better (inverse score)
                time_score = max(0, 10 - (result.response_time_ms / 1000))
                score += time_score
            
            if result.response_quality_score:
                score += result.response_quality_score * 10
            
            model_scores[model_id] = score
        
        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        best_score = model_scores[best_model]
        
        logger.info(f"Selected best model: {best_model} (score: {best_score:.1f})")
        
        # Log the ranking
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info("Model ranking:")
        for i, (model_id, score) in enumerate(sorted_models[:5], 1):
            status = "[OK]" if i == 1 else f"{i}."
            logger.info(f"  {status} {model_id}: {score:.1f}")
        
        return best_model
    
    def get_model_summary(self) -> Dict:
        """Get a summary of model availability and recommendations."""
        results = self.test_all_models()
        
        available_count = sum(1 for r in results.values() if r.available)
        total_count = len(results)
        
        available_models = [
            model_id for model_id, result in results.items()
            if result.available
        ]
        
        unavailable_models = [
            (model_id, result.error_message) 
            for model_id, result in results.items()
            if not result.available
        ]
        
        return {
            "total_models_tested": total_count,
            "available_models": available_count,
            "availability_rate": f"{(available_count/total_count)*100:.1f}%",
            "available_model_ids": available_models,
            "unavailable_models": unavailable_models,
            "recommended_model": self.find_best_available_model(),
            "test_timestamp": time.time()
        }

def test_and_select_model(use_case: str = "general_purpose") -> Optional[str]:
    """
    Convenience function to test and select the best available Anthropic model.
    
    Args:
        use_case: The intended use case for the model
        
    Returns:
        The best available model ID, or None if none available
    """
    tester = AnthropicModelTester()
    return tester.find_best_available_model(use_case=use_case)

if __name__ == "__main__":
    # CLI testing interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Anthropic models and find the best available one")
    parser.add_argument("--use-case", default="general_purpose", help="Intended use case")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache and test all models fresh")
    parser.add_argument("--summary", action="store_true", help="Show detailed summary")
    
    args = parser.parse_args()
    
    tester = AnthropicModelTester()
    
    if args.summary:
        summary = tester.get_model_summary()
        print(json.dumps(summary, indent=2))
    else:
        best_model = tester.find_best_available_model(
            use_case=args.use_case
        )
        if best_model:
            print(f"Best available model: {best_model}")
        else:
            print("No models available!")
