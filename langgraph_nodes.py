"""
LangGraph Nodes - FINAL WORKING VERSION
Uses chat_completion() API format + Lazy loading + Fixed alt sentiment
"""

import os
import json
import time
from typing import Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import InferenceClient
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

from langgraph_state import ReviewState, BatchState
from database_enhanced import EnhancedDatabase

# FIXED: Don't initialize client at module import
_hf_client = None

def get_hf_client():
    """Get or initialize HuggingFace client (lazy loading)"""
    global _hf_client
    
    if _hf_client is not None:
        return _hf_client
    
    # Try to get token from environment
    HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    
    if not HF_TOKEN or HF_TOKEN.strip() == "":
        return None
    
    # Initialize client with token
    print(f"✅ Initializing HF client with token: {HF_TOKEN[:10]}...")
    _hf_client = InferenceClient(token=HF_TOKEN)
    return _hf_client


# Initialize sentiment models (singleton)
_sentiment_models_loaded = False
_best_tokenizer = None
_best_model = None
_alt_tokenizer = None
_alt_model = None

def load_sentiment_models():
    """Load sentiment models once (singleton pattern)"""
    global _sentiment_models_loaded, _best_tokenizer, _best_model, _alt_tokenizer, _alt_model
    
    if _sentiment_models_loaded:
        return
    
    print("   📦 Loading Twitter-BERT models (one-time)...")
    
    # Best Model
    _best_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    _best_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    _best_model.eval()
    
    # Alternate Model - FIXED: Load with low_cpu_mem_usage to avoid meta tensors
    _alt_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    _alt_model = AutoModelForSequenceClassification.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis",
        low_cpu_mem_usage=False  # FIXED: Don't use meta device
    )
    _alt_model.eval()
    
    _sentiment_models_loaded = True
    print("   ✅ Sentiment models loaded!")


# ============================================================================
# STAGE 1: CLASSIFICATION NODE
# ============================================================================

def llm1_classify(review: Dict[str, Any]) -> Dict[str, Any]:
    """LLM1: Type, Department, Priority classification"""
    
    hf_client = get_hf_client()
    
    if hf_client is None:
        return {
            'type': 'unknown',
            'department': 'unknown',
            'priority': 'medium',
            'confidence': 0.0,
            'reasoning': 'HuggingFace API key not set',
            'model': 'Qwen/Qwen2.5-72B-Instruct'
        }
    
    review_text = review.get('review_text', '')
    rating = review.get('rating', 3)
    
    # FIXED: Use chat format with system + user messages
    system_prompt = """You are an expert at classifying customer reviews for theme park and attraction apps.

Classify reviews across these dimensions:

1. TYPE: complaint, praise, suggestion, question, or bug_report
2. DEPARTMENT: engineering, ux, support, or business
3. PRIORITY: critical, high, medium, or low
4. CONFIDENCE: 0.0-1.0
5. REASONING: Brief one-sentence explanation

Respond ONLY in valid JSON format:
{
  "type": "complaint/praise/suggestion/question/bug_report",
  "department": "engineering/ux/support/business",
  "priority": "critical/high/medium/low",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""

    user_prompt = f"""REVIEW:
Rating: {rating}/5
Text: {review_text}

Classify this review:"""

    try:
        print(f"   🔍 Calling Qwen API...")
        
        # FIXED: Use chat_completion instead of text_generation
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=200,
            temperature=0.1
        )
        
        # Extract content from chat response
        content = response.choices[0].message.content
        print(f"   ✅ Got response ({len(content)} chars)")
        
        # Clean and parse JSON
        content_clean = content.strip()
        if content_clean.startswith('```'):
            content_clean = content_clean.split('```')[1]
            if content_clean.startswith('json'):
                content_clean = content_clean[4:]
        content_clean = content_clean.strip()
        
        result = json.loads(content_clean)
        result['model'] = 'Qwen/Qwen2.5-72B-Instruct'
        
        print(f"   ✅ Parsed: {result['type']} → {result['department']}")
        return result
        
    except Exception as e:
        print(f"❌ LLM1 ERROR: {type(e).__name__}: {str(e)}")
        
        return {
            'type': 'unknown',
            'department': 'unknown',
            'priority': 'medium',
            'confidence': 0.0,
            'reasoning': f'API Error: {str(e)}',
            'model': 'Qwen/Qwen2.5-72B-Instruct'
        }


def llm2_analyze(review: Dict[str, Any]) -> Dict[str, Any]:
    """LLM2: User type, Emotion, Context analysis"""
    
    hf_client = get_hf_client()
    
    if hf_client is None:
        return {
            'user_type': 'unknown',
            'emotion': 'unknown',
            'context': 'unknown',
            'confidence': 0.0,
            'reasoning': 'HuggingFace API key not set',
            'model': 'mistralai/Mistral-7B-Instruct-v0.3'
        }
    
    review_text = review.get('review_text', '')
    rating = review.get('rating', 3)
    
    # FIXED: Use chat format
    system_prompt = """You are an expert at understanding customer psychology and emotional context.

Analyze reviews for:
1. USER_TYPE: new_user, regular_user, power_user, or churning_user
2. EMOTION: anger, frustration, joy, satisfaction, disappointment, or confusion
3. CONTEXT: Brief context (1-2 words)
4. CONFIDENCE: 0.0-1.0
5. REASONING: Brief explanation

Respond ONLY in valid JSON format:
{
  "user_type": "new_user/regular_user/power_user/churning_user",
  "emotion": "anger/frustration/joy/satisfaction/disappointment/confusion",
  "context": "brief context",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""

    user_prompt = f"""REVIEW:
Rating: {rating}/5
Text: {review_text}

Analyze this review:"""

    try:
        print(f"   🔍 Calling Mistral API...")
        
        # FIXED: Use chat_completion
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=200,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        print(f"   ✅ Got response ({len(content)} chars)")
        
        # Clean and parse JSON
        content_clean = content.strip()
        if content_clean.startswith('```'):
            content_clean = content_clean.split('```')[1]
            if content_clean.startswith('json'):
                content_clean = content_clean[4:]
        content_clean = content_clean.strip()
        
        result = json.loads(content_clean)
        result['model'] = 'mistralai/Mistral-7B-Instruct-v0.3'
        
        print(f"   ✅ Parsed: {result['user_type']}, {result['emotion']}")
        return result
        
    except Exception as e:
        print(f"❌ LLM2 ERROR: {type(e).__name__}: {str(e)}")
        
        return {
            'user_type': 'unknown',
            'emotion': 'unknown',
            'context': 'unknown',
            'confidence': 0.0,
            'reasoning': f'API Error: {str(e)}',
            'model': 'mistralai/Mistral-7B-Instruct-v0.3'
        }


def manager_synthesize(llm1_result: Dict, llm2_result: Dict, review: Dict) -> Dict[str, Any]:
    """Manager: Synthesize LLM1 and LLM2 results"""
    
    hf_client = get_hf_client()
    
    if hf_client is None:
        return {
            'final_type': llm1_result.get('type', 'unknown'),
            'final_department': llm1_result.get('department', 'unknown'),
            'final_priority': llm1_result.get('priority', 'medium'),
            'synthesis_reasoning': 'HuggingFace API key not set',
            'model': 'meta-llama/Llama-3.3-70B-Instruct'
        }
    
    review_text = review.get('review_text', '')
    rating = review.get('rating', 3)
    
    # FIXED: Use chat format
    system_prompt = """You are a synthesis manager evaluating two AI analyses.

Your task:
1. Validate both analyses
2. Resolve conflicts
3. Make final classification decision
4. Provide synthesis reasoning

Respond ONLY in valid JSON format:
{
  "final_type": "from llm1 or adjusted",
  "final_department": "from llm1 or adjusted",
  "final_priority": "from llm1 or adjusted",
  "synthesis_reasoning": "brief explanation"
}"""

    user_prompt = f"""REVIEW:
Rating: {rating}/5
Text: {review_text}

LLM1 ANALYSIS (Type/Dept/Priority):
{json.dumps(llm1_result, indent=2)}

LLM2 ANALYSIS (User/Emotion/Context):
{json.dumps(llm2_result, indent=2)}

Synthesize these analyses:"""

    try:
        print(f"   🔍 Calling Llama Manager API...")
        
        # FIXED: Use chat_completion
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="meta-llama/Llama-3.3-70B-Instruct",
            max_tokens=200,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        print(f"   ✅ Got response ({len(content)} chars)")
        
        content_clean = content.strip()
        if content_clean.startswith('```'):
            content_clean = content_clean.split('```')[1]
            if content_clean.startswith('json'):
                content_clean = content_clean[4:]
        content_clean = content_clean.strip()
        
        result = json.loads(content_clean)
        result['model'] = 'meta-llama/Llama-3.3-70B-Instruct'
        
        print(f"   ✅ Manager decision: {result['final_type']} → {result['final_department']}")
        return result
        
    except Exception as e:
        print(f"❌ MANAGER ERROR: {type(e).__name__}: {str(e)}")
        
        return {
            'final_type': llm1_result.get('type', 'unknown'),
            'final_department': llm1_result.get('department', 'unknown'),
            'final_priority': llm1_result.get('priority', 'medium'),
            'synthesis_reasoning': f'Manager error: {str(e)}',
            'model': 'meta-llama/Llama-3.3-70B-Instruct'
        }


def stage1_classification_node(state: ReviewState) -> Dict[str, Any]:
    """Stage 1 Node: Classification with PARALLEL execution"""
    print(f"\n      📝 Review ID: {state['review_id']}")
    print(f"      ⏳ STAGE 1: Classification (Parallel LLM1 + LLM2)...")
    
    start_time = time.time()
    review_dict = dict(state)
    
    # PARALLEL EXECUTION
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_llm1 = executor.submit(llm1_classify, review_dict)
        future_llm2 = executor.submit(llm2_analyze, review_dict)
        
        llm1_result = future_llm1.result()
        llm2_result = future_llm2.result()
    
    print(f"         ✅ LLM1: {llm1_result['type']} → {llm1_result['department']} (Priority: {llm1_result['priority']})")
    print(f"         ✅ LLM2: {llm2_result['user_type']}, {llm2_result['emotion']}")
    
    # Manager synthesizes
    print(f"         🤖 Manager synthesizing...")
    manager_result = manager_synthesize(llm1_result, llm2_result, review_dict)
    
    stage1_time = time.time() - start_time
    print(f"      ✅ Stage 1 complete ({stage1_time:.2f}s)")
    
    return {
        "llm1_result": llm1_result,
        "llm2_result": llm2_result,
        "manager_result": manager_result,
        "classification_type": manager_result['final_type'],
        "department": manager_result['final_department'],
        "priority": manager_result['final_priority'],
        "user_type": llm2_result['user_type'],
        "emotion": llm2_result['emotion'],
        "context": llm2_result.get('context', ''),
        "stage1_completed": True,
        "stage1_time": stage1_time,
        "errors": state.get('errors', [])
    }


# ============================================================================
# STAGE 2: SENTIMENT ANALYSIS
# ============================================================================

def analyze_best_sentiment(text: str) -> Dict[str, Any]:
    """Best Model: Twitter-BERT"""
    load_sentiment_models()
    
    try:
        inputs = _best_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = _best_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        
        return {
            'sentiment': label_map[prediction],
            'confidence': confidence,
            'prob_negative': probs[0][0].item(),
            'prob_neutral': probs[0][1].item(),
            'prob_positive': probs[0][2].item(),
            'model': 'twitter-roberta-base-sentiment-latest'
        }
    except Exception as e:
        print(f"❌ Best sentiment ERROR: {e}")
        return {
            'sentiment': 'NEUTRAL',
            'confidence': 0.0,
            'prob_negative': 0.33,
            'prob_neutral': 0.34,
            'prob_positive': 0.33,
            'model': 'error',
            'error': str(e)
        }


def analyze_alt_sentiment(text: str) -> Dict[str, Any]:
    """Alternate Model: BERTweet - FIXED"""
    load_sentiment_models()
    
    try:
        inputs = _alt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = _alt_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        
        return {
            'sentiment': label_map[prediction],
            'confidence': confidence,
            'prob_negative': probs[0][0].item(),
            'prob_neutral': probs[0][1].item(),
            'prob_positive': probs[0][2].item(),
            'model': 'bertweet-base-sentiment-analysis'
        }
    except Exception as e:
        print(f"❌ Alt sentiment ERROR: {e}")
        return {
            'sentiment': 'NEUTRAL',
            'confidence': 0.0,
            'prob_negative': 0.33,
            'prob_neutral': 0.34,
            'prob_positive': 0.33,
            'model': 'error',
            'error': str(e)
        }


def sentiment_layer(best_result: Dict, alt_result: Dict) -> Dict[str, Any]:
    """Sentiment Layer: Combine with confidence weighting"""
    best_sentiment = best_result.get('sentiment')
    best_confidence = best_result.get('confidence', 0.0)
    
    alt_sentiment = alt_result.get('sentiment')
    alt_confidence = alt_result.get('confidence', 0.0)
    
    agreement = (best_sentiment == alt_sentiment)
    
    if agreement:
        final_sentiment = best_sentiment
        combined_confidence = max(best_confidence, alt_confidence)
        agreement_strength = "STRONG"
    else:
        if best_confidence > alt_confidence:
            final_sentiment = best_sentiment
            combined_confidence = best_confidence
        else:
            final_sentiment = alt_sentiment
            combined_confidence = alt_confidence
        agreement_strength = "WEAK"
    
    return {
        'layer_sentiment': final_sentiment,
        'combined_confidence': combined_confidence,
        'agreement': agreement,
        'agreement_strength': agreement_strength
    }


def stage2_sentiment_node(state: ReviewState) -> Dict[str, Any]:
    """Stage 2 Node: Sentiment with PARALLEL execution"""
    print(f"\n      ⏳ STAGE 2: Sentiment Analysis (Parallel Best + Alternate)...")
    
    start_time = time.time()
    review_text = state['review_text']
    
    # PARALLEL EXECUTION
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_best = executor.submit(analyze_best_sentiment, review_text)
        future_alt = executor.submit(analyze_alt_sentiment, review_text)
        
        best_result = future_best.result()
        alt_result = future_alt.result()
    
    print(f"         ✅ Best: {best_result['sentiment']} ({best_result['confidence']:.3f})")
    print(f"         ✅ Alt: {alt_result['sentiment']} ({alt_result['confidence']:.3f})")
    
    # Sentiment Layer combines results
    layer_result = sentiment_layer(best_result, alt_result)
    
    agreement_icon = "✅" if layer_result['agreement'] else "⚠️ "
    print(f"         {agreement_icon} Final: {layer_result['layer_sentiment']} (agreement: {layer_result['agreement']})")
    
    stage2_time = time.time() - start_time
    print(f"      ✅ Stage 2 complete ({stage2_time:.2f}s)")
    
    return {
        "best_sentiment_result": best_result,
        "alt_sentiment_result": alt_result,
        "sentiment_layer_result": layer_result,
        "sentiment": layer_result['layer_sentiment'],
        "sentiment_confidence": layer_result['combined_confidence'],
        "sentiment_agreement": layer_result['agreement'],
        "stage2_completed": True,
        "stage2_time": stage2_time,
        "errors": state.get('errors', [])
    }


# ============================================================================
# STAGE 3: FINALIZATION NODE
# ============================================================================

def stage3_finalization_node(state: ReviewState) -> Dict[str, Any]:
    """Stage 3 Node: Final synthesis with LLM3"""
    print(f"\n      ⏳ STAGE 3: Finalization (LLM3)...")
    
    start_time = time.time()
    
    hf_client = get_hf_client()
    
    if hf_client is None:
        result = {
            'final_sentiment': state.get('sentiment', 'NEUTRAL'),
            'confidence': state.get('sentiment_confidence', 0.0),
            'reasoning': 'Stage 3 skipped - HuggingFace API key not set',
            'validation_notes': 'API key missing',
            'conflicts_found': 'none',
            'action_recommendation': f"Route to {state.get('department', 'support')}",
            'needs_human_review': True,
            'model': 'meta-llama/Llama-3.1-70B-Instruct'
        }
        
        stage3_time = 0.00
        print(f"         ✅ Final: {result['final_sentiment']} ({result.get('confidence', 0):.3f})")
        print(f"         📋 Needs Review: {result.get('needs_human_review', False)}")
        print(f"      ✅ Stage 3 complete ({stage3_time:.2f}s)")
        
        return {
            "final_result": result,
            "final_sentiment": result['final_sentiment'],
            "final_confidence": result['confidence'],
            "reasoning": result['reasoning'],
            "action_recommendation": result['action_recommendation'],
            "conflicts_found": result['conflicts_found'],
            "validation_notes": result['validation_notes'],
            "needs_human_review": result['needs_human_review'],
            "stage3_completed": True,
            "stage3_time": stage3_time,
            "total_time": state.get('stage1_time', 0) + state.get('stage2_time', 0),
            "processing_completed_at": datetime.now().isoformat(),
            "errors": state.get('errors', [])
        }
    
    review_text = state['review_text']
    rating = state['rating']
    
    # FIXED: Use chat format
    system_prompt = """You are a final decision-making AI analyzing customer feedback for a theme park/attraction app.

Your task:
1. Review all data from previous stages
2. Make FINAL sentiment decision
3. Provide comprehensive reasoning
4. Generate action recommendation
5. Flag if human review needed

Respond ONLY in valid JSON format:
{
  "final_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
  "confidence": 0.0-1.0,
  "reasoning": "Comprehensive explanation",
  "validation_notes": "Does classification match sentiment?",
  "conflicts_found": "any conflicts or 'none'",
  "action_recommendation": "Specific action",
  "needs_human_review": true/false
}"""

    user_prompt = f"""REVIEW DATA:
Rating: {rating}/5
Text: {review_text}

STAGE 1 CLASSIFICATION:
- Type: {state.get('classification_type')}
- Department: {state.get('department')}
- Priority: {state.get('priority')}
- User Type: {state.get('user_type')}
- Emotion: {state.get('emotion')}

STAGE 2 SENTIMENT:
- Best: {state['best_sentiment_result'].get('sentiment')} ({state['best_sentiment_result'].get('confidence'):.2f})
- Alternate: {state['alt_sentiment_result'].get('sentiment')} ({state['alt_sentiment_result'].get('confidence'):.2f})
- Agreement: {state.get('sentiment_agreement')}

Make your final decision:"""

    try:
        print(f"   🔍 Calling Llama 70B API...")
        
        # FIXED: Use chat_completion
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="meta-llama/Llama-3.1-70B-Instruct",
            max_tokens=400,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        print(f"   ✅ Got response ({len(content)} chars)")
        
        content_clean = content.strip()
        if content_clean.startswith('```'):
            content_clean = content_clean.split('```')[1]
            if content_clean.startswith('json'):
                content_clean = content_clean[4:]
        content_clean = content_clean.strip()
        
        result = json.loads(content_clean)
        result['model'] = 'meta-llama/Llama-3.1-70B-Instruct'
        
    except Exception as e:
        print(f"❌ STAGE 3 ERROR: {type(e).__name__}: {str(e)}")
        
        result = {
            'final_sentiment': state.get('sentiment', 'NEUTRAL'),
            'confidence': state.get('sentiment_confidence', 0.5),
            'reasoning': f'Error in LLM3: {str(e)}',
            'validation_notes': 'Error',
            'conflicts_found': 'error',
            'action_recommendation': f"Route to {state.get('department')}",
            'needs_human_review': True,
            'model': 'meta-llama/Llama-3.1-70B-Instruct'
        }
    
    stage3_time = time.time() - start_time
    
    print(f"         ✅ Final: {result['final_sentiment']} ({result.get('confidence', 0):.3f})")
    print(f"         📋 Needs Review: {result.get('needs_human_review', False)}")
    print(f"      ✅ Stage 3 complete ({stage3_time:.2f}s)")
    
    # Calculate total time
    total_time = state.get('stage1_time', 0) + state.get('stage2_time', 0) + stage3_time
    
    return {
        "final_result": result,
        "final_sentiment": result['final_sentiment'],
        "final_confidence": result['confidence'],
        "reasoning": result['reasoning'],
        "action_recommendation": result['action_recommendation'],
        "conflicts_found": result['conflicts_found'],
        "validation_notes": result['validation_notes'],
        "needs_human_review": result['needs_human_review'],
        "stage3_completed": True,
        "stage3_time": stage3_time,
        "total_time": total_time,
        "processing_completed_at": datetime.now().isoformat(),
        "errors": state.get('errors', [])
    }


if __name__ == "__main__":
    print("\n✅ LangGraph nodes module loaded!")
    print("   Nodes available:")
    print("   - stage1_classification_node (parallel LLM1+LLM2)")
    print("   - stage2_sentiment_node (parallel Best+Alt)")
    print("   - stage3_finalization_node (LLM3)")