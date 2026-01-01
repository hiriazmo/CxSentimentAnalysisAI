"""
LangGraph State Schema
Defines the state that flows through the graph
"""

from typing import TypedDict, Optional, Dict, Any, List
from datetime import datetime

class ReviewState(TypedDict):
    """
    State schema for review processing graph
    All stages add to this state as it flows through the graph
    """
    
    # Input data
    review: Dict[str, Any]
    review_id: str
    review_text: str
    rating: int
    
    # Stage 1: Classification outputs
    llm1_result: Optional[Dict[str, Any]]
    llm2_result: Optional[Dict[str, Any]]
    manager_result: Optional[Dict[str, Any]]
    
    # Stage 1: Extracted fields for easy access
    classification_type: Optional[str]
    department: Optional[str]
    priority: Optional[str]
    user_type: Optional[str]
    emotion: Optional[str]
    
    # Stage 2: Sentiment outputs
    best_sentiment_result: Optional[Dict[str, Any]]
    alt_sentiment_result: Optional[Dict[str, Any]]
    sentiment_layer_result: Optional[Dict[str, Any]]
    
    # Stage 2: Extracted fields
    sentiment: Optional[str]  # POSITIVE, NEGATIVE, NEUTRAL
    sentiment_confidence: Optional[float]
    sentiment_agreement: Optional[bool]
    
    # Stage 3: Finalization outputs
    final_result: Optional[Dict[str, Any]]
    
    # Stage 3: Extracted fields
    final_sentiment: Optional[str]
    final_confidence: Optional[float]
    reasoning: Optional[str]
    action_recommendation: Optional[str]
    conflicts_found: Optional[str]
    validation_notes: Optional[str]
    
    # Routing decisions
    needs_human_review: bool
    route_to: Optional[str]  # 'human_review', 'complete', 'batch_analysis'
    
    # Processing metadata
    stage1_completed: bool
    stage2_completed: bool
    stage3_completed: bool
    processing_started_at: Optional[str]
    processing_completed_at: Optional[str]
    
    # Timing information
    stage1_time: Optional[float]
    stage2_time: Optional[float]
    stage3_time: Optional[float]
    total_time: Optional[float]
    
    # Error handling
    errors: List[str]
    retry_count: int
    
    # Database sync status
    db_stage1_saved: bool
    db_stage2_saved: bool
    db_stage3_saved: bool


class BatchState(TypedDict):
    """
    State for batch analysis (Stage 4)
    Aggregates results from multiple reviews
    """
    
    # Input
    all_reviews: List[ReviewState]
    total_count: int
    
    # Aggregated metrics
    sentiment_distribution: Optional[Dict[str, int]]
    priority_distribution: Optional[Dict[str, int]]
    department_distribution: Optional[Dict[str, int]]
    emotion_distribution: Optional[Dict[str, int]]
    
    # Analysis outputs
    critical_issues: Optional[List[Dict[str, Any]]]
    quick_wins: Optional[List[Dict[str, Any]]]
    churn_risk: Optional[float]
    model_agreement_rate: Optional[float]
    
    # Recommendations
    recommendations: Optional[List[str]]
    
    # Processing metadata
    batch_started_at: Optional[str]
    batch_completed_at: Optional[str]
    batch_processing_time: Optional[float]


def create_initial_state(review: Dict[str, Any]) -> ReviewState:
    """
    Create initial state for a review
    """
    return ReviewState(
        # Input
        review=review,
        review_id=review.get('review_id', 'unknown'),
        review_text=review.get('review_text', ''),
        rating=review.get('rating', 3),
        
        # Stage 1
        llm1_result=None,
        llm2_result=None,
        manager_result=None,
        classification_type=None,
        department=None,
        priority=None,
        user_type=None,
        emotion=None,
        
        # Stage 2
        best_sentiment_result=None,
        alt_sentiment_result=None,
        sentiment_layer_result=None,
        sentiment=None,
        sentiment_confidence=None,
        sentiment_agreement=None,
        
        # Stage 3
        final_result=None,
        final_sentiment=None,
        final_confidence=None,
        reasoning=None,
        action_recommendation=None,
        conflicts_found=None,
        validation_notes=None,
        
        # Routing
        needs_human_review=False,
        route_to=None,
        
        # Processing metadata
        stage1_completed=False,
        stage2_completed=False,
        stage3_completed=False,
        processing_started_at=datetime.now().isoformat(),
        processing_completed_at=None,
        
        # Timing
        stage1_time=None,
        stage2_time=None,
        stage3_time=None,
        total_time=None,
        
        # Errors
        errors=[],
        retry_count=0,
        
        # Database
        db_stage1_saved=False,
        db_stage2_saved=False,
        db_stage3_saved=False
    )


def create_batch_state(reviews: List[ReviewState]) -> BatchState:
    """
    Create batch state from processed reviews
    """
    return BatchState(
        all_reviews=reviews,
        total_count=len(reviews),
        sentiment_distribution=None,
        priority_distribution=None,
        department_distribution=None,
        emotion_distribution=None,
        critical_issues=None,
        quick_wins=None,
        churn_risk=None,
        model_agreement_rate=None,
        recommendations=None,
        batch_started_at=datetime.now().isoformat(),
        batch_completed_at=None,
        batch_processing_time=None
    )


if __name__ == "__main__":
    # Test state creation
    print("\n" + "="*60)
    print("🧪 TESTING LANGGRAPH STATE")
    print("="*60)
    
    test_review = {
        'review_id': 'test_001',
        'review_text': 'App crashes!',
        'rating': 1
    }
    
    state = create_initial_state(test_review)
    print(f"\n✅ Initial state created for: {state['review_id']}")
    print(f"   Review text: {state['review_text']}")
    print(f"   Stage 1 completed: {state['stage1_completed']}")
    
    print("\n✅ State schema test complete!")
