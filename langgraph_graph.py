"""
LangGraph Graph Definition
Defines the review processing workflow with conditional routing
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal

from langgraph_state import ReviewState, BatchState, create_initial_state
from langgraph_nodes import (
    stage1_classification_node,
    stage2_sentiment_node,
    stage3_finalization_node
)
from stage4_batch_analysis import Stage4BatchAnalysis
from database_enhanced import EnhancedDatabase


# ============================================================================
# DATABASE SYNC NODES
# ============================================================================

def save_stage1_to_db_node(state: ReviewState) -> dict:
    """Save Stage 1 results to database"""
    db = EnhancedDatabase()
    db.connect()
    
    try:
        stage1_data = {
            'llm1_type': state['llm1_result'].get('type'),
            'llm1_department': state['llm1_result'].get('department'),
            'llm1_priority': state['llm1_result'].get('priority'),
            'llm1_confidence': state['llm1_result'].get('confidence'),
            'llm1_reasoning': state['llm1_result'].get('reasoning'),
            
            'llm2_user_type': state['llm2_result'].get('user_type'),
            'llm2_emotion': state['llm2_result'].get('emotion'),
            'llm2_context': state['llm2_result'].get('context'),
            'llm2_confidence': state['llm2_result'].get('confidence'),
            'llm2_reasoning': state['llm2_result'].get('reasoning'),
            
            'manager_classification': str(state['manager_result']),
            'manager_reasoning': state['manager_result'].get('reasoning'),
        }
        
        db.update_stage1(state['review_id'], stage1_data)
        db.close()
        
        return {"db_stage1_saved": True}
    except Exception as e:
        db.close()
        errors = state.get('errors', [])
        errors.append(f"DB Stage 1 save error: {str(e)}")
        return {"errors": errors}


def save_stage2_to_db_node(state: ReviewState) -> dict:
    """Save Stage 2 results to database"""
    db = EnhancedDatabase()
    db.connect()
    
    try:
        stage2_data = {
            'best_sentiment': state['best_sentiment_result']['sentiment'],
            'best_confidence': state['best_sentiment_result']['confidence'],
            'best_prob_positive': state['best_sentiment_result']['prob_positive'],
            'best_prob_neutral': state['best_sentiment_result']['prob_neutral'],
            'best_prob_negative': state['best_sentiment_result']['prob_negative'],
            
            'alt_sentiment': state['alt_sentiment_result']['sentiment'],
            'alt_confidence': state['alt_sentiment_result']['confidence'],
            'alt_prob_positive': state['alt_sentiment_result']['prob_positive'],
            'alt_prob_neutral': state['alt_sentiment_result']['prob_neutral'],
            'alt_prob_negative': state['alt_sentiment_result']['prob_negative'],
            
            'agreement': state['sentiment_agreement'],
            'layer_sentiment': state['sentiment'],
        }
        
        db.update_stage2(state['review_id'], stage2_data)
        db.close()
        
        return {"db_stage2_saved": True}
    except Exception as e:
        db.close()
        errors = state.get('errors', [])
        errors.append(f"DB Stage 2 save error: {str(e)}")
        return {"errors": errors}


def save_stage3_to_db_node(state: ReviewState) -> dict:
    """Save Stage 3 results to database"""
    db = EnhancedDatabase()
    db.connect()
    
    try:
        stage3_data = {
            'final_sentiment': state['final_sentiment'],
            'confidence': state['final_confidence'],
            'reasoning': state['reasoning'],
            'validation_notes': state['validation_notes'],
            'conflicts_found': state['conflicts_found'],
            'action_recommendation': state['action_recommendation'],
            'needs_human_review': state['needs_human_review'],
        }
        
        db.update_stage3(state['review_id'], stage3_data)
        db.close()
        
        return {"db_stage3_saved": True}
    except Exception as e:
        db.close()
        errors = state.get('errors', [])
        errors.append(f"DB Stage 3 save error: {str(e)}")
        return {"errors": errors}


# ============================================================================
# STAGE 4: BATCH ANALYSIS NODE
# ============================================================================

def stage4_batch_analysis_node(state: BatchState) -> dict:
    """
    Stage 4 Node: Batch analysis
    Runs after all reviews are processed
    """
    print(f"\n{'='*70}")
    print(f"📊 STAGE 4: BATCH ANALYSIS")
    print(f"{'='*70}")
    
    stage4 = Stage4BatchAnalysis()
    
    # Convert ReviewState list to dict format for Stage4
    reviews_for_analysis = []
    for review_state in state['all_reviews']:
        review_dict = {
            'review_id': review_state['review_id'],
            'review_text': review_state['review_text'],
            'rating': review_state['rating'],
            'stage1_llm1_type': review_state.get('classification_type'),
            'stage1_llm1_department': review_state.get('department'),
            'stage1_llm1_priority': review_state.get('priority'),
            'stage1_llm2_user_type': review_state.get('user_type'),
            'stage1_llm2_emotion': review_state.get('emotion'),
            'stage2_agreement': review_state.get('sentiment_agreement'),
            'stage3_final_sentiment': review_state.get('final_sentiment'),
            'stage3_needs_human_review': review_state.get('needs_human_review'),
            'stage3_reasoning': review_state.get('reasoning'),
            'stage3_action_recommendation': review_state.get('action_recommendation'),
        }
        reviews_for_analysis.append(review_dict)
    
    # Analyze batch
    insights = stage4.analyze_batch(reviews_for_analysis)
    
    # Save to database
    db = EnhancedDatabase()
    db.connect()
    db.save_batch_insights(insights)
    db.close()
    
    return {
        'sentiment_distribution': insights.get('sentiment_distribution'),
        'priority_distribution': insights.get('priority_distribution'),
        'department_distribution': insights.get('department_distribution'),
        'emotion_distribution': insights.get('emotion_distribution'),
        'critical_issues': insights.get('critical_issues'),
        'quick_wins': insights.get('quick_wins'),
        'churn_risk': insights.get('churn_risk'),
        'model_agreement_rate': insights.get('model_agreement_rate'),
        'recommendations': insights.get('recommendations'),
        'batch_completed_at': insights.get('batch_completed_at')
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_stage3(state: ReviewState) -> Literal["human_review", "complete"]:
    """
    Conditional routing after Stage 3
    Decides if human review is needed
    """
    # Check if human review needed
    if state.get('needs_human_review', False):
        return "human_review"
    
    # Check confidence threshold
    if state.get('final_confidence', 1.0) < 0.5:
        return "human_review"
    
    # Check for conflicts
    if state.get('conflicts_found', 'none') != 'none':
        return "human_review"
    
    # Check priority
    if state.get('priority') == 'critical':
        return "human_review"
    
    return "complete"


def human_review_queue_node(state: ReviewState) -> dict:
    """
    Node for reviews flagged for human review
    Just marks them in the database
    """
    print(f"         🚨 FLAGGED for human review")
    
    # Could integrate with ticketing system, email alerts, etc.
    # For now, just mark in state
    return {
        "route_to": "human_review"
    }


# ============================================================================
# BUILD REVIEW PROCESSING GRAPH
# ============================================================================

def build_review_graph():
    """
    Build the complete review processing graph
    """
    
    # Create graph
    workflow = StateGraph(ReviewState)
    
    # Add all nodes
    workflow.add_node("stage1_classify", stage1_classification_node)
    workflow.add_node("save_stage1", save_stage1_to_db_node)
    
    workflow.add_node("stage2_sentiment", stage2_sentiment_node)
    workflow.add_node("save_stage2", save_stage2_to_db_node)
    
    workflow.add_node("stage3_finalize", stage3_finalization_node)
    workflow.add_node("save_stage3", save_stage3_to_db_node)
    
    workflow.add_node("human_review_queue", human_review_queue_node)
    
    # Add edges (sequential flow through stages)
    workflow.add_edge("stage1_classify", "save_stage1")
    workflow.add_edge("save_stage1", "stage2_sentiment")
    workflow.add_edge("stage2_sentiment", "save_stage2")
    workflow.add_edge("save_stage2", "stage3_finalize")
    workflow.add_edge("stage3_finalize", "save_stage3")
    
    # Add conditional routing after Stage 3
    workflow.add_conditional_edges(
        "save_stage3",
        route_after_stage3,
        {
            "human_review": "human_review_queue",
            "complete": END
        }
    )
    
    # Human review goes to END
    workflow.add_edge("human_review_queue", END)
    
    # Set entry point
    workflow.set_entry_point("stage1_classify")
    
    # Compile with checkpointing
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph


# ============================================================================
# BUILD BATCH ANALYSIS GRAPH (Stage 4)
# ============================================================================

def build_batch_graph():
    """
    Build the batch analysis graph (Stage 4)
    This runs after all reviews are processed
    """
    
    workflow = StateGraph(BatchState)
    
    # Add batch analysis node
    workflow.add_node("stage4_batch", stage4_batch_analysis_node)
    
    # Simple linear flow
    workflow.set_entry_point("stage4_batch")
    workflow.add_edge("stage4_batch", END)
    
    # Compile
    graph = workflow.compile()
    
    return graph


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING LANGGRAPH GRAPH BUILDER")
    print("="*60)
    
    # Build review graph
    print("\n📊 Building review processing graph...")
    review_graph = build_review_graph()
    print("   ✅ Review graph built!")
    
    # Build batch graph
    print("\n📊 Building batch analysis graph...")
    batch_graph = build_batch_graph()
    print("   ✅ Batch graph built!")
    
    print("\n✅ Graph builder test complete!")
