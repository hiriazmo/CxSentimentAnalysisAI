"""
Stage 4: Batch Analysis & Aggregation
- Aggregate insights across all processed reviews
- Identify patterns, trends, critical issues
- Generate actionable recommendations
"""

import json
from typing import Dict, Any, List
from collections import Counter

class Stage4BatchAnalysis:
    """
    Stage 4: Batch-level intelligence and recommendations
    """
    
    def __init__(self):
        print("   📊 Stage 4: Batch Analysis initialized")
    
    def analyze_batch(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a batch of processed reviews
        """
        if not reviews:
            print("      ⚠️  No reviews to analyze")
            return self._empty_insights()
        
        print(f"\n      📊 Analyzing batch of {len(reviews)} reviews...")
        
        # Initialize counters
        total = len(reviews)
        
        # Sentiment distribution
        sentiment_counts = Counter()
        for review in reviews:
            sentiment = review.get('stage3_final_sentiment', 'NEUTRAL')
            sentiment_counts[sentiment] += 1
        
        print(f"         📈 Sentiment: "
              f"POS={sentiment_counts.get('POSITIVE', 0)}, "
              f"NEU={sentiment_counts.get('NEUTRAL', 0)}, "
              f"NEG={sentiment_counts.get('NEGATIVE', 0)}")
        
        # Priority distribution
        priority_counts = Counter()
        for review in reviews:
            priority = review.get('stage1_llm1_priority', 'unknown')
            priority_counts[priority] += 1
        
        print(f"         🎯 Priority: "
              f"Critical={priority_counts.get('critical', 0)}, "
              f"High={priority_counts.get('high', 0)}, "
              f"Medium={priority_counts.get('medium', 0)}, "
              f"Low={priority_counts.get('low', 0)}")
        
        # Department routing
        dept_counts = Counter()
        for review in reviews:
            dept = review.get('stage1_llm1_department', 'unknown')
            dept_counts[dept] += 1
        
        print(f"         🏢 Departments: "
              f"Eng={dept_counts.get('engineering', 0)}, "
              f"UX={dept_counts.get('ux', 0)}, "
              f"Support={dept_counts.get('support', 0)}, "
              f"Business={dept_counts.get('business', 0)}")
        
        # Emotion distribution
        emotion_counts = Counter()
        for review in reviews:
            emotion = review.get('stage1_llm2_emotion', 'unknown')
            emotion_counts[emotion] += 1
        
        # Review type distribution
        type_counts = Counter()
        for review in reviews:
            review_type = review.get('stage1_llm1_type', 'unknown')
            type_counts[review_type] += 1
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(reviews)
        print(f"         🚨 Critical Issues: {len(critical_issues)}")
        
        # Identify quick wins
        quick_wins = self._identify_quick_wins(reviews)
        print(f"         ⚡ Quick Wins: {len(quick_wins)}")
        
        # Calculate churn risk
        churn_risk = self._calculate_churn_risk(reviews)
        print(f"         ⚠️  Churn Risk: {churn_risk:.1f}%")
        
        # Model agreement rate
        agreement_count = sum(1 for r in reviews if r.get('stage2_agreement', False))
        agreement_rate = (agreement_count / total * 100) if total > 0 else 0
        print(f"         🤝 Model Agreement: {agreement_rate:.1f}%")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            sentiment_counts, priority_counts, dept_counts, 
            critical_issues, quick_wins, churn_risk
        )
        
        # Compile batch insights
        insights = {
            'total_reviews': total,
            
            # Sentiment
            'sentiment_positive': sentiment_counts.get('POSITIVE', 0),
            'sentiment_neutral': sentiment_counts.get('NEUTRAL', 0),
            'sentiment_negative': sentiment_counts.get('NEGATIVE', 0),
            'sentiment_distribution': dict(sentiment_counts),
            
            # Priority
            'priority_critical': priority_counts.get('critical', 0),
            'priority_high': priority_counts.get('high', 0),
            'priority_medium': priority_counts.get('medium', 0),
            'priority_low': priority_counts.get('low', 0),
            'priority_distribution': dict(priority_counts),
            
            # Department
            'dept_engineering': dept_counts.get('engineering', 0),
            'dept_ux': dept_counts.get('ux', 0),
            'dept_support': dept_counts.get('support', 0),
            'dept_business': dept_counts.get('business', 0),
            'department_distribution': dict(dept_counts),
            
            # Additional insights
            'emotion_distribution': dict(emotion_counts),
            'type_distribution': dict(type_counts),
            'model_agreement_rate': agreement_rate,
            'churn_risk': churn_risk,
            
            # Actionable lists
            'critical_issues': critical_issues,
            'quick_wins': quick_wins,
            'recommendations': recommendations
        }
        
        return insights
    
    def _identify_critical_issues(self, reviews: List[Dict]) -> List[Dict]:
        """Identify critical issues requiring immediate attention"""
        critical = []
        
        for review in reviews:
            priority = review.get('stage1_llm1_priority', '')
            sentiment = review.get('stage3_final_sentiment', '')
            needs_review = review.get('stage3_needs_human_review', False)
            
            if priority == 'critical' or (sentiment == 'NEGATIVE' and needs_review):
                critical.append({
                    'review_id': review.get('review_id', 'unknown'),
                    'type': review.get('stage1_llm1_type', 'unknown'),
                    'department': review.get('stage1_llm1_department', 'unknown'),
                    'reasoning': review.get('stage3_reasoning', ''),
                    'action': review.get('stage3_action_recommendation', ''),
                    'rating': review.get('rating', 0)
                })
        
        # Sort by rating (lowest first)
        critical.sort(key=lambda x: x['rating'])
        
        return critical[:10]  # Top 10 critical issues
    
    def _identify_quick_wins(self, reviews: List[Dict]) -> List[Dict]:
        """Identify easy-to-fix issues for quick wins"""
        quick_wins = []
        
        for review in reviews:
            review_type = review.get('stage1_llm1_type', '')
            priority = review.get('stage1_llm1_priority', '')
            sentiment = review.get('stage3_final_sentiment', '')
            
            # Suggestions with low priority = quick wins
            if review_type == 'suggestion' and priority in ['low', 'medium']:
                quick_wins.append({
                    'review_id': review.get('review_id', 'unknown'),
                    'suggestion': review.get('review_text', '')[:100],
                    'department': review.get('stage1_llm1_department', 'unknown'),
                    'action': review.get('stage3_action_recommendation', ''),
                    'rating': review.get('rating', 0)
                })
        
        return quick_wins[:10]  # Top 10 quick wins
    
    def _calculate_churn_risk(self, reviews: List[Dict]) -> float:
        """Calculate overall churn risk percentage"""
        if not reviews:
            return 0.0
        
        churn_indicators = 0
        
        for review in reviews:
            user_type = review.get('stage1_llm2_user_type', '')
            sentiment = review.get('stage3_final_sentiment', '')
            rating = review.get('rating', 3)
            
            # Churn indicators
            if user_type == 'churning_user':
                churn_indicators += 2
            elif sentiment == 'NEGATIVE' and rating <= 2:
                churn_indicators += 1
            elif rating == 1:
                churn_indicators += 1
        
        # Calculate percentage
        max_possible = len(reviews) * 2
        churn_risk = (churn_indicators / max_possible * 100) if max_possible > 0 else 0.0
        
        return min(churn_risk, 100.0)
    
    def _generate_recommendations(self, sentiment_counts, priority_counts, 
                                 dept_counts, critical_issues, quick_wins, 
                                 churn_risk) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Sentiment-based
        total = sum(sentiment_counts.values())
        if total > 0:
            neg_pct = (sentiment_counts.get('NEGATIVE', 0) / total * 100)
            if neg_pct > 40:
                recommendations.append(
                    f"🚨 HIGH: {neg_pct:.0f}% negative sentiment. Immediate investigation needed."
                )
            elif neg_pct > 25:
                recommendations.append(
                    f"⚠️  MEDIUM: {neg_pct:.0f}% negative sentiment. Monitor closely."
                )
        
        # Priority-based
        if priority_counts.get('critical', 0) > 0:
            recommendations.append(
                f"🔥 URGENT: {priority_counts['critical']} critical issues require immediate attention."
            )
        
        # Department-based
        if dept_counts:
            top_dept = max(dept_counts, key=dept_counts.get)
            top_count = dept_counts[top_dept]
            recommendations.append(
                f"🎯 FOCUS: {top_count} issues routed to {top_dept} department."
            )
        
        # Churn risk
        if churn_risk > 30:
            recommendations.append(
                f"⚠️  CHURN: {churn_risk:.0f}% churn risk detected. Implement retention strategy."
            )
        
        # Quick wins
        if quick_wins:
            recommendations.append(
                f"⚡ OPPORTUNITY: {len(quick_wins)} quick wins available for easy improvements."
            )
        
        return recommendations
    
    def _empty_insights(self) -> Dict[str, Any]:
        """Return empty insights structure"""
        return {
            'total_reviews': 0,
            'sentiment_positive': 0,
            'sentiment_neutral': 0,
            'sentiment_negative': 0,
            'priority_critical': 0,
            'priority_high': 0,
            'priority_medium': 0,
            'priority_low': 0,
            'dept_engineering': 0,
            'dept_ux': 0,
            'dept_support': 0,
            'dept_business': 0,
            'critical_issues': [],
            'quick_wins': [],
            'recommendations': []
        }


if __name__ == "__main__":
    # Test Stage 4
    print("\n" + "="*60)
    print("🧪 TESTING STAGE 4 BATCH ANALYSIS")
    print("="*60)
    
    # Sample processed reviews
    sample_reviews = [
        {
            'review_id': '001',
            'review_text': 'App crashes!',
            'rating': 1,
            'stage1_llm1_type': 'bug_report',
            'stage1_llm1_department': 'engineering',
            'stage1_llm1_priority': 'critical',
            'stage1_llm2_user_type': 'power_user',
            'stage1_llm2_emotion': 'frustration',
            'stage2_agreement': True,
            'stage3_final_sentiment': 'NEGATIVE',
            'stage3_needs_human_review': True,
            'stage3_reasoning': 'Critical bug',
            'stage3_action_recommendation': 'Fix immediately'
        },
        {
            'review_id': '002',
            'review_text': 'Great app!',
            'rating': 5,
            'stage1_llm1_type': 'praise',
            'stage1_llm1_department': 'ux',
            'stage1_llm1_priority': 'low',
            'stage1_llm2_user_type': 'regular_user',
            'stage1_llm2_emotion': 'joy',
            'stage2_agreement': True,
            'stage3_final_sentiment': 'POSITIVE',
            'stage3_needs_human_review': False
        }
    ]
    
    stage4 = Stage4BatchAnalysis()
    insights = stage4.analyze_batch(sample_reviews)
    
    print("\n📊 BATCH INSIGHTS:")
    print(json.dumps(insights, indent=2))
    print("\n✅ Stage 4 test complete!")
