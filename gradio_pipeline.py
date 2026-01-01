"""
Gradio Pipeline - Streamlined processing for HuggingFace Spaces
Integrates scraping, classification, sentiment, and batch analysis with progress tracking
"""

import os
import sqlite3
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json

# Import existing modules
from stage0_scraper import Stage0WebScraper
from langgraph_state import ReviewState, create_initial_state
from langgraph_graph import build_review_graph, build_batch_graph
from database_enhanced import EnhancedDatabase
from stage4_batch_analysis import Stage4BatchAnalysis


class GradioPipeline:
    """
    Streamlined pipeline for Gradio interface
    Handles scraping, processing, and analysis with progress callbacks
    """
    
    def __init__(self, db_file: str = "review_database.db", review_limit: int = 20):
        self.db_file = db_file
        self.review_limit = review_limit
        
        # Initialize database
        self.db = EnhancedDatabase(db_file)
        self.db.connect()
        self.scraper = Stage0WebScraper(db_file)
        self.scraper.create_reviews_table()
        self.db.enhance_schema()
        
        # Initialize scraper
        
    
        
        # Build graphs
        self.review_graph = build_review_graph()
        self.batch_graph = build_batch_graph()
        
        print("✅ Gradio Pipeline initialized")
    
    def scrape_reviews(
        self,
        app_store_ids: List[str],
        play_store_packages: List[str],
        progress_callback: Optional[Callable] = None
    ) -> int:
        """
        Scrape reviews from App Store and Play Store
        
        Args:
            app_store_ids: List of App Store IDs
            play_store_packages: List of Play Store package names
            progress_callback: Optional Gradio progress callback
        
        Returns:
            Total number of reviews scraped
        """
        total_scraped = 0
        total_apps = len(app_store_ids) + len(play_store_packages)
        
        if total_apps == 0:
            return 0
        
        current_app = 0
        
        # Scrape App Store
        for app_id in app_store_ids:
            current_app += 1
            if progress_callback:
                progress_val = 0.1 + (0.2 * current_app / total_apps)
                progress_callback(
                    progress_val,
                    desc=f"🍎 Scraping App Store ({current_app}/{total_apps}): {app_id}"
                )
            
            try:
                reviews = self.scraper.scrape_app_store_rss(
                    app_id,
                    country="ae",
                    limit=self.review_limit
                )
                saved = self.scraper.save_reviews_to_db(reviews)
                total_scraped += saved
                print(f"   ✅ App Store {app_id}: {saved} reviews")
            except Exception as e:
                print(f"   ❌ App Store {app_id} error: {e}")
                continue
            
            time.sleep(1)  # Rate limiting
        
        # Scrape Play Store
        for package in play_store_packages:
            current_app += 1
            if progress_callback:
                progress_val = 0.1 + (0.2 * current_app / total_apps)
                progress_callback(
                    progress_val,
                    desc=f"🤖 Scraping Play Store ({current_app}/{total_apps}): {package}"
                )
            
            try:
                reviews = self.scraper.scrape_play_store_api(
                    package,
                    country="ae",
                    limit=self.review_limit
                )
                saved = self.scraper.save_reviews_to_db(reviews)
                total_scraped += saved
                print(f"   ✅ Play Store {package}: {saved} reviews")
            except Exception as e:
                print(f"   ❌ Play Store {package} error: {e}")
                continue
            
            time.sleep(1)  # Rate limiting
        
        print(f"\n✅ Total scraped: {total_scraped} reviews")
        return total_scraped
    
    def process_reviews(
        self,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process reviews through Stages 1-3
        
        Args:
            progress_callback: Optional Gradio progress callback
        
        Returns:
            List of processed review dictionaries
        """
        # Get pending reviews
        reviews = self.db.get_pending_reviews(limit=self.review_limit)
        total_reviews = len(reviews)
        
        if total_reviews == 0:
            print("⚠️  No pending reviews to process")
            return []
        
        print(f"\n📊 Processing {total_reviews} reviews...")
        
        processed_states = []
        
        for i, review in enumerate(reviews, 1):
            review_id = review.get('review_id', 'unknown')
            
            if progress_callback:
                progress_val = 0.3 + (0.6 * i / total_reviews)
                progress_callback(
                    progress_val,
                    desc=f"🤖 Processing review {i}/{total_reviews}: {review_id[:20]}..."
                )
            
            try:
                # Create initial state
                state = create_initial_state(review)
                
                # Run through LangGraph
                config = {"configurable": {"thread_id": f"review_{review_id}"}}
                final_state = self.review_graph.invoke(state, config=config)
                
                # Convert state to dict for easier handling
                processed_states.append(dict(final_state))
                
                print(f"   ✅ Review {i}/{total_reviews} processed")
                
            except Exception as e:
                print(f"   ❌ Error processing review {review_id}: {e}")
                continue
        
        print(f"\n✅ Processed {len(processed_states)}/{total_reviews} reviews")
        return processed_states
    
    def analyze_batch(
        self,
        processed_reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run Stage 4: Batch Analysis
        
        Args:
            processed_reviews: List of processed review states
        
        Returns:
            Batch insights dictionary
        """
        if not processed_reviews:
            return {}
        
        print(f"\n📊 Running batch analysis on {len(processed_reviews)} reviews...")
        
        # Convert states to review dicts for Stage 4
        reviews_for_analysis = []
        for state in processed_reviews:
            review_dict = {
                'review_id': state.get('review_id'),
                'review_text': state.get('review_text'),
                'rating': state.get('rating'),
                'stage1_llm1_type': state.get('classification_type'),
                'stage1_llm1_department': state.get('department'),
                'stage1_llm1_priority': state.get('priority'),
                'stage1_llm2_user_type': state.get('user_type'),
                'stage1_llm2_emotion': state.get('emotion'),
                'stage2_agreement': state.get('sentiment_agreement'),
                'stage3_final_sentiment': state.get('final_sentiment'),
                'stage3_needs_human_review': state.get('needs_human_review'),
                'stage3_reasoning': state.get('reasoning'),
                'stage3_action_recommendation': state.get('action_recommendation'),
            }
            reviews_for_analysis.append(review_dict)
        
        # Run Stage 4
        stage4 = Stage4BatchAnalysis()
        insights = stage4.analyze_batch(reviews_for_analysis)
        
        # Save to database
        self.db.save_batch_insights(insights)
        
        print("✅ Batch analysis complete")
        return insights
    
    def get_all_processed_reviews(self) -> List[Dict[str, Any]]:
        """Get all processed reviews from database"""
        return self.db.get_all_processed_reviews()
    
    def close(self):
        """Clean up"""
        self.db.close()


# ============================================================================
# HELPER FUNCTIONS FOR GRADIO
# ============================================================================

def parse_app_store_url(url: str) -> Optional[str]:
    """
    Extract App Store ID from URL or return as-is if already an ID
    
    Examples:
        - "1234567890" -> "1234567890"
        - "https://apps.apple.com/us/app/name/id1234567890" -> "1234567890"
    """
    url = url.strip()
    
    # Check if it's already just a number
    if url.isdigit():
        return url
    
    # Extract from URL
    if 'apps.apple.com' in url:
        parts = url.split('/id')
        if len(parts) > 1:
            app_id = parts[1].split('?')[0].split('/')[0]
            if app_id.isdigit():
                return app_id
    
    # Try to find any number in the string
    import re
    numbers = re.findall(r'\d+', url)
    if numbers:
        # Return the longest number (likely the app ID)
        return max(numbers, key=len)
    
    return None


def parse_play_store_url(url: str) -> Optional[str]:
    """
    Extract package name from Play Store URL or return as-is
    
    Examples:
        - "com.company.app" -> "com.company.app"
        - "https://play.google.com/store/apps/details?id=com.company.app" -> "com.company.app"
    """
    url = url.strip()
    
    # Check if it's already a package name (has dots)
    if '.' in url and not url.startswith('http'):
        return url
    
    # Extract from URL
    if 'play.google.com' in url:
        if 'id=' in url:
            package = url.split('id=')[1].split('&')[0]
            return package
    
    return url if '.' in url else None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING GRADIO PIPELINE")
    print("="*60)
    
    # Test URL parsing
    print("\n📱 Testing URL parsing:")
    
    test_app_urls = [
        "1234567890",
        "https://apps.apple.com/us/app/name/id1234567890",
    ]
    
    for url in test_app_urls:
        app_id = parse_app_store_url(url)
        print(f"   {url} -> {app_id}")
    
    test_play_urls = [
        "com.company.app",
        "https://play.google.com/store/apps/details?id=com.company.app",
    ]
    
    for url in test_play_urls:
        package = parse_play_store_url(url)
        print(f"   {url} -> {package}")
    
    print("\n✅ Gradio pipeline test complete!")
