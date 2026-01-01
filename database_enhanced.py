"""
Enhanced Database Schema for Multi-Stage Review Analysis
Adds Stage 1-4 columns to existing reviews table
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

class EnhancedDatabase:
    """
    Manages enhanced database schema with Stage 1-4 columns
    """
    
    def __init__(self, db_file: str = "review_database.db"):
        self.db_file = db_file
        self.conn = None
        print(f"📁 Database: {db_file}")
    
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        print("✅ Connected to database")
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def enhance_schema(self):
        """
        Add Stage 1-4 columns to existing reviews table
        Non-destructive: keeps all existing data
        """
        print("\n" + "="*60)
        print("🔧 ENHANCING DATABASE SCHEMA")
        print("="*60)
        
        cursor = self.conn.cursor()
        
        # Get existing columns
        cursor.execute("PRAGMA table_info(reviews)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        print(f"📋 Existing columns: {len(existing_columns)}")
        
        # Stage 1: Classification columns
        stage1_columns = [
            ("stage1_llm1_type", "TEXT"),
            ("stage1_llm1_department", "TEXT"),
            ("stage1_llm1_priority", "TEXT"),
            ("stage1_llm1_confidence", "REAL"),
            ("stage1_llm1_reasoning", "TEXT"),
            ("stage1_llm2_user_type", "TEXT"),
            ("stage1_llm2_emotion", "TEXT"),
            ("stage1_llm2_context", "TEXT"),
            ("stage1_llm2_confidence", "REAL"),
            ("stage1_llm2_reasoning", "TEXT"),
            ("stage1_manager_classification", "TEXT"),
            ("stage1_manager_reasoning", "TEXT"),
            ("stage1_completed_at", "TIMESTAMP"),
        ]
        
        # Stage 2: Sentiment columns
        stage2_columns = [
            ("stage2_best_sentiment", "TEXT"),
            ("stage2_best_confidence", "REAL"),
            ("stage2_best_prob_positive", "REAL"),
            ("stage2_best_prob_neutral", "REAL"),
            ("stage2_best_prob_negative", "REAL"),
            ("stage2_alt_sentiment", "TEXT"),
            ("stage2_alt_confidence", "REAL"),
            ("stage2_alt_prob_positive", "REAL"),
            ("stage2_alt_prob_neutral", "REAL"),
            ("stage2_alt_prob_negative", "REAL"),
            ("stage2_agreement", "BOOLEAN"),
            ("stage2_layer_sentiment", "TEXT"),
            ("stage2_completed_at", "TIMESTAMP"),
        ]
        
        # Stage 3: Finalization columns
        stage3_columns = [
            ("stage3_final_sentiment", "TEXT"),
            ("stage3_confidence", "REAL"),
            ("stage3_reasoning", "TEXT"),
            ("stage3_validation_notes", "TEXT"),
            ("stage3_conflicts_found", "TEXT"),
            ("stage3_action_recommendation", "TEXT"),
            ("stage3_needs_human_review", "BOOLEAN"),
            ("stage3_completed_at", "TIMESTAMP"),
        ]
        
        # Processing metadata
        metadata_columns = [
            ("processing_status", "TEXT DEFAULT 'pending'"),
            ("processing_version", "TEXT DEFAULT 'v1.0'"),
            ("processing_started_at", "TIMESTAMP"),
            ("processing_completed_at", "TIMESTAMP"),
        ]
        
        all_new_columns = (
            stage1_columns + 
            stage2_columns + 
            stage3_columns + 
            metadata_columns
        )
        
        # Add columns that don't exist
        added_count = 0
        for col_name, col_type in all_new_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE reviews ADD COLUMN {col_name} {col_type}")
                    added_count += 1
                    print(f"   ✅ Added column: {col_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"   ⚠️  Error adding {col_name}: {e}")
        
        self.conn.commit()
        print(f"\n✅ Schema enhanced: {added_count} new columns added")
        
        # Create logs table for LLM decisions
        self._create_logs_table(cursor)
        
        # Create batch insights table
        self._create_batch_insights_table(cursor)
        
        return added_count
    
    def _create_logs_table(self, cursor):
        """Create table for LLM decision logs"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_decision_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_prompt TEXT,
                output_response TEXT,
                confidence REAL,
                reasoning TEXT,
                processing_time_seconds REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (review_id) REFERENCES reviews(review_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_review_id 
            ON llm_decision_logs(review_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_stage 
            ON llm_decision_logs(stage)
        """)
        
        self.conn.commit()
        print("   ✅ Created llm_decision_logs table")
    
    def _create_batch_insights_table(self, cursor):
        """Create table for batch analytics"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_insights (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date DATE,
                total_reviews INTEGER,
                sentiment_positive INTEGER,
                sentiment_neutral INTEGER,
                sentiment_negative INTEGER,
                priority_critical INTEGER,
                priority_high INTEGER,
                priority_medium INTEGER,
                priority_low INTEGER,
                dept_engineering INTEGER,
                dept_ux INTEGER,
                dept_support INTEGER,
                dept_business INTEGER,
                critical_issues TEXT,
                quick_wins TEXT,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        print("   ✅ Created batch_insights table")
    
    def get_pending_reviews(self, limit: Optional[int] = None) -> List[Dict]:
        """Get reviews that haven't been processed yet"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM reviews 
            WHERE processing_status IS NULL OR processing_status = 'pending'
            ORDER BY scraped_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def update_stage1(self, review_id: str, data: Dict[str, Any]):
        """Update Stage 1 classification data"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE reviews SET
                stage1_llm1_type = ?,
                stage1_llm1_department = ?,
                stage1_llm1_priority = ?,
                stage1_llm1_confidence = ?,
                stage1_llm1_reasoning = ?,
                stage1_llm2_user_type = ?,
                stage1_llm2_emotion = ?,
                stage1_llm2_context = ?,
                stage1_llm2_confidence = ?,
                stage1_llm2_reasoning = ?,
                stage1_manager_classification = ?,
                stage1_manager_reasoning = ?,
                stage1_completed_at = ?,
                processing_status = 'stage1_complete'
            WHERE review_id = ?
        """, (
            data.get('llm1_type'),
            data.get('llm1_department'),
            data.get('llm1_priority'),
            data.get('llm1_confidence'),
            data.get('llm1_reasoning'),
            data.get('llm2_user_type'),
            data.get('llm2_emotion'),
            data.get('llm2_context'),
            data.get('llm2_confidence'),
            data.get('llm2_reasoning'),
            data.get('manager_classification'),
            data.get('manager_reasoning'),
            datetime.now().isoformat(),
            review_id
        ))
        
        self.conn.commit()
    
    def update_stage2(self, review_id: str, data: Dict[str, Any]):
        """Update Stage 2 sentiment data"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE reviews SET
                stage2_best_sentiment = ?,
                stage2_best_confidence = ?,
                stage2_best_prob_positive = ?,
                stage2_best_prob_neutral = ?,
                stage2_best_prob_negative = ?,
                stage2_alt_sentiment = ?,
                stage2_alt_confidence = ?,
                stage2_alt_prob_positive = ?,
                stage2_alt_prob_neutral = ?,
                stage2_alt_prob_negative = ?,
                stage2_agreement = ?,
                stage2_layer_sentiment = ?,
                stage2_completed_at = ?,
                processing_status = 'stage2_complete'
            WHERE review_id = ?
        """, (
            data.get('best_sentiment'),
            data.get('best_confidence'),
            data.get('best_prob_positive'),
            data.get('best_prob_neutral'),
            data.get('best_prob_negative'),
            data.get('alt_sentiment'),
            data.get('alt_confidence'),
            data.get('alt_prob_positive'),
            data.get('alt_prob_neutral'),
            data.get('alt_prob_negative'),
            data.get('agreement'),
            data.get('layer_sentiment'),
            datetime.now().isoformat(),
            review_id
        ))
        
        self.conn.commit()
    
    def update_stage3(self, review_id: str, data: Dict[str, Any]):
        """Update Stage 3 finalization data"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE reviews SET
                stage3_final_sentiment = ?,
                stage3_confidence = ?,
                stage3_reasoning = ?,
                stage3_validation_notes = ?,
                stage3_conflicts_found = ?,
                stage3_action_recommendation = ?,
                stage3_needs_human_review = ?,
                stage3_completed_at = ?,
                processing_status = 'complete',
                processing_completed_at = ?
            WHERE review_id = ?
        """, (
            data.get('final_sentiment'),
            data.get('confidence'),
            data.get('reasoning'),
            data.get('validation_notes'),
            data.get('conflicts_found'),
            data.get('action_recommendation'),
            data.get('needs_human_review'),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            review_id
        ))
        
        self.conn.commit()
    
    def log_llm_decision(self, review_id: str, stage: str, model_name: str, 
                        input_prompt: str, output_response: str, 
                        confidence: float, reasoning: str, processing_time: float):
        """Log LLM decision for audit trail"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO llm_decision_logs 
            (review_id, stage, model_name, input_prompt, output_response, 
             confidence, reasoning, processing_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            review_id, stage, model_name, input_prompt, output_response,
            confidence, reasoning, processing_time
        ))
        
        self.conn.commit()
    
    def get_all_processed_reviews(self) -> List[Dict]:
        """Get all reviews that have been fully processed"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM reviews 
            WHERE processing_status = 'complete'
            ORDER BY processing_completed_at DESC
        """)
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def save_batch_insights(self, insights: Dict[str, Any]):
        """Save batch analytics to database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO batch_insights
            (analysis_date, total_reviews, sentiment_positive, sentiment_neutral,
             sentiment_negative, priority_critical, priority_high, priority_medium,
             priority_low, dept_engineering, dept_ux, dept_support, dept_business,
             critical_issues, quick_wins, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().date(),
            insights.get('total_reviews', 0),
            insights.get('sentiment_positive', 0),
            insights.get('sentiment_neutral', 0),
            insights.get('sentiment_negative', 0),
            insights.get('priority_critical', 0),
            insights.get('priority_high', 0),
            insights.get('priority_medium', 0),
            insights.get('priority_low', 0),
            insights.get('dept_engineering', 0),
            insights.get('dept_ux', 0),
            insights.get('dept_support', 0),
            insights.get('dept_business', 0),
            json.dumps(insights.get('critical_issues', [])),
            json.dumps(insights.get('quick_wins', [])),
            json.dumps(insights.get('recommendations', []))
        ))
        
        self.conn.commit()
        print("   ✅ Batch insights saved to database")
    
    def reset_processing_status(self, limit: Optional[int] = None):
        """Reset processing status to reprocess reviews"""
        cursor = self.conn.cursor()
        
        if limit:
            # Reset only the most recent N reviews
            query = """
                UPDATE reviews 
                SET processing_status = 'pending',
                    processing_started_at = NULL,
                    processing_completed_at = NULL,
                    stage1_completed_at = NULL,
                    stage2_completed_at = NULL,
                    stage3_completed_at = NULL
                WHERE review_id IN (
                    SELECT review_id FROM reviews 
                    ORDER BY scraped_at DESC 
                    LIMIT ?
                )
            """
            cursor.execute(query, (limit,))
        else:
            # Reset all reviews
            query = """
                UPDATE reviews 
                SET processing_status = 'pending',
                    processing_started_at = NULL,
                    processing_completed_at = NULL,
                    stage1_completed_at = NULL,
                    stage2_completed_at = NULL,
                    stage3_completed_at = NULL
            """
            cursor.execute(query)
        
        affected = cursor.rowcount
        self.conn.commit()
        
        if affected > 0:
            print(f"   🔄 Reset {affected} reviews to pending status")
        
        return affected


if __name__ == "__main__":
    # Test database enhancement
    print("\n" + "="*60)
    print("🧪 TESTING DATABASE ENHANCEMENT")
    print("="*60 + "\n")
    
    db = EnhancedDatabase()
    db.connect()
    db.enhance_schema()
    
    # Get pending reviews
    pending = db.get_pending_reviews(limit=5)
    print(f"\n📋 Found {len(pending)} pending reviews")
    
    db.close()
    print("\n✅ Database enhancement test complete!")