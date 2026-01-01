"""
Stage 0: Web Scraping (App Store & Play Store)
Scrapes reviews and stores in database
This integrates with your existing scraper or can be used standalone
"""

import os
import sqlite3
import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import re

class Stage0WebScraper:
    """
    Stage 0: Web scraping for App Store and Play Store reviews
    Integrates with existing database structure
    """
    
    def __init__(self, db_file: str = "review_database.db"):
        self.db_file = db_file
        print(f"   📁 Database: {db_file}")
    
    def create_reviews_table(self):
        """
        Create reviews table if it doesn't exist
        This is your Stage 0 schema
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT UNIQUE,
                product_url TEXT,
                platform TEXT,
                app_name TEXT,
                user_name TEXT,
                review_text TEXT,
                rating INTEGER,
                review_date TEXT,
                app_version TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_review_id 
            ON reviews(review_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_platform 
            ON reviews(platform)
        """)
        
        conn.commit()
        conn.close()
        
        print("   ✅ Reviews table ready (Stage 0)")
    
    def scrape_app_store_rss(self, app_id: str, country: str = "us", 
                              limit: int = 100) -> List[Dict]:
        """
        Scrape App Store reviews using RSS feed
        This is a simple, free method (no API key needed)
        
        Args:
            app_id: App Store app ID (e.g., "1234567890")
            country: Country code (e.g., "us", "ae", "uk")
            limit: Number of reviews to fetch (max 500 per request)
        """
        print(f"   🍎 Scraping App Store: {app_id} ({country})")
        
        # App Store RSS feed URL
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            reviews = []
            entries = data.get('feed', {}).get('entry', [])
            
            # Skip first entry (it's the app info)
            if entries and 'author' not in entries[0]:
                entries = entries[1:]
            
            for entry in entries[:limit]:
                try:
                    review = {
                        'review_id': entry.get('id', {}).get('label', ''),
                        'platform': 'app_store',
                        'app_name': data.get('feed', {}).get('title', {}).get('label', 'Unknown'),
                        'user_name': entry.get('author', {}).get('name', {}).get('label', 'Anonymous'),
                        'review_text': entry.get('content', {}).get('label', ''),
                        'rating': int(entry.get('im:rating', {}).get('label', '3')),
                        'review_date': entry.get('updated', {}).get('label', ''),
                        'app_version': entry.get('im:version', {}).get('label', ''),
                        'product_url': entry.get('link', {}).get('attributes', {}).get('href', '')
                    }
                    reviews.append(review)
                except Exception as e:
                    print(f"      ⚠️  Error parsing review: {e}")
                    continue
            
            print(f"      ✅ Scraped {len(reviews)} reviews")
            return reviews
            
        except Exception as e:
            print(f"      ❌ Error scraping App Store: {e}")
            return []
    
    def scrape_play_store_api(self, app_id: str, country: str = "us", 
                               limit: int = 100) -> List[Dict]:
        """
        Scrape Google Play Store reviews
        Note: This is a simplified version. For production, use google-play-scraper library
        
        Args:
            app_id: Play Store package name (e.g., "com.company.app")
            country: Country code
            limit: Number of reviews to fetch
        """
        print(f"   🤖 Scraping Play Store: {app_id} ({country})")
        
        try:
            # Using unofficial API endpoint (works without auth)
            # For production, recommend: pip install google-play-scraper
            from google_play_scraper import Sort, reviews_all
            
            result = reviews_all(
                app_id,
                sleep_milliseconds=0,
                lang='en',
                country=country,
                sort=Sort.NEWEST
            )
            
            reviews = []
            for r in result[:limit]:
                review = {
                    'review_id': r.get('reviewId', ''),
                    'platform': 'play_store',
                    'app_name': app_id,
                    'user_name': r.get('userName', 'Anonymous'),
                    'review_text': r.get('content', ''),
                    'rating': r.get('score', 3),
                    'review_date': r.get('at', '').isoformat() if r.get('at') else '',
                    'app_version': r.get('reviewCreatedVersion', ''),
                    'product_url': f"https://play.google.com/store/apps/details?id={app_id}"
                }
                reviews.append(review)
            
            print(f"      ✅ Scraped {len(reviews)} reviews")
            return reviews
            
        except ImportError:
            print("      ⚠️  google-play-scraper not installed")
            print("      Run: pip install google-play-scraper")
            return []
        except Exception as e:
            print(f"      ❌ Error scraping Play Store: {e}")
            return []
    
    def save_reviews_to_db(self, reviews: List[Dict]) -> int:
        """
        Save scraped reviews to database
        Returns number of new reviews saved
        """
        if not reviews:
            return 0
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        saved_count = 0
        
        for review in reviews:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO reviews 
                    (review_id, product_url, platform, app_name, user_name, 
                     review_text, rating, review_date, app_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    review.get('review_id'),
                    review.get('product_url', ''),
                    review.get('platform'),
                    review.get('app_name', ''),
                    review.get('user_name'),
                    review.get('review_text'),
                    review.get('rating'),
                    review.get('review_date', ''),
                    review.get('app_version', '')
                ))
                
                if cursor.rowcount > 0:
                    saved_count += 1
                    
            except Exception as e:
                print(f"      ⚠️  Error saving review: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        print(f"      ✅ Saved {saved_count} new reviews to database")
        return saved_count
    
    def scrape_from_urls_file(self, urls_file: str = "urls.txt") -> int:
        """
        Scrape reviews from URLs listed in a text file
        
        URLs file format (one per line):
        app_store:1234567890:us
        play_store:com.company.app:us
        """
        print(f"\n   📄 Reading URLs from: {urls_file}")
        
        if not os.path.exists(urls_file):
            print(f"      ⚠️  File not found: {urls_file}")
            return 0
        
        total_saved = 0
        
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"      ✅ Found {len(urls)} URLs")
        
        for i, url in enumerate(urls, 1):
            print(f"\n   [{i}/{len(urls)}] Processing: {url}")
            
            parts = url.split(':')
            if len(parts) < 2:
                print(f"      ⚠️  Invalid format: {url}")
                continue
            
            platform = parts[0].lower()
            app_id = parts[1]
            country = parts[2] if len(parts) > 2 else 'us'
            
            if platform == 'app_store':
                reviews = self.scrape_app_store_rss(app_id, country)
            elif platform == 'play_store':
                reviews = self.scrape_play_store_api(app_id, country)
            else:
                print(f"      ⚠️  Unknown platform: {platform}")
                continue
            
            saved = self.save_reviews_to_db(reviews)
            total_saved += saved
            
            # Be nice to servers
            time.sleep(2)
        
        return total_saved
    
    def get_review_count(self) -> int:
        """Get total number of reviews in database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reviews")
        count = cursor.fetchone()[0]
        conn.close()
        return count


if __name__ == "__main__":
    # Run Stage 0 scraper - reads from urls.txt
    print("\n" + "="*70)
    print("🕷️  STAGE 0: WEB SCRAPER")
    print("="*70)
    
    scraper = Stage0WebScraper(db_file="review_database.db")
    
    # Create table if not exists
    print("\n📁 Ensuring database table exists...")
    scraper.create_reviews_table()
    
    # Scrape from urls.txt
    print("\n🔄 Starting scraping from urls.txt...")
    total_saved = scraper.scrape_from_urls_file("urls.txt")
    
    # Show results
    total_reviews = scraper.get_review_count()
    
    print("\n" + "="*70)
    print("✅ SCRAPING COMPLETE!")
    print("="*70)
    print(f"📊 New reviews saved: {total_saved}")
    print(f"📊 Total reviews in database: {total_reviews}")
    print("\n🎯 Next step: Run the analysis!")
    print("   python main_langgraph.py")
    print("="*70 + "\n")