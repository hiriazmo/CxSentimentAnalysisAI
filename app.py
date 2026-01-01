"""
HuggingFace Spaces - Review Intelligence System (Streamlit)
Complete app with URL input, progress tracking, and interactive dashboard
FIXED VERSION - Better UI contrast + Proper field mapping
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from typing import List, Dict, Optional
import time

from gradio_pipeline import GradioPipeline


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Review Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED Custom CSS - Better Contrast
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    
    /* FIXED: Metric cards with better contrast */
    .stMetric {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #60a5fa;
    }
    
    .stMetric label {
        color: #dbeafe !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 36px !important;
        font-weight: bold !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #93c5fd !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    
    .success-box {
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }




    
    .success-box h1 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Better table styling */
    .dataframe {
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

if 'results' not in st.session_state:
    st.session_state.results = None

if 'insights' not in st.session_state:
    st.session_state.insights = None

if 'scraped_count' not in st.session_state:
    st.session_state.scraped_count = 0



# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_reviews_streamlit(app_store_urls: str, play_store_urls: str, 
                              hf_api_key: str, review_limit: int):
    """
    Process reviews with Streamlit progress tracking
    """
    
    # Validate inputs
    if not hf_api_key or not hf_api_key.strip():
        st.error("❌ Please provide your HuggingFace API key")
        return False
    
    if not app_store_urls.strip() and not play_store_urls.strip():
        st.error("❌ Please provide at least one App Store or Play Store URL")
        return False
    
    try:
        # Set API key
        os.environ['HUGGINGFACE_API_KEY'] = hf_api_key.strip()
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize pipeline
        status_text.text("🚀 Initializing pipeline...")
        progress_bar.progress(5)
        pipeline = GradioPipeline(review_limit=review_limit)
        
        # Parse URLs
        app_urls = [url.strip() for url in app_store_urls.split('\n') if url.strip()]
        play_urls = [url.strip() for url in play_store_urls.split('\n') if url.strip()]
        
        # Stage 0: Scraping
        status_text.text("🕷️ Scraping reviews from stores...")
        progress_bar.progress(10)
        
        scraped_count = 0
        total_apps = len(app_urls) + len(play_urls)
        
        for i, app_id in enumerate(app_urls, 1):
            status_text.text(f"🍎 Scraping App Store ({i}/{total_apps}): {app_id}")
            reviews = pipeline.scraper.scrape_app_store_rss(app_id, country="ae", limit=review_limit)
            saved = pipeline.scraper.save_reviews_to_db(reviews)
            scraped_count += saved
            progress_bar.progress(10 + int(20 * i / total_apps))
            time.sleep(1)
        
        for i, package in enumerate(play_urls, 1):
            status_text.text(f"🤖 Scraping Play Store ({i}/{total_apps}): {package}")
            reviews = pipeline.scraper.scrape_play_store_api(package, country="ae", limit=review_limit)
            saved = pipeline.scraper.save_reviews_to_db(reviews)
            scraped_count += saved
            progress_bar.progress(10 + int(20 * (len(app_urls) + i) / total_apps))
            time.sleep(1)
        
        if scraped_count == 0:
            st.warning("⚠️ No reviews scraped. Please check your URLs and try again.")
            progress_bar.empty()
            status_text.empty()
            return False
        
        st.session_state.scraped_count = scraped_count
        
        # Stage 1-3: Processing
        status_text.text("🤖 Processing reviews with AI models...")
        progress_bar.progress(30)
        
        reviews = pipeline.db.get_pending_reviews(limit=review_limit)
        total_reviews = len(reviews)
        
        print(f"📊 DEBUG: Found {total_reviews} reviews to process")
        
        processed_states = []
        
        for i, review in enumerate(reviews, 1):
            review_id = review.get('review_id', 'unknown')[:20]
            status_text.text(f"🤖 Processing review {i}/{total_reviews}: {review_id}...")
            progress_bar.progress(30 + int(60 * i / total_reviews))
            
            try:
                from langgraph_state import create_initial_state
                state = create_initial_state(review)
                config = {"configurable": {"thread_id": f"review_{review.get('review_id')}"}}
                final_state = pipeline.review_graph.invoke(state, config=config)
                
                # Convert to dict
                state_dict = dict(final_state)
                processed_states.append(state_dict)
                
                # DEBUG: Print what we got
                print(f"✅ Processed {review_id}:")
                print(f"   Type: {state_dict.get('classification_type', 'MISSING')}")
                print(f"   Dept: {state_dict.get('department', 'MISSING')}")
                print(f"   Sentiment: {state_dict.get('final_sentiment', 'MISSING')}")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing review: {str(e)}")
                print(f"❌ ERROR: {e}")
                import traceback
                print(traceback.format_exc())
                continue
        
        if len(processed_states) == 0:
            st.error("❌ No reviews were processed successfully.")
            progress_bar.empty()
            status_text.empty()
            return False
        
        # Stage 4: Batch Analysis
        status_text.text("📊 Generating batch insights...")
        progress_bar.progress(90)

        
        insights = pipeline.analyze_batch(processed_states)

        
        # Store in session state
        st.session_state.results = processed_states
        st.session_state.insights = insights
        st.session_state.processing_complete = True
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False



# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_summary_section(scraped_count: int, results: List[Dict], insights: Dict):
    """Create summary metrics section"""
    
    total = len(results)
    positive = insights.get('sentiment_distribution', {}).get('POSITIVE', 0)
    neutral = insights.get('sentiment_distribution', {}).get('NEUTRAL', 0)
    negative = insights.get('sentiment_distribution', {}).get('NEGATIVE', 0)
    critical = insights.get('priority_distribution', {}).get('critical', 0)
    churn_risk = insights.get('churn_risk', 0)
    
    # Success header
    st.markdown(
        f"""
        <div class="success-box">
            <h1 style="margin: 0;">✅ Analysis Complete!</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Review Intelligence System Results
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Metrics with better styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Reviews", total, f"Scraped: {scraped_count}")
    
    with col2:
        pos_pct = (positive / total * 100) if total > 0 else 0
        st.metric("😊 Positive", positive, f"{pos_pct:.1f}%")
    
    with col3:
        neg_pct = (negative / total * 100) if total > 0 else 0
        st.metric("😞 Negative", negative, f"{neg_pct:.1f}%")
    
    with col4:
        st.metric("🚨 Critical", critical, "⚠️" if critical > 0 else "✅")
    
    with col5:
        st.metric("📉 Churn Risk", f"{churn_risk:.1f}%", 
                 "🔴 High" if churn_risk > 30 else "🟢 Low")
    
    # Recommendations
    if insights.get('recommendations'):
        st.markdown("### 💡 Key Recommendations")
        for rec in insights.get('recommendations', []):
            st.info(rec)



def create_sentiment_chart(insights: Dict):
    """Create sentiment distribution donut chart"""
    sentiment_dist = insights.get('sentiment_distribution', {})
    
    labels = list(sentiment_dist.keys())
    values = list(sentiment_dist.values())
    colors = ['#10b981', '#f59e0b', '#ef4444']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="😊 Sentiment Distribution",
        showlegend=True,
        height=400
    )
    
    return fig



def create_priority_chart(insights: Dict):
    """Create priority distribution bar chart"""
    priority_dist = insights.get('priority_distribution', {})
    
    priority_order = ['critical', 'high', 'medium', 'low']
    labels = [p for p in priority_order if p in priority_dist]
    values = [priority_dist.get(p, 0) for p in labels]
    colors = ['#dc2626', '#f59e0b', '#3b82f6', '#10b981']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors[:len(labels)],
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="🎯 Priority Levels",
        xaxis_title="Priority",
        yaxis_title="Count",
        height=400
    )
    
    return fig



def create_department_chart(insights: Dict):
    """Create department routing horizontal bar chart"""
    dept_dist = insights.get('department_distribution', {})
    
    labels = list(dept_dist.keys())
    values = list(dept_dist.values())
    
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color='#667eea',
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="🏢 Department Routing",
        xaxis_title="Number of Issues",
        yaxis_title="Department",
        height=400
    )
    
    return fig








def create_emotion_chart(insights: Dict):
    """Create emotion distribution chart"""
    emotion_dist = insights.get('emotion_distribution', {})
    
    labels = list(emotion_dist.keys())
    values = list(emotion_dist.values())
    
    fig = px.bar(
        x=labels,
        y=values,
        labels={'x': 'Emotion', 'y': 'Count'},
        color=values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title="😊 Emotional Analysis",
        xaxis_title="Emotion Type",
        yaxis_title="Number of Reviews",
        height=300,
        showlegend=False
    )
    
    return fig



def create_reviews_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    FIXED: Create DataFrame with proper field mapping
    Checks both state field names AND database field names
    """
    
    df_data = []
    for review in results:
        # FIXED: Check state fields FIRST, fall back to database fields
        df_data.append({
            'Review ID': review.get('review_id', 'N/A')[:20],
            'Rating': review.get('rating', 0),
            'Review': (review.get('review_text', 'N/A') or '')[:100] + '...',
            'Sentiment': review.get('final_sentiment', review.get('stage3_final_sentiment', 'N/A')),
            'Type': review.get('classification_type', review.get('stage1_llm1_type', 'N/A')),
            'Department': review.get('department', review.get('stage1_llm1_department', 'N/A')),
            'Priority': review.get('priority', review.get('stage1_llm1_priority', 'N/A')),
            'Emotion': review.get('emotion', review.get('stage1_llm2_emotion', 'N/A')),
            'Needs Review': '🚨 Yes' if review.get('needs_human_review', review.get('stage3_needs_human_review')) else '✅ No'
        })
    
    return pd.DataFrame(df_data)



# ============================================================================
# MAIN APP
# ============================================================================


def main():
    """Main Streamlit app"""
    
    # Title
    st.title("🎯 Review Intelligence System")
    st.markdown("### Multi-Stage AI Analysis Dashboard")
    st.markdown("Powered by **LangGraph** + **HuggingFace** • 4-Stage Processing Pipeline")
    st.markdown("---")
    
    # Sidebar - Input or View Mode
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        if st.session_state.processing_complete:
            st.success("✅ Analysis Complete!")
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.results = None
                st.session_state.insights = None
                st.rerun()
        else:
            st.info("👈 Enter URLs below to start")
        
        # Database Management Section
        st.markdown("---")
        st.markdown("### 🗄️ Database Management")
        
        # Show current database stats
        try:
            from database_enhanced import EnhancedDatabase
            db = EnhancedDatabase()
            db.connect()
            cursor = db.conn.execute("SELECT COUNT(*) FROM reviews")
            count = cursor.fetchone()[0]
            db.close()
            
            st.metric("Total Reviews in DB", count)
            
            if count > 0:
                st.caption(f"💡 Database contains {count} reviews from previous analyses")
        except:
            st.metric("Total Reviews in DB", 0)
        
        # Reset Database Button
        if st.button("🗑️ Reset Database", type="secondary", use_container_width=True, 
                    help="Delete all reviews and start fresh. Useful when switching between different apps."):
            import os
            if os.path.exists("review_database.db"):
                os.remove("review_database.db")
                st.success("✅ Database deleted! Ready for fresh analysis.")
                time.sleep(1)
                st.rerun()
            else:
                st.info("ℹ️ No database found to delete")
    
    # Main content - Input or Results
    if not st.session_state.processing_complete:
        show_input_form()
    else:
        show_results_dashboard()






def show_input_form():
    """Show input form for URLs and API key"""
    
    st.markdown("### 📝 Step 1: Enter Store URLs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🍎 App Store IDs")
        st.markdown(
            """
            **Format:** Just paste the app ID
            - Example: `1158907446` (UAE)
            - Example: `1234567890` (US)
            """
        )
        app_store_urls = st.text_area(
            "App Store IDs (one per line)",
            placeholder="1158907446\n1234567890",
            height=150,
            key="app_urls"
        )
    
    with col2:
        st.markdown("#### 🤖 Play Store Packages")
        st.markdown(
            """
            **Format:** Package name
            - Example: `com.yas.app`
            - Example: `com.company.app`
            """
        )
        play_store_urls = st.text_area(
            "Play Store Package Names (one per line)",
            placeholder="com.yas.app\ncom.company.app",
            height=150,
            key="play_urls"
        )
    
    st.markdown("---")
    st.markdown("### 🔑 Step 2: Configure Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        hf_api_key = st.text_input(
            "🔑 HuggingFace API Key",
            type="password",
            placeholder="hf_...",
            help="Get your key from: https://huggingface.co/settings/tokens",
            key="hf_key"
        )
    
    with col2:
        review_limit = st.slider(
            "📊 Reviews per App",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="More reviews = longer processing time",
            key="review_limit"
        )
    
    st.markdown("---")
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                success = process_reviews_streamlit(
                    app_store_urls,
                    play_store_urls,
                    hf_api_key,
                    review_limit
                )
                
                if success:
                    st.balloons()
                    st.rerun()
    
    # Documentation
    with st.expander("📚 How to Use"):
        st.markdown("""
        ### 📖 Quick Guide
        
        **1. Get HuggingFace API Key:**
        - Visit: https://huggingface.co/settings/tokens
        - Create new token (Read access)
        - Copy token (starts with `hf_`)
        
        **2. Enter URLs:**
        - **App Store**: Just the ID number (e.g., `1234567890`)
        - **Play Store**: Package name (e.g., `com.company.app`)
        - One per line
        
        **3. Click Start:**
        - Watch progress bar
        - Wait for completion (~7 sec per review)
        - View results automatically
        
        ### 🏗️ What Happens:
        - 🕷️ **Stage 0**: Scrapes reviews from stores
        - 🤖 **Stage 1**: Classifies with 3 AI models (Type, Department, Priority)
        - 😊 **Stage 2**: Analyzes sentiment with dual BERT models
        - 📊 **Stage 3**: Synthesizes insights and recommendations
        - 💡 **Stage 4**: Generates batch analytics
        
        ### ⚡ Performance:
        - ~7 seconds per review
        - 7 AI models working together
        - Parallel execution for speed
        """)



def show_results_dashboard():
    """Show results dashboard with charts and tables"""
    
    results = st.session_state.results
    insights = st.session_state.insights
    scraped_count = st.session_state.scraped_count
    
    # Summary section
    create_summary_section(scraped_count, results, insights)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sentiment Analysis",
        "🚨 Critical Issues",
        "📋 All Reviews",
        "📥 Export"
    ])
    
    # TAB 1: Sentiment Analysis
    with tab1:
        st.header("📊 Sentiment Analysis Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sentiment = create_sentiment_chart(insights)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            fig_priority = create_priority_chart(insights)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        st.markdown("### 🏢 Department Routing")
        fig_dept = create_department_chart(insights)
        st.plotly_chart(fig_dept, use_container_width=True)
        
        st.markdown("### 😊 Emotional Analysis")
        fig_emotion = create_emotion_chart(insights)
        st.plotly_chart(fig_emotion, use_container_width=True)
    
    # TAB 2: Critical Issues
    with tab2:
        st.header("🚨 Critical Issues Requiring Attention")
        
        # Filter critical reviews
        critical_reviews = [
            r for r in results
            if (r.get('priority') == 'critical' or 
                r.get('stage1_llm1_priority') == 'critical' or
                r.get('needs_human_review', r.get('stage3_needs_human_review')) or
                (r.get('final_sentiment', r.get('stage3_final_sentiment')) == 'NEGATIVE' and r.get('rating', 5) <= 2))
        ]
        
        if len(critical_reviews) == 0:
            st.success("✅ No critical issues found! All reviews are in good shape.")
        else:
            st.warning(f"Found {len(critical_reviews)} critical issues")
            
            for review in critical_reviews:
                with st.expander(
                    f"⚠️ {review.get('review_id', 'Unknown')[:30]} - "
                    f"Rating: {review.get('rating', 'N/A')}/5"
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Review Text:**")
                        st.write(review.get('review_text', 'No text available'))
                        
                        st.markdown("**Reasoning:**")
                        reasoning = review.get('reasoning', review.get('stage3_reasoning', 'No reasoning available'))
                        st.info(reasoning)
                    
                    with col2:
                        st.markdown("**Classification:**")
                        st.write(f"📌 Type: {review.get('classification_type', review.get('stage1_llm1_type', 'N/A'))}")
                        st.write(f"🏢 Department: {review.get('department', review.get('stage1_llm1_department', 'N/A'))}")
                        st.write(f"🎯 Priority: {review.get('priority', review.get('stage1_llm1_priority', 'N/A'))}")
                        st.write(f"😔 Emotion: {review.get('emotion', review.get('stage1_llm2_emotion', 'N/A'))}")
                        st.write(f"💭 Sentiment: {review.get('final_sentiment', review.get('stage3_final_sentiment', 'N/A'))}")
                        
                        st.markdown("**Action:**")
                        action = review.get('action_recommendation', review.get('stage3_action_recommendation', 'No action specified'))
                        st.error(action)
    
    # TAB 3: All Reviews
    with tab3:
        st.header("📋 Detailed Review Analysis")
        
        # Create DataFrame
        df = create_reviews_dataframe(results)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=df['Sentiment'].unique(),
                default=df['Sentiment'].unique()
            )
        
        with col2:
            dept_filter = st.multiselect(
                "Filter by Department",
                options=df['Department'].unique(),
                default=df['Department'].unique()
            )
        
        with col3:
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=df['Priority'].unique(),
                default=df['Priority'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['Sentiment'].isin(sentiment_filter)) &
            (df['Department'].isin(dept_filter)) &
            (df['Priority'].isin(priority_filter))
        ]
        
        st.info(f"Showing {len(filtered_df)} of {len(df)} reviews")
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=600
        )
    
    # TAB 4: Export
    with tab4:
        st.header("📥 Export Results")
        
        st.markdown("### Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 CSV Export")
            st.write("Download complete analysis with all classifications")
            
            df = create_reviews_dataframe(results)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 📋 JSON Export")
            st.write("Download raw data with all details")
            
            import json
            json_data = json.dumps({
                'results': results,
                'insights': insights,
                'scraped_count': scraped_count,
                'export_date': datetime.now().isoformat()
            }, indent=2)
            
            st.download_button(
                label="📥 Download JSON Data",
                data=json_data,
                file_name=f"review_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reviews Analyzed", len(results))
        
        with col2:
            positive = insights.get('sentiment_distribution', {}).get('POSITIVE', 0)
            total = len(results)
            pct = (positive / total * 100) if total > 0 else 0
            st.metric("Positive Rate", f"{pct:.1f}%")
        
        with col3:
            critical = insights.get('priority_distribution', {}).get('critical', 0)
            st.metric("Critical Issues", critical)


# ============================================================================
# FOOTER
# ============================================================================

def show_footer():
    """Show footer with credits"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Powered by Multi-Stage AI Pipeline | 
            Stage 1: Classification (Qwen, Mistral, Llama) | 
            Stage 2: Sentiment (Twitter-BERT) | 
            Stage 3: Finalization (Llama 70B) | 
            Stage 4: Batch Analysis</p>
            <p>Built with ❤️ using LangGraph + HuggingFace + Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
    show_footer()