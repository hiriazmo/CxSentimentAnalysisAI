---
title: CxSentimentAnalysisAI
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
sdk_version: 1.52.2
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).


---
title: Review Intelligence System
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🎯 Review Intelligence System

**Multi-Agent AI-Powered Review Analysis Platform**

Analyze customer reviews from App Store and Play Store with 7 specialized AI models working in parallel.

## 🚀 What This Does

This application provides **intelligent, multi-stage analysis** of customer reviews using a sophisticated AI pipeline:

- 📱 Scrapes reviews from **App Store** and **Play Store**
- 🤖 Classifies reviews by **type**, **department**, and **priority**
- 😊 Analyzes **sentiment** with dual BERT models
- 👥 Identifies **user types** and **emotional states**
- 📊 Generates **actionable insights** and **batch analytics**
- 🎯 Routes issues to appropriate teams

Perfect for **product managers**, **UX teams**, **support teams**, and **business analysts**.

## ✨ Key Features

### 4-Stage AI Pipeline

| Stage | What It Does | Models Used |
|-------|-------------|-------------|
| **Stage 0** | Web Scraping | App Store RSS & Play Store API |
| **Stage 1** | Classification | Qwen 72B + Llama 3B + Llama 70B |
| **Stage 2** | Sentiment | Twitter-RoBERTa + BERTweet |
| **Stage 3** | Synthesis | Llama 70B |
| **Stage 4** | Analytics | Statistical aggregation |

### What You Get

- ✅ **Review Type**: praise, complaint, suggestion, question, bug_report
- ✅ **Department**: engineering, ux, support, business
- ✅ **Priority**: critical, high, medium, low
- ✅ **Sentiment**: POSITIVE, NEUTRAL, NEGATIVE (with confidence)
- ✅ **Emotion**: joy, satisfaction, frustration, anger, disappointment
- ✅ **User Type**: new_user, regular_user, power_user, churning_user
- ✅ **Actions**: Specific recommendations for each review
- ✅ **Analytics**: Churn risk, critical issues, quick wins

## 🎬 How to Use

### Step 1: Get HuggingFace API Key

1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token with **Read** access
3. Copy token (starts with `hf_`)

### Step 2: Enter App URLs

**App Store:**
- Format: Just the app ID number
- Example: `1022164656`
- Find in URL: `apps.apple.com/app/id1022164656`

**Play Store:**
- Format: Package name
- Example: `com.disney.wdpro.dlr`
- Find in URL: `play.google.com/store/apps/details?id=com.disney.wdpro.dlr`

### Step 3: Run Analysis

1. Paste HuggingFace API key
2. Enter URLs (one per line)
3. Choose reviews per app (5-100)
4. Click **"🚀 Start Analysis"**
5. Wait ~7 seconds per review
6. View results!

### Step 4: Manage Database

- **Reset Database**: Click when analyzing different apps
- **Keep Database**: Don't reset to track trends over time

## 💡 Use Cases

**Product Management**
- Identify critical issues
- Prioritize feature requests
- Track sentiment trends

**UX/Design Teams**
- Find usability issues
- Discover improvement ideas
- Understand user emotions

**Support Teams**
- Route issues automatically
- Categorize requests
- Identify quick wins

**Business Analytics**
- Measure satisfaction
- Calculate churn risk
- Track competitive position

## 🏗️ Technical Details

**AI Models:**
1. Qwen/Qwen2.5-72B-Instruct - Classification
2. meta-llama/Llama-3.2-3B-Instruct - User analysis
3. meta-llama/Llama-3.3-70B-Instruct - Synthesis
4. cardiffnlp/twitter-roberta-base-sentiment-latest - Sentiment
5. finiteautomata/bertweet-base-sentiment-analysis - Validation
6. meta-llama/Llama-3.1-70B-Instruct - Final reasoning

**Technology Stack:**
- Frontend: Streamlit
- AI: LangGraph + HuggingFace Inference API
- Database: SQLite (49 columns)
- Visualization: Plotly

**Performance:**
- ⚡ ~7 seconds per review
- 🔄 Parallel processing
- 🎯 100% model agreement

## 📊 Sample Output

```
Dashboard Metrics:
📊 Total Reviews: 20
😊 Positive: 15 (75%)
😞 Negative: 4 (20%)
🚨 Critical: 0
📉 Churn Risk: 7.5%

Department Routing:
🏢 Engineering: 4
🎨 UX: 9
💼 Business: 6
```

## 🔐 Privacy & Data

- ✅ All processing on HuggingFace servers
- ✅ No permanent data storage
- ✅ Public reviews only
- ✅ Reset database anytime
- ✅ Export your data

## 🙋 Support

For issues:
1. Check HuggingFace API key is valid
2. Verify URL format is correct
3. Try resetting database
4. Check internet connection

---

**Made with ❤️ for Product Teams**

⭐ Star this space if you find it useful!