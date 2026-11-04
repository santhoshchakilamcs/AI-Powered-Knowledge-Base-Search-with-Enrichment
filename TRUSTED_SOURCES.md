# 🔒 Trusted External Sources for Auto-Enrichment

## Overview

The AI-Powered Knowledge Base now supports **auto-enrichment from multiple trusted external sources**. When information is missing from your uploaded documents, the system can automatically fetch relevant content from verified, authoritative sources.

---

## 🌟 Available Trusted Sources

### 1. **Wikipedia** 📚
- **Type**: General Knowledge Encyclopedia
- **Priority**: 1 (Highest)
- **Best For**: General topics, definitions, historical information
- **API**: Wikipedia API (free, no key required)
- **Status**: ✅ Enabled by default

**Example Use Cases:**
- "What is quantum computing?"
- "Explain blockchain technology"
- "Who was Marie Curie?"

---

### 2. **arXiv** 🎓
- **Type**: Academic Papers & Research
- **Priority**: 2
- **Best For**: Scientific research, technical papers, academic topics
- **API**: arXiv.org API (free, no key required)
- **Status**: ✅ Enabled by default
- **Coverage**: Physics, Mathematics, Computer Science, Biology, Finance, Statistics

**Example Use Cases:**
- "Latest research on neural networks"
- "Quantum entanglement papers"
- "Machine learning optimization techniques"

---

### 3. **PubMed** 🏥
- **Type**: Medical & Health Research
- **Priority**: 3
- **Best For**: Medical information, health research, clinical studies
- **API**: NCBI E-utilities API (free, no key required)
- **Status**: ✅ Enabled by default
- **Coverage**: Biomedical literature, clinical trials, health sciences

**Example Use Cases:**
- "COVID-19 vaccine research"
- "Treatment options for diabetes"
- "Latest cancer research findings"

---

### 4. **Web Search** 🌐
- **Type**: General Web Search
- **Priority**: 4 (Fallback)
- **Best For**: Current events, general information, recent news
- **API**: DuckDuckGo Search (free, privacy-focused)
- **Status**: ✅ Enabled by default

**Example Use Cases:**
- "Latest news on AI regulations"
- "Current stock market trends"
- "Recent technology announcements"

---

## 🎯 How It Works

### Automatic Source Selection

The system automatically selects the most appropriate trusted source based on:

1. **Query Context**: Analyzes your question to determine the topic
2. **Priority Order**: Tries higher-priority sources first
3. **Availability**: Uses only enabled sources
4. **Relevance**: Fetches content that fills knowledge gaps

### Enrichment Process

```
User Query → Check Documents → Missing Info Detected
    ↓
Select Trusted Sources (by priority)
    ↓
Fetch from Wikipedia → arXiv → PubMed → Web Search
    ↓
Add to Knowledge Base → Re-run Search → Enhanced Answer
```

---

## 🔧 Configuration

### Enable/Disable Sources

Edit `app/services/enrichment_engine.py`:

```python
TRUSTED_SOURCES = {
    'wikipedia': {
        'enabled': True,  # Set to False to disable
        'priority': 1
    },
    'arxiv': {
        'enabled': True,  # Set to False to disable
        'priority': 2
    },
    'pubmed': {
        'enabled': True,  # Set to False to disable
        'priority': 3
    },
    'web_search': {
        'enabled': True,  # Set to False to disable
        'priority': 4
    }
}
```

### Adjust Max Sources

In `app/services/rag_pipeline.py`:

```python
enrichment_results = await enrichment_engine.auto_enrich(
    query=query,
    missing_info=missing_info,
    max_sources=3  # Change this number (1-5 recommended)
)
```

---

## 📊 Source Metadata

Each enriched document includes:

- **Source Type**: `wikipedia`, `arxiv`, `pubmed`, or `web_search`
- **Title**: Original article/paper title
- **URL**: Direct link to the source
- **Authors**: (for arXiv and PubMed)
- **Enriched Flag**: `enriched: true` in metadata

---

## 🎨 UI Display

### Enrichment Suggestions
```
🎯 Enrichment Suggestions

✅ Auto-enriched from arXiv: Neural Network Optimization
   🔗 View Source (link to arXiv paper)

✅ Auto-enriched from PubMed: COVID-19 Vaccine Efficacy
   🔗 View Source (link to PubMed article)
```

### Sources Section
```
📚 Sources

🌐 arXiv: Neural Network Optimization  [95% relevant] [External Source]
   Neural networks are computational models inspired by...
   🔗 View Full Source (link to paper)

🌐 PubMed: COVID-19 Vaccine Efficacy  [92% relevant] [External Source]
   This study examines the efficacy of mRNA vaccines...
   🔗 View Full Source (link to article)
```

---

## 🚀 Usage Examples

### Example 1: Academic Research

**Query**: "Explain transformer architecture in deep learning"

**Auto-Enrichment**:
1. ✅ Wikipedia: "Transformer (machine learning model)"
2. ✅ arXiv: "Attention Is All You Need" (original paper)
3. ✅ arXiv: "BERT: Pre-training of Deep Bidirectional Transformers"

**Result**: Comprehensive answer with academic sources and citations

---

### Example 2: Medical Information

**Query**: "What are the side effects of statins?"

**Auto-Enrichment**:
1. ✅ Wikipedia: "Statin"
2. ✅ PubMed: "Adverse Effects of Statin Therapy"
3. ✅ PubMed: "Long-term Safety of Statins"

**Result**: Evidence-based medical information with research citations

---

### Example 3: Current Events

**Query**: "Latest developments in quantum computing"

**Auto-Enrichment**:
1. ✅ Wikipedia: "Quantum computing"
2. ✅ arXiv: Recent quantum computing papers
3. ✅ Web Search: Latest news articles

**Result**: Up-to-date information from multiple trusted sources

---

## 🔐 Privacy & Security

### Data Handling
- ✅ All API calls are made server-side
- ✅ No user data is sent to external sources
- ✅ Only search queries are transmitted
- ✅ Enriched content is stored locally in your vector database

### API Rate Limits
- **Wikipedia**: No strict limits, respectful delays implemented
- **arXiv**: No authentication required, 0.5s delay between requests
- **PubMed**: No authentication required, 0.5s delay between requests
- **DuckDuckGo**: Rate-limited by the service, respectful usage

---

## 🛠️ Adding Custom Trusted Sources

### Step 1: Define the Source

Add to `TRUSTED_SOURCES` in `enrichment_engine.py`:

```python
'your_source': {
    'enabled': True,
    'name': 'Your Source Name',
    'description': 'Description of the source',
    'api_url': 'https://api.yoursource.com',
    'priority': 5
}
```

### Step 2: Implement the Enrichment Method

```python
async def _enrich_from_your_source(
    self,
    query: str,
    missing_info: List[str],
    max_sources: int
) -> Dict[str, List]:
    """Fetch information from your custom source."""
    results = {'sources': [], 'content': []}
    
    # Your implementation here
    # Make API calls, parse responses, etc.
    
    return results
```

### Step 3: Add to auto_enrich Method

```python
elif source_type == 'your_source':
    results = await self._enrich_from_your_source(query, missing_info, remaining)
```

---

## 📈 Monitoring & Logs

Check server logs for enrichment activity:

```
INFO - Auto-enrichment enabled. Attempting to fetch external content
INFO - Fetched arXiv paper: Attention Is All You Need
INFO - Fetched PubMed article: COVID-19 Vaccine Efficacy
INFO - Successfully enriched knowledge base with 3 items from trusted sources
INFO - Re-running search after auto-enrichment
INFO - Re-generated answer with enriched content. New confidence: 0.95
```

---

## ✅ Best Practices

1. **Enable Relevant Sources**: Only enable sources relevant to your domain
2. **Set Appropriate Priorities**: Higher priority = tried first
3. **Monitor API Usage**: Check logs for rate limiting issues
4. **Review Enriched Content**: Verify quality of auto-enriched information
5. **Clear Old Enrichments**: Periodically clean up outdated enriched content

---

## 🆘 Troubleshooting

### Issue: No enrichment happening
- ✅ Check that auto-enrichment checkbox is enabled in UI
- ✅ Verify sources are enabled in configuration
- ✅ Check server logs for errors

### Issue: Irrelevant sources fetched
- ✅ Adjust source priorities
- ✅ Disable irrelevant sources
- ✅ Refine your query to be more specific

### Issue: API errors
- ✅ Check internet connectivity
- ✅ Verify API endpoints are accessible
- ✅ Check for rate limiting in logs

---

## 📚 References

- **Wikipedia API**: https://www.mediawiki.org/wiki/API:Main_page
- **arXiv API**: https://arxiv.org/help/api/
- **PubMed E-utilities**: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **DuckDuckGo Search**: https://github.com/deedy5/duckduckgo_search

---

**Last Updated**: 2025-10-07
**Version**: 2.0 - Multi-Source Trusted Enrichment

