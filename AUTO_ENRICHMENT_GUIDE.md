# Auto-Enrichment Feature Guide

## 🎉 What Was Fixed

The auto-enrichment feature is now **fully functional**! Previously, it was only generating suggestions but not actually fetching content from external sources.

### Changes Made:

1. **Integrated Enrichment Engine** - The RAG pipeline now calls the enrichment engine to fetch content from Wikipedia and DuckDuckGo
2. **Automatic Re-search** - After enriching the knowledge base, the system automatically re-runs the search to include the new content in the answer
3. **Score Normalization** - Fixed the relevance score validation error by normalizing scores to be between 0 and 1

---

## 🚀 How Auto-Enrichment Works

When you enable "Auto-enrichment from external sources":

1. **Initial Search** - The system searches your uploaded documents
2. **Completeness Check** - The AI determines if the answer is complete
3. **Auto-Enrichment** - If incomplete, it automatically:
   - Searches Wikipedia for relevant articles
   - Searches the web using DuckDuckGo
   - Extracts and adds the content to your knowledge base
4. **Re-search** - Searches again with the newly enriched content
5. **Enhanced Answer** - Provides a better answer using both your documents and external sources

---

## 📝 How to Test It

### Test Case 1: Question Not in Your Documents

1. **Upload a document** about a specific topic (e.g., AWS Bedrock)
2. **Ask a question** that requires additional context not in your document
   - Example: "What are the latest AI regulations in Europe?"
3. **Enable auto-enrichment** checkbox
4. **Submit the query**
5. **Observe**: The system will fetch information from Wikipedia/web and include it in the answer

### Test Case 2: Partial Information

1. **Upload a document** with partial information
2. **Ask a comprehensive question** that needs more context
   - Example: "Compare AWS Bedrock with OpenAI's API in terms of pricing and features"
3. **Enable auto-enrichment**
4. **Submit the query**
5. **Observe**: The system will enrich with external pricing/feature information

---

## 🔍 What to Look For

### In the UI:

- **Enrichment Suggestions** section will show:
  - ✅ "Auto-enriched from: Wikipedia: [Article Name]"
  - ✅ "Auto-enriched from: Web: [Source Title]"
- **Higher confidence scores** after enrichment
- **More comprehensive answers** combining your docs + external sources

### In the Server Logs:

Look for these log messages:
```
INFO - Auto-enrichment enabled. Attempting to fetch external content
INFO - Fetched Wikipedia page: [Page Title]
INFO - Fetched web result: [Result Title]
INFO - Successfully enriched knowledge base with X items
INFO - Re-running search after auto-enrichment
INFO - Re-generated answer with enriched content
```

---

## 🛠️ Technical Details

### External Sources Used:

1. **Wikipedia** - For encyclopedic information
   - Uses the `wikipedia` Python package
   - Fetches article summaries
   - Includes source URLs

2. **DuckDuckGo Search** - For web search results
   - Uses the `duckduckgo-search` Python package
   - Fetches top search results
   - Includes snippets and URLs

### Enriched Content Storage:

- Enriched content is added to the same vector store as your documents
- Marked with `enriched: True` in metadata
- Includes source information (Wikipedia URL, web URL)
- Persists across sessions

---

## 🎯 Example Queries to Test

### Good Test Queries (will trigger enrichment):

1. **"What is quantum computing?"** - If you don't have quantum computing docs
2. **"Who invented the internet?"** - Historical information
3. **"What are the latest developments in AI?"** - Current events
4. **"Explain blockchain technology"** - Technical concepts
5. **"What is the capital of France?"** - General knowledge

### Queries That Won't Trigger Enrichment:

- Questions fully answered by your uploaded documents
- Questions where the AI is confident with existing information

---

## 📊 Monitoring Enrichment

### Check Enriched Documents:

The enriched content is stored in your vector database and will appear in future searches. You can identify enriched documents by:

- Filename starts with "wikipedia:" or "web_search:"
- Metadata includes `enriched: True`
- Includes `original_query` showing what triggered the enrichment

### View in Logs:

```bash
# Watch the logs in real-time
tail -f logs/app.log

# Or check the terminal where the server is running
```

---

## ⚙️ Configuration

### Adjust Enrichment Behavior:

Edit `app/services/rag_pipeline.py`:

```python
# Line ~217: Change max_sources to fetch more/fewer external sources
enrichment_results = await enrichment_engine.auto_enrich(
    query=query,
    missing_info=missing_info,
    max_sources=2  # Change this number (default: 2)
)
```

### Disable Specific Sources:

Edit `app/services/enrichment_engine.py`:

```python
# Set to False to disable
WIKIPEDIA_AVAILABLE = True  # Set to False to disable Wikipedia
DUCKDUCKGO_AVAILABLE = True  # Set to False to disable web search
```

---

## 🐛 Troubleshooting

### Auto-enrichment not working?

1. **Check the checkbox** - Make sure "Enable auto-enrichment" is checked
2. **Check logs** - Look for error messages in the server logs
3. **Test with a clear gap** - Ask a question that's definitely not in your docs
4. **Verify packages** - Ensure `wikipedia` and `duckduckgo-search` are installed

### No external content added?

- The AI might think your documents are sufficient (high confidence)
- The external sources might not have relevant information
- Check if `is_complete: false` in the response

### Rate limiting?

- Wikipedia and DuckDuckGo have rate limits
- The system includes delays to be respectful
- If you hit limits, wait a few minutes and try again

---

## 🎓 Best Practices

1. **Use for knowledge gaps** - Best when your documents don't have complete information
2. **Verify external content** - Always verify important information from external sources
3. **Monitor storage** - Enriched content accumulates in your vector store
4. **Clear periodically** - Consider clearing enriched content if it becomes stale

---

## 📈 Future Enhancements

Potential improvements:
- Add more external sources (arXiv, PubMed, etc.)
- Implement source credibility scoring
- Add user control over which sources to use
- Implement content freshness checks
- Add citation formatting

---

## ✅ Summary

The auto-enrichment feature is now **fully operational** and will:
- ✅ Automatically detect incomplete answers
- ✅ Fetch relevant content from Wikipedia and the web
- ✅ Add it to your knowledge base
- ✅ Re-generate answers with the enriched content
- ✅ Show you what sources were used

**Try it now by asking a question that's not fully covered in your documents!** 🚀

