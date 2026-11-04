# Quick Start Guide

Get your RAG Knowledge Base up and running in 5 minutes!

## Prerequisites

- **Python 3.8+** installed
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Terminal/Command Line** access

## Step 1: Setup (2 minutes)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Generate a `.env` file

## Step 2: Configure (1 minute)

Open the `.env` file and add your OpenAI API key:

```bash
# Edit .env file
nano .env

# Or use your preferred editor
code .env
```

Add your API key:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Save and close the file.

## Step 3: Run (1 minute)

```bash
# Make run script executable
chmod +x run.sh

# Start the application
./run.sh
```

You should see:
```
🚀 Starting AI-Powered Knowledge Base...
🌐 Starting server on http://localhost:8000
📚 API documentation: http://localhost:8000/docs
```

## Step 4: Use (1 minute)

### Option A: Web Interface

1. Open your browser to: **http://localhost:8000**
2. Upload a document (PDF, TXT, or DOCX)
3. Ask a question in the search box
4. Get your AI-generated answer!

### Option B: API

```bash
# Upload a document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@examples/sample_document.txt"

# Ask a question
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?"}'
```

## Step 5: Test with Sample Data

```bash
# Run the example test script
python examples/test_api.py
```

This will:
- Upload the sample document
- Run several test queries
- Demonstrate all features
- Show you how the system works

## What You Get

### ✅ Core Features
- **Document Upload**: PDF, TXT, DOCX support
- **Natural Language Search**: Ask questions in plain English
- **AI Answers**: GPT-4 powered responses
- **Confidence Scores**: Know how reliable each answer is
- **Source References**: See which documents were used

### ✅ Advanced Features
- **Completeness Detection**: System tells you when info is missing
- **Enrichment Suggestions**: Get tips to improve your knowledge base
- **Auto-Enrichment**: Fetch info from Wikipedia/web automatically
- **Rating System**: Rate answers to improve the system

## Example Workflow

### 1. Upload Documents
```bash
# Upload company handbook
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@handbook.pdf"

# Upload policy document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@policies.docx"
```

### 2. Ask Questions
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vacation policy?",
    "top_k": 5,
    "enable_auto_enrichment": false
  }'
```

### 3. Get Structured Response
```json
{
  "query": "What is the vacation policy?",
  "answer": "According to the employee handbook, new employees receive 15 days of vacation per year...",
  "confidence": 0.92,
  "is_complete": true,
  "sources": [
    {
      "document_name": "handbook.pdf",
      "relevance_score": 0.95,
      "content": "Vacation Policy: New employees..."
    }
  ],
  "missing_info": [],
  "enrichment_suggestions": []
}
```

### 4. Rate the Answer
```bash
curl -X POST "http://localhost:8000/api/rate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vacation policy?",
    "answer": "According to...",
    "rating": 5,
    "feedback": "Very helpful!"
  }'
```

## Understanding the Response

### Confidence Score
- **0.8 - 1.0** 🟢 High confidence - answer is well-supported
- **0.5 - 0.8** 🟡 Medium confidence - answer is partially supported
- **0.0 - 0.5** 🔴 Low confidence - limited information

### Completeness
- **Complete** ✅ All information available
- **Incomplete** ⚠️ Some information missing (check `missing_info`)

### When Answer is Incomplete
The system will:
1. Tell you what information is missing
2. Suggest documents to upload
3. Offer to fetch info from external sources (if enabled)

## Common Use Cases

### 1. Company Knowledge Base
Upload: Employee handbooks, policies, procedures
Ask: "What is the remote work policy?", "How do I request time off?"

### 2. Product Documentation
Upload: User guides, API docs, FAQs
Ask: "How do I configure X?", "What are the system requirements?"

### 3. Research Assistant
Upload: Research papers, articles, notes
Ask: "What are the main findings?", "How does X relate to Y?"

### 4. Legal/Compliance
Upload: Contracts, regulations, guidelines
Ask: "What are the requirements for X?", "Is Y compliant?"

## Tips for Best Results

### 📄 Document Upload
- Upload related documents together
- Use clear, descriptive filenames
- Ensure documents contain actual text (not just images)
- Break very large documents into sections

### 🔍 Asking Questions
- Be specific and clear
- Use natural language
- Include context when needed
- Try different phrasings if needed

### 🎯 Improving Accuracy
- Upload more relevant documents
- Use enrichment suggestions
- Enable auto-enrichment for general knowledge
- Rate answers to help improve the system

## Troubleshooting

### "No documents found"
**Solution**: Upload documents first before searching

### Low confidence scores
**Solution**: 
- Upload more relevant documents
- Enable auto-enrichment
- Rephrase your question

### "Connection refused"
**Solution**: Make sure the server is running (`./run.sh`)

### "Invalid API key"
**Solution**: Check your `.env` file has the correct OpenAI API key

## Next Steps

### 📚 Learn More
- Read [USAGE.md](USAGE.md) for detailed usage guide
- Check [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for architecture
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment

### 🔧 Customize
Edit `.env` to customize:
- LLM model (GPT-4, GPT-3.5-turbo)
- Chunk size and overlap
- Number of results retrieved
- Confidence threshold

### 🚀 Deploy
- Deploy to AWS, Google Cloud, Heroku, etc.
- See [DEPLOYMENT.md](DEPLOYMENT.md) for guides
- Add authentication for production
- Set up monitoring and backups

### 🧪 Develop
- Run tests: `pytest`
- Check coverage: `pytest --cov=app`
- Add new features
- Customize prompts in `app/services/rag_pipeline.py`

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Support

### Resources
- 📖 [README.md](README.md) - Project overview
- 📘 [USAGE.md](USAGE.md) - Detailed usage
- 🏗️ [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Architecture
- 🚀 [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- ✅ [FEATURES.md](FEATURES.md) - Feature checklist

### Getting Help
1. Check the documentation
2. Review error logs
3. Search existing issues
4. Create a new issue with details

## Success Checklist

- [ ] Setup completed successfully
- [ ] `.env` configured with API key
- [ ] Server running on http://localhost:8000
- [ ] Sample document uploaded
- [ ] Test query executed
- [ ] Received AI-generated answer
- [ ] Explored web interface
- [ ] Checked API documentation

## You're Ready! 🎉

Your AI-Powered Knowledge Base is now running!

Start uploading documents and asking questions to experience the power of RAG technology.

---

**Need help?** Check the documentation or create an issue.
**Want to contribute?** Pull requests are welcome!

