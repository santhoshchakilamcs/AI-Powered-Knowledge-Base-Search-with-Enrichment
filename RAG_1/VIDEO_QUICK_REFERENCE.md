# 🎬 Video Recording Quick Reference Card

Print this out or keep it on a second screen while recording!

---

## ⏱️ TIMING CHEAT SHEET

| Time | What to Show | What to Say |
|------|--------------|-------------|
| **0:00** | Title slide | "Hi! AI-Powered Knowledge Base..." |
| **0:30** | Upload document | "Let me show you how it works..." |
| **0:50** | Query: "What is in my document?" | "Notice the HIGH confidence..." |
| **1:15** | Query: "Who is Marie Curie?" | "Watch the auto-enrichment..." |
| **2:00** | `rag_pipeline.py` lines 93-118 | "The brain of the system..." |
| **2:20** | `schemas.py` lines 45-56 | "Strongly typed responses..." |
| **2:40** | `enrichment_engine.py` lines 20-35 | "Priority-based sources..." |
| **3:00** | Innovation slide | "Four key innovations..." |
| **4:15** | Trade-offs slide | "Let's be honest..." |
| **4:45** | Results slide | "The results..." |

---

## 📂 FILES TO OPEN (In Order)

### **Browser Tabs:**
1. http://localhost:8000 (main UI)
2. http://localhost:8000/docs (optional - API docs)

### **Code Editor Tabs (Open in this order):**
1. `README.md`
2. `app/services/rag_pipeline.py`
3. `app/models/schemas.py`
4. `app/services/enrichment_engine.py`
5. `tests/test_rag_pipeline.py`
6. `FEATURES.md`

---

## 🎯 DEMO QUERIES (Copy-Paste Ready)

### **Query 1: Complete Answer (from your documents)**
```
What information is in my document?
```
**Expected:** High confidence, Complete badge, Your document in sources

### **Query 2: Auto-Enrichment (not in documents)**
```
Who is Marie Curie?
```
**Expected:** Auto-enrichment notification, Wikipedia source, External badge

### **Backup Queries (if needed):**
```
Who is Albert Einstein?
What is quantum computing?
Tell me about the Eiffel Tower?
Who won the Nobel Prize in Physics in 2023?
```

---

## 💻 CODE SNIPPETS TO HIGHLIGHT

### **1. Auto-Enrichment Logic** (`rag_pipeline.py` lines 93-118)
```python
# Step 4: Automatic enrichment - fetch from external sources ONLY if answer is incomplete
if not llm_response.is_complete and llm_response.missing_info:
    # Answer is incomplete - automatically fetch from trusted external sources
    logger.info(f"Answer incomplete (confidence: {llm_response.confidence}). Auto-fetching from external sources")
    
    llm_response.enrichment_suggestions = await self._generate_enrichment_suggestions(
        query=query,
        answer=llm_response.answer,
        missing_info=llm_response.missing_info,
        enable_auto_enrichment=True  # Always enable when answer is incomplete
    )
```

**Say:** "This is where the magic happens - if the answer is incomplete, it automatically triggers enrichment."

---

### **2. Structured Output** (`schemas.py` lines 45-56)
```python
class SearchResponse(BaseModel):
    """Response model for search queries with structured output."""
    query: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_complete: bool
    sources: List[SourceReference]
    missing_info: List[str] = Field(default_factory=list)
    enrichment_suggestions: List[EnrichmentSuggestion] = Field(default_factory=list)
    auto_enrichment_applied: bool = False
```

**Say:** "Everything is strongly typed - confidence from 0 to 1, boolean completeness flag, missing info list."

---

### **3. Trusted Sources** (`enrichment_engine.py` lines 20-35)
```python
TRUSTED_SOURCES = {
    "wikipedia": {
        "priority": 1,
        "description": "General knowledge encyclopedia",
        "enabled": True
    },
    "arxiv": {
        "priority": 2,
        "description": "Academic papers and research",
        "enabled": True
    },
    "pubmed": {
        "priority": 3,
        "description": "Medical and life sciences research",
        "enabled": True
    },
}
```

**Say:** "Priority-based system - Wikipedia first, then arXiv for academic, PubMed for medical."

---

## 🎤 KEY PHRASES TO USE

### **Opening:**
- "Self-assessing AI that knows when it doesn't know"
- "Automatically fills knowledge gaps"
- "Production-ready, not just a prototype"

### **During Demo:**
- "Watch what happens..."
- "Notice the confidence score..."
- "See the auto-enrichment notification?"
- "This is the magic of intelligent RAG"

### **During Code:**
- "This is where the magic happens..."
- "The brain of the system..."
- "Strongly typed with Pydantic..."
- "Priority-based, curated sources"

### **Innovations:**
- "The AI knows when it doesn't know"
- "Smart enrichment, not blind fetching"
- "Graceful handling of irrelevant documents"
- "Production-ready from day one"

### **Trade-offs:**
- "Let's be honest about limitations..."
- "Built in 24 hours, scoped appropriately"
- "Architecture designed for easy extension"

### **Closing:**
- "Demonstrates system design thinking"
- "Production-ready engineering"
- "Thank you, happy to answer questions"

---

## 🎨 UI ELEMENTS TO HIGHLIGHT

### **During Upload:**
- [ ] "Choose File" button
- [ ] Upload progress
- [ ] Document appearing in list
- [ ] Document name and size

### **During Complete Answer:**
- [ ] Search box
- [ ] "Search" button
- [ ] Answer text
- [ ] Confidence bar (should be high/green)
- [ ] "Complete" badge (green)
- [ ] Sources section
- [ ] Your document name
- [ ] Relevance score

### **During Auto-Enrichment:**
- [ ] Search box with new query
- [ ] Lower confidence initially
- [ ] Auto-enrichment notification (purple box)
- [ ] "Auto-enriched with information from: Wikipedia"
- [ ] External sources in sources section
- [ ] "External Source" badge (orange)
- [ ] Wikipedia link
- [ ] Final high confidence

---

## 🚨 COMMON MISTAKES TO AVOID

- ❌ Speaking too fast
- ❌ Not showing cursor
- ❌ Code font too small
- ❌ Forgetting to highlight key elements
- ❌ Not pausing after important points
- ❌ Skipping the "why" behind decisions
- ❌ Not showing enthusiasm
- ❌ Going over 5 minutes

---

## ✅ PRE-RECORDING CHECKLIST

### **Technical Setup:**
- [ ] Server running: `./run.sh`
- [ ] Browser open to http://localhost:8000
- [ ] All code files open in editor
- [ ] Screen recording software ready
- [ ] Microphone tested
- [ ] Audio levels checked
- [ ] Notifications disabled
- [ ] Other apps closed

### **Content Ready:**
- [ ] `examples/sample_document.txt` ready
- [ ] Queries copied to clipboard
- [ ] Slides created (if using)
- [ ] Script reviewed
- [ ] Timing practiced

### **Environment:**
- [ ] Quiet room
- [ ] Good lighting (if showing face)
- [ ] Water nearby
- [ ] Comfortable seating
- [ ] No interruptions

---

## 🎬 RECORDING FLOW

1. **Start recording**
2. **Count down: 3, 2, 1...**
3. **Show title slide (or speak title)**
4. **Follow script timing**
5. **Pause 2 seconds at end**
6. **Stop recording**

---

## 📊 POST-RECORDING CHECKLIST

- [ ] Review entire video
- [ ] Check audio quality
- [ ] Verify all demos worked
- [ ] Confirm timing (under 5 minutes)
- [ ] Add captions/subtitles
- [ ] Add transitions (optional)
- [ ] Add background music (optional, low volume)
- [ ] Export in HD (1080p)
- [ ] Test playback on different devices

---

## 🎯 SUCCESS METRICS

Your video should show:
- ✅ Working upload
- ✅ High confidence answer from documents
- ✅ Auto-enrichment in action
- ✅ Clean code structure
- ✅ Test coverage
- ✅ Honest trade-offs
- ✅ Professional delivery
- ✅ Under 5 minutes

---

## 🆘 IF SOMETHING GOES WRONG

### **Demo doesn't work:**
- Have backup screen recording ready
- Or explain what should happen
- Stay calm and professional

### **Forget what to say:**
- Pause, take a breath
- Refer to this card
- Continue confidently

### **Go over time:**
- Speed up trade-offs section
- Skip optional details
- Focus on key innovations

---

## 📞 EMERGENCY CONTACTS

- **Script:** `VIDEO_SCRIPT.md`
- **Slides:** `VIDEO_SLIDES.md`
- **Full README:** `README.md`

---

## 🎉 FINAL TIPS

1. **Smile** - Even if not on camera, it affects your voice
2. **Breathe** - Pause between sections
3. **Enthusiasm** - Show excitement about your work
4. **Confidence** - You built this, own it!
5. **Clarity** - Speak clearly, especially technical terms
6. **Pace** - 150-160 words per minute
7. **Practice** - Do a dry run first

---

**You've got this! 🚀**

**Remember:** This is YOUR project. You understand it better than anyone. Show your passion and expertise!

---

## 🎬 QUICK START COMMANDS

```bash
# Start server
cd /Users/suprabathc/Documents/augment-projects/RAG_1
./run.sh

# Open browser
open http://localhost:8000

# Open code editor
code .
```

---

**Good luck with your recording! 🎥✨**

