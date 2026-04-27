# 🎯 Smart Career Decision System — Multi-Agent AI

> **Prototype / Proof of Concept** developed as part of the AI Agents Program at SDAIA Academy,
> focusing on LangGraph-based multi-agent workflows.

---

## 📌 Short Description

A smart AI system that helps students and graduates compare technical career paths
and choose the best fit based on their skills, interests, and career goal —
powered by a 5-agent LangGraph pipeline.

---

## 🔴 Problem

Many students and graduates struggle to choose between technical career paths
such as AI Engineer, Data Scientist, Machine Learning Engineer, and Data Engineer.

- Job titles are often similar and confusing.
- Each path requires very different skills and knowledge.
- There is an overwhelming amount of information online.
- This leads to poor career decisions and wasted time.

---

## ✅ Solution

A multi-agent AI system that:
1. Parses and understands the user's profile (skills, interests, goal).
2. Researches each selected career path using a local knowledge base and optional web search.
3. Analyzes the user's strengths and skill gaps.
4. Computes objective match scores (0–100) per path.
5. Generates a clear, reasoned final recommendation with actionable next steps.

---

## 🏗️ Architecture

```
User Input (Gradio UI)
        │
        ▼
┌─────────────────────┐
│   1. Planner Agent  │  Parses input · Normalizes skills & interests
│                     │  Decides if web research is needed
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│   2. Research Agent                                         │
│                                                             │
│   Tools:                                                    │
│   ├── get_local_career_info  → local KB per path            │
│   ├── web_search             → DuckDuckGo (if needed)       │
│   └── fetch_url              → page content (if useful)     │
│                                                             │
│   Pattern: ReAct loop (up to 12 iterations)                 │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   3. Profile Analyzer Agent  │  Identifies strengths, gaps, personality fit
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   4. Scorer Agent            │  score_path tool
│                              │  Formula: 60% skill + 40% interest → 0–100
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   5. Decision Agent          │  Synthesizes all outputs
│                              │  Anchors best_path to top Scorer result
│                              │  Generates reason · gaps · next steps
└────────┬─────────────────────┘
         │
         ▼
Gradio Output (formatted text report)
```

---

## 🤖 Agents Explained

| # | Agent | Method | Role |
|---|-------|--------|------|
| 1 | **Planner Agent** | `with_structured_output(PlannerOutput)` | Parses user input, normalizes skills/interests, decides if web research is needed |
| 2 | **Research Agent** | `bind_tools([...])` + ReAct loop | Calls tools to gather data per path; handles failures gracefully |
| 3 | **Profile Analyzer Agent** | `with_structured_output(ProfileOutput)` | Identifies strengths, weaknesses, and personality fit |
| 4 | **Scorer Agent** | `score_path` tool (deterministic) | Computes objective 0–100 scores: 60% skills + 40% interests |
| 5 | **Decision Agent** | `llm.invoke` + safe JSON parser | Synthesizes all outputs into a final recommendation |

---

## 🛠️ Tools

| Tool | Used By | Description |
|------|---------|-------------|
| `get_local_career_info` | Research Agent | Returns structured data from the local career knowledge base |
| `web_search` | Research Agent | Searches DuckDuckGo for current market info (no API key needed) |
| `fetch_url` | Research Agent | Fetches and extracts readable text from a URL |
| `score_path` | Scorer Agent | Deterministic fuzzy-match scoring: 60% skill + 40% interest alignment |

---

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **LangChain** | LLM abstraction, tool calling, structured output |
| **LangGraph** | Multi-agent pipeline (StateGraph, MemorySaver) |
| **OpenAI GPT-4o-mini** | Language model powering all agents |
| **Pydantic v2** | Schema validation for agent outputs |
| **Gradio** | Web UI |
| **DuckDuckGo Search** | Free web search (no API key required) |
| **BeautifulSoup4** | HTML parsing for fetch_url tool |
| **python-dotenv** | Secure API key loading |

---

## ✨ Features

- **5-agent sequential pipeline** managed by LangGraph
- **ReAct-style tool-calling loop** in the Research Agent
- **Deterministic scoring** — objective, reproducible results
- **Graceful error handling** — web search failure won't crash the system
- **Safe JSON parsing** — fallback output if LLM response is malformed
- **Agent pipeline trace** — shows exactly what each agent did
- **Simple, clean Gradio UI** — one form, one result, no technical jargon
- **Secure API key loading** via `.env` file

---

## 🗂️ Project Structure

```
smart-career-decision-system/
│
├── app.py               ← Main application (all agents, tools, UI)
├── requirements.txt     ← Python dependencies
├── .env.example         ← Template for environment variables
├── .gitignore           ← Excludes .env and cache files
├── README.md            ← This file
│
└── docs/
    └── project_explanation.md   ← Detailed technical explanation
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-career-decision-system.git
cd smart-career-decision-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env
# Open .env and replace: OPENAI_API_KEY=your_key_here
```

### 5. Run the app

```bash
python app.py
```

Open your browser at: **[http://localhost:7860](https://791b2ab450ebd12ac7.gradio.live)**

---

## 💡 Example Input

| Field | Example Value |
|-------|--------------|
| Career Paths | AI Engineer, Data Scientist |
| Current Skills | Python, SQL, pandas, statistics, FastAPI |
| Interests | Building AI products, data analysis, experimenting with LLMs |
| Career Goal | I want to transition into AI and build real-world products within 6 months |

---

## 📊 Example Output

```
══════════════════════════════════════════════════════════════
  ✅  RECOMMENDED PATH:  AI ENGINEER
══════════════════════════════════════════════════════════════

📌 WHY THIS PATH?

   Your Python skills and strong interest in building AI products align
   closely with the AI Engineer role. Your FastAPI experience is directly
   applicable to deploying AI services, and your enthusiasm for LLMs
   gives you a strong foundation for this path.

📊 MATCH SCORES  (Skill 60% + Interest 40%)

   AI Engineer                      ████████████████░░░░  78/100  ◀ BEST FIT
                                    Skills: 45/60   Interests: 33/40
   Data Scientist                   ████████████░░░░░░░░  62/100
                                    Skills: 38/60   Interests: 24/40

💪 YOUR STRENGTHS

   • Python proficiency with practical project experience
   • Existing FastAPI knowledge applicable to AI service deployment
   • Strong interest in LLMs and AI product development

⚠️  AREAS TO DEVELOP

   • Prompt engineering and LLM evaluation techniques
   • RAG system design and vector database integration
   • Production deployment and CI/CD for AI systems

🎯 TOP SKILL GAPS FOR AI ENGINEER

   • LLM tooling (LangChain / LangGraph)
   • Vector databases and RAG architecture
   • AI evaluation and observability

🚀 RECOMMENDED NEXT STEPS

   1. Complete a LangChain and LangGraph course (DeepLearning.AI)
   2. Build a RAG project using OpenAI and a vector database
   3. Deploy one AI-powered API using FastAPI on a cloud platform
   4. Practice prompt engineering and build an evaluation pipeline

─── Agent Pipeline Trace ──────────────────────────────────
   ✅ Planner Agent: 2 path(s) | 5 skill(s) | web_research=no
   ✅ Research Agent: tools → [get_local_career_info]
   ✅ Profile Analyzer: 3 strength(s) | 3 gap(s) identified
   ✅ Scorer Agent: AI Engineer=78 | Data Scientist=62
   ✅ Decision Agent: recommended → AI Engineer
══════════════════════════════════════════════════════════════
```

---

## 🔮 Future Improvements

- Integrate real-time job market data (LinkedIn, Indeed APIs)
- Add more career paths beyond AI and data fields
- Connect with LinkedIn or GitHub for automatic skill extraction
- Add support for Arabic language input
- Deploy to Hugging Face Spaces for public access
- Add a conversational follow-up Q&A after the recommendation

---

## 🎓 Program Note

> This project was developed as part of the **AI Agents Program at SDAIA Academy**,
> focusing on LangGraph-based multi-agent workflows.
> It is a **prototype / proof of concept** and is not intended for production use.

---

## 📄 License

This project is for educational purposes only.
