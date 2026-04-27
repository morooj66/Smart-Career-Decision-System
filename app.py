"""
Smart Career Decision System — Multi-Agent AI
=============================================
A 5-agent LangGraph pipeline that helps students and graduates
compare technical career paths and choose the best fit.

Pipeline:
    User Input → Planner → Research → Profile Analyzer → Scorer → Decision → Gradio Output

Agents  : Planner · Research · Profile Analyzer · Scorer · Decision
Tools   : get_local_career_info · web_search · fetch_url · score_path
Stack   : LangChain · LangGraph · OpenAI · Pydantic · Gradio

NOTE: This is a prototype / proof of concept developed as part of an
AI Agents program at SDAIA Academy, focusing on LangGraph-based
multi-agent workflows.
"""

# ── Standard library ──────────────────────────────────────────
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, TypedDict

# ── Third-party ───────────────────────────────────────────────
import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

# ── LangChain / LangGraph ─────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────

load_dotenv()  # reads .env file if present

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n"
        "Create a .env file with:  OPENAI_API_KEY=your_key_here\n"
        "or export it as an environment variable."
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# ─────────────────────────────────────────────────────────────
# 2. LOCAL KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────

LOCAL_CAREER_KB: Dict[str, Dict[str, Any]] = {
    "Data Scientist": {
        "summary": (
            "Analyzes data to generate insights, build statistical models, "
            "and communicate findings to guide business decisions."
        ),
        "core_skills": [
            "Python", "SQL", "statistics", "experiment design",
            "data visualization", "communication", "machine learning basics",
        ],
        "typical_projects": [
            "A/B testing", "forecasting", "customer segmentation",
            "dashboards & insights", "churn prediction",
        ],
        "tools": [
            "pandas", "numpy", "scikit-learn", "Jupyter",
            "Tableau / Power BI", "BigQuery / Snowflake",
        ],
        "signals_of_fit": [
            "enjoys analysis", "curious about patterns",
            "likes communicating insights", "loves data storytelling",
        ],
        "learning_path": [
            "Python + SQL fundamentals",
            "Statistics & probability",
            "Machine learning basics",
            "Portfolio projects",
            "Data storytelling",
        ],
        "avg_salary_usd": "95,000 – 140,000",
        "demand_trend": "High and stable",
    },
    "AI Engineer": {
        "summary": (
            "Builds and ships AI-powered products using LLMs, tool calling, "
            "evaluation pipelines, and production integration."
        ),
        "core_skills": [
            "Python", "APIs", "prompt engineering", "LLM tooling",
            "evaluation", "software engineering", "RAG systems",
        ],
        "typical_projects": [
            "RAG / chat assistants", "tool-using agents",
            "AI features in apps", "eval pipelines", "fine-tuning workflows",
        ],
        "tools": [
            "LangChain / LangGraph", "FastAPI", "OpenAI / LLM APIs",
            "vector databases", "CI/CD", "observability tools",
        ],
        "signals_of_fit": [
            "likes product building", "enjoys rapid iteration",
            "comfortable with ambiguity", "excited about LLMs",
        ],
        "learning_path": [
            "Python + backend basics",
            "LLM fundamentals & prompting",
            "Tool calling & agents",
            "RAG systems",
            "Evals & deployment patterns",
        ],
        "avg_salary_usd": "130,000 – 200,000",
        "demand_trend": "Very high and fast-growing",
    },
    "Machine Learning Engineer": {
        "summary": (
            "Designs, trains, and deploys ML models with a focus on "
            "reliability, performance, and production-grade pipelines."
        ),
        "core_skills": [
            "Python", "ML algorithms", "deep learning", "data pipelines",
            "model serving", "MLOps", "system design",
        ],
        "typical_projects": [
            "recommendation systems", "image / text classification",
            "model deployment", "feature pipelines", "A/B experiments",
        ],
        "tools": [
            "PyTorch / TensorFlow", "MLflow", "Docker",
            "Kubernetes", "Airflow", "feature stores",
        ],
        "signals_of_fit": [
            "enjoys engineering + modeling", "likes optimization",
            "cares about production reliability", "enjoys math",
        ],
        "learning_path": [
            "ML foundations & math",
            "Deep learning",
            "Training loops & experiments",
            "Model deployment",
            "Monitoring & drift detection",
        ],
        "avg_salary_usd": "120,000 – 180,000",
        "demand_trend": "High and growing",
    },
    "Data Engineer": {
        "summary": (
            "Builds and maintains data infrastructure, pipelines, "
            "and warehouses that power analytics and ML systems."
        ),
        "core_skills": [
            "Python", "SQL", "ETL/ELT pipelines", "cloud platforms",
            "Spark", "data modeling", "system design",
        ],
        "typical_projects": [
            "data pipelines", "data warehouse design",
            "streaming systems", "data lake architecture", "data quality checks",
        ],
        "tools": [
            "Airflow", "dbt", "Apache Spark", "Kafka",
            "AWS / GCP / Azure", "Databricks",
        ],
        "signals_of_fit": [
            "likes infrastructure", "enjoys problem-solving at scale",
            "values reliability and correctness", "likes backend systems",
        ],
        "learning_path": [
            "SQL + Python",
            "ETL basics & cloud storage",
            "Orchestration (Airflow / Prefect)",
            "Streaming & real-time data",
            "Data modeling & dbt",
        ],
        "avg_salary_usd": "110,000 – 160,000",
        "demand_trend": "High and stable",
    },
}

ALL_PATHS: List[str] = list(LOCAL_CAREER_KB.keys())

# ─────────────────────────────────────────────────────────────
# 3. PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    selected_paths: List[str] = Field(
        ..., description="Career paths the user wants to compare."
    )
    current_skills: List[str] = Field(
        ..., description="User's current skills — normalized, deduplicated list."
    )
    interests: List[str] = Field(
        ..., description="User's interests — normalized, deduplicated list."
    )
    goal: str = Field(..., description="User's stated career goal.")
    needs_web_research: bool = Field(
        ...,
        description=(
            "True only if the user explicitly asks about current salaries / "
            "market demand, or the paths are very niche / emerging. "
            "Otherwise false — use the local knowledge base."
        ),
    )


class ProfileOutput(BaseModel):
    strengths: List[str] = Field(
        ..., description="User's existing strengths relevant to the selected paths."
    )
    weaknesses: List[str] = Field(
        ..., description="Clear skill or experience gaps compared to the selected paths."
    )
    personality_fit: str = Field(
        ...,
        description=(
            "One sentence: which type of role best matches the user's "
            "personality and working style."
        ),
    )


# ─────────────────────────────────────────────────────────────
# 4. LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    user_input:      Dict[str, Any]
    planner_output:  Optional[Dict[str, Any]]
    research_output: Optional[Dict[str, Any]]
    profile_output:  Optional[Dict[str, Any]]
    scores:          Optional[List[Dict[str, Any]]]
    decision_output: Optional[Dict[str, Any]]
    status_log:      List[str]

# ─────────────────────────────────────────────────────────────
# 5. TOOLS
# ─────────────────────────────────────────────────────────────

@tool
def get_local_career_info(path_name: str) -> dict:
    """
    Return structured career path data from the local knowledge base.
    Always call this first for every selected path before any web search.
    """
    data = LOCAL_CAREER_KB.get(path_name)
    if not data:
        available = ", ".join(LOCAL_CAREER_KB.keys())
        return {
            "error": f"Path '{path_name}' not found. Available: {available}",
            "ok": False,
        }
    return {"path_name": path_name, "ok": True, **data}


@tool
def web_search(query: str, max_results: int = 4) -> dict:
    """
    Search the web for current career market information using DuckDuckGo.
    Use only when needs_web_research is true. Handles failures gracefully.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "href":    r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return {"ok": True, "query": query, "results": results}
    except Exception as exc:
        return {
            "ok": False,
            "query": query,
            "error": str(exc),
            "results": [],
            "fallback": "Web search failed — using local knowledge base only.",
        }


@tool
def fetch_url(url: str, max_chars: int = 3000) -> dict:
    """
    Fetch and extract readable text from a URL.
    Strips scripts, styles, and nav elements. Truncates to max_chars.
    Handles network errors and non-200 responses gracefully.
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CareerBot/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return {
            "url":  url,
            "ok":   True,
            "text": text[:max_chars],
        }
    except requests.exceptions.Timeout:
        return {"url": url, "ok": False, "error": "Request timed out after 10s."}
    except requests.exceptions.HTTPError as exc:
        return {"url": url, "ok": False, "error": f"HTTP {exc.response.status_code}"}
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc)}


@tool
def score_path(
    user_skills: List[str],
    user_interests: List[str],
    path_name: str,
    path_data: dict,
) -> dict:
    """
    Compute a deterministic match score (0–100) between the user profile
    and a career path.

    Scoring formula:
        Skill alignment    → up to 60 points  (60% weight)
        Interest alignment → up to 40 points  (40% weight)

    Uses fuzzy partial-word matching to handle synonyms and variations.
    """

    def normalize(items: list) -> set:
        return {str(x).strip().lower() for x in items if str(x).strip()}

    def fuzzy_overlap(user_set: set, path_set: set) -> int:
        """Count how many user items fuzzy-match at least one path item."""
        matched = 0
        for u in user_set:
            for p in path_set:
                if u in p or p in u:
                    matched += 1
                    break
        return matched

    skills    = normalize(user_skills)
    interests = normalize(user_interests)
    core      = normalize(path_data.get("core_skills", []))
    signals   = normalize(path_data.get("signals_of_fit", []))

    if not core and not signals:
        return {
            "path_name":   path_name,
            "score":       0.0,
            "breakdown":   {"skill_alignment": 0.0, "interest_alignment": 0.0},
            "explanation": "No core_skills or signals_of_fit found in path data.",
        }

    skill_match    = fuzzy_overlap(skills, core)
    interest_match = fuzzy_overlap(interests, signals)

    skill_score    = min(60.0, (skill_match    / max(1, min(7, len(core))))    * 60.0)
    interest_score = min(40.0, (interest_match / max(1, min(4, len(signals)))) * 40.0)
    total          = round(skill_score + interest_score, 2)

    return {
        "path_name": path_name,
        "score":     total,
        "breakdown": {
            "skill_alignment":    round(skill_score, 2),
            "interest_alignment": round(interest_score, 2),
        },
        "explanation": (
            f"Matched {skill_match}/{len(core)} core skills → {skill_score:.1f}/60 pts; "
            f"matched {interest_match}/{len(signals)} interest signals → {interest_score:.1f}/40 pts."
        ),
    }


# ─── Tool registry (used by Research Agent) ──────────────────

_RESEARCH_TOOL_MAP = {
    "get_local_career_info": get_local_career_info,
    "web_search":            web_search,
    "fetch_url":             fetch_url,
}


def _execute_tool_calls(tool_calls: list) -> List[ToolMessage]:
    """
    Execute a list of LLM tool_calls and return ToolMessage objects.
    Never raises — all errors are captured inside the ToolMessage content.
    """
    messages = []
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args", {})
        tid  = call.get("id", str(uuid.uuid4()))
        fn   = _RESEARCH_TOOL_MAP.get(name)
        try:
            result = fn.invoke(args) if fn else {"error": f"Unknown tool: {name}", "ok": False}
        except Exception as exc:
            result = {"error": str(exc), "ok": False}
        messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tid,
                name=name,
            )
        )
    return messages

# ─────────────────────────────────────────────────────────────
# 6. AGENT SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are the Planner Agent in a 5-agent career decision system.

Responsibilities:
1. Parse the user's input: selected_paths, current skills, interests, goal.
2. Normalize skills and interests into clean, deduplicated lowercase lists.
3. Set needs_web_research = true ONLY if:
   - The user explicitly asks about current salaries or job market trends, OR
   - The selected paths are highly niche or emerging fields.
   Otherwise always set it to false — the local knowledge base is sufficient.

Return only valid structured output matching the schema. No extra text.
"""

RESEARCH_SYSTEM = """\
You are the Research Agent in a 5-agent career decision system.

Responsibilities:
1. Call get_local_career_info for EVERY selected career path — no exceptions.
2. If needs_web_research is true, call web_search for each path to get
   current market demand and salary information.
3. Optionally call fetch_url on one promising search result for extra depth.
4. If a tool returns ok=false, continue with the data you have — do not crash.
5. After all tools are done, return ONLY the following JSON (no markdown):

{
  "per_path_findings": {
    "Path Name": [
      {
        "topic": "Role Summary",
        "key_points": ["point 1", "point 2"],
        "sources": ["local_kb"]
      }
    ]
  },
  "global_notes": ["general market observation"],
  "source_quality_score": 0.85
}

Rules:
- Always call get_local_career_info first for every path.
- Return ONLY valid JSON. No markdown fences, no preamble, no explanation.
"""

PROFILE_SYSTEM = """\
You are the Profile Analyzer Agent in a 5-agent career decision system.

Given the user's skills, interests, goal, and the research summary per path,
produce a concise structured profile analysis:

- strengths: 3–5 things the user already has that are relevant to the paths.
- weaknesses: 3–5 clear skill or experience gaps compared to path requirements.
- personality_fit: exactly one sentence on which type of role suits the user best.

Return only valid structured output matching the schema. No extra text.
"""

DECISION_SYSTEM = """\
You are the Decision Agent in a 5-agent career decision system.

You receive:
  - planner_output   : parsed user profile (skills, interests, goal)
  - research_output  : structured findings per career path
  - profile_output   : strengths, weaknesses, personality fit
  - deterministic_scores : objective numeric scores per path (0-100)

Your job:
1. The path with the HIGHEST deterministic score is the anchor — use it as best_path.
2. Write a clear, specific reason (2–3 sentences) that references the user's
   actual skills and how they match the best path requirements.
3. List exactly 3 skill gaps the user must close for the best path.
4. Provide 4 concrete, actionable next steps.

Rules:
- Only recommend paths that appear in deterministic_scores.
- Do NOT invent new paths or rename existing ones.
- Return ONLY valid JSON matching this exact schema — no markdown, no preamble:

{
  "best_path": "exact path name",
  "reason": "2-3 sentences",
  "skill_gaps": ["gap 1", "gap 2", "gap 3"],
  "next_steps": ["step 1", "step 2", "step 3", "step 4"]
}
"""

# ─────────────────────────────────────────────────────────────
# 7. UTILITY: SAFE JSON PARSER
# ─────────────────────────────────────────────────────────────

def _safe_parse_json(text: str, fallback: dict) -> dict:
    """
    Try to parse JSON from an LLM response.
    Strips markdown fences, then tries full parse, then regex extraction.
    Returns fallback dict if all attempts fail.
    """
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
    return fallback

# ─────────────────────────────────────────────────────────────
# 8. LLM INSTANCES PER AGENT
# ─────────────────────────────────────────────────────────────

_planner_llm  = llm.with_structured_output(PlannerOutput)
_research_llm = llm.bind_tools([get_local_career_info, web_search, fetch_url])
_profile_llm  = llm.with_structured_output(ProfileOutput)
# Decision agent uses plain llm → parses JSON from content string

# ─────────────────────────────────────────────────────────────
# 9. AGENT NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def planner_node(state: WorkflowState) -> WorkflowState:
    """
    Agent 1 — Planner Agent
    ───────────────────────
    Parses and normalizes user input into a structured PlannerOutput.
    Decides whether web research is needed for the Research Agent.

    Method : llm.with_structured_output(PlannerOutput)
    Tools  : none
    """
    ui = state["user_input"]
    user_text = (
        f"Selected paths: {', '.join(ui.get('selected_paths', []))}\n"
        f"Skills: {ui.get('current_skills', '')}\n"
        f"Interests: {ui.get('interests', '')}\n"
        f"Goal: {ui.get('goal', '')}"
    )

    result: PlannerOutput = _planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_text),
    ])

    state["planner_output"] = result.model_dump()
    state["status_log"].append(
        f"✅ Planner Agent: {len(result.selected_paths)} path(s) | "
        f"{len(result.current_skills)} skill(s) | "
        f"web_research={'yes' if result.needs_web_research else 'no'}"
    )
    return state


def research_node(state: WorkflowState) -> WorkflowState:
    """
    Agent 2 — Research Agent
    ────────────────────────
    Gathers structured data for every selected career path using a
    ReAct-style tool-calling loop (max 12 iterations).

    Method : llm.bind_tools([...]) + manual tool execution loop
    Tools  : get_local_career_info, web_search, fetch_url
    """
    plan = PlannerOutput(**state["planner_output"])

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=json.dumps({
            "selected_paths":     plan.selected_paths,
            "needs_web_research": plan.needs_web_research,
        })),
    ]

    tools_used: List[str] = []

    for iteration in range(12):
        ai_msg = _research_llm.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            break  # LLM finished — no more tool calls

        for tc in tool_calls:
            tools_used.append(tc.get("name", "unknown"))

        messages.extend(_execute_tool_calls(tool_calls))

    # Extract final content from the last AI message
    final_text = getattr(ai_msg, "content", "") or ""

    fallback_research = {
        "per_path_findings": {
            path: [{"topic": "Local KB", "key_points": ["See local knowledge base."], "sources": ["local_kb"]}]
            for path in plan.selected_paths
        },
        "global_notes": ["Web research unavailable — using local knowledge base."],
        "source_quality_score": 0.6,
    }

    research_data = _safe_parse_json(final_text, fallback_research)

    state["research_output"] = research_data
    unique_tools = list(dict.fromkeys(tools_used))
    state["status_log"].append(
        f"✅ Research Agent: tools → [{', '.join(unique_tools) or 'none'}]"
    )
    return state


def profile_analyzer_node(state: WorkflowState) -> WorkflowState:
    """
    Agent 3 — Profile Analyzer Agent
    ─────────────────────────────────
    Analyzes the user's strengths, weaknesses, and personality fit
    against the selected career paths.

    Method : llm.with_structured_output(ProfileOutput)
    Tools  : none
    """
    plan     = state["planner_output"]
    research = state["research_output"] or {}

    # Build a compact research summary (top 2 findings per path)
    research_summary: Dict[str, list] = {}
    for path, findings in research.get("per_path_findings", {}).items():
        research_summary[path] = [
            f.get("key_points", [])[:2]
            for f in (findings or [])[:2]
        ]

    payload = {
        "user_skills":       plan.get("current_skills", []),
        "user_interests":    plan.get("interests", []),
        "user_goal":         plan.get("goal", ""),
        "selected_paths":    plan.get("selected_paths", []),
        "research_summary":  research_summary,
    }

    result: ProfileOutput = _profile_llm.invoke([
        SystemMessage(content=PROFILE_SYSTEM),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])

    state["profile_output"] = result.model_dump()
    state["status_log"].append(
        f"✅ Profile Analyzer: {len(result.strengths)} strength(s) | "
        f"{len(result.weaknesses)} gap(s) identified"
    )
    return state


def scorer_node(state: WorkflowState) -> WorkflowState:
    """
    Agent 4 — Scorer Agent
    ──────────────────────
    Computes deterministic match scores for every selected career path.
    Formula: 60% skill alignment + 40% interest alignment → total 0–100.

    Method : calls score_path tool directly for each path
    Tools  : score_path
    """
    plan   = PlannerOutput(**state["planner_output"])
    scores = []

    for path_name in plan.selected_paths:
        path_data = LOCAL_CAREER_KB.get(path_name, {})
        result = score_path.invoke({
            "user_skills":    plan.current_skills,
            "user_interests": plan.interests,
            "path_name":      path_name,
            "path_data":      path_data,
        })
        scores.append(result)

    # Sort descending by score
    scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    state["scores"] = scores

    summary = " | ".join(
        f"{s['path_name']}={s['score']:.0f}" for s in scores
    )
    state["status_log"].append(f"✅ Scorer Agent: {summary}")
    return state


def decision_node(state: WorkflowState) -> WorkflowState:
    """
    Agent 5 — Decision Agent
    ────────────────────────
    Synthesizes all previous outputs into a final career recommendation.
    The best_path is anchored to the highest Scorer Agent score.

    Method : llm.invoke → _safe_parse_json
    Tools  : none (reads from state only)
    """
    scores = state.get("scores") or []
    top_path = scores[0]["path_name"] if scores else "Unknown"

    payload = {
        "planner_output":       state.get("planner_output"),
        "research_output":      state.get("research_output"),
        "profile_output":       state.get("profile_output"),
        "deterministic_scores": scores,
    }

    ai_msg = llm.invoke([
        SystemMessage(content=DECISION_SYSTEM),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)[:14000]),
    ])

    raw_text = getattr(ai_msg, "content", "") or ""

    fallback_decision = {
        "best_path":  top_path,
        "reason":     f"Based on your profile, {top_path} is your strongest match.",
        "skill_gaps": ["See path requirements for details."],
        "next_steps": [
            "Explore online courses for this path.",
            "Build a portfolio project.",
            "Connect with professionals in this field.",
            "Review the learning path in the knowledge base.",
        ],
    }

    decision = _safe_parse_json(raw_text, fallback_decision)

    # Anchor best_path to top Scorer result — prevents LLM hallucination
    decision["best_path"] = top_path

    state["decision_output"] = decision
    state["status_log"].append(
        f"✅ Decision Agent: recommended → {decision['best_path']}"
    )
    return state

# ─────────────────────────────────────────────────────────────
# 10. BUILD LANGGRAPH PIPELINE
# ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """
    Build and compile the 5-node sequential LangGraph pipeline.

    Graph:
        planner → research → profile_analyzer → scorer → decision → END
    """
    builder = StateGraph(WorkflowState)

    builder.add_node("planner",          planner_node)
    builder.add_node("research",         research_node)
    builder.add_node("profile_analyzer", profile_analyzer_node)
    builder.add_node("scorer",           scorer_node)
    builder.add_node("decision",         decision_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner",          "research")
    builder.add_edge("research",         "profile_analyzer")
    builder.add_edge("profile_analyzer", "scorer")
    builder.add_edge("scorer",           "decision")
    builder.add_edge("decision",         END)

    return builder.compile(checkpointer=MemorySaver())


_graph = _build_graph()

# ─────────────────────────────────────────────────────────────
# 11. WORKFLOW RUNNER
# ─────────────────────────────────────────────────────────────

def run_workflow(
    selected_paths: List[str],
    current_skills: str,
    interests: str,
    goal: str,
) -> WorkflowState:
    """Run the full 5-agent pipeline and return the final WorkflowState."""
    init_state: WorkflowState = {
        "user_input": {
            "selected_paths": selected_paths,
            "current_skills": current_skills,
            "interests":      interests,
            "goal":           goal,
        },
        "planner_output":  None,
        "research_output": None,
        "profile_output":  None,
        "scores":          None,
        "decision_output": None,
        "status_log":      [],
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return _graph.invoke(init_state, config=config)

# ─────────────────────────────────────────────────────────────
# 12. FORMAT RESULT FOR GRADIO
# ─────────────────────────────────────────────────────────────

def _format_result(state: WorkflowState) -> str:
    """Convert the final WorkflowState into a human-readable text report."""
    decision = state.get("decision_output") or {}
    scores   = state.get("scores") or []
    profile  = state.get("profile_output") or {}
    log      = state.get("status_log") or []

    best = decision.get("best_path", "—")
    lines: List[str] = []

    # Header
    lines += [
        "=" * 62,
        f"  ✅  RECOMMENDED PATH:  {best.upper()}",
        "=" * 62,
    ]

    # Reason
    reason = decision.get("reason", "")
    if reason:
        lines += ["\n📌 WHY THIS PATH?\n", f"   {reason}"]

    # Match scores with ASCII bar chart
    if scores:
        lines.append("\n📊 MATCH SCORES  (Skill 60% + Interest 40%)\n")
        for s in scores:
            pct       = s.get("score", 0.0)
            filled    = int(pct / 5)
            bar       = "█" * filled + "░" * (20 - filled)
            marker    = "  ◀ BEST FIT" if s["path_name"] == best else ""
            sk        = s.get("breakdown", {}).get("skill_alignment", 0)
            intr      = s.get("breakdown", {}).get("interest_alignment", 0)
            lines.append(f"   {s['path_name']:<32} {bar}  {pct:.0f}/100{marker}")
            lines.append(f"   {'':32} Skills: {sk:.0f}/60   Interests: {intr:.0f}/40")

    # Strengths
    strengths = profile.get("strengths", [])
    if strengths:
        lines.append("\n💪 YOUR STRENGTHS\n")
        lines += [f"   • {s}" for s in strengths]

    # Weaknesses from profile
    weaknesses = profile.get("weaknesses", [])
    if weaknesses:
        lines.append("\n⚠️  AREAS TO DEVELOP\n")
        lines += [f"   • {w}" for w in weaknesses]

    # Skill gaps from decision
    skill_gaps = decision.get("skill_gaps", [])
    if skill_gaps:
        lines.append(f"\n🎯 TOP SKILL GAPS FOR {best.upper()}\n")
        lines += [f"   • {g}" for g in skill_gaps]

    # Next steps
    next_steps = decision.get("next_steps", [])
    if next_steps:
        lines.append("\n🚀 RECOMMENDED NEXT STEPS\n")
        lines += [f"   {i}. {step}" for i, step in enumerate(next_steps, 1)]

    # Agent trace
    if log:
        lines.append("\n─── Agent Pipeline Trace " + "─" * 37)
        lines += [f"   {entry}" for entry in log]

    lines.append("=" * 62)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# 13. GRADIO HANDLER
# ─────────────────────────────────────────────────────────────

def gradio_submit(
    selected_paths: List[str],
    current_skills: str,
    interests: str,
    goal: str,
) -> str:
    """Validate inputs, run the pipeline, and return a formatted result."""
    # Input validation
    if not selected_paths:
        return "❌ Please select at least one career path to compare."
    if not current_skills or not current_skills.strip():
        return "❌ Please enter your current skills."
    if not interests or not interests.strip():
        return "❌ Please describe your interests."
    if not goal or not goal.strip():
        return "❌ Please describe your career goal."
    if len(current_skills.strip()) < 3:
        return "❌ Please provide more detail about your skills (at least 3 characters)."

    try:
        state = run_workflow(selected_paths, current_skills, interests, goal)
        return _format_result(state)
    except Exception as exc:
        return (
            f"❌ An error occurred during analysis.\n\n"
            f"Details: {str(exc)}\n\n"
            "Please check your API key and try again."
        )

# ─────────────────────────────────────────────────────────────
# 14. GRADIO UI
# ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Smart Career Decision System",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🎯 Smart Career Decision System
        ### Powered by Multi-Agent AI &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; OpenAI

        Enter your profile below and click **Run Analysis**.
        Five specialized AI agents will analyze your profile and recommend
        the best technical career path for you.

        > ⚠️ This is a **prototype / proof of concept** developed as part of an AI Agents program at SDAIA Academy.
        """
    )

    with gr.Row():

        # ── Left column: inputs ──────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### 1 · Select Paths to Compare")
            inp_paths = gr.CheckboxGroup(
                choices=ALL_PATHS,
                value=["AI Engineer", "Data Scientist"],
                label="Career Paths",
            )

            gr.Markdown("### 2 · Your Profile")
            inp_skills = gr.Textbox(
                label="Current Skills",
                lines=3,
                placeholder="e.g. Python, SQL, pandas, statistics, FastAPI, data visualization ...",
            )
            inp_interests = gr.Textbox(
                label="Interests",
                lines=3,
                placeholder=(
                    "e.g. building AI products, analyzing large datasets, "
                    "working with pipelines, experimenting with LLMs ..."
                ),
            )
            inp_goal = gr.Textbox(
                label="Career Goal",
                lines=2,
                placeholder=(
                    "e.g. I want to transition into AI engineering and build "
                    "real-world LLM-powered products within 6 months."
                ),
            )

            run_btn = gr.Button(
                "▶  Run Multi-Agent Analysis",
                variant="primary",
                size="lg",
            )

        # ── Right column: output ─────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### 3 · Analysis Result")
            out_result = gr.Textbox(
                label="Recommendation",
                lines=35,
                show_copy_button=True,
                interactive=False,
                placeholder=(
                    "Your personalized career recommendation will appear here\n"
                    "after the 5-agent pipeline completes.\n\n"
                    "Pipeline:\n"
                    "  1. Planner Agent        — parses & normalizes your input\n"
                    "  2. Research Agent       — gathers data per career path\n"
                    "  3. Profile Analyzer     — identifies strengths & gaps\n"
                    "  4. Scorer Agent         — computes match scores (0–100)\n"
                    "  5. Decision Agent       — generates final recommendation"
                ),
            )

    run_btn.click(
        fn=gradio_submit,
        inputs=[inp_paths, inp_skills, inp_interests, inp_goal],
        outputs=[out_result],
    )

    gr.Markdown(
        """
        ---
        **How it works**

        Your input flows through a sequential **LangGraph** pipeline:
        `Planner → Research → Profile Analyzer → Scorer → Decision`

        - The **Research Agent** calls tools (`get_local_career_info`, `web_search`, `fetch_url`) in a ReAct loop.
        - The **Scorer Agent** computes deterministic scores: 60% skill alignment + 40% interest alignment.
        - The **Decision Agent** synthesizes all outputs into a final recommendation with actionable next steps.

        *Developed as part of the AI Agents Program — SDAIA Academy.*
        """
    )

# ─────────────────────────────────────────────────────────────
# 15. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
    )
