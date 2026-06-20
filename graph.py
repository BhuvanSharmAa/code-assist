"""
LangGraph multi-agent code review pipeline.

"""

import json
import os
from typing import TypedDict, List, Optional
from groq import Groq
from langgraph.graph import StateGraph, END
import base64
import requests

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 2


# State

class AgentState(TypedDict):
    code: str
    language: str
    active_agents: List[str]
    security_output: Optional[str]
    quality_output: Optional[str]
    logic_output: Optional[str]
    retry_count: int
    validation_passed: bool
    final_output: Optional[str]
    agent_trace: List[str]


# Helpers

def llm(system: str, user: str, max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def code_block(code: str, language: str) -> str:
    return f"Language: {language}\n\n```\n{code}\n```"


# Node 1: Router

ROUTER_SYSTEM = """You are a code analysis router. Given a code snippet, decide which specialist agents are needed.
Return ONLY a JSON object like: {"agents": ["security", "quality", "logic"]}

Rules:
- Always include "quality"
- Include "security" if code handles user input, auth, file I/O, network calls, or SQL
- Include "logic" if code has algorithms, loops, recursion, or complex branching
- You may return 1, 2, or all 3 agents"""

def router_node(state: AgentState) -> AgentState:
    raw = llm(ROUTER_SYSTEM, code_block(state["code"], state["language"]), max_tokens=100)
    try:
        parsed = json.loads(raw)
        agents = [a for a in parsed.get("agents", []) if a in ("security", "quality", "logic")]
    except Exception:
        agents = ["quality", "logic"]
    if not agents:
        agents = ["quality"]
    trace = state.get("agent_trace", [])
    trace.append(f"Router selected: {', '.join(agents)}")
    return {**state, "active_agents": agents, "agent_trace": trace}


# Node 2a: Security Agent

SECURITY_SYSTEM = """You are a security-focused code reviewer. Analyze the code for:
- Injection vulnerabilities (SQL, command, XSS)
- Insecure data handling or hardcoded secrets
- Missing input validation or authentication issues
- Unsafe library usage

Format your response in markdown with:
## Security findings
List each issue with severity: [HIGH/MEDIUM/LOW] and a one-line fix suggestion.
If no issues found, say "No security issues detected." """

def security_agent(state: AgentState) -> AgentState:
    if "security" not in state.get("active_agents", []):
        return state
    output = llm(SECURITY_SYSTEM, code_block(state["code"], state["language"]))
    trace = state.get("agent_trace", [])
    trace.append("Security agent: completed")
    return {**state, "security_output": output, "agent_trace": trace}


# Node 2b: Quality Agent

QUALITY_SYSTEM = """You are a code quality reviewer. Analyze the code for:
- Naming conventions and readability
- Code duplication or unnecessary complexity
- Missing error handling or edge cases
- Violations of SOLID principles or language idioms

Format your response in markdown with:
## Quality findings
List each issue as a bullet. Rate overall quality: [POOR/FAIR/GOOD/EXCELLENT]."""

def quality_agent(state: AgentState) -> AgentState:
    if "quality" not in state.get("active_agents", []):
        return state
    output = llm(QUALITY_SYSTEM, code_block(state["code"], state["language"]))
    trace = state.get("agent_trace", [])
    trace.append("Quality agent: completed")
    return {**state, "quality_output": output, "agent_trace": trace}


# Node 2c: Logic Agent

LOGIC_SYSTEM = """You are a logic and correctness reviewer. Analyze the code for:
- Logical bugs or off-by-one errors
- Unhandled edge cases (empty input, null, overflow)
- Algorithm correctness and time/space complexity
- Race conditions or state mutation issues

Format your response in markdown with:
## Logic findings
List each issue with a clear explanation. Include Big-O if relevant."""

def logic_agent(state: AgentState) -> AgentState:
    if "logic" not in state.get("active_agents", []):
        return state
    output = llm(LOGIC_SYSTEM, code_block(state["code"], state["language"]))
    trace = state.get("agent_trace", [])
    trace.append("Logic agent: completed")
    return {**state, "logic_output": output, "agent_trace": trace}


# Node 3: Validator

VALIDATOR_SYSTEM = """You are a quality gate for code review outputs.
Given agent outputs, decide if they are useful and substantive.

Return ONLY JSON: {"passed": true} or {"passed": false, "reason": "brief reason"}

Fail if: any active agent returned an empty response, or the response is just boilerplate with no specifics about the actual code."""

def validator_node(state: AgentState) -> AgentState:
    active = state.get("active_agents", [])
    outputs = []
    if "security" in active and state.get("security_output"):
        outputs.append(f"Security:\n{state['security_output'][:300]}")
    if "quality" in active and state.get("quality_output"):
        outputs.append(f"Quality:\n{state['quality_output'][:300]}")
    if "logic" in active and state.get("logic_output"):
        outputs.append(f"Logic:\n{state['logic_output'][:300]}")

    if not outputs:
        trace = state.get("agent_trace", [])
        trace.append("Validator: no outputs — failing")
        return {**state, "validation_passed": False, "agent_trace": trace}

    combined = "\n\n".join(outputs)
    raw = llm(VALIDATOR_SYSTEM, combined, max_tokens=80)
    try:
        result = json.loads(raw)
        passed = result.get("passed", False)
        reason = result.get("reason", "")
    except Exception:
        passed = True
        reason = ""

    trace = state.get("agent_trace", [])
    retry = state.get("retry_count", 0)
    if passed:
        trace.append("Validator: PASSED")
    else:
        trace.append(f"Validator: FAILED (retry {retry+1}/{MAX_RETRIES}) — {reason}")

    return {
        **state,
        "validation_passed": passed,
        "retry_count": retry + (0 if passed else 1),
        "agent_trace": trace,
    }


def should_retry(state: AgentState) -> str:
    if not state.get("validation_passed") and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "synthesize"


# Node 4: Synthesizer

SYNTHESIZER_SYSTEM = """You are a senior code reviewer synthesizing findings from multiple specialist agents.
Combine their outputs into a single coherent, prioritized review.

Structure your output as:
# Code review summary

## Critical issues
## Improvements
## What is good
## Recommended next steps

Avoid repeating the same issue twice. Rank by severity. Be specific and reference actual code patterns."""

def synthesizer_node(state: AgentState) -> AgentState:
    parts = []
    if state.get("security_output"):
        parts.append(f"**Security agent findings:**\n{state['security_output']}")
    if state.get("quality_output"):
        parts.append(f"**Quality agent findings:**\n{state['quality_output']}")
    if state.get("logic_output"):
        parts.append(f"**Logic agent findings:**\n{state['logic_output']}")

    combined = "\n\n---\n\n".join(parts)
    final = llm(SYNTHESIZER_SYSTEM, combined, max_tokens=1200)
    trace = state.get("agent_trace", [])
    trace.append("Synthesizer: completed")
    return {**state, "final_output": final, "agent_trace": trace}


# Build Graph

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("security_agent", security_agent)
    g.add_node("quality_agent", quality_agent)
    g.add_node("logic_agent", logic_agent)
    g.add_node("validator", validator_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("router")

    g.add_edge("router", "security_agent")
    g.add_edge("security_agent", "quality_agent")
    g.add_edge("quality_agent", "logic_agent")
    g.add_edge("logic_agent", "validator")

    g.add_conditional_edges(
        "validator",
        should_retry,
        {
            "retry": "security_agent",
            "synthesize": "synthesizer",
        },
    )

    g.add_edge("synthesizer", END)
    return g.compile()


GRAPH = build_graph()


def run_graph(code: str, language: str) -> AgentState:
    initial_state: AgentState = {
        "code": code,
        "language": language,
        "active_agents": [],
        "security_output": None,
        "quality_output": None,
        "logic_output": None,
        "retry_count": 0,
        "validation_passed": False,
        "final_output": None,
        "agent_trace": [],
    }
    return GRAPH.invoke(initial_state)

png_bytes = GRAPH.get_graph().draw_mermaid_png()


with open("pipeline_graph.png", "wb") as f:
    f.write(png_bytes)