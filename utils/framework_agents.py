import json
from typing import TypedDict

from utils.agent import build_casino_analysis
from utils.agent import run_casino_agent
from utils.llm import DEFAULT_GROQ_MODEL
from utils.llm import generate_agent_summary
from utils.llm import get_api_key

try:
    from crewai import Agent
    from crewai import Crew
    from crewai import LLM
    from crewai import Process
    from crewai import Task
except Exception:
    Agent = Crew = LLM = Process = Task = None

try:
    import litellm
except ImportError:
    litellm = None

try:
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph
except Exception:
    END = START = StateGraph = None

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
except Exception:
    ChatPromptTemplate = ChatGroq = StrOutputParser = None

try:
    from llama_index.core import PromptTemplate
    from llama_index.llms.groq import Groq as LlamaIndexGroq
except Exception:
    PromptTemplate = LlamaIndexGroq = None


class WorkflowState(TypedDict, total=False):
    model: object
    player_data: dict
    game_plan: dict
    goal: str
    analysis: dict


def _build_analysis_context(analysis):
    return json.dumps(
        {
            "prediction": analysis["prediction"]["label"],
            "confidence": analysis["confidence"],
            "risk": analysis["risk"],
            "strategy": analysis["strategy"],
            "optimization": analysis["optimization"],
            "loss_prediction": analysis["loss_prediction"],
            "addiction_risk": analysis["addiction_risk"],
            "alerts": analysis["alerts"],
            "financial_guard": analysis["financial_guard"],
            "intervention": analysis["intervention"],
            "retrieval": analysis["retrieval"],
            "final_action": analysis["final_action"],
        },
        indent=2,
        default=str,
    )


def get_framework_status():
    return {
        "custom": True,
        "langgraph": StateGraph is not None,
        "crewai": all(item is not None for item in (Agent, Crew, LLM, Process, Task))
        and litellm is not None,
        "langchain": all(
            item is not None
            for item in (ChatPromptTemplate, ChatGroq, StrOutputParser)
        ),
        "llamaindex": all(item is not None for item in (PromptTemplate, LlamaIndexGroq)),
    }


def run_langgraph_agent(model, player_data, game_plan, goal):
    if StateGraph is None or START is None or END is None:
        raise RuntimeError("LangGraph is not installed.")

    def analyze_node(state: WorkflowState):
        analysis = build_casino_analysis(
            state["model"],
            state["player_data"],
            state["game_plan"],
            state["goal"],
        )
        analysis["framework"] = "langgraph"
        return {"analysis": analysis}

    def summarize_node(state: WorkflowState):
        analysis = state["analysis"]
        analysis["summary"] = generate_agent_summary(analysis)
        return {"analysis": analysis}

    graph = StateGraph(WorkflowState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)

    compiled_graph = graph.compile()
    result = compiled_graph.invoke(
        {
            "model": model,
            "player_data": player_data,
            "game_plan": game_plan,
            "goal": goal,
        }
    )
    return result["analysis"]


def run_crewai_agent(model, player_data, game_plan, goal):
    if not all(item is not None for item in (Agent, Crew, LLM, Process, Task)):
        raise RuntimeError("CrewAI is not installed.")
    if litellm is None:
        analysis = build_casino_analysis(model, player_data, game_plan, goal)
        analysis["framework"] = "crewai_fallback"
        analysis["summary"] = generate_agent_summary(analysis)
        analysis["framework_note"] = (
            "CrewAI selected, but LiteLLM is not installed in the environment. "
            "Install `litellm` to enable CrewAI orchestration."
        )
        return analysis

    analysis = build_casino_analysis(model, player_data, game_plan, goal)
    api_key = get_api_key()

    if not api_key:
        analysis["framework"] = "crewai_fallback"
        analysis["summary"] = generate_agent_summary(analysis)
        analysis["framework_note"] = "CrewAI selected, but no GROQ_API_KEY was found."
        return analysis

    llm = LLM(
        model=f"groq/{DEFAULT_GROQ_MODEL}",
        temperature=0.2,
    )

    risk_agent = Agent(
        role="Responsible gambling risk analyst",
        goal="Identify the most important behavioral, financial, and safety risks.",
        backstory="You specialize in player risk patterns and early warning signals.",
        llm=llm,
        verbose=False,
    )
    strategy_agent = Agent(
        role="Casino strategy analyst",
        goal="Turn the analysis into a practical and concise decision summary.",
        backstory="You combine game strategy with player-safety awareness.",
        llm=llm,
        verbose=False,
    )

    analysis_context = _build_analysis_context(analysis)

    risk_task = Task(
        description=(
            "Review this casino player analysis and write a short risk brief.\n"
            f"{analysis_context}"
        ),
        expected_output="A concise risk-focused note with the top warnings and why they matter.",
        agent=risk_agent,
    )
    strategy_task = Task(
        description=(
            "Using the same analysis context and the previous risk brief, write a final summary "
            "covering prediction, risk, strategy, and what action to take.\n"
            f"{analysis_context}"
        ),
        expected_output="A crisp final answer suitable for showing directly in the app.",
        agent=strategy_agent,
    )

    crew = Crew(
        agents=[risk_agent, strategy_agent],
        tasks=[risk_task, strategy_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()

    analysis["framework"] = "crewai"
    analysis["summary"] = str(result)
    return analysis


def run_langchain_agent(model, player_data, game_plan, goal):
    if not all(item is not None for item in (ChatPromptTemplate, ChatGroq, StrOutputParser)):
        raise RuntimeError("LangChain is not installed.")

    analysis = build_casino_analysis(model, player_data, game_plan, goal)
    api_key = get_api_key()

    if not api_key:
        analysis["framework"] = "langchain_fallback"
        analysis["summary"] = generate_agent_summary(analysis)
        analysis["framework_note"] = "LangChain selected, but no GROQ_API_KEY was found."
        return analysis

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a casino analytics copilot. Summarize the analysis clearly, "
                "highlight the player label, top risks, best strategy, and the safest next action.",
            ),
            (
                "human",
                "Goal: {goal}\nAnalysis:\n{analysis_context}",
            ),
        ]
    )
    llm = ChatGroq(
        model=DEFAULT_GROQ_MODEL,
        temperature=0.2,
        api_key=api_key,
    )
    chain = prompt | llm | StrOutputParser()

    analysis["framework"] = "langchain"
    analysis["summary"] = chain.invoke(
        {
            "goal": goal,
            "analysis_context": _build_analysis_context(analysis),
        }
    )
    return analysis


def run_llamaindex_agent(model, player_data, game_plan, goal):
    if not all(item is not None for item in (PromptTemplate, LlamaIndexGroq)):
        raise RuntimeError("LlamaIndex is not installed.")

    analysis = build_casino_analysis(model, player_data, game_plan, goal)
    api_key = get_api_key()

    if not api_key:
        analysis["framework"] = "llamaindex_fallback"
        analysis["summary"] = generate_agent_summary(analysis)
        analysis["framework_note"] = "LlamaIndex selected, but no GROQ_API_KEY was found."
        return analysis

    template = PromptTemplate(
        "You are a casino analytics copilot.\n"
        "Write a polished summary for the user.\n"
        "Focus on player type, confidence, biggest risks, strategy recommendation, and the safest next action.\n\n"
        "Goal: {goal}\n"
        "Analysis:\n{analysis_context}\n"
    )
    llm = LlamaIndexGroq(
        model=DEFAULT_GROQ_MODEL,
        api_key=api_key,
        temperature=0.2,
    )

    response = llm.complete(
        template.format(
            goal=goal,
            analysis_context=_build_analysis_context(analysis),
        )
    )

    analysis["framework"] = "llamaindex"
    analysis["summary"] = str(getattr(response, "text", response))
    return analysis


def run_analysis(model, player_data, game_plan, goal, framework="custom"):
    selected_framework = framework.lower()

    try:
        if selected_framework == "langgraph":
            return run_langgraph_agent(model, player_data, game_plan, goal)
        if selected_framework == "crewai":
            return run_crewai_agent(model, player_data, game_plan, goal)
        if selected_framework == "langchain":
            return run_langchain_agent(model, player_data, game_plan, goal)
        if selected_framework == "llamaindex":
            return run_llamaindex_agent(model, player_data, game_plan, goal)
        return run_casino_agent(model, player_data, game_plan, goal)
    except Exception as exc:
        analysis = run_casino_agent(model, player_data, game_plan, goal)
        analysis["framework"] = f"{selected_framework}_fallback"
        analysis["framework_note"] = str(exc)
        return analysis
