import pickle
import random
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.framework_agents import get_framework_status
from utils.framework_agents import run_analysis


model = pickle.load(open("model/model.pkl", "rb"))

st.set_page_config(
    page_title="Vegas AI Intelligence Studio",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(9, 19, 34, 0.82);
        --panel-strong: rgba(13, 27, 46, 0.95);
        --border: rgba(116, 149, 181, 0.18);
        --text: #edf4ff;
        --muted: #9db2ca;
        --accent: #ff8a3d;
        --accent-2: #4dd0b6;
        --danger: #ff6b6b;
        --warning: #ffd166;
        --shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 138, 61, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(77, 208, 182, 0.16), transparent 24%),
            linear-gradient(180deg, #08101d 0%, #0d1728 45%, #08111d 100%);
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 64px 64px;
        mask-image: linear-gradient(180deg, transparent, rgba(0,0,0,0.75), transparent);
        opacity: 0.22;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 17, 29, 0.98), rgba(11, 26, 44, 0.98));
        border-right: 1px solid var(--border);
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255, 138, 61, 0.14), rgba(7, 17, 31, 0.92) 38%, rgba(77, 208, 182, 0.12));
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        overflow: hidden;
        position: relative;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -60px;
        top: -60px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 138, 61, 0.35), transparent 65%);
        filter: blur(6px);
        animation: drift 7s ease-in-out infinite alternate;
    }

    .hero-kicker {
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: #ffd7bf;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.05;
        color: var(--text);
        margin-bottom: 0.5rem;
        text-shadow: 0 0 18px rgba(255, 138, 61, 0.2);
    }

    .hero-copy {
        font-size: 1rem;
        color: var(--muted);
        max-width: 760px;
    }

    .glass-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.6fr 0.95fr;
        gap: 1rem;
        align-items: stretch;
    }

    .scene-card {
        position: relative;
        border-radius: 22px;
        min-height: 220px;
        border: 1px solid rgba(255,255,255,0.08);
        background:
            radial-gradient(circle at 30% 25%, rgba(255, 209, 102, 0.22), transparent 24%),
            radial-gradient(circle at 70% 40%, rgba(255, 107, 107, 0.22), transparent 20%),
            linear-gradient(180deg, rgba(17, 10, 18, 0.94), rgba(9, 17, 31, 0.98));
        overflow: hidden;
    }

    .scene-card::before {
        content: "";
        position: absolute;
        left: -8%;
        right: -8%;
        bottom: 0;
        height: 48%;
        background:
            linear-gradient(180deg, transparent, rgba(0,0,0,0.18)),
            linear-gradient(90deg,
                rgba(255, 138, 61, 0.16) 0 8%,
                transparent 8% 12%,
                rgba(77, 208, 182, 0.12) 12% 18%,
                transparent 18% 24%,
                rgba(123, 169, 255, 0.14) 24% 30%,
                transparent 30% 100%);
        clip-path: polygon(0 100%, 0 42%, 8% 35%, 12% 58%, 20% 30%, 27% 62%, 35% 44%, 41% 68%, 49% 28%, 56% 74%, 64% 39%, 70% 60%, 78% 33%, 85% 65%, 93% 42%, 100% 54%, 100% 100%);
        opacity: 0.92;
    }

    .scene-card::after {
        content: "♠  ♥  ♦  ♣";
        position: absolute;
        top: 14px;
        right: 18px;
        color: rgba(255,255,255,0.24);
        letter-spacing: 0.3rem;
        font-size: 1.1rem;
        animation: glowPulse 2.6s ease-in-out infinite;
    }

    .scene-title {
        position: absolute;
        left: 18px;
        top: 18px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }

    .scene-copy {
        position: absolute;
        left: 18px;
        top: 56px;
        max-width: 280px;
        color: #d8e3ef;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .scene-wheel {
        position: absolute;
        right: 22px;
        bottom: 18px;
        width: 108px;
        height: 108px;
        border-radius: 50%;
        background:
            conic-gradient(
                #ff6b6b 0 14%,
                #111a25 14% 28%,
                #4dd0b6 28% 42%,
                #111a25 42% 56%,
                #ffd166 56% 70%,
                #111a25 70% 84%,
                #7ba9ff 84% 100%
            );
        border: 5px solid rgba(255,255,255,0.08);
        box-shadow: 0 0 28px rgba(255, 138, 61, 0.28);
        animation: spinSlow 10s linear infinite;
    }

    .scene-wheel::after {
        content: "";
        position: absolute;
        inset: 26px;
        border-radius: 50%;
        background: rgba(9, 18, 33, 0.96);
        border: 2px solid rgba(255,255,255,0.06);
    }

    .marquee {
        overflow: hidden;
        position: relative;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        margin: 1rem 0 0.35rem;
    }

    .marquee-track {
        display: flex;
        width: max-content;
        animation: marquee 22s linear infinite;
        white-space: nowrap;
    }

    .marquee-item {
        padding: 0.82rem 1.15rem;
        color: #ffd7bf;
        font-size: 0.92rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(14, 27, 46, 0.95), rgba(9, 18, 33, 0.95));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem;
        min-height: 124px;
        box-shadow: var(--shadow);
        transition: transform 180ms ease, border-color 180ms ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 138, 61, 0.32);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .metric-subtle {
        color: #c6d4e5;
        font-size: 0.92rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }

    .pill {
        padding: 0.4rem 0.72rem;
        border-radius: 999px;
        font-size: 0.84rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.05);
    }

    .pill-ready {
        color: #bff7df;
        background: rgba(77, 208, 182, 0.12);
    }

    .pill-off {
        color: #ffd1d1;
        background: rgba(255, 107, 107, 0.12);
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.65rem;
    }

    .slot-board {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255, 138, 61, 0.16), rgba(13, 27, 46, 0.94));
        box-shadow: inset 0 0 40px rgba(255, 138, 61, 0.08), 0 18px 34px rgba(0, 0, 0, 0.22);
    }

    .insight-box {
        background: var(--panel-strong);
        border-left: 4px solid var(--accent);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        color: var(--text);
    }

    .stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 14px;
        border: 0;
        background: linear-gradient(135deg, #ff8a3d, #ff6b6b);
        color: white;
        font-weight: 700;
        box-shadow: 0 16px 30px rgba(255, 107, 107, 0.22);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border);
        padding: 0.45rem 1rem;
    }

    .mini-banner {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .mini-banner-card {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        background: linear-gradient(180deg, rgba(20, 24, 40, 0.9), rgba(12, 20, 34, 0.95));
        border: 1px solid rgba(255,255,255,0.07);
        position: relative;
        overflow: hidden;
    }

    .mini-banner-card::after {
        content: "";
        position: absolute;
        inset: auto -20% -55% auto;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,138,61,0.28), transparent 60%);
    }

    .mini-banner-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #fff1e8;
    }

    .mini-banner-label {
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .floating-note {
        display: inline-block;
        padding: 0.48rem 0.78rem;
        border-radius: 999px;
        background: rgba(255, 138, 61, 0.12);
        border: 1px solid rgba(255, 138, 61, 0.22);
        color: #ffe1cf;
        animation: glowPulse 2.4s ease-in-out infinite;
    }

    @keyframes marquee {
        from { transform: translateX(0); }
        to { transform: translateX(-50%); }
    }

    @keyframes spinSlow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 0 rgba(255,138,61,0); opacity: 0.88; }
        50% { box-shadow: 0 0 24px rgba(255,138,61,0.22); opacity: 1; }
    }

    @keyframes drift {
        from { transform: translateY(0px) translateX(0px); }
        to { transform: translateY(18px) translateX(-12px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_currency(value):
    return f"₹{value:,.2f}"


def stream_text(text, delay=0.008):
    for word in text.split():
        yield word + " "
        time.sleep(delay)


def slot_animation():
    symbols = ["🍒", "💎", "7️⃣", "🔔"]
    slot = st.empty()
    for _ in range(15):
        spin = [random.choice(symbols) for _ in range(3)]
        slot.markdown(
            f"<div class='slot-board'>{'  |  '.join(spin)}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.08)
    final = [random.choice(symbols) for _ in range(3)]
    slot.markdown(
        f"<div class='slot-board'>{'  |  '.join(final)}</div>",
        unsafe_allow_html=True,
    )


framework_status = get_framework_status()
framework_labels = {
    "custom": "Custom",
    "langgraph": "LangGraph",
    "crewai": "CrewAI",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
}

if "leaderboard" not in st.session_state:
    st.session_state["leaderboard"] = []

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None


st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Sin City Command Deck</div>
                <div class="hero-title">Vegas AI Intelligence Studio</div>
                <div class="hero-copy">
                    A cinematic casino control room for player analysis, risk forecasting, strategy optimization,
                    and framework comparison. The goal is to feel custom-built, not generic.
                </div>
                <div style="margin-top:1rem;">
                    <span class="floating-note">Live floor simulation</span>
                </div>
            </div>
            <div class="scene-card">
                <div class="scene-title">Neon Table Monitor</div>
                <div class="scene-copy">
                    Blackjacks, roulette swings, hot streaks, confidence shifts, and safety alerts all in one premium surface.
                </div>
                <div class="scene-wheel"></div>
            </div>
        </div>
        <div class="marquee">
            <div class="marquee-track">
                <div class="marquee-item">High Roller Intelligence</div>
                <div class="marquee-item">Variance Tracking</div>
                <div class="marquee-item">Table Allocation Engine</div>
                <div class="marquee-item">Responsible Play Signals</div>
                <div class="marquee-item">Semantic Strategy Match</div>
                <div class="marquee-item">High Roller Intelligence</div>
                <div class="marquee-item">Variance Tracking</div>
                <div class="marquee-item">Table Allocation Engine</div>
                <div class="marquee-item">Responsible Play Signals</div>
                <div class="marquee-item">Semantic Strategy Match</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Control Center")
    st.caption("Tune the player profile, select the orchestration framework, and run a full analysis.")

    framework = st.selectbox(
        "Agent Framework",
        options=list(framework_labels.keys()),
        format_func=lambda key: framework_labels[key],
    )

    if st.button("Simulate Player"):
        total_games = random.randint(50, 300)
        # Distribute games across different types
        blackjack_games = random.randint(0, total_games // 4)
        poker_games = random.randint(0, total_games // 4)
        roulette_games = random.randint(0, total_games // 4)
        slots_games = total_games - blackjack_games - poker_games - roulette_games
        
        st.session_state["sim"] = {
            "num_games": total_games,
            "win_rate": round(random.uniform(0.45, 0.6), 2),
            "profit": random.randint(-5000, 8000),
            "avg_bet": random.randint(50, 500),
            "streak": random.randint(1, 15),
            "blackjack": blackjack_games,
            "poker": poker_games,
            "roulette": roulette_games,
            "slots": slots_games,
        }

    sim = st.session_state.get("sim", {})

    st.markdown("### Game Rounds")
    blackjack_rounds = st.number_input("Blackjack", 0, 500, sim.get("blackjack", 25))
    poker_rounds = st.number_input("Poker", 0, 500, sim.get("poker", 25))
    roulette_rounds = st.number_input("Roulette", 0, 500, sim.get("roulette", 25))
    slots_rounds = st.number_input("Slots", 0, 500, sim.get("slots", 25))

    # Calculate total games from rounds
    total_games = blackjack_rounds + poker_rounds + roulette_rounds + slots_rounds

    st.markdown("### Player Inputs")
    st.markdown(f"**Total Games: {total_games}** (sum of all game rounds)")
    num_games = total_games  # Use calculated total
    win_rate = st.slider("Win Rate", 0.0, 1.0, sim.get("win_rate", 0.5), step=0.01)
    profit = st.number_input("Total Profit", -10000, 10000, sim.get("profit", 0))
    avg_bet = st.number_input("Average Bet", 10, 1000, sim.get("avg_bet", 100))
    streak = st.number_input("Win Streak", 1, 20, sim.get("streak", 5))

    analyze = st.button("Analyze Player")


status_markup = "".join(
    [
        (
            f"<span class='pill pill-ready'>{framework_labels[name]} ready</span>"
            if available
            else f"<span class='pill pill-off'>{framework_labels[name]} unavailable</span>"
        )
        for name, available in framework_status.items()
    ]
)

st.markdown(
    f"""
    <div class="glass-card">
        <div class="section-title">Framework Availability</div>
        <div class="pill-row">{status_markup}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


if analyze:
    player_data = {
        "num_games": num_games,
        "win_rate": win_rate,
        "profit": profit,
        "avg_bet": avg_bet,
        "streak": streak,
    }
    game_plan = {
        "blackjack": blackjack_rounds,
        "poker": poker_rounds,
        "roulette": roulette_rounds,
        "slots": slots_rounds,
    }

    with st.container():
        st.markdown("### Live Simulation")
        slot_animation()
        progress = st.progress(0, text="Running analysis agents...")
        for i in range(100):
            time.sleep(0.008)
            progress.progress(i + 1, text="Running analysis agents...")

    st.session_state["analysis_result"] = run_analysis(
        model=model,
        player_data=player_data,
        game_plan=game_plan,
        goal="Analyze this player and recommend the best casino strategy",
        framework=framework,
    )
    
    # Save current inputs to session state for persistence
    st.session_state["sim"] = {
        "num_games": num_games,
        "win_rate": win_rate,
        "profit": profit,
        "avg_bet": avg_bet,
        "streak": streak,
        "blackjack": blackjack_rounds,
        "poker": poker_rounds,
        "roulette": roulette_rounds,
        "slots": slots_rounds,
    }
    
    st.session_state["leaderboard"].append(
        {
            "Score": st.session_state["analysis_result"]["score"],
            "Type": st.session_state["analysis_result"]["prediction"]["label"],
        }
    )

result = st.session_state.get("analysis_result")

if result is None:
    st.info("Enter player details in the sidebar and click Analyze Player to generate the dashboard.")
else:
    profit_history = result["profit_history"]
    variance = result["player_data"]["variance"]
    label = result["prediction"]["label"]
    confidence = result["confidence"]
    score = result["score"]
    total_expected = result["strategy"]["total_expected"]
    risk_level = result["risk"]["level"]
    addiction_risk = result["addiction_risk"]
    loss_prediction = result["loss_prediction"]
    allocation = result["optimization"]["allocation"]
    expected_returns = result["strategy"]["expected_returns"]
    retrieval = result["retrieval"]
    benchmark_metrics = result["benchmark_metrics"]
    explainability = result["explainability"]

    st.markdown(
        f"""
        <div class="mini-banner">
            <div class="mini-banner-card">
                <div class="mini-banner-label">Framework</div>
                <div class="mini-banner-value">{result.get('framework', 'custom').title()}</div>
            </div>
            <div class="mini-banner-card">
                <div class="mini-banner-label">Trend</div>
                <div class="mini-banner-value">{result['simulation']['trend']}</div>
            </div>
            <div class="mini-banner-card">
                <div class="mini-banner-label">Safe Budget</div>
                <div class="mini-banner-value">{format_currency(result['financial_guard']['safe_budget'])}</div>
            </div>
            <div class="mini-banner-card">
                <div class="mini-banner-label">Worst Case</div>
                <div class="mini-banner-value">{format_currency(loss_prediction['projected_worst_case'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metrics = [
        ("Player Label", label, f"Confidence {confidence:.2f}%"),
        ("Variance", f"{variance:,.2f}", f"Risk level {risk_level}"),
        ("Expected Return", format_currency(total_expected), result["strategy"]["recommendation"]),
        ("Score", f"{score:.2f}", result["final_action"]),
    ]
    for col, metric in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{metric[0]}</div>
                    <div class="metric-value">{metric[1]}</div>
                    <div class="metric-subtle">{metric[2]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result.get("framework_note"):
        st.warning(result["framework_note"])

    tab_overview, tab_strategy, tab_risk, tab_search, tab_leaderboard = st.tabs(
        ["Overview", "Strategy", "Risk", "Semantic Search", "Leaderboard"]
    )

    with tab_overview:
        left_col, right_col = st.columns([1.6, 1])
        with left_col:
            history_df = pd.DataFrame(
                {"Game": range(len(profit_history)), "Profit": profit_history}
            )
            history_fig = px.line(
                history_df,
                x="Game",
                y="Profit",
                template="plotly_dark",
                color_discrete_sequence=["#ff8a3d"],
            )
            history_fig.update_layout(
                title="Profit History Simulation",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(history_fig, use_container_width=True)

            trajectory_df = pd.DataFrame(
                {
                    "Segment": ["Opening", "Mid Session", "Late Session", "Projection"],
                    "Value": [
                        float(profit_history[0]) if len(profit_history) else 0,
                        float(profit_history[len(profit_history) // 3]) if len(profit_history) > 2 else 0,
                        float(profit_history[(2 * len(profit_history)) // 3]) if len(profit_history) > 3 else 0,
                        float(result["simulation"]["future_profit"]),
                    ],
                }
            )
            trajectory_fig = px.area(
                trajectory_df,
                x="Segment",
                y="Value",
                template="plotly_dark",
                color_discrete_sequence=["#4dd0b6"],
            )
            trajectory_fig.update_layout(
                title="Casino Session Momentum",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(trajectory_fig, use_container_width=True)

        with right_col:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={"text": "Prediction Confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#4dd0b6"},
                        "bgcolor": "rgba(255,255,255,0.06)",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,107,107,0.18)"},
                            {"range": [50, 75], "color": "rgba(255,209,102,0.18)"},
                            {"range": [75, 100], "color": "rgba(77,208,182,0.22)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#edf4ff"},
            )
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>Framework:</strong> {result.get('framework', 'custom')}<br>
                    <strong>Future Trend:</strong> {result['simulation']['trend']}<br>
                    <strong>Projected Future Profit:</strong> {format_currency(result['simulation']['future_profit'])}<br>
                    <strong>Safe Budget:</strong> {format_currency(result['financial_guard']['safe_budget'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            vibe_df = pd.DataFrame(
                {
                    "Axis": ["Skill", "Risk", "Discipline", "Volatility", "Recovery"],
                    "Value": [
                        confidence,
                        min(result["risk"]["risk_index"] / 200, 100),
                        max(20, 100 - addiction_risk["score"]),
                        min(100, variance / 150),
                        max(10, 100 - min(abs(loss_prediction["expected_loss"]) / 50, 90)),
                    ],
                }
            )
            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=vibe_df["Value"],
                    theta=vibe_df["Axis"],
                    fill="toself",
                    line=dict(color="#ff8a3d"),
                    fillcolor="rgba(255,138,61,0.22)",
                    name="Player Aura",
                )
            )
            radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#edf4ff"),
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                height=340,
                title="Player Aura Map",
                font={"color": "#edf4ff"},
            )
            st.plotly_chart(radar, use_container_width=True)

        st.markdown("### AI Insight")
        with st.container():
            st.write_stream(stream_text(result["summary"]))

        with st.expander("Model Explainability"):
            st.write("Top supporting signals")
            for factor in explainability["supporting_factors"]:
                st.write(
                    f"- {factor['label'].title()}: this profile is closer to the {label.lower()} baseline than the opposite class."
                )

            if explainability["cautionary_factors"]:
                st.write("Signals that reduce certainty")
                for factor in explainability["cautionary_factors"]:
                    st.write(
                        f"- {factor['label'].title()}: this feature is less aligned with the predicted class, so confidence should be tempered."
                    )

        with st.expander("Benchmark Metrics"):
            holdout = benchmark_metrics["holdout"]
            cross_validation = benchmark_metrics["cross_validation"]
            overall = benchmark_metrics["overall"]
            stress_test = benchmark_metrics["stress_test"]

            st.caption(
                f"Dataset rows: {overall['dataset_rows']} | "
                f"Out-of-fold accuracy: {overall['oof_accuracy']}% | "
                f"5-fold cross-validation for more realistic evaluation."
            )

            st.write("Hold-out split")
            bench_cols = st.columns(6)
            bench_cols[0].metric("Accuracy", f"{holdout['accuracy']}%")
            bench_cols[1].metric("Precision", f"{holdout['precision']}%")
            bench_cols[2].metric("Recall", f"{holdout['recall']}%")
            bench_cols[3].metric("F1 Score", f"{holdout['f1']}%")
            bench_cols[4].metric("Brier", f"{holdout['brier']}")
            bench_cols[5].metric("Stress Acc.", f"{stress_test['accuracy']}%")

            st.write("Cross-validation mean ± std")
            cv_cols = st.columns(4)
            cv_cols[0].metric(
                "Accuracy CV",
                f"{cross_validation['accuracy']['mean']}%",
                f"±{cross_validation['accuracy']['std']}",
            )
            cv_cols[1].metric(
                "Precision CV",
                f"{cross_validation['precision']['mean']}%",
                f"±{cross_validation['precision']['std']}",
            )
            cv_cols[2].metric(
                "Recall CV",
                f"{cross_validation['recall']['mean']}%",
                f"±{cross_validation['recall']['std']}",
            )
            cv_cols[3].metric(
                "F1 CV",
                f"{cross_validation['f1']['mean']}%",
                f"±{cross_validation['f1']['std']}",
            )

            st.caption(
                f"Held-out test size: {holdout['test_size']} rows. "
                f"Class balance: Skilled {overall['class_balance']['skilled_pct']}%, "
                f"Lucky {overall['class_balance']['lucky_pct']}%."
            )
            st.caption(f"Robustness check: {stress_test['scenario']}")
            cm = holdout["confusion_matrix"]
            cm_df = pd.DataFrame(
                [
                    ["Lucky", cm["true_lucky"], cm["false_skilled"]],
                    ["Skilled", cm["false_lucky"], cm["true_skilled"]],
                ],
                columns=["Actual", "Predicted Lucky", "Predicted Skilled"],
            )
            st.dataframe(cm_df, use_container_width=True)

            calibration = benchmark_metrics["calibration"]
            # Use clean_data calibration for realistic evaluation
            clean_cal = calibration.get("clean_data", calibration.get("predicted"))
            
            if isinstance(clean_cal, dict):
                pred_probs = clean_cal["predicted"]
                obs_rates = clean_cal["observed"]
            else:
                # Fallback for old format
                pred_probs = calibration["predicted"]
                obs_rates = calibration["observed"]
            
            calibration_df = pd.DataFrame(
                {
                    "Predicted Probability": pred_probs,
                    "Observed Positive Rate": obs_rates,
                }
            )
            calibration_fig = px.line(
                calibration_df,
                x="Predicted Probability",
                y="Observed Positive Rate",
                markers=True,
                template="plotly_dark",
                color_discrete_sequence=["#ffd166"],
            )
            calibration_fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=100,
                y1=100,
                line=dict(color="#4dd0b6", dash="dash"),
            )
            calibration_fig.update_layout(
                title="Calibration Curve",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Mean predicted probability (%)",
                yaxis_title="Observed positive rate (%)",
            )
            st.plotly_chart(calibration_fig, use_container_width=True)

        with st.expander("Limitations"):
            for item in benchmark_metrics["limitations"]:
                st.write(f"- {item}")

    with tab_strategy:
        strategy_cols = st.columns(2)
        with strategy_cols[0]:
            st.markdown("### Expected Returns by Game")
            returns_df = pd.DataFrame(
                {
                    "Game": [game.title() for game in expected_returns.keys()],
                    "Expected Return": list(expected_returns.values()),
                }
            )
            returns_fig = px.bar(
                returns_df,
                x="Game",
                y="Expected Return",
                template="plotly_dark",
                color="Expected Return",
                color_continuous_scale=["#ff6b6b", "#ffd166", "#4dd0b6"],
            )
            returns_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(returns_fig, use_container_width=True)

        with strategy_cols[1]:
            st.markdown("### Optimal Allocation")
            allocation_df = pd.DataFrame(
                {
                    "Game": [game.title() for game in allocation.keys()],
                    "Rounds": list(allocation.values()),
                }
            )
            allocation_fig = px.pie(
                allocation_df,
                names="Game",
                values="Rounds",
                hole=0.55,
                color_discrete_sequence=["#ff8a3d", "#4dd0b6", "#ffd166", "#7ba9ff"],
            )
            allocation_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#edf4ff"},
            )
            st.plotly_chart(allocation_fig, use_container_width=True)

        st.markdown("### Strategy Advisor")
        for game, value in expected_returns.items():
            st.write(f"{game.title()}: {format_currency(value)}")
        st.success(f"Recommended focus: {result['strategy']['best_game'].title()}")
        st.caption(f"Total expected outcome: {format_currency(total_expected)}")

    with tab_risk:
        risk_cols = st.columns(3)
        risk_cols[0].metric("Behavioral Risk", addiction_risk["level"], f"{addiction_risk['score']}/100")
        risk_cols[1].metric("Projected Worst Loss", format_currency(loss_prediction["projected_worst_case"]), loss_prediction["level"])
        risk_cols[2].metric("Risk Index", f"{result['risk']['risk_index']:,.2f}", risk_level)

        st.markdown("### Real-World Impact")
        if addiction_risk["level"] == "High":
            st.error(f"Addiction Risk: {addiction_risk['level']} ({addiction_risk['score']}/100)")
        elif addiction_risk["level"] == "Moderate":
            st.warning(f"Addiction Risk: {addiction_risk['level']} ({addiction_risk['score']}/100)")
        else:
            st.success(f"Addiction Risk: {addiction_risk['level']} ({addiction_risk['score']}/100)")

        st.caption(addiction_risk["disclaimer"])
        if addiction_risk["factors"]:
            st.write("Key factors")
            for factor in addiction_risk["factors"]:
                st.write(f"- {factor}")

        st.markdown("### Loss Prediction")
        st.write(f"Expected loss: {format_currency(loss_prediction['expected_loss'])}")
        st.write(
            f"Projected worst-case loss: {format_currency(loss_prediction['projected_worst_case'])}"
        )
        st.write(f"Loss risk level: {loss_prediction['level']}")

        st.markdown("### Intervention Guidance")
        for action in result["intervention"]:
            st.write(f"- {action}")

        st.markdown("### Risk Alerts")
        for alert in result["alerts"]:
            if "No major" in alert:
                st.success(alert)
            else:
                st.warning(alert)

    with tab_search:
        st.caption(f"Search mode: {retrieval['mode']}")
        for item in retrieval["results"]:
            with st.container():
                st.markdown(f"### {item['title']}")
                st.caption(f"Source: {item['source']} | Similarity: {item['score']:.2f}")
                st.write(item["text"])

    with tab_leaderboard:
        leaderboard_df = pd.DataFrame(st.session_state["leaderboard"])
        leaderboard_df = leaderboard_df.sort_values(by="Score", ascending=False)
        st.dataframe(leaderboard_df.head(10), use_container_width=True)
