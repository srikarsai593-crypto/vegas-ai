import random
import numpy as np
from utils.evaluation import explain_prediction_drivers
from utils.evaluation import get_reference_metrics
from utils.features import create_input_features
from utils.llm import generate_agent_summary
from utils.semantic_search import build_player_query, semantic_search


GAME_RETURNS = {
    "blackjack": 0.48,
    "poker": 0.52,
    "roulette": 0.47,
    "slots": 0.45,
}

OPTIMAL_WEIGHTS = {
    "blackjack": 0.3,
    "poker": 0.4,
    "roulette": 0.2,
    "slots": 0.1,
}


# 🎯 SIMULATION
def simulate_profit_history(num_games, profit, scale=100):
    return np.random.normal(
        loc=profit / max(num_games, 1),
        scale=scale,
        size=num_games,
    )


def calculate_variance(profit_history):
    return float(np.var(profit_history))


def simulation_agent(player_data):
    future_profit = (
        player_data["num_games"]
        * player_data["avg_bet"]
        * (player_data["win_rate"] - 0.5)
    )

    return {
        "future_profit": round(future_profit, 2),
        "trend": "Positive" if future_profit > 0 else "Negative",
    }


# 🧠 BEHAVIOR
def predict_player_type(model, player_data):
    features = create_input_features(
        player_data["num_games"],
        player_data["win_rate"],
        player_data["variance"],
        player_data["profit"],
        player_data["avg_bet"],
        player_data["streak"],
    )

    result = model.predict([features])[0]

    return {
        "features": features,
        "result": int(result),
        "label": "Skilled" if result == 1 else "Lucky",
    }


# 🎯 CONFIDENCE
def calculate_confidence(model, features, player_data):
    try:
        proba = model.predict_proba([features])[0]
        top_proba = float(max(proba))
        sorted_proba = sorted([float(value) for value in proba], reverse=True)
        margin = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else top_proba
    except Exception:
        top_proba = 0.68
        margin = 0.18

    sample_factor = min(player_data["num_games"] / 300, 1.0)
    variance_penalty = min(player_data["variance"] / 20000, 1.0)
    streak_factor = min(player_data["streak"] / 20, 1.0)

    confidence = (
        40
        + (top_proba * 35)
        + (margin * 25)
        + (sample_factor * 10)
        + (streak_factor * 5)
        - (variance_penalty * 15)
    )

    # Avoid forcing a narrow confidence ceiling;
    # let the model express higher certainty when data supports it.
    return round(max(40, min(confidence, 99)), 2)


# ⚠ RISK
def calculate_risk(player_data):
    variance = player_data["variance"]
    avg_bet = player_data["avg_bet"]

    if variance < 5000 and avg_bet < 200:
        level = "Low"
    elif variance < 10000 and avg_bet < 500:
        level = "Medium"
    else:
        level = "High"

    return {
        "level": level,
        "risk_index": round(avg_bet * variance, 2),
    }


# 🚨 ADDICTION
def detect_addiction_risk(player_data, game_plan):
    total_rounds = sum(game_plan.values())
    score = 0
    factors = []

    if player_data["streak"] >= 10:
        score += 25
        factors.append("long streak")

    if player_data["avg_bet"] >= 400:
        score += 25
        factors.append("high bet size")

    if player_data["num_games"] >= 250 or total_rounds >= 250:
        score += 20
        factors.append("high volume")

    if player_data["profit"] < -2000:
        score += 20
        factors.append("existing losses")

    if player_data["variance"] >= 10000:
        score += 10
        factors.append("high volatility")

    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "level": level,
        "score": min(score, 100),
        "factors": factors,
        "disclaimer": "Behavioral heuristic only",
    }


# 📉 LOSS (FINAL FIXED VERSION)
def predict_loss_risk(player_data, game_plan):
    total_rounds = max(sum(game_plan.values()), 1)

    house_edge = sum(
        rounds * (0.5 - GAME_RETURNS[game])
        for game, rounds in game_plan.items()
    ) / total_rounds

    expected_loss = total_rounds * player_data["avg_bet"] * house_edge
    projected_worst_case = expected_loss + np.sqrt(player_data["variance"])

    if projected_worst_case > 5000:
        level = "High"
    elif projected_worst_case > 2000:
        level = "Medium"
    else:
        level = "Low"

    return {
        "expected_loss": round(expected_loss, 2),
        "projected_worst_case": round(projected_worst_case, 2),
        "level": level,
    }


# 🚨 ALERTS
def generate_risk_alerts(player_data, risk, addiction_risk, loss_prediction):
    alerts = []

    if addiction_risk["level"] != "Low":
        alerts.append(f"Addiction Risk: {addiction_risk['level']}")

    if loss_prediction["projected_worst_case"] > 3000:
        alerts.append("High potential loss detected")

    if risk["level"] == "High":
        alerts.append("High volatility risk")

    if not alerts:
        alerts.append("No major risk detected")

    return alerts


# 🎰 STRATEGY
def calculate_strategy(game_plan, avg_bet):
    expected_returns = {}

    for game, rounds in game_plan.items():
        expected_returns[game] = rounds * avg_bet * (GAME_RETURNS[game] - 0.5)

    best_game = max(expected_returns, key=expected_returns.get)

    return {
        "expected_returns": expected_returns,
        "total_expected": sum(expected_returns.values()),
        "best_game": best_game,
        "recommendation": f"Focus on {best_game}",
    }


# 🎯 OPTIMIZER
def optimize_game_mix(game_plan):
    total_rounds = sum(game_plan.values())

    return {
        "total_rounds": total_rounds,
        "allocation": {
            g: int(total_rounds * w)
            for g, w in OPTIMAL_WEIGHTS.items()
        },
    }


# 💰 FINANCIAL
def financial_guard_agent(player_data):
    return {
        "safe_budget": round(player_data["avg_bet"] * player_data["num_games"] * 0.2, 2),
    }


# 🚨 INTERVENTION
def intervention_agent(addiction_risk, loss_prediction):
    actions = []

    if addiction_risk["level"] == "High":
        actions.append("Stop playing immediately")

    if loss_prediction["projected_worst_case"] > 3000:
        actions.append("Reduce betting size")

    if not actions:
        actions.append("No intervention needed")

    return actions


# 📊 SCORE
def calculate_score(player_data):
    return round(
        (player_data["win_rate"] * 100)
        - (player_data["variance"] * 0.005)
        + (player_data["profit"] / 1000),
        2,
    )


# 🧠 CONTROLLER
def build_casino_analysis(model, player_data, game_plan, goal):
    profit_history = simulate_profit_history(
        player_data["num_games"], player_data["profit"]
    )

    enriched_player_data = {
        **player_data,
        "variance": calculate_variance(profit_history),
    }

    prediction = predict_player_type(model, enriched_player_data)
    confidence = calculate_confidence(model, prediction["features"], enriched_player_data)
    risk = calculate_risk(enriched_player_data)
    strategy = calculate_strategy(game_plan, enriched_player_data["avg_bet"])
    optimization = optimize_game_mix(game_plan)
    simulation = simulation_agent(enriched_player_data)

    score = calculate_score(enriched_player_data)

    addiction_risk = detect_addiction_risk(enriched_player_data, game_plan)
    loss_prediction = predict_loss_risk(enriched_player_data, game_plan)
    alerts = generate_risk_alerts(enriched_player_data, risk, addiction_risk, loss_prediction)

    financial_guard = financial_guard_agent(enriched_player_data)
    intervention = intervention_agent(addiction_risk, loss_prediction)

    retrieval_query = build_player_query(
        goal, enriched_player_data, prediction["label"], risk["level"]
    )
    retrieval = semantic_search(retrieval_query, top_k=4)
    explainability = explain_prediction_drivers(enriched_player_data, prediction)
    benchmark_metrics = get_reference_metrics()

    final_action = (
        "Reduce play immediately" if risk["level"] == "High"
        else "Proceed with optimized strategy"
    )

    analysis = {
        "goal": goal,
        "game_plan": game_plan,
        "player_data": enriched_player_data,
        "prediction": prediction,
        "confidence": confidence,
        "risk": risk,
        "strategy": strategy,
        "optimization": optimization,
        "simulation": simulation,
        "score": score,
        "addiction_risk": addiction_risk,
        "loss_prediction": loss_prediction,
        "alerts": alerts,
        "financial_guard": financial_guard,
        "intervention": intervention,
        "retrieval": retrieval,
        "explainability": explainability,
        "benchmark_metrics": benchmark_metrics,
        "final_action": final_action,
        "profit_history": profit_history,
    }

    return analysis


def run_casino_agent(model, player_data, game_plan, goal):
    analysis = build_casino_analysis(model, player_data, game_plan, goal)
    analysis["framework"] = "custom"
    analysis["summary"] = generate_agent_summary(analysis)
    return analysis
