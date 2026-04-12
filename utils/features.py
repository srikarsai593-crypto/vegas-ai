def create_features(df):
    df["profit_per_game"] = df["profit"] / df["num_games"]
    df["consistency_score"] = 1 / (df["variance"] + 0.001)
    df["risk_index"] = df["avg_bet"] * df["variance"]
    return df


def create_input_features(num_games, win_rate, variance, profit, avg_bet, streak):
    profit_per_game = profit / num_games
    consistency_score = 1 / (variance + 0.001)
    risk_index = avg_bet * variance

    return [
        num_games,
        win_rate,
        variance,
        profit,
        avg_bet,
        streak,
        profit_per_game,
        consistency_score,
        risk_index
    ]