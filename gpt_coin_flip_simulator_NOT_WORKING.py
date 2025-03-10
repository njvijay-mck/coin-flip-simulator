import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Biased Coin Flip Function ---
def flip_biased_coin(p_heads=0.6):
    return 'Heads' if np.random.random() < p_heads else 'Tails'

# Simulation Logic
def run_simulation(initial_capital, p_heads, fractional_kelly, flips, cap):
    capital = initial_capital
    capital_history = [capital]
    bet_fraction = fractional_kelly * (2 * p_heads - 1)

    for flip in range(flips):
        bet_amount = capital * bet_fraction
        outcome = "Heads" if np.random.random() < p_heads else "Tails"

        if outcome == "Heads":
            capital *= (1 + bet_fraction)
        else:
            capital *= (1 - bet_fraction)

        capital = min(capital, cap)
        capital_history.append(capital)

        if capital <= 0.01 or capital >= cap:
            break

    return capital_history

# Streamlit UI
st.title("Biased Coin Betting Simulation with Kelly Criterion")

# Sidebar Inputs
st.sidebar.header("Simulation Parameters")
initial_capital = st.sidebar.number_input("Initial Capital", value=25.0, min_value=1.0)
p_heads = st.sidebar.slider("Probability of Heads", 0.5, 0.9, 0.6, 0.01)
fractional_kelly = st.sidebar.slider("Fractional Kelly (%)", 5, 100, 20) / 100
number_of_flips = st.sidebar.slider("Number of Flips", 10, 500, 300)
num_simulations = st.sidebar.selectbox("Number of Simulations", [100, 500, 1000, 5000])
cap = st.sidebar.number_input("Maximum Payout (Cap)", min_value=50, max_value=10000, value=250)

params = {
    "initial_capital": initial_capital,
    "p_heads": 0.6,
    "fractional_kelly": fractional_kelly,
    "flips": number_of_flips,
    "cap": cap
}

# Single Simulation Example
st.subheader("Example Single Simulation")
single_run = run_simulation(**params)

fig_single = go.Figure()
fig_single.add_trace(
    go.Scatter(y=single_run, mode='lines+markers', name='Wealth')
)
fig.update_layout(
    title="Wealth Progression (Single Run)",
    xaxis_title="Flip Number",
    yaxis_title="Capital ($)",
)
st.plotly_chart(fig)

# Monte Carlo Simulations
st.subheader("Monte Carlo Simulations")

final_wealths = []
for _ in range(num_simulations):
    history = run_simulation(**params)
    final_wealths.append(history[-1])

# Results Visualization
fig_hist = go.Figure()
fig.add_trace(go.Histogram(x=final_wealths, nbinsx=50))

fig.update_layout(
    title=f"Distribution of Final Wealth after {num_simulations} Simulations",
    xaxis_title="Final Wealth ($)",
    yaxis_title="Frequency",
)
st.plotly_chart(fig)

# Calculate Metrics
mean_final = np.mean(final_wealths)
median_final = np.median(final_wealths)
std_final = np.std(final_wealths)
prob_cap = np.mean(np.array(final_wealths) >= cap)
prob_ruin = np.mean(np.array(final_wealths) <= 1.0)

st.subheader("Summary Metrics")
st.markdown(f"""
- **Mean Final Wealth**: ${mean_final:.2f}
- **Median Final Wealth**: ${median_final:.2f}
- **Volatility (Std Dev)**: ${std_final:.2f}
- **Probability of Hitting Cap (${cap})**: {prob_cap:.2%}
- **Probability of Ruin**: {prob_ruin:.2%}
""")
