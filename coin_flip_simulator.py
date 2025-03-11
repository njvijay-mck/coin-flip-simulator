import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Trading Simulation with Kelly Criterion",
    page_icon="🪙",
    layout="wide"
)

# App title and description
st.title("Trading Simulation with Kelly Criterion")
st.markdown("""
This application simulates a trading strategy using a weighted coin flip model with Kelly criterion position sizing.
You can adjust parameters such as probability bias, position size, Kelly fraction, and risk-reward ratio.
Monte Carlo simulation is available to analyze the variability of outcomes.
""")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Probability of heads parameter
heads_prob = st.sidebar.slider(
    "Probability of Heads",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.01,
    help="Set the probability of getting heads when flipping the coin"
)

# Bet on heads or tails
bet_on = st.sidebar.radio(
    "Bet On",
    options=["Heads", "Tails"],
    index=0,
    help="Choose whether to bet on heads or tails"
)

# Initial capital
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Set your initial capital amount"
)

# Risk-reward ratio
risk_reward_ratio = st.sidebar.slider(
    "Risk-Reward Ratio (Reward:Risk)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Set the risk-reward ratio (e.g., 2.0 means potential gain is twice the potential loss)"
)

# Kelly fraction
use_kelly = st.sidebar.checkbox(
    "Use Kelly Criterion",
    value=True,
    help="Use Kelly criterion for position sizing"
)

kelly_fraction = st.sidebar.slider(
    "Kelly Fraction",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Fraction of the full Kelly criterion to use (0.5 = half Kelly)"
)

# Number of flips parameter
num_flips = st.sidebar.number_input(
    "Number of Flips",
    min_value=1,
    max_value=10000,
    value=100,
    step=10,
    help="Set the number of times to flip the coin"
)

# Monte Carlo parameters
st.sidebar.header("Monte Carlo Parameters")

run_monte_carlo = st.sidebar.checkbox(
    "Run Monte Carlo Simulation",
    value=False,
    help="Run multiple simulations to analyze the distribution of outcomes"
)

num_simulations = st.sidebar.number_input(
    "Number of Simulations",
    min_value=10,
    max_value=10000,
    value=100,
    step=10,
    help="Set the number of Monte Carlo simulations to run",
    disabled=not run_monte_carlo
)

# Function to calculate Kelly criterion
def calculate_kelly(win_prob, win_loss_ratio):
    """
    Calculate the optimal Kelly criterion bet size
    
    Parameters:
    win_prob (float): Probability of winning
    win_loss_ratio (float): Ratio of win amount to loss amount
    
    Returns:
    float: Optimal fraction of bankroll to bet
    """
    # Kelly formula: f* = (p * b - q) / b
    # where p = probability of winning, q = probability of losing (1-p),
    # b = win/loss ratio
    
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    
    q = 1 - win_prob
    kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
    
    # Kelly can be negative, which means don't bet
    return max(0, kelly)

# Function to simulate coin flips
def simulate_coin_flips(heads_probability, num_flips):
    """
    Simulate coin flips
    
    Parameters:
    heads_probability (float): Probability of getting heads
    num_flips (int): Number of flips to simulate
    
    Returns:
    list: List of 1 (heads) or 0 (tails)
    """
    # Generate random numbers between 0 and 1
    random_values = np.random.random(num_flips)
    
    # Convert to heads (True) or tails (False) based on probability
    flips = random_values < heads_probability
    
    # Convert to 1 for heads, 0 for tails for easier analysis
    return [1 if flip else 0 for flip in flips]

# Function to simulate trading with Kelly criterion
def simulate_trading(flips, bet_on_heads, initial_capital, risk_reward_ratio, kelly_fraction, use_kelly=True):
    """
    Simulate trading with Kelly criterion
    
    Parameters:
    flips (list): List of coin flip results (1 for heads, 0 for tails)
    bet_on_heads (bool): True if betting on heads, False if betting on tails
    initial_capital (float): Initial capital
    risk_reward_ratio (float): Risk-reward ratio
    kelly_fraction (float): Fraction of Kelly to use
    use_kelly (bool): Whether to use Kelly criterion
    
    Returns:
    pd.DataFrame: DataFrame with simulation results
    """
    # Calculate win probability based on bet choice
    win_prob = heads_prob if bet_on_heads else 1 - heads_prob
    
    # Calculate Kelly bet size
    full_kelly = calculate_kelly(win_prob, risk_reward_ratio)
    bet_fraction = full_kelly * kelly_fraction if use_kelly else 0.01  # Default to 1% if not using Kelly
    
    # Initialize results
    results = []
    current_capital = initial_capital
    max_capital = initial_capital
    cumulative_heads = 0
    cumulative_tails = 0
    
    for i, flip in enumerate(flips):
        # Record wealth before flip
        wealth_before = current_capital
        
        # Calculate bet amount
        bet_amount = current_capital * bet_fraction
        
        # Determine if bet wins
        is_heads = flip == 1
        cumulative_heads += 1 if is_heads else 0
        cumulative_tails += 0 if is_heads else 1
        
        win = (is_heads and bet_on_heads) or (not is_heads and not bet_on_heads)
        
        # Update capital
        if win:
            current_capital += bet_amount * risk_reward_ratio
        else:
            current_capital -= bet_amount
        
        # Update max capital for drawdown calculation
        max_capital = max(max_capital, current_capital)
        
        # Calculate drawdown
        drawdown = (max_capital - current_capital) / max_capital * 100 if max_capital > 0 else 0
        
        # Calculate ROI
        roi = (current_capital - initial_capital) / initial_capital * 100
        
        # Add to results
        results.append({
            'Flip Number': i + 1,
            'Bet Fraction': bet_fraction,
            'Bet Amount': bet_amount,
            'Outcome': 'Heads' if is_heads else 'Tails',
            'Win': win,
            'Wealth Before Flip': wealth_before,
            'Wealth After Flip': current_capital,
            'Cumulative ROI (%)': roi,
            'Drawdown (%)': drawdown,
            'Cumulative Heads': cumulative_heads,
            'Cumulative Tails': cumulative_tails
        })
    
    return pd.DataFrame(results)

# Function to run Monte Carlo simulations
def run_monte_carlo_simulations(heads_probability, bet_on_heads, initial_capital, 
                               risk_reward_ratio, kelly_fraction, use_kelly, 
                               num_flips, num_simulations):
    """
    Run multiple trading simulations to analyze the distribution of outcomes
    
    Parameters:
    heads_probability (float): Probability of getting heads
    bet_on_heads (bool): True if betting on heads, False if betting on tails
    initial_capital (float): Initial capital
    risk_reward_ratio (float): Risk-reward ratio
    kelly_fraction (float): Fraction of Kelly to use
    use_kelly (bool): Whether to use Kelly criterion
    num_flips (int): Number of flips per simulation
    num_simulations (int): Number of simulations to run
    
    Returns:
    dict: Dictionary with simulation results
    """
    # Initialize results
    final_capitals = []
    max_drawdowns = []
    roi_values = []
    wealth_trajectories = []
    
    # Run simulations
    for i in range(num_simulations):
        # Simulate coin flips
        flips = simulate_coin_flips(heads_probability, num_flips)
        
        # Run trading simulation
        results = simulate_trading(flips, bet_on_heads, initial_capital, 
                                  risk_reward_ratio, kelly_fraction, use_kelly)
        
        # Extract key metrics
        final_capital = results['Wealth After Flip'].iloc[-1]
        max_drawdown = results['Drawdown (%)'].max()
        roi = (final_capital - initial_capital) / initial_capital * 100
        
        # Store results
        final_capitals.append(final_capital)
        max_drawdowns.append(max_drawdown)
        roi_values.append(roi)
        
        # Store wealth trajectory
        wealth_trajectories.append(results['Wealth After Flip'].tolist())
    
    # Convert wealth trajectories to DataFrame for easier plotting
    wealth_df = pd.DataFrame(wealth_trajectories).T
    wealth_df.index = range(1, num_flips + 1)
    
    return {
        'final_capitals': final_capitals,
        'max_drawdowns': max_drawdowns,
        'roi_values': roi_values,
        'wealth_trajectories': wealth_df
    }

# Button to run simulation
if st.sidebar.button("Run Simulation"):
    # Run the simulation
    st.session_state.flips = simulate_coin_flips(heads_prob, num_flips)
    st.session_state.simulation_run = True
    
    # Calculate statistics
    heads_count = sum(st.session_state.flips)
    tails_count = num_flips - heads_count
    actual_heads_prob = heads_count / num_flips
    
    # Determine win probability based on bet choice
    win_prob = heads_prob if bet_on == "Heads" else 1 - heads_prob
    
    # Calculate Kelly bet size
    full_kelly = calculate_kelly(win_prob, risk_reward_ratio)
    actual_kelly = full_kelly * kelly_fraction
    
    # Run trading simulation
    bet_on_heads = bet_on == "Heads"
    st.session_state.trading_results = simulate_trading(
        st.session_state.flips,
        bet_on_heads,
        initial_capital,
        risk_reward_ratio,
        kelly_fraction,
        use_kelly
    )
    
    # Display statistics
    st.subheader("Simulation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Heads Count", heads_count)
    with col2:
        st.metric("Tails Count", tails_count)
    with col3:
        st.metric("Actual Heads Probability", f"{actual_heads_prob:.4f}")
    with col4:
        st.metric("Kelly Bet Size", f"{actual_kelly:.4f}" if use_kelly else "Not Used")
    
    # Display final results
    final_capital = st.session_state.trading_results['Wealth After Flip'].iloc[-1]
    total_roi = (final_capital - initial_capital) / initial_capital * 100
    max_drawdown = st.session_state.trading_results['Drawdown (%)'].max()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Capital", f"${final_capital:.2f}", f"{final_capital - initial_capital:+.2f}")
    with col2:
        st.metric("Total ROI", f"{total_roi:.2f}%")
    with col3:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
    
    # Display trading results table
    st.subheader("Trading Results")
    display_cols = [
        'Flip Number', 'Bet Fraction', 'Bet Amount', 'Outcome', 'Win',
        'Wealth Before Flip', 'Wealth After Flip', 'Cumulative ROI (%)',
        'Drawdown (%)', 'Cumulative Heads', 'Cumulative Tails'
    ]
    
    # Format the table
    formatted_results = st.session_state.trading_results.copy()
    formatted_results['Bet Fraction'] = formatted_results['Bet Fraction'].apply(lambda x: f"{x:.4f}")
    formatted_results['Bet Amount'] = formatted_results['Bet Amount'].apply(lambda x: f"${x:.2f}")
    formatted_results['Wealth Before Flip'] = formatted_results['Wealth Before Flip'].apply(lambda x: f"${x:.2f}")
    formatted_results['Wealth After Flip'] = formatted_results['Wealth After Flip'].apply(lambda x: f"${x:.2f}")
    formatted_results['Cumulative ROI (%)'] = formatted_results['Cumulative ROI (%)'].apply(lambda x: f"{x:.2f}%")
    formatted_results['Drawdown (%)'] = formatted_results['Drawdown (%)'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(formatted_results[display_cols], use_container_width=True)
    
    # Plot 1: Wealth over time
    fig1 = px.line(
        st.session_state.trading_results,
        x='Flip Number',
        y='Wealth After Flip',
        title="Capital Growth Over Time"
    )
    fig1.update_layout(
        xaxis_title="Number of Flips",
        yaxis_title="Capital ($)",
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 2: Drawdown over time
    fig2 = px.line(
        st.session_state.trading_results,
        x='Flip Number',
        y='Drawdown (%)',
        title="Drawdown Over Time"
    )
    fig2.update_layout(
        xaxis_title="Number of Flips",
        yaxis_title="Drawdown (%)",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Plot 3: Win/Loss distribution
    win_count = st.session_state.trading_results['Win'].sum()
    loss_count = len(st.session_state.trading_results) - win_count
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=['Wins', 'Losses'],
        y=[win_count, loss_count],
        marker_color=['green', 'red']
    ))
    fig3.update_layout(
        title="Win/Loss Distribution",
        xaxis_title="Outcome",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Plot 4: Cumulative ROI
    fig4 = px.line(
        st.session_state.trading_results,
        x='Flip Number',
        y='Cumulative ROI (%)',
        title="Cumulative Return on Investment"
    )
    fig4.update_layout(
        xaxis_title="Number of Flips",
        yaxis_title="ROI (%)",
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Plot 5: Heads vs Tails over time
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=st.session_state.trading_results['Flip Number'],
        y=st.session_state.trading_results['Cumulative Heads'],
        mode='lines',
        name='Heads'
    ))
    fig5.add_trace(go.Scatter(
        x=st.session_state.trading_results['Flip Number'],
        y=st.session_state.trading_results['Cumulative Tails'],
        mode='lines',
        name='Tails'
    ))
    fig5.update_layout(
        title="Cumulative Heads vs Tails",
        xaxis_title="Number of Flips",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    if run_monte_carlo:
        # Run Monte Carlo simulations
        monte_carlo_results = run_monte_carlo_simulations(
            heads_prob, bet_on_heads, initial_capital, risk_reward_ratio, 
            kelly_fraction, use_kelly, num_flips, num_simulations
        )
        
        # Display Monte Carlo results
        st.subheader("Monte Carlo Simulation Results")
        
        # Plot 6: Distribution of final capitals
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(
            x=monte_carlo_results['final_capitals'],
            nbinsx=20,
            marker_color='blue'
        ))
        fig6.update_layout(
            title="Distribution of Final Capitals",
            xaxis_title="Final Capital ($)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig6, use_container_width=True)
        
        # Plot 7: Distribution of maximum drawdowns
        fig7 = go.Figure()
        fig7.add_trace(go.Histogram(
            x=monte_carlo_results['max_drawdowns'],
            nbinsx=20,
            marker_color='red'
        ))
        fig7.update_layout(
            title="Distribution of Maximum Drawdowns",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig7, use_container_width=True)
        
        # Plot 8: Distribution of ROI values
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(
            x=monte_carlo_results['roi_values'],
            nbinsx=20,
            marker_color='green'
        ))
        fig8.update_layout(
            title="Distribution of ROI Values",
            xaxis_title="ROI (%)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # Plot 9: Wealth trajectories
        fig9 = go.Figure()
        
        # Plot a subset of trajectories if there are many simulations
        plot_count = min(num_simulations, 50)  # Limit to 50 lines for better visualization
        
        # Add median trajectory
        median_trajectory = monte_carlo_results['wealth_trajectories'].median(axis=1)
        fig9.add_trace(go.Scatter(
            x=monte_carlo_results['wealth_trajectories'].index,
            y=median_trajectory,
            mode='lines',
            line=dict(color='black', width=3),
            name="Median"
        ))
        
        # Add percentile trajectories
        percentile_10 = monte_carlo_results['wealth_trajectories'].quantile(0.1, axis=1)
        percentile_90 = monte_carlo_results['wealth_trajectories'].quantile(0.9, axis=1)
        
        fig9.add_trace(go.Scatter(
            x=monte_carlo_results['wealth_trajectories'].index,
            y=percentile_10,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name="10th Percentile"
        ))
        
        fig9.add_trace(go.Scatter(
            x=monte_carlo_results['wealth_trajectories'].index,
            y=percentile_90,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name="90th Percentile"
        ))
        
        # Add individual trajectories with low opacity
        for i in range(plot_count):
            fig9.add_trace(go.Scatter(
                x=monte_carlo_results['wealth_trajectories'].index,
                y=monte_carlo_results['wealth_trajectories'].iloc[:, i],
                mode='lines',
                line=dict(color='blue', width=0.5),
                opacity=0.2,
                showlegend=False
            ))
        
        fig9.update_layout(
            title="Monte Carlo Wealth Trajectories",
            xaxis_title="Number of Flips",
            yaxis_title="Capital ($)",
            height=500
        )
        st.plotly_chart(fig9, use_container_width=True)
        
        # Display Monte Carlo statistics
        st.subheader("Monte Carlo Statistics")
        
        # Calculate statistics
        final_capital_mean = np.mean(monte_carlo_results['final_capitals'])
        final_capital_median = np.median(monte_carlo_results['final_capitals'])
        final_capital_std = np.std(monte_carlo_results['final_capitals'])
        final_capital_min = np.min(monte_carlo_results['final_capitals'])
        final_capital_max = np.max(monte_carlo_results['final_capitals'])
        
        roi_mean = np.mean(monte_carlo_results['roi_values'])
        roi_median = np.median(monte_carlo_results['roi_values'])
        roi_std = np.std(monte_carlo_results['roi_values'])
        
        drawdown_mean = np.mean(monte_carlo_results['max_drawdowns'])
        drawdown_median = np.median(monte_carlo_results['max_drawdowns'])
        drawdown_std = np.std(monte_carlo_results['max_drawdowns'])
        
        # Calculate probability of profit
        profit_probability = sum(1 for roi in monte_carlo_results['roi_values'] if roi > 0) / len(monte_carlo_results['roi_values']) * 100
        
        # Calculate probability of ruin (capital below a threshold, e.g., 10% of initial)
        ruin_threshold = initial_capital * 0.1
        ruin_probability = sum(1 for cap in monte_carlo_results['final_capitals'] if cap < ruin_threshold) / len(monte_carlo_results['final_capitals']) * 100
        
        # Display statistics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Final Capital", f"${final_capital_mean:.2f}")
            st.metric("Median Final Capital", f"${final_capital_median:.2f}")
            st.metric("Std Dev of Final Capital", f"${final_capital_std:.2f}")
            st.metric("Min Final Capital", f"${final_capital_min:.2f}")
            st.metric("Max Final Capital", f"${final_capital_max:.2f}")
        
        with col2:
            st.metric("Mean ROI", f"{roi_mean:.2f}%")
            st.metric("Median ROI", f"{roi_median:.2f}%")
            st.metric("Std Dev of ROI", f"{roi_std:.2f}%")
            st.metric("Probability of Profit", f"{profit_probability:.2f}%")
        
        with col3:
            st.metric("Mean Max Drawdown", f"{drawdown_mean:.2f}%")
            st.metric("Median Max Drawdown", f"{drawdown_median:.2f}%")
            st.metric("Std Dev of Max Drawdown", f"{drawdown_std:.2f}%")
            st.metric("Probability of Ruin (<10% capital)", f"{ruin_probability:.2f}%")
        
        # Create a box plot comparing the distributions
        fig10 = make_subplots(rows=1, cols=3, 
                             subplot_titles=("Final Capital Distribution", 
                                            "ROI Distribution (%)", 
                                            "Max Drawdown Distribution (%)"))
        
        fig10.add_trace(
            go.Box(y=monte_carlo_results['final_capitals'], name="Final Capital", marker_color="blue"),
            row=1, col=1
        )
        
        fig10.add_trace(
            go.Box(y=monte_carlo_results['roi_values'], name="ROI (%)", marker_color="green"),
            row=1, col=2
        )
        
        fig10.add_trace(
            go.Box(y=monte_carlo_results['max_drawdowns'], name="Max Drawdown (%)", marker_color="red"),
            row=1, col=3
        )
        
        fig10.update_layout(
            height=400,
            title_text="Distribution Comparison (Box Plots)",
            showlegend=False
        )
        
        st.plotly_chart(fig10, use_container_width=True)

# Initialize session state if not already done
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

# Display instructions if simulation hasn't been run yet
if not st.session_state.simulation_run:
    st.info("""
    This simulation uses the Kelly criterion for position sizing in a coin-flip trading model.
    
    **How to use:**
    1. Set the probability of heads using the slider
    2. Choose whether to bet on heads or tails
    3. Set your initial capital
    4. Adjust the risk-reward ratio
    5. Choose whether to use Kelly criterion and set the Kelly fraction
    6. Set the number of flips to simulate
    7. Click 'Run Simulation' to see the results
    
    The Kelly criterion calculates the optimal fraction of your capital to bet in order to maximize long-term growth.
    Using a fractional Kelly (less than 1.0) reduces risk while still providing good returns.
    """)
    
    # Kelly criterion explanation
    st.subheader("Understanding the Kelly Criterion")
    st.markdown("""
    The Kelly criterion is a formula used to determine the optimal size of a series of bets to maximize the logarithm of wealth. The formula is:
    
    **f* = (p × b - q) / b**
    
    Where:
    - f* is the fraction of the current bankroll to wager
    - p is the probability of winning
    - q is the probability of losing (1-p)
    - b is the net odds received on the wager (payout per unit wagered)
    
    For example, if you have a 60% chance of winning (p = 0.6), a 40% chance of losing (q = 0.4), and a risk-reward ratio of 1:1 (b = 1), the Kelly criterion suggests betting 20% of your bankroll on each trade.
    
    Most professional investors use a "fractional Kelly" approach (e.g., Half-Kelly) to reduce volatility while still achieving good long-term growth.
    """)
