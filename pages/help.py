import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Trading Simulation - Help",
    page_icon="❓",
    layout="wide"
)

# App title and description
st.title("Trading Simulation Help Guide")
st.markdown("""
This guide explains the concepts, parameters, and methodology used in the Trading Simulation with Kelly Criterion application.
""")

# Create tabs for different help sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Kelly Criterion", "Position Sizing", "Parameters", "Monte Carlo"])

with tab1:
    st.header("Application Overview")
    
    st.subheader("Goal of the Application")
    st.markdown("""
    The primary goal of this application is to simulate trading outcomes using a simplified coin flip model. While real trading is much more complex, 
    this model allows us to understand fundamental concepts like:
    
    - Position sizing strategies
    - Risk management techniques
    - The impact of win probability on long-term performance
    - The effect of risk-reward ratios on capital growth
    - How drawdowns affect overall performance
    
    By simulating thousands of trades (coin flips) with different parameters, you can gain insights into how various trading strategies might perform over time.
    """)
    
    st.subheader("The Coin Flip Model")
    st.markdown("""
    In our simulation, each coin flip represents a single trade:
    
    - **Heads**: Represents a winning trade
    - **Tails**: Represents a losing trade
    - **Probability of Heads**: Represents your win rate (trading edge)
    - **Bet Amount**: Represents your position size for each trade
    
    While this is a simplification, it captures the essential probabilistic nature of trading. Even with a positive edge (probability > 50%), 
    poor position sizing can lead to ruin, while proper position sizing with a modest edge can lead to substantial growth.
    """)
    
    st.subheader("Key Metrics")
    st.markdown("""
    The application tracks several key metrics:
    
    - **Capital Growth**: How your trading capital changes over time
    - **Return on Investment (ROI)**: Percentage gain or loss on initial capital
    - **Drawdown**: Peak-to-trough decline in capital, measuring risk
    - **Win/Loss Distribution**: Pattern of winning and losing trades
    - **Final Capital**: Ending capital after all simulated trades
    """)

with tab2:
    st.header("Kelly Criterion Explained")
    
    st.subheader("What is the Kelly Criterion?")
    st.markdown("""
    The Kelly Criterion is a mathematical formula used to determine the optimal size of a series of bets or investments. 
    It was formulated by John L. Kelly Jr., a researcher at Bell Labs, in 1956.
    
    The formula aims to maximize the long-term growth rate of capital by finding the optimal fraction of capital to risk on each bet or trade.
    """)
    
    st.subheader("The Kelly Formula")
    st.markdown("""
    The basic Kelly formula is:
    
    **f* = (p × b - q) / b**
    
    Where:
    - **f*** is the optimal fraction of your capital to bet
    - **p** is the probability of winning
    - **q** is the probability of losing (1-p)
    - **b** is the odds received on the bet (the win/loss ratio)
    
    In trading terms:
    - **p** is your win rate (percentage of winning trades)
    - **b** is your risk-reward ratio (how much you win when right vs. how much you lose when wrong)
    """)
    
    st.subheader("Interpreting Kelly Results")
    st.markdown("""
    - **Positive Kelly**: When the Kelly formula returns a positive value, it suggests you have a positive edge and should bet that fraction of your capital
    - **Zero Kelly**: When the formula returns zero, it suggests you have no edge and should not bet
    - **Negative Kelly**: When the formula returns a negative value, it suggests you have a negative edge and should consider betting on the opposite outcome
    
    For example, if the Kelly formula returns 0.25 (or 25%), it suggests betting 25% of your capital on each trade would maximize long-term growth.
    """)
    
    st.subheader("Full Kelly vs. Fractional Kelly")
    st.markdown("""
    In practice, many traders and investors use a "Fractional Kelly" approach:
    
    - **Full Kelly (100%)**: Theoretically optimal for maximizing long-term growth, but can lead to significant volatility and drawdowns
    - **Half Kelly (50%)**: Reduces the bet size to half of the Kelly recommendation, significantly reducing volatility while still capturing about 75% of the growth rate
    - **Quarter Kelly (25%)**: Further reduces volatility at the expense of growth rate
    
    The application allows you to adjust the Kelly fraction to find your personal risk tolerance sweet spot.
    """)
    
    st.subheader("Limitations of Kelly")
    st.markdown("""
    While powerful, the Kelly Criterion has limitations:
    
    1. It assumes you know your exact win probability and risk-reward ratio, which is rarely the case in real trading
    2. It doesn't account for correlation between trades
    3. It optimizes for geometric growth, which may not align with all trading goals
    4. It can lead to very large position sizes when edge is significant
    
    This is why many professional traders use a fractional Kelly approach as a starting point, then adjust based on other factors.
    """)

with tab3:
    st.header("Position Sizing Strategies")
    
    st.subheader("Why Position Sizing Matters")
    st.markdown("""
    Position sizing is arguably the most important aspect of trading that many traders overlook. Even with a profitable strategy:
    
    - **Too small** positions won't generate meaningful returns
    - **Too large** positions can lead to ruin through drawdowns
    - **Inconsistent** position sizing makes performance analysis difficult
    
    The right position sizing strategy can help you:
    1. Survive long enough for your edge to play out
    2. Maximize returns while controlling risk
    3. Reduce emotional decision-making
    """)
    
    st.subheader("Position Sizing Methods")
    st.markdown("""
    The application implements the Kelly Criterion, but other common position sizing methods include:
    
    - **Fixed Percentage**: Risking a fixed percentage of capital on each trade
    - **Fixed Dollar Amount**: Risking the same dollar amount on each trade
    - **Volatility-Based**: Adjusting position size based on market volatility
    - **Optimal f**: A variation of Kelly that uses historical data
    
    Each method has advantages and disadvantages depending on market conditions and trading style.
    """)
    
    st.subheader("Risk of Ruin")
    st.markdown("""
    Risk of ruin is the probability that you'll lose your entire trading capital. It's influenced by:
    
    - Win rate
    - Risk-reward ratio
    - Position sizing
    - Starting capital
    
    The Monte Carlo simulation in this application helps visualize this risk by showing the distribution of possible outcomes and the probability of falling below a critical threshold.
    """)

with tab4:
    st.header("Application Parameters Explained")
    
    st.subheader("Probability of Heads")
    st.markdown("""
    This parameter sets the probability of getting heads when flipping the coin, representing your win rate in trading.
    
    - **Value Range**: 0.0 to 1.0
    - **Default**: 0.55 (55% win rate)
    - **Interpretation**: 
        - 0.5 = Fair coin (no edge)
        - > 0.5 = Edge when betting on heads
        - < 0.5 = Edge when betting on tails
    
    In real trading, developing a strategy with a consistent edge above 50% is challenging but essential for long-term success.
    """)
    
    st.subheader("Bet On")
    st.markdown("""
    This parameter determines whether you're betting on heads or tails for each flip.
    
    - **Options**: Heads or Tails
    - **Default**: Heads
    - **Strategy**: Generally, you should bet on whichever outcome has a higher probability (> 0.5)
    
    This is analogous to going long (betting on price increase) or short (betting on price decrease) in trading.
    """)
    
    st.subheader("Initial Capital")
    st.markdown("""
    This parameter sets your starting capital for the simulation.
    
    - **Value Range**: $100 to $1,000,000
    - **Default**: $10,000
    
    While the absolute amount doesn't affect percentage returns, it does impact the absolute dollar value of each bet and can influence psychological factors in real trading.
    """)
    
    st.subheader("Risk-Reward Ratio")
    st.markdown("""
    This parameter sets the ratio between potential gain and potential loss on each trade.
    
    - **Value Range**: 0.1 to 5.0
    - **Default**: 1.0 (equal risk and reward)
    - **Interpretation**:
        - 1.0 = Win and lose the same amount
        - 2.0 = Win twice as much as you lose
        - 0.5 = Win half as much as you lose
    
    In trading, a higher risk-reward ratio can compensate for a lower win rate. Many successful traders aim for risk-reward ratios above 1.0.
    """)
    
    st.subheader("Use Kelly Criterion")
    st.markdown("""
    This checkbox determines whether to use the Kelly formula for position sizing.
    
    - **Default**: Enabled
    - **When Disabled**: Uses a fixed 1% position size instead
    
    This allows you to compare Kelly-based position sizing with a simple fixed percentage approach.
    """)
    
    st.subheader("Kelly Fraction")
    st.markdown("""
    This parameter adjusts what fraction of the full Kelly recommendation to use.
    
    - **Value Range**: 0.0 to 1.0
    - **Default**: 0.5 (Half Kelly)
    - **Interpretation**:
        - 1.0 = Full Kelly (theoretically optimal for growth, but high volatility)
        - 0.5 = Half Kelly (reduced volatility with still good growth)
        - 0.25 = Quarter Kelly (much lower volatility with moderate growth)
    
    Most professional traders use between 0.25 and 0.5 of the Kelly recommendation to reduce volatility.
    """)
    
    st.subheader("Number of Flips")
    st.markdown("""
    This parameter sets how many coin flips (trades) to simulate.
    
    - **Value Range**: 1 to 10,000
    - **Default**: 100
    
    More flips provide a better representation of long-term performance but may take longer to simulate.
    """)

with tab5:
    st.header("Monte Carlo Simulation")
    
    st.subheader("What is Monte Carlo Simulation?")
    st.markdown("""
    Monte Carlo simulation is a computational technique that uses random sampling to obtain numerical results. 
    The underlying concept is to use randomness to solve problems that might be deterministic in principle.
    
    In trading, Monte Carlo simulations help understand the range of possible outcomes given a set of parameters and probabilities.
    """)
    
    st.subheader("Why Use Monte Carlo in Trading?")
    st.markdown("""
    Even with a positive edge, trading outcomes can vary significantly due to random sequences of wins and losses. 
    Monte Carlo simulation helps:
    
    1. Understand the distribution of possible outcomes
    2. Estimate the probability of reaching specific profit targets
    3. Calculate the probability of drawdowns exceeding comfort levels
    4. Test the robustness of a trading strategy across different market conditions
    5. Make more informed risk management decisions
    """)
    
    st.subheader("Interpreting Monte Carlo Results")
    st.markdown("""
    The application provides several visualizations and statistics from the Monte Carlo simulation:
    
    - **Distribution of Final Capitals**: Shows the range of possible ending capital values
    - **Distribution of Maximum Drawdowns**: Shows how deep drawdowns might get
    - **Distribution of ROI Values**: Shows the range of possible returns
    - **Wealth Trajectories**: Shows how capital might evolve over time, with percentile bands
    - **Statistical Metrics**: Includes mean, median, standard deviation, and probabilities of profit and ruin
    
    Key insights to look for:
    
    - **Wide Distribution**: Indicates high volatility and uncertainty
    - **Skewed Distribution**: May indicate potential for outlier results (very good or very bad)
    - **High Probability of Ruin**: Suggests the strategy may be too risky
    - **Wide Percentile Bands**: Indicates high path dependency and potential emotional challenges
    """)
    
    st.subheader("Number of Simulations")
    st.markdown("""
    This parameter controls how many separate simulations to run in the Monte Carlo analysis.
    
    - **Value Range**: 10 to 1,000
    - **Default**: 100
    - **Recommendation**: Higher values (500+) provide more reliable distributions but take longer to compute
    
    Each simulation runs the full number of flips with a different random sequence of outcomes, creating a distribution of possible results.
    """)

# Footer
st.markdown("---")
st.markdown("""
### Additional Resources

For more information on trading concepts and risk management, consider these resources:

- [Investopedia: Kelly Criterion](https://www.investopedia.com/articles/trading/04/091504.asp)
- [Wikipedia: Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Trading Mathematics: Position Sizing](https://www.tradingmathematics.com/position-sizing/)
- [Risk of Ruin Calculator](https://www.moneymanagement-trading.com/risk-of-ruin-calculator/)

Remember that while this simulation provides valuable insights, real trading involves additional complexities including transaction costs, slippage, market impact, and psychological factors.
""")
