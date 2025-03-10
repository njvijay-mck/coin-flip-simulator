# Trading Simulation with Kelly Criterion

This application simulates a trading strategy using a weighted coin flip model with Kelly criterion position sizing. It's part of a larger trading simulation project that will eventually include more sophisticated Monte Carlo simulations.

## Features

- Simulate trading with a weighted coin flip model (adjustable probability bias)
- Choose to bet on either heads or tails
- Implement Kelly criterion for optimal position sizing
- Adjust risk-reward ratio for each trade
- Use fractional Kelly to reduce volatility
- Track performance metrics including:
  - Capital growth
  - Return on investment (ROI)
  - Maximum drawdown
  - Win/loss ratio
- Visualize results with interactive Plotly charts
- **Monte Carlo simulation** to analyze the distribution of possible outcomes:
  - Distribution of final capital values
  - Distribution of maximum drawdowns
  - Distribution of ROI values
  - Wealth trajectory analysis with percentile bands
  - Comprehensive statistical analysis including probability of profit and ruin

## Installation

1. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```
streamlit run coin_flip_simulator.py
```

2. Use the sidebar to adjust parameters:
   - Set the probability of heads (0.0 to 1.0)
   - Choose whether to bet on heads or tails
   - Set your initial capital
   - Adjust the risk-reward ratio
   - Set the Kelly fraction (0.0 to 1.0)
   - Set the number of flips to simulate
   
3. Click "Run Simulation" to see the results

## Understanding Kelly Criterion

The Kelly criterion is a formula used to determine the optimal size of a series of bets to maximize the logarithm of wealth. The formula is:

**f* = (p × b - q) / b**

Where:
- f* is the fraction of the current bankroll to wager
- p is the probability of winning
- q is the probability of losing (1-p)
- b is the net odds received on the wager (payout per unit wagered)

For example, if you have a 60% chance of winning (p = 0.6), a 40% chance of losing (q = 0.4), and a risk-reward ratio of 1:1 (b = 1), the Kelly criterion suggests betting 20% of your bankroll on each trade.

Most professional investors use a "fractional Kelly" approach (e.g., Half-Kelly) to reduce volatility while still achieving good long-term growth.

## Visualizations

The application provides several visualizations:
- Capital growth over time
- Drawdown over time
- Win/loss distribution
- Cumulative return on investment
- Cumulative heads vs tails counts
