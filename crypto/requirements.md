

**Project Title:** Deep Hedging Engine: Professional-Grade Proof of Concept (V2)

### 1. Executive Summary
This document outlines the technical requirements for a professional-grade deep hedging proof-of-concept. The primary goal is to build a system that can learn, backtest, and evaluate a cost-aware hedging policy for Bitcoin options using the underlying perpetual swap. The system must be able to operate in two evaluation modes: long-volatility (buying options and hedging) and short-volatility (selling options and hedging).

The ultimate decision criterion for success is the strategy's ability to outperform a well-tuned, discrete Black-Scholes delta hedging baseline, specifically by demonstrating superior **tail-risk reduction (measured by CVaR/Expected Shortfall)** and total PnL after accounting for all realistic market frictions.

---

### 2. Background: What is Deep Hedging?
(This section remains the same as the previous version, providing the core concept.)

Traditional hedging strategies (like Black-Scholes delta hedging) rely on simplified mathematical models that often ignore real-world frictions like transaction costs, market impact, and discrete hedging intervals.

**Deep Hedging** is a framework that uses a neural network to learn a hedging strategy directly from simulated market data. The key advantage is that we can incorporate realistic frictions into the training simulation. The network's goal is not to predict prices, but to find a sequence of trades that best replicates an option's payoff while minimizing a specified risk measure (e.g., the cost and variance of the hedging PnL).

---

### 3. System Architecture & Components

#### 3.1. Data Layer
The foundation of the system is a robust data pipeline capable of ingesting, cleaning, and storing high-resolution market data.
* **Data Ingestion:** The system must be able to process and store the following from Deribit:
    * **Perpetual/Futures:** Tick-level trades, best bid/ask quotes, L2 order book depth (top 5-10 levels), and funding rates with their precise timestamps.
    * **Options:** Tick-level trades, quotes (bid/ask, implied volatility), and Greeks (if available from the API).
* **Data Processing & Storage:**
    * All data must be timestamped, de-duplicated, and aligned.
    * Data should be stored in a columnar format (e.g., Parquet), partitioned by date and symbol for efficient querying.

#### 3.2. Market Simulator & Execution Engine
This core component simulates the market environment with a high degree of realism. It will have two primary modes:
1.  **Historical Replay:** For backtesting, the engine will replay historical data precisely as it occurred.
2.  **Stochastic Simulation:** For training, the engine will generate market scenarios using models (e.g., Heston, GBM with jumps) where parameters are randomized per episode to ensure the agent learns a robust policy (**domain randomization**).

The simulator must model the following **market frictions**:
* **Costs:** Configurable maker/taker fees, historical bid-ask spreads, and a market impact model (e.g., square-root model).
* **Execution:** Model slippage as a function of order size versus available depth. For this PoC, focus on Market Orders; the architecture should allow for Limit Orders and queue modeling in the future.
* **Funding:** Accurately account for funding payments on the perpetual swap based on historical rates.
* **Latency:** Introduce a configurable, random delay between a trading signal and its simulated execution.

#### 3.3. Portfolio and Risk Engine
This module tracks the state of the trading portfolio and enforces risk limits.
* **Position Management:** Maintain the state of the option position (long or short) and the hedging position in the perpetual swap.
* **PnL Accounting:** Provide a detailed, real-time breakdown of PnL, separating the option's PnL from the hedge's PnL and itemizing all friction costs (fees, slippage, funding, etc.).
* **Risk Constraints:** Implement and enforce configurable risk limits, including maximum position size, daily loss limits, and a master kill-switch.

#### 3.4. Policy / Agent Module (The "Brain")
This is the deep hedging model itself.
* **State Vector:** The model will receive a rich set of inputs at each timestep, including: time-to-expiry, underlying price, volatility estimates, implied volatility, option Greeks, current hedge position, recent trade sizes, market spread, and funding rate.
* **Actions:** The model's output will be the new target hedge position in the perpetual swap.
* **Objective Functions:** The training process will optimize a configurable objective. The primary target will be a **Mean-CVaR** objective (e.g., minimize `mean(Loss) + CVaR_95%(Loss)`), which directly targets tail risk.

---

### 4. Baselines for Comparison
A new policy is only valuable if it improves upon existing methods. The system must implement the following baselines for rigorous comparison:
1.  **Tuned Discrete Delta Hedge:** A standard Black-Scholes delta hedging strategy with a "no-trade band." The re-hedge is only triggered if the delta deviation exceeds a threshold tuned to transaction costs.
2.  **Do-Nothing / Static Hedge:** A baseline to measure the cost of doing nothing or holding a static initial hedge.

---

### 5. Evaluation, Reporting & Acceptance Criteria

#### 5.1. Evaluation Protocol
* **Backtesting:** The primary evaluation method will be a walk-forward backtest on out-of-sample historical data, covering multiple market regimes (bull, bear, volatile, quiet).
* **Stress Tests:** The system must be able to run stress-test scenarios where frictions are amplified (e.g., 2x spreads, 2x market impact, sudden funding rate spikes) to test the policy's robustness.

#### 5.2. Key Metrics & Reporting
The final output will be a report that compares the Deep Hedging policy against the baselines across these key metrics:
* **PnL & Risk:** Total PnL after all costs, **Expected Shortfall (CVaR at 1% and 5%)**, PnL variance, and max drawdown.
* **Cost Analysis:** A detailed breakdown of costs from fees, spread crossing, slippage/impact, and funding.
* **Trading Activity:** Total turnover and trade frequency.

#### 5.3. Acceptance Criteria (Success Gates)
The PoC will be considered successful if the deep hedging policy demonstrates:
1.  A statistically significant **reduction in 5% Expected Shortfall (ES/CVaR) of at least 15-30%** compared to the tuned delta-hedge baseline.
2.  Comparable or better net PnL after all costs.
3.  Robustness under stress tests (i.e., performance does not collapse when frictions are doubled).

---

### 6. Scope & Deliverables for this PoC

* **Initial Scope:**
    * **Assets:** Deribit BTC weekly and monthly options (ATM strikes) and the BTC perpetual swap.
    * **Simulator:** Implement the **Historical Replay** simulator with accurate modeling of fees, spreads, market impact, and funding.
    * **Policies:** Implement the **Mean-CVaR Deep Hedging Agent** and the **Tuned Discrete Delta Hedge** baseline.
* **Deliverables:**
    1.  **Codebase:** A well-structured Python codebase for all system components.
    2.  **Configuration:** All parameters (frictions, risk limits, model objectives) must be configurable via YAML or JSON files.
    3.  **Documentation:** A `README.md` explaining how to set up the environment, run a backtest, and interpret the results.
    4.  **Example Run:** A script and config file to replicate a backtest for one long-vol and one short-vol case on a specific BTC weekly option.
    5.  **Final Report:** A summary report of the example run, presenting the metrics and acceptance criteria.
    6.  **Paper Trading Adapter:** An operational adapter to the Deribit Testnet API with a functioning kill-switch.