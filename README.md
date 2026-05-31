# S.A.F.E. Engine (Sector-Adaptive Fundamental Engine)

A stochastic equity valuation and risk-management framework built in Python.

## 📌 Overview
The S.A.F.E. framework moves beyond static accounting multiples by integrating stochastic intrinsic valuation with deterministic tail-risk stress testing. It utilizes a 1,000-iteration Monte Carlo simulation to project discounted cash flows (DCF) and programmatically enforces a Hansen-Jagannathan (HJ) Reality Gate to filter out overvalued momentum assets.

## 🚀 Core Features
* **Stochastic Valuation:** Replaces static WACC and terminal growth inputs with probability density functions.
* **Hansen-Jagannathan Constraint:** Treats WACC as a Stochastic Discount Factor (SDF) to ensure simulated risk premiums are mathematically justified by historical Sharpe Ratios.
* **Black Swan Stress Testing:** Calculates a strict "Crisis Value" floor simulating a severe macro-downturn.
* **Sector-Adaptive Grading:** Adjusts fundamental scoring weights dynamically based on GICS sectors.
* **Capital Allocation:** Utilizes a Half-Kelly Criterion model to size positions optimally based on win probability and downside risk.

## ⚠️ Disclaimer
This repository is for academic and quantitative research purposes only. It is not financial advice.
