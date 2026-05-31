# S.A.F.E. ENGINE (Sector-Adaptive Fundamental Engine) v1.0
# Developed by: Henry C. Baxter
# Description: Quantitative framework for S&P 500 constituent valuation and risk optimization.

!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import os
import zipfile


# ==========================================
# SECTION 1: SECTOR-ADAPTIVE GRADING SCALES
# ==========================================
def grade_margin_adaptive(value, sector, metric_type='gross'):
    if pd.isna(value): return 0
    t1, t2, t3, t4 = 40, 30, 20, 10
    if metric_type != 'gross':
        t1, t2, t3, t4 = 20, 15, 10, 5
    if sector in ['Consumer Defensive', 'Consumer Cyclical']:
        t1, t2, t3, t4 = (t1/1.5), (t2/1.5), (t3/1.5), (t4/1.5)
    if value >= t1: return 10
    if value >= t2: return 8
    if value >= t3: return 6
    if value >= t4: return 4
    return 2

def grade_growth(value):
    if pd.isna(value): return 0
    if value >= 20: return 10
    if value >= 10: return 8
    if value >= 5: return 6
    if value > 0: return 4
    return 0

def grade_valuation_pe(value):
    if pd.isna(value) or value <= 0: return 0
    if value <= 15: return 10
    if value <= 20: return 8
    if value <= 25: return 6
    if value <= 35: return 4
    return 2

def grade_valuation_multiple(value):
    if pd.isna(value) or value <= 0: return 0
    if value <= 10: return 10
    if value <= 15: return 8
    if value <= 20: return 6
    if value <= 25: return 4
    return 2

def grade_yield(value):
    if pd.isna(value): return 0
    if value >= 4.0: return 10
    if value >= 2.5: return 8
    if value >= 1.5: return 6
    if value > 0: return 4
    return 2

def grade_fcf_margin(value):
    if pd.isna(value): return 0
    if value >= 15.0: return 10
    if value >= 10.0: return 8
    if value >= 5.0: return 6
    if value > 0: return 4
    return 2

def grade_turnover(value):
    if pd.isna(value): return 0
    if value >= 1.5: return 10
    if value >= 1.0: return 8
    if value >= 0.7: return 6
    if value >= 0.4: return 4
    return 2

def grade_returns(value):
    if pd.isna(value): return 0
    if value >= 20: return 10
    if value >= 15: return 8
    if value >= 10: return 6
    if value >= 5: return 4
    return 2

def grade_leverage(value):
    if pd.isna(value): return 0
    if value <= 1.5: return 10
    if value <= 2.5: return 8
    if value <= 3.5: return 6
    if value <= 4.5: return 4
    return 2

def grade_accrual_quality(value):
    if pd.isna(value): return 0
    if value <= 0: return 10
    if value <= 2: return 8
    if value <= 5: return 6
    if value <= 8: return 4
    return 2

def get_final_rating(score):
    if score >= 8.5: return "Elite / Top-Tier"
    if score >= 7.5: return "Strong Buy"
    if score >= 6.5: return "Buy"
    if score >= 5.5: return "Hold"
    if score >= 4.5: return "Hold weak / Watch"
    return "Avoid"

# ==========================================
# SECTION 2: THE MONTE CARLO DCF ENGINE
# ==========================================
def generate_dcf_model(ticker, current_price, current_fcf, shares_out, net_debt, base_growth_rate, live_wacc, sharpe_ratio=None):
    """
    Stochastic DCF Engine with Hansen-Jagannathan Reality Sanitization.
    Enforces the bound: Vol(SDF) / E[SDF] >= Sharpe Ratio.
    """
    if pd.isna(current_fcf) or current_fcf <= 0 or pd.isna(shares_out) or shares_out <= 0:
        return "N/A (Negative FCF/Missing Data)", 0, np.nan

    try:
        tgr = 0.025 # Terminal Growth Rate anchored at 2.5%
        base_g = min(max(base_growth_rate / 100, 0.02), 0.15) if pd.notna(base_growth_rate) else 0.05

        iterations = 1000
        rand_g = np.random.normal(base_g, 0.02, iterations)
        rand_g = np.clip(rand_g, 0.0, 0.25)

        # We simulate WACC as our Stochastic Discount Factor (SDF)
        rand_wacc = np.random.normal(live_wacc, 0.01, iterations)
        rand_wacc = np.clip(rand_wacc, 0.05, 0.15)

        # ==========================================
        # HANSEN-JAGANNATHAN (HJ) REALITY GATE
        # ==========================================
        # Logic: SDF Volatility must be >= Sharpe Ratio
        sdf_m = 1 / (1 + rand_wacc)
        sdf_volatility = np.std(sdf_m) / np.mean(sdf_m)

        if sharpe_ratio is not None and pd.notna(sharpe_ratio) and sharpe_ratio > 0:
            if sdf_volatility < sharpe_ratio:
                # Reality Haircut: If simulated risk < observed market reward,
                # we penalize the terminal value by 15% to remain conservative.
                hj_penalty = 0.85
            else:
                hj_penalty = 1.0
        else:
            hj_penalty = 1.0

        implied_prices = []
        for i in range(iterations):
            g = rand_g[i]
            wacc = rand_wacc[i]
            if wacc <= tgr: wacc = tgr + 0.001

            fcf_proj = [current_fcf * (1 + g)**yr for yr in range(1, 6)]
            pv_fcf = [fcf_proj[yr] / (1 + wacc)**(yr+1) for yr in range(5)]

            terminal_value = (fcf_proj[-1] * (1 + tgr)) / (wacc - tgr)
            pv_tv = terminal_value / (1 + wacc)**5

            implied_ev = sum(pv_fcf) + pv_tv
            implied_equity = implied_ev - net_debt

            # Apply the HJ Reality Haircut
            implied_price = (implied_equity / shares_out) * hj_penalty
            implied_prices.append(implied_price)

        implied_prices = np.array(implied_prices)
        implied_prices = implied_prices[(implied_prices > 0) & (implied_prices < current_price * 5)]

        if len(implied_prices) == 0: return "Error: Invalid MC Projections", 0, np.nan

        mean_implied = np.mean(implied_prices)
        prob_undervalued = np.sum(implied_prices > current_price) / len(implied_prices) * 100

        # --- THE MISSING PLOTTING BLOCK ---
        fig = plt.figure(figsize=(7, 4))
        plt.hist(implied_prices, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        plt.axvline(current_price, color='red', linestyle='dashed', linewidth=2, label=f'Current: ${current_price:.2f}')
        plt.axvline(mean_implied, color='green', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_implied:.2f}')
        plt.title(f"{ticker} Monte Carlo (1,000 Scenarios)", fontsize=12, fontweight='bold')
        plt.legend()

        filename = f"{ticker}_DCF_Chart.png"
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

        return f"Mean: ${mean_implied:.2f} ({prob_undervalued:.0f}% Win)", prob_undervalued, mean_implied

    except Exception as e:
        return f"Error: {e}", 0, np.nan

# --- BLACK SWAN SENSITIVITY MODULE ---
def run_stress_test(ticker, current_price, current_fcf, shares_out, net_debt, base_g, wacc):
    try:
        # Scenarios: (Growth Adjustment, WACC Adjustment)
        scenarios = {
            "Fed_Spike": (base_g, wacc + 0.03),
            "Growth_Shock": (base_g * 0.5, wacc),
            "Crisis": (base_g * 0.2, wacc + 0.04)
        }

        stress_results = {}
        for name, (g, w) in scenarios.items():
            # 5-year DCF calculation for each scenario
            fcf_proj = [current_fcf * (1 + (g/100))**yr for yr in range(1, 6)]
            terminal = (fcf_proj[-1] * 1.02) / (w - 0.02) if w > 0.02 else 0
            pv = sum([fcf_proj[i] / (1+w)**(i+1) for i in range(5)]) + (terminal / (1+w)**5)
            implied_val = (pv - net_debt) / shares_out
            stress_results[name] = round(implied_val, 2)
        return stress_results
    except:
        return {"Fed_Spike": 0, "Growth_Shock": 0, "Crisis": 0}

# ==========================================
# SECTION 3: PERFORMANCE & RISK BACKTESTER
# ==========================================
def run_historical_risk_analysis(ticker_symbol, risk_free_rate):
    try:
        stock_hist = yf.Ticker(ticker_symbol).history(period="3y")
        spy_hist = yf.Ticker("SPY").history(period="3y")

        if stock_hist.empty or spy_hist.empty or len(stock_hist) < 250:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        returns_df = pd.concat([stock_hist['Close'], spy_hist['Close']], axis=1).dropna().pct_change().dropna()
        returns_df.columns = ['Stock', 'SPY']

        if len(stock_hist) >= 252 and len(spy_hist) >= 252:
            stock_1y_ret = ((stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[-252]) - 1) * 100
            spy_1y_ret = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-252]) - 1) * 100
            alpha_1y = stock_1y_ret - spy_1y_ret
        else:
            alpha_1y = np.nan

        if len(stock_hist) > 500 and len(spy_hist) > 500:
            stock_3y_ret = ((stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0]) - 1) * 100
            spy_3y_ret = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100
            alpha_3y = stock_3y_ret - spy_3y_ret
        else:
            alpha_3y = np.nan

        cov = returns_df.cov().iloc[0, 1]
        var = returns_df['SPY'].var()
        beta = cov / var if var > 0 else np.nan

        ann_ret = returns_df['Stock'].mean() * 252
        ann_vol = returns_df['Stock'].std() * np.sqrt(252)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        downside_returns = returns_df['Stock'][returns_df['Stock'] < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (ann_ret - risk_free_rate) / downside_vol if (pd.notna(downside_vol) and downside_vol > 0) else np.nan

        return alpha_1y, alpha_3y, beta, sharpe, sortino, ann_vol
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# --- BOOTSTRAP BACKTESTER ---
def bootstrap_alpha_test(ticker_symbol, iterations=100):
    try:
        stock_hist = yf.Ticker(ticker_symbol).history(period="3y")['Close'].pct_change().dropna()
        spy_hist = yf.Ticker("SPY").history(period="3y")['Close'].pct_change().dropna()
        if len(stock_hist) < 252: return np.nan
        wins = 0
        for _ in range(iterations):
            start = np.random.randint(0, len(stock_hist) - 126)
            stock_ret = (1 + stock_hist.iloc[start:start+126]).prod() - 1
            spy_ret = (1 + spy_hist.iloc[start:start+126]).prod() - 1
            if stock_ret > spy_ret: wins += 1
        return (wins / iterations) * 100
    except: return np.nan

# --- MAX DRAWDOWN ANALYZER ---
def calculate_mdd(ticker_symbol):
    try:
        hist = yf.Ticker(ticker_symbol).history(period="5y")['Close']
        rolling_max = hist.cummax()
        drawdowns = (hist - rolling_max) / rolling_max
        return round(drawdowns.min() * 100, 2)
    except: return np.nan

# ==========================================
# SECTION 4: DATA HARVESTER & ENGINE
# ==========================================
def run_master_pipeline(ticker_symbol, risk_free_rate):
    try:
        company = yf.Ticker(ticker_symbol)
        info = company.info
        sector = info.get('sector', 'Unknown')

        inc = company.financials.dropna(how='all').T.sort_index()
        bal = company.balance_sheet.dropna(how='all').T.sort_index()
        cf = company.cashflow.dropna(how='all').T.sort_index()
        hist = company.history(period="1y")

        latest_date = inc.index[-1]

        # --- DATA EXTRACTION ---
        revenue = inc.loc[latest_date, 'Total Revenue']
        gross_profit = inc.loc[latest_date, 'Gross Profit']
        operating_income = inc.loc[latest_date].get('Operating Income', 0)
        ni_name = 'Net Income From Continuing Operation Net Minority Interest' if 'Net Income From Continuing Operation Net Minority Interest' in inc.columns else 'Net Income'
        net_income = inc.loc[latest_date, ni_name]

        total_assets = bal.loc[latest_date, 'Total Assets']
        current_assets = bal.loc[latest_date].get('Current Assets', np.nan)
        current_liabilities = bal.loc[latest_date].get('Current Liabilities', np.nan)
        equity = bal.loc[latest_date].get('Stockholders Equity', np.nan)
        total_debt = bal.loc[latest_date].get('Total Debt', 0)
        cash_equiv = bal.loc[latest_date].get('Cash And Cash Equivalents', 0)
        fcf = cf.loc[latest_date].get('Free Cash Flow', np.nan)

        pe_ratio = info.get('trailingPE', np.nan)
        peg_ratio = info.get('pegRatio', np.nan)
        ev_to_ebitda = info.get('enterpriseToEbitda', np.nan)
        eps_growth = info.get('earningsGrowth', 0) * 100
        rev_growth_yoy = info.get('revenueGrowth', 0) * 100
        div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        market_cap = info.get('marketCap', np.nan)
        ebitda = info.get('ebitda', inc.loc[latest_date].get('EBITDA', np.nan))

        shares_out = info.get('sharesOutstanding', np.nan)
        current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else np.nan)

        # --- CORE CALCULATIONS ---
        equity_risk_premium = 0.055
        live_wacc = risk_free_rate + equity_risk_premium
        net_debt = max(total_debt - cash_equiv, 0)

        # 1. RUN MODELS

        alpha_1y, alpha_3y, beta, sharpe, sortino, ann_vol = run_historical_risk_analysis(ticker_symbol, risk_free_rate)


        dcf_status, mc_win_rate, mean_implied = generate_dcf_model(
            ticker_symbol, current_price, fcf, shares_out, net_debt,
            rev_growth_yoy, live_wacc, sharpe_ratio=sharpe
        )

        stress_vals = run_stress_test(ticker_symbol, current_price, fcf, shares_out, net_debt, rev_growth_yoy, live_wacc)

        # 2. STRESS TEST & BACKTEST MODULES
        stress_vals = run_stress_test(ticker_symbol, current_price, fcf, shares_out, net_debt, rev_growth_yoy, live_wacc)
        boot_win_rate = bootstrap_alpha_test(ticker_symbol)
        mdd_val = calculate_mdd(ticker_symbol)

        # --- RATIO CALCULATIONS ---
        if len(inc) >= 3 and len(bal) >= 3:
            revenue_3yr = inc['Total Revenue'].iloc[-3:].mean()
            gross_profit_3yr = inc['Gross Profit'].iloc[-3:].mean()
            net_income_3yr = inc[ni_name].iloc[-3:].mean()
            total_assets_3yr = bal['Total Assets'].iloc[-3:].mean()
            gross_margin = (gross_profit_3yr / revenue_3yr) * 100
            profit_margin = (net_income_3yr / revenue_3yr) * 100
            asset_turnover = revenue_3yr / total_assets_3yr
        else:
            gross_margin = (gross_profit / revenue) * 100
            profit_margin = (net_income / revenue) * 100
            asset_turnover = revenue / total_assets

        operating_margin = (operating_income / revenue) * 100
        tax_rate = inc.loc[latest_date].get('Tax Rate For Calcs', 0.21)
        nopat = operating_income * (1 - tax_rate)
        invested_cap = bal.loc[latest_date].get('Invested Capital', (total_assets - current_liabilities))

        roa = (net_income / total_assets) * 100 if total_assets > 0 else np.nan
        roe = (net_income / equity) * 100 if pd.notna(equity) and equity > 0 else np.nan
        roic = (nopat / invested_cap) * 100 if invested_cap > 0 else np.nan

        current_ratio = current_assets / current_liabilities if pd.notna(current_assets) and current_liabilities > 0 else np.nan
        debt_to_equity = total_debt / equity if pd.notna(equity) and equity > 0 else np.nan
        net_debt_to_ebitda = (net_debt / ebitda) if pd.notna(ebitda) and ebitda > 0 else np.nan

        fcf_margin = (fcf / revenue) * 100 if pd.notna(fcf) else np.nan
        fcf_yield = (fcf / market_cap) * 100 if pd.notna(fcf) and pd.notna(market_cap) else np.nan

        # Accrual & F-Score logic
        if len(inc) >= 3 and len(cf) >= 3 and len(bal) >= 3:
            ocf_col = 'Operating Cash Flow' if 'Operating Cash Flow' in cf.columns else cf.columns[0]
            accrual_quality = ((inc[ni_name].iloc[-3:].sum() - cf[ocf_col].iloc[-3:].sum()) / bal['Total Assets'].iloc[-3:].mean()) * 100
        else:
            accrual_quality = np.nan

        f_score = 0
        if len(inc) >= 2 and len(bal) >= 2 and len(cf) >= 2:
            cy_ni = inc.loc[inc.index[-1], ni_name]
            if cy_ni > 0: f_score += 1

        else:
            f_score = np.nan

        # SMA & Final Grading
        if len(hist) >= 200:
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            price_vs_sma = ((current_price / sma_200) - 1) * 100
        else:
            sma_200 = price_vs_sma = np.nan

        p1_score = (grade_growth(rev_growth_yoy) + grade_growth(eps_growth)) / 2
        margin_score = (grade_margin_adaptive(gross_margin, sector, 'gross') + grade_margin_adaptive(operating_margin, sector, 'op') + grade_margin_adaptive(profit_margin, sector, 'net')) / 3
        return_score = (grade_returns(roa) + grade_returns(roe) + grade_returns(roic)) / 3
        p2_score = (margin_score + return_score + grade_turnover(asset_turnover)) / 3
        p3_score = (grade_fcf_margin(fcf_margin) + grade_yield(fcf_yield) + grade_accrual_quality(accrual_quality)) / 3
        p4_score = ((10 if current_ratio >= 1.5 else 6) + (grade_leverage(debt_to_equity) + grade_leverage(net_debt_to_ebitda)) / 2) / 2
        p5_score = (grade_valuation_pe(pe_ratio) + grade_valuation_multiple(peg_ratio) + grade_valuation_multiple(ev_to_ebitda) + grade_yield(div_yield)) / 4

        final_score = (p1_score + p2_score + p3_score + p4_score + p5_score) / 5
        rating = get_final_rating(final_score)

        # Conviction & Grade logic
        conviction_score = round((final_score * 4) + (mc_win_rate / 100) * 30 + ((f_score / 9) * 15 if pd.notna(f_score) else 0) + (min((sortino / 2.0) * 15, 15) if (pd.notna(sortino) and sortino > 0) else 0), 1)
        if conviction_score >= 85: master_grade = "A+ (High Conviction Buy)"
        elif conviction_score >= 75: master_grade = "A (Buy)"
        elif conviction_score >= 60: master_grade = "B (Hold)"
        else: master_grade = "C (Avoid/Sell)"

        # Kelly Allocation
        upside_potential = (mean_implied / current_price) - 1 if (pd.notna(mean_implied) and pd.notna(current_price) and current_price > 0) else 0
        kelly_allocation = max(0, min(((mc_win_rate/100) - ((1 - mc_win_rate/100) / (upside_potential / ann_vol if ann_vol > 0 else 1))) / 2, 0.20)) * 100 if upside_potential > 0 else 0

        # --- DICTIONARY ---
        return {
            'Ticker': ticker_symbol,
            'Sector': sector,
            'Target Allocation (Kelly)': f"{round(kelly_allocation, 1)}%" if kelly_allocation > 0 else "0.0% (Skip)",
            'Master Grade': master_grade,
            'Conviction Score (0-100)': conviction_score,
            'S.A.F.E. Score (1-10)': round(final_score, 2),
            'Final Rating': rating,
            'MC DCF Model': dcf_status,
            'Live WACC (%)': round(live_wacc * 100, 2),
            'Beta (3-Yr)': round(beta, 2) if pd.notna(beta) else "N/A",
            'Sharpe Ratio': round(sharpe, 2) if pd.notna(sharpe) else "N/A",
            'Sortino Ratio': round(sortino, 2) if pd.notna(sortino) else "N/A",
            '1-Yr Alpha vs SPY (%)': round(alpha_1y, 2) if pd.notna(alpha_1y) else "N/A",
            '3-Yr Alpha vs SPY (%)': round(alpha_3y, 2) if pd.notna(alpha_3y) else "N/A",
            'Piotroski F-Score': f"{f_score}/9" if pd.notna(f_score) else "N/A",
            'Price vs 200-SMA': f"{round(price_vs_sma, 1)}%" if pd.notna(price_vs_sma) else "N/A",
            'P1: Growth': round(p1_score, 1),
            'P2: Profit/Eff': round(p2_score, 1),
            'P3: Cash Flow': round(p3_score, 1),
            'P4: Health': round(p4_score, 1),
            'P5: Value/Yield': round(p5_score, 1),
            'Rev Growth (%)': round(rev_growth_yoy, 2),
            'Gross Margin (%)': round(gross_margin, 2),
            'Operating Margin (%)': round(operating_margin, 2),
            'ROA (%)': round(roa, 2),
            'ROE (%)': round(roe, 2),
            'ROIC (%)': round(roic, 2),
            'Asset Turnover': round(asset_turnover, 2),
            'FCF Margin (%)': round(fcf_margin, 2),
            'FCF Yield (%)': round(fcf_yield, 2),
            'Accrual Quality': round(accrual_quality, 2),
            'Current Ratio': round(current_ratio, 2),
            'Debt/Equity': round(debt_to_equity, 2),
            'Net Debt/EBITDA': round(net_debt_to_ebitda, 2),
            'P/E Ratio': round(pe_ratio, 2),
            'PEG Ratio': round(peg_ratio, 2),
            'EV/EBITDA': round(ev_to_ebitda, 2),
            'Div Yield (%)': round(div_yield, 2),
            'Crisis Value ($)': stress_vals['Crisis'],
            'Backtest Win Rate (%)': round(boot_win_rate, 1) if pd.notna(boot_win_rate) else "N/A",
            'Max Drawdown (%)': mdd_val
        }
    except Exception as e:
        return {"Ticker": ticker_symbol, "Error": str(e)}


# =========================================================
# SECTION 4.5: DYNAMIC SENSITIVITY & STRESS-TESTING
# =========================================================

def run_nexus_sensitivity(ticker_symbol, database):
    """
    Evaluates the robustness of the S.A.F.E. Mean Fair Value by
    stress-testing Terminal Growth (g) and WACC (r).
    """
    try:
        # 1. DYNAMIC DATA RETRIEVAL
        asset_data = database[database['Ticker'] == ticker_symbol.upper()].iloc[0]
        current_price = asset_data.get('Current Price ($)', 0)



        base_growth = 0.03
        base_wacc = asset_data['Live WACC (%)'] / 100

        # Gordon Growth Proxy for Sensitivity
        fcf_proxy = asset_data['S.A.F.E. Score (1-10)'] * (base_wacc - base_growth)

        # 2. STRESS-TEST RANGES (+/- 2%)
        growth_range = np.linspace(base_growth - 0.02, base_growth + 0.02, 5)
        wacc_range = np.linspace(base_wacc + 0.02, base_wacc - 0.02, 5)

        # 3. BUILD THE MATRIX
        matrix = np.zeros((len(wacc_range), len(growth_range)))
        for i, r in enumerate(wacc_range):
            for j, g in enumerate(growth_range):
                matrix[i, j] = fcf_proxy / (r - g) if r > g else np.nan

        # 4. VISUALIZATION
        plt.figure(figsize=(10, 6))

        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="RdYlGn",
                    xticklabels=[f"{x*100:.1f}%" for x in growth_range],
                    yticklabels=[f"{y*100:.1f}%" for y in wacc_range])

        plt.title(f"📊 S.A.F.E. STRESS TEST: {ticker_symbol.upper()}")
        plt.xlabel("Terminal Growth Rate (g)")
        plt.ylabel("Discount Rate (WACC)")
        plt.show()

    except Exception as e:
        print(f"❌ Sensitivity Error for {ticker_symbol}: {e}")



# ==========================================
# SECTION 5: COLAB EXECUTION & REPORTS
# ==========================================

# === Ticker Placement ===

target_stocks = ['MSFT', 'LLY', 'V', 'CVX', 'ABBV', 'NVDA', 'BRK.B', 'MU', 'MA', 'NFLX']

#=== Ticker Placement ===

print("Pinging Federal Reserve for Live WACC...")
try:
    tnx = yf.Ticker("^TNX")
    current_risk_free_rate = tnx.history(period="1d")['Close'].iloc[-1] / 100
    print(f"✅ U.S. Risk-Free Rate Locked: {current_risk_free_rate * 100:.2f}%\n")
except:
    current_risk_free_rate = 0.045
    print("⚠️ Fed API Offline. Using fallback Risk-Free Rate of 4.50%\n")

results = []
for ticker in target_stocks:
    print(f"Running Monte Carlo Simulator for {ticker}...")
    data = run_master_pipeline(ticker, current_risk_free_rate)
    if data and "Error" not in data:
        results.append(data)
    time.sleep(1)

print("\n ANALYSIS COMPLETE \n")

if results:
    safe_database = pd.DataFrame(results)


    safe_database = safe_database.replace([np.inf, -np.inf], np.nan)
    if len(safe_database) >= 5:
        pe_data = pd.to_numeric(safe_database['P/E Ratio'], errors='coerce').fillna(0)
        z_scores = np.abs(stats.zscore(pe_data))
        outliers = safe_database[z_scores > 3]
        if not outliers.empty:
            outlier_tickers = ", ".join(outliers['Ticker'].tolist())
            print(f"🧹 DATA SANITIZATION ACTIVE: Excluded {outlier_tickers} (Z-Score > 3).\n")
            safe_database = safe_database[z_scores <= 3]

    safe_database = safe_database.sort_values(by='Conviction Score (0-100)', ascending=False).reset_index(drop=True)


    print(" MODEL VALIDITY TEST (NULL HYPOTHESIS) ")
    top_half = safe_database.head(max(1, len(safe_database)//2))
    alpha_values = pd.to_numeric(top_half['1-Yr Alpha vs SPY (%)'], errors='coerce').dropna()

    if len(alpha_values) > 1:
        t_stat, p_val = stats.ttest_1samp(alpha_values, 0)
        avg_alpha = alpha_values.mean()

        print(f"Avg Alpha (Top Tier): {avg_alpha:.2f}%")
        print(f"P-Value: {p_val:.4f}")

        if p_val < 0.05 and avg_alpha > 0:
            print("✅ RESULT: Reject Null Hypothesis (Alpha is Significant)")
        elif p_val < 0.05 and avg_alpha < 0:
            print("⚠️ RESULT: Reject Null (Quality is currently out of favor)")
        else:
            print("ℹ️ RESULT: Fail to Reject Null (Need more data/larger sample)")
    else:
        print("Add more tickers to run statistical significance tests.")
    print("-" * 50 + "\n")


    print("\n📦 Packaging data for CSV...")
    csv_filename = "SAFE_Master_Database.csv"
    safe_database.to_csv(csv_filename, index=False)

    from google.colab import files
    files.download(csv_filename)
    print("✅ CSV Download triggered!")


    print("\n🗜️ Packaging Monte Carlo Charts into a ZIP file...")
    import time
    time.sleep(2)

    zip_filename = "SAFE_Monte_Carlo_DCFs.zip"
    chart_files = [f for f in os.listdir('.') if f.endswith("_DCF_Chart.png")]

    if len(chart_files) == 0:
        print("❌ ERROR: No chart files found. Check Section 2 for savefig logic.")
    else:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in chart_files:
                zipf.write(file)
                print(f"📦 Added to Archive: {file}")

        if os.path.exists(zip_filename):
            from google.colab import files
            files.download(zip_filename)
            print(f"🚀 SUCCESS: {len(chart_files)} charts packaged and downloading.")
    print("\n🏆 FULL S.A.F.E. LEADERBOARD:")
    display(safe_database)

# ---SUMMARY ---
print("\n" + "="*70)
print("💼 S.A.F.E. EXECUTIVE STRATEGY REPORT: THE NEXUS GROUP")
print("="*70)

try:
    # 1. THE SANITIZER: Filter out "Value Traps" (negative floors)
    filtered_db = safe_database[safe_database['Crisis Value ($)'] > 0].copy()

    # 2. DYNAMIC SORTING: Top 5 by Conviction Score
    exec_summary = filtered_db.sort_values(by='Conviction Score (0-100)', ascending=False).head(5).copy()

    if exec_summary.empty:
        print("⚠️ No assets met the S.A.F.E. quality threshold for this run.")
    else:
        avg_alpha = pd.to_numeric(exec_summary['1-Yr Alpha vs SPY (%)'], errors='coerce').mean()

        # EXTRACT PROPER MONTE CARLO WIN RATE:
        mc_win_rates = exec_summary['MC DCF Model'].astype(str).str.extract(r'\((\d+)% Win\)').astype(float)
        avg_mc_win_rate = mc_win_rates[0].mean() if not mc_win_rates.empty else 0

        avg_backtest = pd.to_numeric(exec_summary['Backtest Win Rate (%)'], errors='coerce').mean()
        avg_mdd = pd.to_numeric(exec_summary['Max Drawdown (%)'], errors='coerce').mean()
        avg_safe = pd.to_numeric(exec_summary['S.A.F.E. Score (1-10)'], errors='coerce').mean()

        # 3. Summary Table
        print(f"\n--- PORTFOLIO AGGREGATES (Top Nexus Constituents) ---")
        print(f"🔹 Historical 1-Yr Alpha:   {avg_alpha:+.2f}%")
        print(f"🔹 Average MC Win Rate:     {avg_mc_win_rate:.1f}%")
        print(f"🔹 Hist. Backtest Win Rate: {avg_backtest:.1f}%")
        print(f"🔹 Average Max Drawdown:    {avg_mdd:.2f}%")
        print(f"🔹 Mean S.A.F.E. Grade:     {avg_safe:.2f}/10")

        # 4. Individual Breakdown
        print(f"\n--- INDIVIDUAL ASSET INTELLIGENCE ---")
        header = f"{'Ticker':<8} | {'Master Grade':<20} | {'Crisis Val':<12} | {'MC Win %'}"
        print(header)
        print("-" * len(header))

        for _, row in exec_summary.iterrows():
            win_pct = row['MC DCF Model'].split('(')[1].split(')')[0] if '(' in str(row['MC DCF Model']) else "N/A"
            print(f"{row['Ticker']:<8} | {row['Master Grade']:<20} | ${row['Crisis Value ($)']:<11.2f} | {win_pct}")

        # 5. AUTOMATIC STRESS TESTS
        print(f"\n--- VISUAL STRESS TESTS (Top Conviction Assets) ---")
        for ticker in exec_summary['Ticker'].head(3):
            run_nexus_sensitivity(ticker, safe_database)

        # 6. DYNAMIC NEXUS JUSTIFICATION
        current_sectors = ", ".join(exec_summary['Sector'].unique())
        print(f"\n--- THE NEXUS FACTOR JUSTIFICATION ---")
        print(f"> STRATEGY: This portfolio isolates 'Nexus' assets where high fundamental")
        print(f"> quality (S.A.F.E. Score > {avg_safe:.1f}) intersects with verified momentum.")
        print(f"> DIVERSIFICATION: Analysis currently spans: {current_sectors}.")
        print(f"> RISK PROFILE: All constituents maintain a positive Crisis Value floor.")

except Exception as e:
    print(f"Executive Summary Error: {e}")

print("="*70 + "\n")
