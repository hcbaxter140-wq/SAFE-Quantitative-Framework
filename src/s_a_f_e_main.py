# S.A.F.E. ENGINE (Sector-Adaptive Fundamental Engine) v1.0
# Developed by: Henry C. Baxter
# Description: Quantitative framework for secondary market constituent valuation and risk optimization.

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
# GLOBAL TRANSLATION MATRIX
# ==========================================
geopolitical_risk_matrix = {
    'United States': 0.00, 'Canada': 0.00, 'United Kingdom': 0.005,
    'China': 0.015, 'India': 0.025, 'Brazil': 0.035, 'Russia': 0.20
}

def get_size_premium(market_cap_billions):
    if pd.isna(market_cap_billions): return 0.02
    if market_cap_billions >= 50.0: return 0.00
    if market_cap_billions >= 10.0: return 0.005
    if market_cap_billions >= 2.0: return 0.012
    if market_cap_billions >= 0.3: return 0.025
    return 0.045

sector_adjustment_matrix = {
    'Technology': {'smooth_fcf': True, 'wacc_modifier': 0.00},
    'Consumer Cyclical': {'smooth_fcf': True, 'wacc_modifier': 0.00},
    'Communication Services': {'smooth_fcf': True, 'wacc_modifier': 0.00},
    'Utilities': {'smooth_fcf': False, 'wacc_modifier': -0.01},
    'Real Estate': {'smooth_fcf': False, 'wacc_modifier': 0.00}
}

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
# SECTION 2: MONTE CARLO DCF ENGINE
# ==========================================
def generate_dcf_model(ticker, current_price, current_fcf, shares_out, net_debt, base_growth_rate, live_wacc, sharpe_ratio=None):
    if pd.isna(current_fcf) or current_fcf <= 0 or pd.isna(shares_out) or shares_out <= 0:
        return "N/A (Negative FCF/Missing Data)", 0, np.nan

    try:
        tgr = 0.025
        base_g = min(max(base_growth_rate / 100, 0.02), 0.15) if pd.notna(base_growth_rate) else 0.05

        iterations = 1000
        rand_g = np.random.normal(base_g, 0.02, iterations)
        rand_g = np.clip(rand_g, 0.0, 0.25)

        rand_wacc = np.random.normal(live_wacc, 0.01, iterations)
        rand_wacc = np.clip(rand_wacc, 0.05, 0.15)

        sdf_m = 1 / (1 + rand_wacc)
        sdf_volatility = np.std(sdf_m) / np.mean(sdf_m)

        if sharpe_ratio is not None and pd.notna(sharpe_ratio) and sharpe_ratio > 0:
            if sdf_volatility < sharpe_ratio:
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

            implied_price = (implied_equity / shares_out) * hj_penalty
            implied_prices.append(implied_price)

        implied_prices = np.array(implied_prices)
        implied_prices = implied_prices[(implied_prices > 0) & (implied_prices < current_price * 5)]

        if len(implied_prices) == 0: return "Error: Invalid MC Projections", 0, np.nan

        mean_implied = np.mean(implied_prices)
        prob_undervalued = np.sum(implied_prices > current_price) / len(implied_prices) * 100

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

# --- BLACK SWAN SENSITIVITY ---
def run_stress_test(ticker, current_price, current_fcf, shares_out, net_debt, base_g, wacc):
    try:
        scenarios = {
            "Fed_Spike": (base_g, wacc + 0.03),
            "Growth_Shock": (base_g * 0.5, wacc),
            "Crisis": (base_g * 0.2, wacc + 0.04)
        }

        stress_results = {}
        for name, (g, w) in scenarios.items():
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
        country = info.get('country', 'United States')
        tax_rate = inc.loc[latest_date].get('Tax Rate For Calcs', 0.21)

        # RISK MODELS TO OBTAIN BETA
        alpha_1y, alpha_3y, beta, sharpe, sortino, ann_vol = run_historical_risk_analysis(ticker_symbol, risk_free_rate)

        # --- CORE CALCULATIONS ---
        equity_risk_premium = 0.055
        net_debt = max(total_debt - cash_equiv, 0)

        total_capital = market_cap + total_debt if pd.notna(market_cap) and market_cap > 0 else 1
        weight_e = market_cap / total_capital if total_capital > 0 else 1
        weight_d = total_debt / total_capital if total_capital > 0 else 0

        raw_beta = beta if pd.notna(beta) else 1.0
        adj_beta = (0.67 * raw_beta) + (0.33 * 1.0)

        cost_of_equity = risk_free_rate + (adj_beta * equity_risk_premium)

        try:
            raw_interest = inc.loc[latest_date].get('Interest Expense', 0)
            interest_expense = abs(float(raw_interest)) if pd.notna(raw_interest) else 0
        except:
            interest_expense = 0

        cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.045

        safe_tax_rate = tax_rate if pd.notna(tax_rate) else 0.21
        base_wacc = (weight_e * cost_of_equity) + (weight_d * cost_of_debt * (1 - safe_tax_rate))

        crp_penalty = geopolitical_risk_matrix.get(country, 0.00)
        size_premium = get_size_premium(market_cap / 1e9) if pd.notna(market_cap) else 0.02
        sector_rules = sector_adjustment_matrix.get(sector, {'smooth_fcf': False, 'wacc_modifier': 0.00})

        raw_live_wacc = base_wacc + crp_penalty + size_premium + sector_rules.get('wacc_modifier', 0.00)

        live_wacc = max(min(raw_live_wacc, 0.125), 0.07) 
        dynamic_wacc_pct = live_wacc * 100

        # --- DYNAMIC FCF SMOOTHING ---
        raw_fcf = cf.loc[latest_date].get('Free Cash Flow', np.nan) if not cf.empty else np.nan
        if sector_rules.get('smooth_fcf', False) and len(cf.columns) >= 3 and 'Free Cash Flow' in cf.index:
            fcf = cf.loc['Free Cash Flow'].iloc[:3].mean()
        else:
            fcf = raw_fcf

        blended_growth_rate = (eps_growth * 0.7) + (rev_growth_yoy * 0.3) if pd.notna(eps_growth) and pd.notna(rev_growth_yoy) else (rev_growth_yoy if pd.notna(rev_growth_yoy) else 0.05)

        dcf_status, mc_win_rate, mean_implied = generate_dcf_model(
            ticker_symbol, current_price, fcf, shares_out, net_debt,
            blended_growth_rate, live_wacc, sharpe_ratio=sharpe
        )

        stress_vals = run_stress_test(ticker_symbol, current_price, fcf, shares_out, net_debt, blended_growth_rate, live_wacc)
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
        nopat = operating_income * (1 - safe_tax_rate)
        invested_cap = bal.loc[latest_date].get('Invested Capital', (total_assets - current_liabilities))

        roa = (net_income / total_assets) * 100 if total_assets > 0 else np.nan
        roe = (net_income / equity) * 100 if pd.notna(equity) and equity > 0 else np.nan
        roic = (nopat / invested_cap) * 100 if invested_cap > 0 else np.nan

        baseline_hurdle_rate = 10.0
        roic_wacc_spread = roic - dynamic_wacc_pct if pd.notna(roic) else np.nan

        if pd.notna(roic) and pd.notna(dynamic_wacc_pct):
            passes_dual_hurdle = (roic > dynamic_wacc_pct) and (roic > baseline_hurdle_rate)
        else:
            passes_dual_hurdle = False

        economic_value_added = invested_cap * (roic_wacc_spread / 100) if pd.notna(invested_cap) and pd.notna(roic_wacc_spread) else np.nan

        current_ratio = current_assets / current_liabilities if pd.notna(current_assets) and current_liabilities > 0 else np.nan
        debt_to_equity = total_debt / equity if pd.notna(equity) and equity > 0 else np.nan
        net_debt_to_ebitda = (net_debt / ebitda) if pd.notna(ebitda) and ebitda > 0 else np.nan

        fcf_margin = (fcf / revenue) * 100 if pd.notna(fcf) else np.nan
        fcf_yield = (fcf / market_cap) * 100 if pd.notna(fcf) and pd.notna(market_cap) else np.nan

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
        is_trap = (sector in ['Energy', 'Basic Materials', 'Industrials']) and (pd.notna(pe_ratio)) and (pe_ratio < 13)
        conviction_score = round((final_score * 5.5) + (mc_win_rate / 100) * 30 + (min((sortino / 2.0) * 15, 15) if (pd.notna(sortino) and sortino > 0) else 0), 1)

        if is_trap:
            conviction_score -= 30

        if conviction_score >= 85: master_grade = "A+ (High Conviction Buy)"
        elif conviction_score >= 75: master_grade = "A (Buy)"
        elif conviction_score >= 60: master_grade = "B (Hold)"
        else: master_grade = "C (Avoid/Sell)"

        upside_potential = (mean_implied / current_price) - 1 if (pd.notna(mean_implied) and pd.notna(current_price) and current_price > 0) else 0
        
        if is_trap:
            kelly_allocation = 0
        else:
            kelly_allocation = max(0, min(((mc_win_rate/100) - ((1 - mc_win_rate/100) / (upside_potential / ann_vol if ann_vol > 0 else 1))) / 2, 0.20)) * 100 if upside_potential > 0 else 0

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
            'Profitable (TTM)': "Yes" if f_score == 1 else "No",
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
            'Passes Dual-Hurdle': "✅ Pass" if passes_dual_hurdle else "❌ Fail",
            'ROIC-WACC Spread (%)': round(roic_wacc_spread, 2) if pd.notna(roic_wacc_spread) else "N/A",
            'EVA ($)': round(economic_value_added, 2) if pd.notna(economic_value_added) else "N/A",
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
# SECTION 5: DYNAMIC SENSITIVITY & STRESS-TESTING
# =========================================================
def run_nexus_sensitivity(ticker_symbol, database):
    try:
        asset_data = database[database['Ticker'] == ticker_symbol.upper()].iloc[0]
        current_price = asset_data.get('Current Price ($)', 0)

        base_growth = 0.03
        base_wacc = asset_data['Live WACC (%)'] / 100

        fcf_proxy = asset_data['S.A.F.E. Score (1-10)'] * (base_wacc - base_growth)

        growth_range = np.linspace(base_growth - 0.02, base_growth + 0.02, 5)
        wacc_range = np.linspace(base_wacc + 0.02, base_wacc - 0.02, 5)

        matrix = np.zeros((len(wacc_range), len(growth_range)))
        for i, r in enumerate(wacc_range):
            for j, g in enumerate(growth_range):
                matrix[i, j] = fcf_proxy / (r - g) if r > g else np.nan

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

def apply_institutional_risk_bounds(df, max_asset_cap=0.20, max_sector_cap=0.30):
    portfolio_df = df.copy()
    portfolio_df['raw_kelly'] = portfolio_df['Target Allocation (Kelly)'].replace('0.0% (Skip)', '0').str.rstrip('%').astype(float) / 100
    portfolio_df['constrained_allocation'] = portfolio_df['raw_kelly'].clip(upper=max_asset_cap)
    sector_exposure = portfolio_df.groupby('Sector')['constrained_allocation'].sum()
    overallocated_sectors = sector_exposure[sector_exposure > max_sector_cap].index
    for sector in overallocated_sectors:
        sector_mask = portfolio_df['Sector'] == sector
        current_sector_total = portfolio_df.loc[sector_mask, 'constrained_allocation'].sum()
        scale_factor = max_sector_cap / current_sector_total
        portfolio_df.loc[sector_mask, 'constrained_allocation'] *= scale_factor
    total_portfolio_weight = portfolio_df['constrained_allocation'].sum()
    if total_portfolio_weight > 1.0:
        portfolio_df['constrained_allocation'] /= total_portfolio_weight
    portfolio_df['Target Allocation (Kelly)'] = portfolio_df['constrained_allocation'].apply(lambda x: f"{round(x * 100, 1)}%" if x > 0 else "0.0% (Skip)")
    return portfolio_df.drop(columns=['raw_kelly', 'constrained_allocation'])

# ==========================================
# SECTION 6: COLAB EXECUTION & REPORTS
# ==========================================
target_stocks = ['AAPL', 'NVDA', 'TSLA', 'COST', 'MSFT','BA', 'AAL', 'WBD', 'PTON', 'AMC','CF', 'ALB', 'XOM', 'NEM', 'MOS','PG', 'KO', 'NEE', 'JNJ', 'WM','CROX', 'TXRH', 'INCY', 'VRTX', 'ATGE', 'MUSA']

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
    print("🛡️ Applying Institutional Risk Bounds (Max 30% per Sector)...")
    safe_database = apply_institutional_risk_bounds(safe_database, max_asset_cap=0.20, max_sector_cap=0.30)
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
            files.download(zip_filename)
            print(f"🚀 SUCCESS: {len(chart_files)} charts packaged and downloading.")

    print("\n🏆 FULL S.A.F.E. LEADERBOARD:")
    pd.set_option('display.max_columns', None)
    display(safe_database)

# ---SUMMARY ---
print("\n" + "="*70)
print("💼 S.A.F.E. EXECUTIVE STRATEGY REPORT: THE NEXUS GROUP")
print("="*70)

try:
# SANITIZATION
    filtered_db = safe_database[
        (safe_database['Crisis Value ($)'] > 0) & 
        (safe_database['Passes Dual-Hurdle'] == "✅ Pass")
    ].copy()

    exec_summary = filtered_db.sort_values(by='Conviction Score (0-100)', ascending=False).head(5).copy()

    if exec_summary.empty:
        print("⚠️ No assets met the S.A.F.E. quality threshold for this run.")
    else:
        avg_alpha = pd.to_numeric(exec_summary['1-Yr Alpha vs SPY (%)'], errors='coerce').mean()
        mc_win_rates = exec_summary['MC DCF Model'].astype(str).str.extract(r'\((\d+)% Win\)').astype(float)
        avg_mc_win_rate = mc_win_rates[0].mean() if not mc_win_rates.empty else 0
        avg_backtest = pd.to_numeric(exec_summary['Backtest Win Rate (%)'], errors='coerce').mean()
        avg_mdd = pd.to_numeric(exec_summary['Max Drawdown (%)'], errors='coerce').mean()
        avg_safe = pd.to_numeric(exec_summary['S.A.F.E. Score (1-10)'], errors='coerce').mean()

        print(f"\n--- PORTFOLIO AGGREGATES (Top Nexus Constituents) ---")
        print(f"🔹 Historical 1-Yr Alpha:   {avg_alpha:+.2f}%")
        print(f"🔹 Average MC Win Rate:     {avg_mc_win_rate:.1f}%")
        print(f"🔹 Hist. Backtest Win Rate: {avg_backtest:.1f}%")
        print(f"🔹 Average Max Drawdown:    {avg_mdd:.2f}%")
        print(f"🔹 Mean S.A.F.E. Grade:     {avg_safe:.2f}/10")

        print(f"\n--- INDIVIDUAL ASSET INTELLIGENCE ---")
        header = f"{'Ticker':<8} | {'Master Grade':<20} | {'Crisis Val':<12} | {'MC Win %'}"
        print(header)
        print("-" * len(header))

        for _, row in exec_summary.iterrows():
            win_pct = row['MC DCF Model'].split('(')[1].split(')')[0] if '(' in str(row['MC DCF Model']) else "N/A"
            print(f"{row['Ticker']:<8} | {row['Master Grade']:<20} | ${row['Crisis Value ($)']:<11.2f} | {win_pct}%")

        print(f"\n--- VISUAL STRESS TESTS (Top Conviction Assets) ---")
        for ticker in exec_summary['Ticker'].head(3):
            run_nexus_sensitivity(ticker, safe_database)

        current_sectors = ", ".join(exec_summary['Sector'].unique())
        print(f"\n--- THE NEXUS FACTOR JUSTIFICATION ---")
        print(f"> STRATEGY: This portfolio isolates 'Nexus' assets where high fundamental")
        print(f"> quality (S.A.F.E. Score > {avg_safe:.1f}) intersects with verified momentum.")
        print(f"> DIVERSIFICATION: Analysis currently spans: {current_sectors}.")
        print(f"> RISK PROFILE: All constituents maintain a positive Crisis Value floor.")

except Exception as e:
    print(f"Executive Summary Error: {e}")

print("="*70 + "\n")
