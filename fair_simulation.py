"""
========================================================================================
Carremax Cyber Risk Quantification (CRQ) - FAIR Monte Carlo Simulation
========================================================================================
Author: Risk Quantification & Cybersecurity Strategy Team
Framework: FAIR (Factor Analysis of Information Risk) / ISO 27005 / Open FAIR
Case: Carremax Sports Retail (€40B Revenue Tier)
========================================================================================
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

BOARD_RISK_APPETITE = 50_000_000  # €50M Board limit
MITIGATION_INVESTMENT = 2_000_000  # €2M Security Automation & OT Visibility Project
# --------------------------------------------------------------------------------------
# 1. Core Distribution Generators & Helper Functions
# --------------------------------------------------------------------------------------
def rpert(n, min_val, mode_val, max_val, lam=4):
    """
    Generate n random samples from a PERT (Project Evaluation and Review Technique) distribution.
    Widely used in FAIR modeling to capture expert estimates (min, most likely, max).
    """
    if min_val == max_val:
        return np.full(n, min_val)
    mu = (min_val + lam * mode_val + max_val) / (lam + 2)
    v = (mu - min_val) / (max_val - min_val)
    w = (max_val - mu) / (max_val - min_val)
    alpha = v * (lam + 2)
    beta = w * (lam + 2)
    return min_val + (max_val - min_val) * stats.beta.rvs(alpha, beta, size=n)

def gauss(n, min_val, mode_val, max_val):
    """
    Generate gaussian variable truncated between min and max with average at mode
    """
    if (max_val + min_val )/2 < mode_val:
        sigma = (max_val*1.1 - min_val) / 6
    else:
        sigma = (max_val - min_val*0.9) / 6
    mu = mode_val
    res = np.random.randn(n)
    X = np.array(mu + res * sigma)
    for i in range(len(X)):
        X[i] = max(X[i], 0)
    return X

def calculate_downtime_loss(duration_weeks):
    """
    Non-linear business interruption mapping based on Carremax executive interview:
      - 0 to 1 Week:  €0 to €15M
      - 1 to 2 Weeks: €15M to €30M (Limit of degraded backup operations)
      - 2 to 3 Weeks: €30M to €50M (Reaches Board Risk Appetite of €50M)
      - 3 to 4 Weeks: €50M to €90M (Catastrophic supply chain / ERP failure)
      - > 4 Weeks:    €90M + €40M/week thereafter
    """
    x = np.atleast_1d(duration_weeks)
    loss = np.zeros_like(x, dtype=float)
    
    pal1 = 15_000_000
    pal2 = 30_000_000
    pal3 = 50_000_000
    pal4 = 90_000_000
    
    
    palier_1 = gauss(1, pal1*0.75, pal1, pal1*1.25)[0]
    palier_2 = gauss(1, pal2*0.75, pal2, pal2*1.25)[0]
    palier_3 = gauss(1, pal3*0.75, pal3, pal3*1.25)[0]
    palier_4 = gauss(1, pal4*0.75, pal4, pal4*1.25)[0]
    
    mask0 = (x <= 1.0)
    loss[mask0] = x[mask0] * palier_1
    
    mask1 = (x > 1.0) & (x <= 2.0)
    loss[mask1] = palier_1 + (x[mask1] - 1.0) * (palier_2 - palier_1)
    
    mask2 = (x > 2.0) & (x <= 3.0)
    loss[mask2] = palier_2 + (x[mask2] - 2.0) * (palier_3 - palier_2)
    
    mask3 = (x > 3.0) & (x <= 4.0)
    loss[mask3] = palier_3 + (x[mask3] - 3.0) * (palier_4 - palier_3)
    
    mask4 = (x > 4.0)
    loss[mask4] = palier_4 + (x[mask4] - 4.0) * palier_3
    
    return loss if isinstance(duration_weeks, np.ndarray) else float(loss[0])


# --------------------------------------------------------------------------------------
# 2. FAIR Model Simulation Engine
# --------------------------------------------------------------------------------------
def run_fair_simulation(
    n_sims=10000,
    # Threat Event Frequency (TEF) - attempts per year
    tef_min=1.0, tef_mode=1.5, tef_max=2.0,
    # Vulnerability (Vuln) - probability attempt breaches defenses
    vuln_min=0.16, vuln_mode=0.21, vuln_max=0.26,
    # Downtime Duration (weeks)
    fixed_downtime=False, downtime_min=1.0, downtime_mode=2.0, downtime_max=4.5,
    # Recovery / Incident Response & Forensics Costs (€)
    ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
    # Secondary Losses: Legal, GDPR, Refunds (€)
    secondary_min=0, secondary_mode=2_000_000, secondary_max=5_000_000,
    # Ransom Cap and Extortion Behavior
    ransom_cap=20_000_000, ransom_min = 5_000_000, ransom_max = 40_000_000,
    # Insurance Parameters
    insurance_weekly_payout=500_000,
    insurance_max_weeks=8,
    insurance_annual_premium=300_000,
    seed=42
):
    """
    Executes an end-to-end FAIR Monte Carlo simulation across n_sims annual iterations.
    """
    np.random.seed(seed)
    
    # 1. Threat Event Frequency (TEF) sampling
    tef_samples = gauss(n_sims, tef_min, tef_mode, tef_max)
    annual_threat_counts = np.random.poisson(tef_samples)
    
    # Container records for annual aggregation
    annual_loss_events = np.zeros(n_sims, dtype=int)
    annual_gross_loss = np.zeros(n_sims, dtype=float)
    annual_net_loss = np.zeros(n_sims, dtype=float)
    annual_bi_loss = np.zeros(n_sims, dtype=float)
    annual_ir_loss = np.zeros(n_sims, dtype=float)
    annual_ransom_loss = np.zeros(n_sims, dtype=float)
    annual_sec_loss = np.zeros(n_sims, dtype=float)
    annual_insurance_payout = np.zeros(n_sims, dtype=float)
    annual_downtime_weeks = np.zeros(n_sims, dtype=float)

    for i, threats in enumerate(annual_threat_counts):
        if threats == 0:
            annual_net_loss[i] = insurance_annual_premium
            continue
            
        # Sample vulnerability for each threat attempt
        vulns = gauss(threats, vuln_min, vuln_mode, vuln_max)
        breaches = np.random.rand(threats) < vulns
        n_breaches = int(np.sum(breaches))
        annual_loss_events[i] = n_breaches
        
        if n_breaches == 0:
            annual_net_loss[i] = insurance_annual_premium
            continue
            
        # Simulate each successful breach / loss event
        if fixed_downtime:
            downtimes = np.full(n_breaches, downtime_mode)
        else:
            downtimes = gauss(n_breaches, downtime_min, downtime_mode, downtime_max)
        bi_losses = calculate_downtime_loss(downtimes)
        ir_losses = gauss(n_breaches, ir_min, ir_mode, ir_max)
        
        # Ransom extortion: triggered with higher likelihood when downtime exceeds 2 weeks
        ransom_losses = np.zeros(n_breaches)
        for j, dt in enumerate(downtimes):
            # If outage extends beyond backup threshold (2 weeks), pressure to pay escalates
            p_pay = 0.50 if dt <= 2.0 else (0.70 if dt < 3.0 else 0.90)
            # Sample ransom demand / settlement with the €20M willingness cap as average
            ransom_amount = float(gauss(1, ransom_min, ransom_cap, ransom_max)[0])
            if np.random.rand() < p_pay and ransom_amount < ransom_cap:
                # Ransom is paid, but capped at the organization's willingness to pay
                ransom_losses[j] = ransom_amount

        sec_other_losses = gauss(n_breaches, secondary_min, secondary_mode, secondary_max)
        
        # Insurance payout calculation per event
        # €500k/week capped at 8 weeks (€4.0M max per event)
        ins_payouts = np.minimum(downtimes * insurance_weekly_payout, insurance_weekly_payout * insurance_max_weeks)
        
        # Event totals
        event_gross = bi_losses + ir_losses + ransom_losses + sec_other_losses
        event_net = event_gross - ins_payouts
        
        # Accumulate annual figures
        annual_bi_loss[i] = np.sum(bi_losses)
        annual_ir_loss[i] = np.sum(ir_losses)
        annual_ransom_loss[i] = np.sum(ransom_losses)
        annual_sec_loss[i] = np.sum(sec_other_losses)
        annual_insurance_payout[i] = np.sum(ins_payouts)
        annual_downtime_weeks[i] = np.sum(downtimes)
        
        annual_gross_loss[i] = np.sum(event_gross)
        # Net loss includes fixed annual insurance premium
        annual_net_loss[i] = np.sum(event_net) + insurance_annual_premium

    results = {
        "n_sims": n_sims,
        "annual_threat_counts": annual_threat_counts,
        "annual_loss_events": annual_loss_events,
        "annual_gross_loss": annual_gross_loss,
        "annual_net_loss": annual_net_loss,
        "annual_bi_loss": annual_bi_loss,
        "annual_ir_loss": annual_ir_loss,
        "annual_ransom_loss": annual_ransom_loss,
        "annual_sec_loss": annual_sec_loss,
        "annual_insurance_payout": annual_insurance_payout,
        "annual_downtime_weeks": annual_downtime_weeks,
        # Executive Summary Metrics
        "ael_gross": float(np.mean(annual_gross_loss)),
        "ael_net": float(np.mean(annual_net_loss)),
        "var_90_net": float(np.percentile(annual_net_loss, 90)),
        "var_95_net": float(np.percentile(annual_net_loss, 95)),
        "var_99_net": float(np.percentile(annual_net_loss, 99)),
        "max_loss_net": float(np.max(annual_net_loss)),
        "prob_exceed_50m": float(np.mean(annual_net_loss > 50_000_000)),
        "prob_at_least_one_breach": float(np.mean(annual_loss_events > 0)),
        "expected_downtime_weeks": float(np.mean(annual_downtime_weeks[annual_loss_events > 0])) if np.any(annual_loss_events > 0) else 0.0,
        "expected_insurance_recovery": float(np.mean(annual_insurance_payout[annual_loss_events > 0])) if np.any(annual_loss_events > 0) else 0.0,
    }
    return results

def show_results(baseline1, mitigated1, mitigated2 = None, name_sim = ""):
    """
    Display the simulation results for both baseline and mitigated scenarios.
    """
    # ----------------------------------------------------------------------------------
    # C. Executive Metrics & ROSI Calculation
    # ----------------------------------------------------------------------------------
    risk_reduction_ael = baseline1["ael_net"] - mitigated1["ael_net"]
    rosi = ((risk_reduction_ael - MITIGATION_INVESTMENT) / MITIGATION_INVESTMENT) * 100.0
    
    print("-" * 80)
    print(" 1. AS-IS BASELINE RISK PROFILE (CURRENT STATE)")
    print("-" * 80)
    print(f" • Annual Probability of at Least 1 Breach: {baseline1['prob_at_least_one_breach']:.1%}")
    print(f" • Annualized Expected Loss (Net AEL):       €{baseline1['ael_net']:,.0f}")
    print(f" • 90% Value at Risk (90% VaR):             €{baseline1['var_90_net']:,.0f}")
    print(f" • 95% Value at Risk (95% VaR):             €{baseline1['var_95_net']:,.0f}")
    print(f" • 99% Value at Risk (99% VaR):             €{baseline1['var_99_net']:,.0f}")
    print(f" • Maximum Simulated Single-Year Loss:      €{baseline1['max_loss_net']:,.0f}")
    print(f" • Probability of Exceeding Board Appetite: {baseline1['prob_exceed_50m']:.1%}  (Appetite: €50M)")
    print(f" • Average Breach Downtime:                 {baseline1['expected_downtime_weeks']:.2f} weeks")
    print(f" • Average Insurance Payout per Breach:     €{baseline1['expected_insurance_recovery']:,.0f}")

    print("\n" + "-" * 80)
    print(" 2. TO-BE MITIGATED PROFILE (€2M AUTOMATION + 100% EDR + OT CONTROL)")
    print("-" * 80)
    print(f" • Annual Probability of at Least 1 Breach: {mitigated1['prob_at_least_one_breach']:.1%}")
    print(f" • Post-Mitigation Net AEL:                 €{mitigated1['ael_net']:,.0f}")
    print(f" • Post-Mitigation 90% VaR:                 €{mitigated1['var_90_net']:,.0f}")
    print(f" • Post-Mitigation 95% VaR:                 €{mitigated1['var_95_net']:,.0f}")
    print(f" • Post-Mitigation 99% VaR:                 €{mitigated1['var_99_net']:,.0f}")
    print(f" • Probability of Exceeding Board Appetite: {mitigated1['prob_exceed_50m']:.1%}  (Near Zero)")
    
    print("\n" + "-" * 80)
    print(" 3. STRATEGIC BUSINESS CASE & RETURN ON SECURITY INVESTMENT (ROSI)")
    print("-" * 80)
    print(f" • Annual Net Risk Reduction (Δ AEL):       €{risk_reduction_ael:,.0f}")
    print(f" • Proposed Security Investment:            €{MITIGATION_INVESTMENT:,.0f}")
    print(f" • Return on Security Investment (ROSI):    {rosi:.1f}%")
    print(f" • Net Annual Benefit:                      €{(risk_reduction_ael - MITIGATION_INVESTMENT):,.0f} / year")
    print("=" * 80)
    
    # ----------------------------------------------------------------------------------
    # D. Multi-Panel Visualizations
    # ----------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    # --- Plot 1: Loss Exceedance Curve (LEC) ---
    sorted_base = np.sort(baseline1["annual_net_loss"])
    prob_base = 1.0 - np.arange(1, baseline1["n_sims"] + 1) / baseline1["n_sims"]
    
    sorted_mit = np.sort(mitigated1["annual_net_loss"])
    prob_mit = 1.0 - np.arange(1, mitigated1["n_sims"] + 1) / mitigated1["n_sims"]
    
    ax1.plot(sorted_base / 1e6, prob_base, label="Baseline (As-Is Current State)", color="#005587", linewidth=2.8)
    mitigated_results = [
    mitigated1,
    mitigated2
    ]

    labels = [
        "Mitigated with all solutions",
        "Mitigated with 99'%'EDR"
    ]

    for mitigated, label in zip(mitigated_results, labels):

        sorted_mit = np.sort(mitigated["annual_net_loss"])

        prob_mit = (
            1.0
            - np.arange(1, mitigated["n_sims"] + 1)
            / mitigated["n_sims"]
        )

        ax1.plot(
            sorted_mit / 1e6,
            prob_mit,
            label=label,
            linewidth=2.5
        )
    # Reference vertical lines
    ax1.axvline(x=BOARD_RISK_APPETITE / 1e6, color="#e3000f", linestyle="--", linewidth=2.2,
                label=f"Board Risk Appetite (€{BOARD_RISK_APPETITE/1e6:.0f}M)")
    ax1.axvline(x=baseline1["var_90_net"] / 1e6, color="#005587", linestyle=":", linewidth=1.8,
                label=f"Baseline 90% VaR (€{baseline1['var_90_net']/1e6:.1f}M)")
    ax1.axvline(x=mitigated1["var_90_net"] / 1e6, color="#86bc25", linestyle=":", linewidth=1.8,
                label=f"Mitigated 90% VaR (€{mitigated1['var_90_net']/1e6:.1f}M)")

    ax1.set_title("Loss Exceedance Curve (LEC) - As-Is vs. To-Be", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Annual Net Financial Loss (€ Millions)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability of Exceedance", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 110)
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)
    
    # --- Plot 2: Average Loss Component Breakdown (When Breach Occurs) ---
    categories = ["Business\nInterruption", "IR & Forensics\nRecovery", "Extortion\n(Ransom)", "Secondary\n(GDPR/Refunds)"]
    breach_filter = baseline1["annual_loss_events"] > 0
    mean_bi = np.mean(baseline1["annual_bi_loss"][breach_filter]) / 1e6
    mean_ir = np.mean(baseline1["annual_ir_loss"][breach_filter]) / 1e6
    mean_ransom = np.mean(baseline1["annual_ransom_loss"][breach_filter]) / 1e6
    mean_sec = np.mean(baseline1["annual_sec_loss"][breach_filter]) / 1e6
    
    values = [mean_bi, mean_ir, mean_ransom, mean_sec]
    colors = ["#005587", "#43b02a", "#d9381e", "#768692"]
    
    bars = ax2.bar(categories, values, color=colors, width=0.55, edgecolor="#222", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"€{h:.1f}M",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
                     
    ax2.set_title("Baseline Loss Breakdown (Average per Breach Event)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("Average Cost (€ Millions)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(values) * 1.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    
    # Annotate insurance recovery
    mean_ins = np.mean(baseline1["annual_insurance_payout"][breach_filter]) / 1e6
    ax2.text(0.95, 0.90, f"Cyber Insurance Offset: -€{mean_ins:.2f}M / event\n(Capped at €4.0M)",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3f7", edgecolor="#005587", alpha=0.9))

    plt.tight_layout()
    plt.savefig("lec_curve_output"+name_sim+".png", dpi=160)
    print("\n[SUCCESS] Charts saved to 'lec_curve_output.png'")



# --------------------------------------------------------------------------------------
# 3. Main Execution & Comparison (As-Is Baseline vs. To-Be Mitigation)
# --------------------------------------------------------------------------------------
def main():
    
    
    print("=" * 80)
    print(" CARREMAX CYBER RISK QUANTIFICATION (FAIR MONTE CARLO SIMULATION)")
    print("=" * 80)
    print(f"Iterations: 10,000 | Board Risk Appetite: €{BOARD_RISK_APPETITE:,.0f}\n")
    
    print("\n" + "=" * 80)
    print(" A/ Variable duration downtime ")
    print("=" * 80)
    
    # ----------------------------------------------------------------------------------
    # A. Baseline Simulation (As-Is: 90% EDR, Manual IR, 0% OT Visibility)
    # ----------------------------------------------------------------------------------
    baseline1 = run_fair_simulation(
        n_sims=10000,
        tef_min=1.0, tef_mode=1.5, tef_max=2.0,
        vuln_min=0.16, vuln_mode=0.21, vuln_max=0.26,
        downtime_min=1.0, downtime_mode=2.0, downtime_max=4.5,
        ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
        secondary_min=0, secondary_mode=2_000_000, secondary_max=5_000_000,
        ransom_cap=20_000_000,
        insurance_weekly_payout=500_000,
        insurance_max_weeks=8,
        insurance_annual_premium=300_000,
        seed=42
    )
    
    # ----------------------------------------------------------------------------------
    # B. Mitigated Simulation (To-Be: 100% EDR, Automated IR, Supply-Chain OT Visibility)
    # ----------------------------------------------------------------------------------
    # 1. Vuln drops drastically 
    # 2. Downtime duration drops due to automated recovery
    # 3. IR/Forensics cost drops due to automated playbooks
    mitigated_total = run_fair_simulation(
        n_sims=10000,
        tef_min=1.0, tef_mode=1.5, tef_max=2.0,
        vuln_min=0.09, vuln_mode=0.12, vuln_max=0.16,
        downtime_min=0.5, downtime_mode=1.0, downtime_max=1.5,
        ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
        secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
        ransom_cap=20_000_000,
        insurance_weekly_payout=500_000,
        insurance_max_weeks=8,
        insurance_annual_premium=300_000,
        seed=42
    )
    
    # mitigated1_1 = run_fair_simulation(
    #     n_sims=10000,
    #     tef_min=1.0, tef_mode=1.5, tef_max=2.0,
    #     vuln_min=0.144, vuln_mode=0.189, vuln_max=0.234,
    #     downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
    #     ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
    #     secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
    #     ransom_cap=20_000_000,
    #     insurance_weekly_payout=500_000,
    #     insurance_max_weeks=8,
    #     insurance_annual_premium=300_000,
    #     seed=42
    # )
    
    mitigated1_2 = run_fair_simulation(
            n_sims=10000,
            tef_min=1.0, tef_mode=1.5, tef_max=2.0,
            vuln_min=0.1088, vuln_mode=0.1428, vuln_max=0.1768,
            downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
        ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
            secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
            ransom_cap=20_000_000,
            insurance_weekly_payout=500_000,
            insurance_max_weeks=8,
            insurance_annual_premium=300_000,
            seed=42
        )
    
    # mitigated1_3 = run_fair_simulation(
    #         n_sims=10000,
    #         tef_min=1.0, tef_mode=1.5, tef_max=2.0,
    #         vuln_min=0.16, vuln_mode=0.21, vuln_max=0.26,
    #         downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
    #         ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
    #         secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
    #         ransom_cap=20_000_000,
    #         insurance_weekly_payout=500_000,
    #         insurance_max_weeks=8,
    #         insurance_annual_premium=300_000,
    #         seed=42
    #     )
    
    show_results(baseline1, mitigated_total, mitigated1_2, "_var_downtime")
    
    print("\n" + "=" * 80)
    print(" B/ Fixed duration downtime ")
    print("=" * 80)
    
    # ----------------------------------------------------------------------------------
    # A. Baseline Simulation (As-Is: 90% EDR, Manual IR, 0% OT Visibility)
    # ----------------------------------------------------------------------------------
    baseline2 = run_fair_simulation(
        n_sims=10000,
        tef_min=1.0, tef_mode=1.5, tef_max=2.0,
        vuln_min=0.16, vuln_mode=0.21, vuln_max=0.26,
        fixed_downtime=True, downtime_min=1.0, downtime_mode=2.0, downtime_max=4.5,
        ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
        secondary_min=0, secondary_mode=2_000_000, secondary_max=5_000_000,
        ransom_cap=20_000_000,
        insurance_weekly_payout=500_000,
        insurance_max_weeks=8,
        insurance_annual_premium=300_000,
        seed=42
    )
    
    # ----------------------------------------------------------------------------------
    # B. Mitigated Simulation (To-Be: 100% EDR, Automated IR, Supply-Chain OT Visibility)
    # ----------------------------------------------------------------------------------
    # 1. Vuln drops drastically 
    # 2. Downtime duration drops due to automated recovery
    # 3. IR/Forensics cost drops due to automated playbooks
    mitigated_total_2 = run_fair_simulation(
        n_sims=10000,
        tef_min=1.0, tef_mode=1.5, tef_max=2.0,
        vuln_min=0.09, vuln_mode=0.12, vuln_max=0.16,
        fixed_downtime=True, downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
        ir_min=2_000_000, ir_mode=4_500_000, ir_max=7_000_000,
        secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
        ransom_cap=20_000_000,
        insurance_weekly_payout=500_000,
        insurance_max_weeks=8,
        insurance_annual_premium=300_000,
        seed=42
    )
    
    # mitigated2_1 = run_fair_simulation(
    #         n_sims=10000,
    #         tef_min=1.0, tef_mode=1.5, tef_max=2.0,
    #         vuln_min=0.144, vuln_mode=0.189, vuln_max=0.234,
    #         downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
    #         ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
    #         secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
    #         ransom_cap=20_000_000,
    #         insurance_weekly_payout=500_000,
    #         insurance_max_weeks=8,
    #         insurance_annual_premium=300_000,
    #         seed=42
    #     )
        
    mitigated2_2 = run_fair_simulation(
                    n_sims=10000,
                    tef_min=1.0, tef_mode=1.5, tef_max=2.0,
                    vuln_min=0.1088, vuln_mode=0.1428, vuln_max=0.1768,
                    downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
                ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
                    secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
                    ransom_cap=20_000_000,
                    insurance_weekly_payout=500_000,
                    insurance_max_weeks=8,
                    insurance_annual_premium=300_000,
                    seed=42
                )
        
    # mitigated2_3 = run_fair_simulation(
    #                 n_sims=10000,
    #                 tef_min=1.0, tef_mode=1.5, tef_max=2.0,
    #                 vuln_min=0.16, vuln_mode=0.21, vuln_max=0.26,
    #                 downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
    #                 ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
    #                 secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
    #                 ransom_cap=20_000_000,
    #                 insurance_weekly_payout=500_000,
    #                 insurance_max_weeks=8,
    #                 insurance_annual_premium=300_000,
    #                 seed=42
    #             )
    
    show_results(baseline2, mitigated_total_2, mitigated2_2, "_fixed_downtime")
    
    
    return baseline1, mitigated_total, baseline2, mitigated_total_2


if __name__ == "__main__":
    main()
