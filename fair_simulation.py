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
    
    mask0 = (x <= 1.0)
    loss[mask0] = x[mask0] * 15_000_000
    
    mask1 = (x > 1.0) & (x <= 2.0)
    loss[mask1] = 15_000_000 + (x[mask1] - 1.0) * (30_000_000 - 15_000_000)
    
    mask2 = (x > 2.0) & (x <= 3.0)
    loss[mask2] = 30_000_000 + (x[mask2] - 2.0) * (50_000_000 - 30_000_000)
    
    mask3 = (x > 3.0) & (x <= 4.0)
    loss[mask3] = 50_000_000 + (x[mask3] - 3.0) * (90_000_000 - 50_000_000)
    
    mask4 = (x > 4.0)
    loss[mask4] = 90_000_000 + (x[mask4] - 4.0) * 40_000_000
    
    return loss if isinstance(duration_weeks, np.ndarray) else float(loss[0])


# --------------------------------------------------------------------------------------
# 2. FAIR Model Simulation Engine
# --------------------------------------------------------------------------------------
def run_fair_simulation(
    n_sims=10000,
    # Threat Event Frequency (TEF) - attempts per year
    tef_min=1.0, tef_mode=1.5, tef_max=2.0,
    # Vulnerability (Vuln) - probability attempt breaches defenses
    vuln_min=0.15, vuln_mode=0.35, vuln_max=0.60,
    # Downtime Duration (weeks)
    downtime_min=1.0, downtime_mode=2.0, downtime_max=4.5,
    # Recovery / Incident Response & Forensics Costs (€)
    ir_min=5_000_000, ir_mode=10_000_000, ir_max=15_000_000,
    # Secondary Losses: Legal, GDPR, Refunds (€)
    secondary_min=0, secondary_mode=2_000_000, secondary_max=5_000_000,
    # Ransom Cap and Extortion Behavior
    ransom_cap=20_000_000,
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
    tef_samples = rpert(n_sims, tef_min, tef_mode, tef_max)
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
        vulns = rpert(threats, vuln_min, vuln_mode, vuln_max)
        breaches = np.random.rand(threats) < vulns
        n_breaches = int(np.sum(breaches))
        annual_loss_events[i] = n_breaches
        
        if n_breaches == 0:
            annual_net_loss[i] = insurance_annual_premium
            continue
            
        # Simulate each successful breach / loss event
        downtimes = rpert(n_breaches, downtime_min, downtime_mode, downtime_max)
        bi_losses = calculate_downtime_loss(downtimes)
        ir_losses = rpert(n_breaches, ir_min, ir_mode, ir_max)
        
        # Ransom extortion: triggered with higher likelihood when downtime exceeds 2 weeks
        ransom_losses = np.zeros(n_breaches)
        for j, dt in enumerate(downtimes):
            # If outage extends beyond backup threshold (2 weeks), pressure to pay escalates
            p_pay = 0.25 if dt < 2.0 else (0.65 if dt < 3.0 else 0.90)
            if np.random.rand() < p_pay:
                # Sample ransom demand / settlement up to the €20M willingness cap
                ransom_losses[j] = float(rpert(1, 5_000_000, 12_000_000, ransom_cap)[0])
                
        sec_other_losses = rpert(n_breaches, secondary_min, secondary_mode, secondary_max)
        
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


# --------------------------------------------------------------------------------------
# 3. Main Execution & Comparison (As-Is Baseline vs. To-Be Mitigation)
# --------------------------------------------------------------------------------------
def main():
    BOARD_RISK_APPETITE = 50_000_000  # €50M Board limit
    MITIGATION_INVESTMENT = 2_000_000  # €2M Security Automation & OT Visibility Project
    
    print("=" * 80)
    print(" CARREMAX CYBER RISK QUANTIFICATION (FAIR MONTE CARLO SIMULATION)")
    print("=" * 80)
    print(f"Iterations: 10,000 | Board Risk Appetite: €{BOARD_RISK_APPETITE:,.0f}\n")
    
    # ----------------------------------------------------------------------------------
    # A. Baseline Simulation (As-Is: 90% EDR, Manual IR, 0% OT Visibility)
    # ----------------------------------------------------------------------------------
    baseline = run_fair_simulation(
        n_sims=10000,
        tef_min=1.0, tef_mode=1.5, tef_max=2.0,
        vuln_min=0.15, vuln_mode=0.35, vuln_max=0.60,
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
    # With €2M investment:
    # 1. Vuln drops drastically (mode 15%, max 28%)
    # 2. Downtime duration drops due to automated recovery (mode 1.0 wk, max 2.2 wks)
    # 3. IR/Forensics cost drops due to automated playbooks (€2M - €7M)
    mitigated = run_fair_simulation(
        n_sims=10000,
        tef_min=1.0, tef_mode=1.5, tef_max=2.0,
        vuln_min=0.05, vuln_mode=0.15, vuln_max=0.28,
        downtime_min=0.5, downtime_mode=1.0, downtime_max=2.2,
        ir_min=2_000_000, ir_mode=4_500_000, ir_max=7_000_000,
        secondary_min=0, secondary_mode=1_000_000, secondary_max=3_000_000,
        ransom_cap=20_000_000,
        insurance_weekly_payout=500_000,
        insurance_max_weeks=8,
        insurance_annual_premium=300_000,
        seed=42
    )
    
    # ----------------------------------------------------------------------------------
    # C. Executive Metrics & ROSI Calculation
    # ----------------------------------------------------------------------------------
    risk_reduction_ael = baseline["ael_net"] - mitigated["ael_net"]
    rosi = ((risk_reduction_ael - MITIGATION_INVESTMENT) / MITIGATION_INVESTMENT) * 100.0
    
    print("-" * 80)
    print(" 1. AS-IS BASELINE RISK PROFILE (CURRENT STATE)")
    print("-" * 80)
    print(f" • Annual Probability of at Least 1 Breach: {baseline['prob_at_least_one_breach']:.1%}")
    print(f" • Annualized Expected Loss (Net AEL):       €{baseline['ael_net']:,.0f}")
    print(f" • 90% Value at Risk (90% VaR):             €{baseline['var_90_net']:,.0f}")
    print(f" • 95% Value at Risk (95% VaR):             €{baseline['var_95_net']:,.0f}")
    print(f" • Maximum Simulated Single-Year Loss:      €{baseline['max_loss_net']:,.0f}")
    print(f" • Probability of Exceeding Board Appetite: {baseline['prob_exceed_50m']:.1%}  (Appetite: €50M)")
    print(f" • Average Breach Downtime:                 {baseline['expected_downtime_weeks']:.2f} weeks")
    print(f" • Average Insurance Payout per Breach:     €{baseline['expected_insurance_recovery']:,.0f}")

    print("\n" + "-" * 80)
    print(" 2. TO-BE MITIGATED PROFILE (€2M AUTOMATION + 100% EDR + OT CONTROL)")
    print("-" * 80)
    print(f" • Annual Probability of at Least 1 Breach: {mitigated['prob_at_least_one_breach']:.1%}")
    print(f" • Post-Mitigation Net AEL:                 €{mitigated['ael_net']:,.0f}")
    print(f" • Post-Mitigation 90% VaR:                 €{mitigated['var_90_net']:,.0f}")
    print(f" • Post-Mitigation 95% VaR:                 €{mitigated['var_95_net']:,.0f}")
    print(f" • Probability of Exceeding Board Appetite: {mitigated['prob_exceed_50m']:.1%}  (Near Zero)")
    
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
    sorted_base = np.sort(baseline["annual_net_loss"])
    prob_base = 1.0 - np.arange(1, baseline["n_sims"] + 1) / baseline["n_sims"]
    
    sorted_mit = np.sort(mitigated["annual_net_loss"])
    prob_mit = 1.0 - np.arange(1, mitigated["n_sims"] + 1) / mitigated["n_sims"]
    
    ax1.plot(sorted_base / 1e6, prob_base, label="Baseline (As-Is Current State)", color="#005587", linewidth=2.8)
    ax1.plot(sorted_mit / 1e6, prob_mit, label="Mitigated (To-Be €2M Roadmap)", color="#86bc25", linewidth=2.8)
    
    # Reference vertical lines
    ax1.axvline(x=BOARD_RISK_APPETITE / 1e6, color="#e3000f", linestyle="--", linewidth=2.2,
                label=f"Board Risk Appetite (€{BOARD_RISK_APPETITE/1e6:.0f}M)")
    ax1.axvline(x=baseline["var_90_net"] / 1e6, color="#005587", linestyle=":", linewidth=1.8,
                label=f"Baseline 90% VaR (€{baseline['var_90_net']/1e6:.1f}M)")
    ax1.axvline(x=mitigated["var_90_net"] / 1e6, color="#86bc25", linestyle=":", linewidth=1.8,
                label=f"Mitigated 90% VaR (€{mitigated['var_90_net']/1e6:.1f}M)")

    ax1.set_title("Loss Exceedance Curve (LEC) - As-Is vs. To-Be", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Annual Net Financial Loss (€ Millions)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability of Exceedance", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 110)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)
    
    # --- Plot 2: Average Loss Component Breakdown (When Breach Occurs) ---
    categories = ["Business\nInterruption", "IR & Forensics\nRecovery", "Extortion\n(Ransom)", "Secondary\n(GDPR/Refunds)"]
    breach_filter = baseline["annual_loss_events"] > 0
    mean_bi = np.mean(baseline["annual_bi_loss"][breach_filter]) / 1e6
    mean_ir = np.mean(baseline["annual_ir_loss"][breach_filter]) / 1e6
    mean_ransom = np.mean(baseline["annual_ransom_loss"][breach_filter]) / 1e6
    mean_sec = np.mean(baseline["annual_sec_loss"][breach_filter]) / 1e6
    
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
    mean_ins = np.mean(baseline["annual_insurance_payout"][breach_filter]) / 1e6
    ax2.text(0.95, 0.90, f"Cyber Insurance Offset: -€{mean_ins:.2f}M / event\n(Capped at €4.0M)",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3f7", edgecolor="#005587", alpha=0.9))

    plt.tight_layout()
    plt.savefig("lec_curve_output.png", dpi=160)
    print("\n[SUCCESS] Charts saved to 'lec_curve_output.png'")
    
    return baseline, mitigated


if __name__ == "__main__":
    main()
