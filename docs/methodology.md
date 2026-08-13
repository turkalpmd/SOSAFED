# Statistical Methodology: Phase 2 Causal Evaluation

This document describes the statistical methods used to evaluate the real-world impact of SO-SAFED's AI-optimized physician staffing during the Phase 2 prospective deployment (December 2024 -- May 2025).

## Study Design

**Quasi-experimental split-month design:**
- **Intervention period** (days 1--15 of each month): AI-optimized physician allocation based on TiDE-RIN forecasts. The number of physicians (3--6) was set according to predicted patient volume.
- **Control period** (days 16--31): Fixed 4-physician staffing (historical standard).
- **Analysis window:** 16:00--24:00 shift (peak demand period).
- **Duration:** 6 months (December 2024 -- May 2025), yielding 1,456 analysis hours.

## 0. Cohort and Negative-Boarding Cleaning

A 31% subset of timestamp-complete after-hours visits (4,293 of 13,919) carried a **non-physical negative boarding time** (disposition timestamp earlier than examination-end timestamp), a timestamp-entry error that inflated boarding-derived effect estimates in earlier analyses. All boarding-derived results in Sections 1--3 below use the **cleaned cohort (n = 9,626; 4,711 intervention / 4,915 control)**. Demographic characteristics and diagnostic test-utilization rates (not boarding-derived) use the full timestamp-complete cohort (n = 13,935).

## 1. Propensity Score Matching (PSM)

### Rationale
Patient volume was not significantly different between intervention and control days (p = 0.49), but observed confounders (arrival hour, day of week, month, acuity, age, sex, daily volume) could still bias a naive comparison. PSM eliminates this concern by matching patients on observed confounders. Two specifications are reported.

### Method
1. **Propensity score estimation:** Logistic regression predicting assignment to intervention vs. control.
   - **Fully adjusted (primary, requested during peer review):** age, sex, daily after-hours volume, arrival hour, day of week, calendar month, full 5-level CTAS (T1--T5).
   - **Parsimonious (sensitivity):** arrival hour, day of week, month, binary triage (high-risk T1--T3 vs. low-risk T4--T5).
2. **Matching:** 1:1 nearest-neighbour matching on the logit propensity score (fully adjusted model: caliper 0.2 SD).
3. **Balance verification:** Standardized mean difference (SMD) < 0.1 for all covariates post-matching; residual imbalance on daily volume resolved via stabilized IPTW.
4. **Effect estimation:** Average Treatment Effect on the Treated (ATT) = mean/median boarding time difference between matched intervention and control patients.
5. **Inference:** Mann-Whitney U test on matched pairs; E-value (VanderWeele-Ding) for unmeasured confounding.

### Key Result
- **Fully adjusted PSM (primary): 4,708 matched pairs**, ATT = **-7.9 min mean / -10.0 min median** (p = 0.003), corroborated by stabilized IPTW (-8.5 min, worst weighted SMD 0.01). E-value 1.28 (mean) / 1.32 (median).
- **Parsimonious PSM (sensitivity): 4,711 matched pairs**, ATT = -18.4 min mean / -21.1 min median (p < 0.001).
- All covariates balanced post-matching (|SMD| < 0.10), except daily volume under the fully adjusted NN match (-0.111), which stabilized IPTW resolves (weighted SMD -0.002).
- **Note:** The previously reported ATT of -31.9 minutes was computed before the negative-boarding cleaning and is an **artefact of the 31% timestamp-error records** (uncleaned median pair difference -71.5 min, clinically implausible). It has been retired in favor of the fully adjusted estimate above.

## 2. Interrupted Time Series (ITS)

### Rationale
ITS is the gold standard for evaluating the temporal impact of health interventions in quasi-experimental designs (widely used in JAMA, BMJ, Lancet).

### Method
Segmented regression model:

```
boarding_time = beta_0 + beta_1(time) + beta_2(intervention) + beta_3(time_after_intervention) + epsilon
```

Where:
- `beta_1`: pre-intervention temporal trend
- `beta_2`: immediate level change at intervention onset (step effect)
- `beta_3`: change in slope post-intervention (sustained effect)

### Key Result (cleaned cohort)
| Parameter | Coefficient | P-value | Interpretation |
|-----------|-------------|---------|----------------|
| Baseline trend | -0.14 | 0.025 | Pre-intervention boarding had a slight downward trend |
| **Immediate effect** | **-24.2 min** | **0.011** | AI caused an instant ~24-minute drop |
| Trend change | +0.19 | 0.039 | Attenuation over time |

- ITS is the most robust estimator to the negative-boarding cleaning (-22.1 min uncleaned to -24.2 min cleaned), because it aggregates boarding to hourly means, within which the timestamp-error records are partially averaged out.
- Immediate effect supports temporal causality per Bradford Hill criteria.

## 3. Heterogeneous Treatment Effects (HTE)

### Rationale
Not all patients benefit equally. Subgroup analysis identifies which populations gain the most, informing targeted optimization strategies.

### Method
Stratified Mann-Whitney U tests comparing intervention vs. control within each subgroup (cleaned cohort).

### Results by Triage Risk

| Subgroup | Intervention | Control | Difference | P-value |
|----------|-------------|---------|------------|---------|
| **Low Risk (T4--T5)** | 170.7 min | 179.8 min | **-9.2 min** | **0.010** |
| High Risk (T1--T3) | 211.5 min | 214.0 min | -2.5 min | 0.73 (NS) |

**Mechanism:** Low-risk patients are most sensitive to capacity expansion because they queue behind high-risk patients under fixed staffing. Additional physicians absorb the low-risk backlog.

### Results by Time of Day

| Period | Difference | P-value |
|--------|------------|---------|
| **16:00--19:00 (early evening)** | **-14.7 min** | **0.004** |
| 20:00--24:00 (late evening) | -3.1 min | 0.41 (NS) |

**Mechanism:** Early evening coincides with the primary ED surge (post-work/school arrivals). AI staffing anticipates this surge; fixed staffing cannot. The late-evening subgroup was significant in the uncleaned analysis (-6.3 min, p = 0.012) but loses significance after the negative-boarding cleaning; direction is preserved but the finding should be reported as non-significant, likely underpowered.

## 4. Spillover / Contamination Analysis

### Rationale
If staff in the control period learned from the intervention period (e.g., adopted workflow changes), the control would not be a true counterfactual.

### Method
Spearman rank correlation between time index and mean boarding time within the control period only, computed on the **daily** control-period series (canonical, n = 92 days) for consistency with the power analysis. A significant negative trend would suggest contamination.

### Result
- **Daily series: Spearman r = -0.097, p = 0.36** (n = 92 days); the original hourly computation gave r = -0.049, p = 0.185 (n = 734 hours).
- No significant temporal trend in the control period.
- **Power:** The daily test (80% power, two-sided alpha = 0.05) was powered to detect a monotonic trend of |r| >= 0.29 (~0.56 min/day, ~52 min over the 92-day window); the observed trend is well below this threshold.
- **Conclusion:** The control period remained stable; comparison is clean and unbiased.

## 5. Patient-Safety Screen

### Rationale
Demand-responsive staffing sometimes assigns fewer than the standard four physicians; a safety screen checks this did not come at the cost of early returns or unsafe discharges.

### Method
Descriptive comparison on the full after-hours index-visit registry (N = 15,380; every triaged visit is at risk of an early return regardless of whether its boarding timestamps were analyzable).

### Result
| Endpoint | Intervention | Control | P-value |
|----------|---------------|---------|---------|
| 72-hour ED return | 9.92% | 10.02% | 0.86 |
| 7-day ED return | 14.33% | 14.33% | 1.00 |
| Discharge against advice / unauthorized exit | 0.039% (n=3) | 0.104% (n=8) | 0.23 |

No signal that operational gains came at the expense of short-term patient safety. In-ED mortality and documented adverse events are not available as structured fields in the operational extract and will be captured prospectively via the institutional mortality registry and incident-reporting system in a planned multi-center extension.

## 6. Census-Staffing Alignment (npj-Style Analysis)

Following the methodology of Hu et al. (*npj Health Systems*, 2025), we computed the Mean Squared Difference (MSD) between hourly patient census and staffing capacity (physicians x ideal ratio of 16 patients/physician).

Lower MSD indicates tighter alignment between workload and capacity.

### Additional Analyses
- **Tradeoff curves:** Daily boarding time vs. physician cost scatter plots, showing intervention points cluster at lower boarding time with comparable cost.
- **Covariate-controlled regression:** OLS regression of daily boarding time on intervention indicator, physician count, patient volume, acuity mix, and day-of-week dummies.
- **Deviation analysis:** Intervention period achieved 100% staffing fidelity (schedules delivered 5 days in advance); control period showed systematic demand-capacity mismatch.

## 7. Convergent Validity

Multiple independent methods converge on the same conclusion (cleaned cohort):

| Method | Effect Size | P-value | What It Measures |
|--------|------------|---------|------------------|
| Unadjusted (Mann-Whitney) | -8.2 min | 0.012 | Simple group difference |
| **PSM, fully adjusted (primary)** | **-7.9 / -10.0 min** | **0.003** | Confounding-adjusted effect |
| PSM, parsimonious (sensitivity) | -18.4 / -21.1 min | < 0.001 | Confounding-adjusted effect, fewer covariates |
| ITS (temporal, sensitivity) | -24.2 min | 0.011 | Immediate causal impact |

All methods show a **statistically significant reduction in boarding time** during AI-optimized periods, providing robust evidence for the causal efficacy of demand-responsive staffing. The fully adjusted PSM estimate (~8--10 minutes) is recommended as the primary confounding-adjusted effect; the earlier -31.9 min figure was an artefact of uncleaned negative-boarding timestamp errors (Section 1) and has been retired.

## References

1. Akbasli IT, Birbilen AZ, Teksam O. Artificial intelligence-driven forecasting and shift optimization for pediatric emergency department crowding. *JAMIA Open*. 2025;8(2):ooae138. doi:10.1093/jamiaopen/ooae138
2. Hu et al. Implementing a prediction-driven framework for emergency department nurse staffing. *npj Health Systems*. 2025. doi:10.1038/s44401-025-00019-2
3. Das A et al. Long-term Forecasting with TiDE: Time-series Dense Encoder. arXiv:2304.08424, 2023.
4. Kim T et al. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022.
