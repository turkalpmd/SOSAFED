"""
SO-SAFED Statistical Analysis Module

Implements the causal evaluation framework for the Phase 2 prospective
deployment, including:

- Propensity Score Matching (PSM)
- Interrupted Time Series (ITS)
- Heterogeneous Treatment Effects (HTE)
- Spillover / Contamination Analysis

Methods follow gold-standard epidemiological approaches (Bradford Hill criteria)
as described in the companion manuscript.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class BoardingTimeAnalysis:
    """Causal evaluation of AI-optimized staffing on ED boarding time.

    The study uses a split-month quasi-experimental design:
    - Days 1-15: AI-optimized physician staffing (intervention)
    - Days 16-31: Fixed 4-physician staffing (control)
    - Analysis window: 16:00-24:00 shift
    """

    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Patient-level data with columns:
            - arrival: datetime of triage arrival
            - departure: datetime of exam completion
            - boarding_time: minutes (hospital_los - exam_los)
            - triage_risk: 'High Risk' or 'Low Risk'
            - arrival_hour: hour of arrival (16-23)
            - day_of_week: 0=Monday ... 6=Sunday
            - month_num: month integer
            - half: 'Intervention' or 'Control'
        """
        self.data = data.copy()

    # ------------------------------------------------------------------
    # Propensity Score Matching
    # ------------------------------------------------------------------

    def propensity_score_matching(
        self,
        covariates: list[str] = None,
        caliper: Optional[float] = None,
    ) -> dict:
        """1:1 nearest-neighbour PSM to estimate the ATT.

        Parameters
        ----------
        covariates
            Matching variables. Default: arrival_hour, day_of_week, month_num,
            triage_binary.
        caliper
            Maximum distance for matching (None = no caliper).

        Returns
        -------
        dict
            ATT, p-value, matched pair count, balance diagnostics (SMD).
        """
        if covariates is None:
            covariates = ["arrival_hour", "day_of_week", "month_num", "triage_binary"]

        df = self.data.copy()
        df["treatment"] = (df["half"] == "Intervention").astype(int)

        if "triage_binary" not in df.columns and "triage_risk" in df.columns:
            df["triage_binary"] = (df["triage_risk"] == "High Risk").astype(int)

        X = df[covariates].values
        y = df["treatment"].values

        # Propensity score via logistic regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        ps = lr.predict_proba(X)[:, 1]
        df["propensity_score"] = ps

        # 1:1 nearest-neighbour matching
        treated = df[df["treatment"] == 1].copy()
        control = df[df["treatment"] == 0].copy()

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(control[["propensity_score"]].values)
        distances, indices = nn.kneighbors(treated[["propensity_score"]].values)

        if caliper is not None:
            mask = distances.flatten() <= caliper
            treated = treated[mask]
            indices = indices[mask]

        matched_control = control.iloc[indices.flatten()]

        # Balance check: Standardized Mean Difference
        smd_results = {}
        for cov in covariates:
            t_mean = treated[cov].mean()
            c_mean = matched_control[cov].mean()
            t_std = treated[cov].std()
            c_std = matched_control[cov].std()
            pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
            smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
            smd_results[cov] = {"smd_after": round(smd, 4), "balanced": abs(smd) < 0.1}

        # ATT estimation
        att = treated["boarding_time"].mean() - matched_control["boarding_time"].mean()
        u_stat, p_value = stats.mannwhitneyu(
            treated["boarding_time"], matched_control["boarding_time"], alternative="two-sided"
        )

        return {
            "att": round(att, 1),
            "p_value": p_value,
            "n_pairs": len(treated),
            "treated_mean": round(treated["boarding_time"].mean(), 1),
            "control_mean": round(matched_control["boarding_time"].mean(), 1),
            "balance_smd": smd_results,
        }

    # ------------------------------------------------------------------
    # Interrupted Time Series
    # ------------------------------------------------------------------

    def interrupted_time_series(self, outcome_col: str = "boarding_time") -> dict:
        """Segmented regression ITS analysis.

        Model: outcome ~ time + intervention + time_after_intervention

        Returns
        -------
        dict
            Coefficients, p-values, R-squared, and interpretation.
        """
        daily = (
            self.data.groupby(self.data["arrival"].dt.date)
            .agg(
                outcome=(outcome_col, "mean"),
                n_patients=("arrival", "count"),
            )
            .reset_index()
        )
        daily.columns = ["date", "outcome", "n_patients"]
        daily = daily.sort_values("date").reset_index(drop=True)

        daily["time"] = range(len(daily))
        midpoint = len(daily) // 2
        daily["intervention"] = (daily["time"] >= midpoint).astype(int)
        daily["time_after"] = daily["time"] - midpoint
        daily["time_after"] = daily["time_after"].clip(lower=0)

        model = smf.ols("outcome ~ time + intervention + time_after", data=daily).fit()

        return {
            "baseline_trend": {
                "coef": round(model.params["time"], 3),
                "p_value": round(model.pvalues["time"], 4),
            },
            "immediate_effect": {
                "coef": round(model.params["intervention"], 1),
                "p_value": round(model.pvalues["intervention"], 4),
            },
            "trend_change": {
                "coef": round(model.params["time_after"], 3),
                "p_value": round(model.pvalues["time_after"], 4),
            },
            "r_squared": round(model.rsquared, 4),
            "n_observations": len(daily),
        }

    # ------------------------------------------------------------------
    # Heterogeneous Treatment Effects
    # ------------------------------------------------------------------

    def heterogeneous_effects(self) -> dict:
        """Subgroup analysis by triage risk and time-of-day.

        Returns
        -------
        dict
            Treatment effects for each subgroup with p-values.
        """
        results = {}

        # By triage risk
        for risk in ["High Risk", "Low Risk"]:
            sub = self.data[self.data["triage_risk"] == risk]
            interv = sub[sub["half"] == "Intervention"]["boarding_time"]
            ctrl = sub[sub["half"] == "Control"]["boarding_time"]
            diff = interv.mean() - ctrl.mean()
            _, p_val = stats.mannwhitneyu(interv, ctrl, alternative="two-sided")
            results[f"triage_{risk.lower().replace(' ', '_')}"] = {
                "intervention_mean": round(interv.mean(), 1),
                "control_mean": round(ctrl.mean(), 1),
                "difference": round(diff, 1),
                "p_value": p_val,
                "n_intervention": len(interv),
                "n_control": len(ctrl),
            }

        # By time period
        time_groups = {"early_evening_16_19": (16, 19), "late_evening_20_24": (20, 23)}
        for name, (start, end) in time_groups.items():
            sub = self.data[self.data["arrival_hour"].between(start, end)]
            interv = sub[sub["half"] == "Intervention"]["boarding_time"]
            ctrl = sub[sub["half"] == "Control"]["boarding_time"]
            diff = interv.mean() - ctrl.mean()
            _, p_val = stats.mannwhitneyu(interv, ctrl, alternative="two-sided")
            results[name] = {
                "intervention_mean": round(interv.mean(), 1),
                "control_mean": round(ctrl.mean(), 1),
                "difference": round(diff, 1),
                "p_value": p_val,
                "n_intervention": len(interv),
                "n_control": len(ctrl),
            }

        return results

    # ------------------------------------------------------------------
    # Spillover / Contamination Analysis
    # ------------------------------------------------------------------

    def spillover_analysis(self) -> dict:
        """Test whether the control period was contaminated by intervention effects.

        Uses Spearman correlation of control-period boarding time over time.
        A significant negative trend would suggest spillover (staff learned
        from AI period).

        Returns
        -------
        dict
            Spearman r, p-value, and contamination assessment.
        """
        control = self.data[self.data["half"] == "Control"].copy()
        daily_ctrl = (
            control.groupby(control["arrival"].dt.date)["boarding_time"]
            .mean()
            .reset_index()
        )
        daily_ctrl.columns = ["date", "boarding_time"]
        daily_ctrl = daily_ctrl.sort_values("date").reset_index(drop=True)
        daily_ctrl["time_index"] = range(len(daily_ctrl))

        r, p_val = stats.spearmanr(daily_ctrl["time_index"], daily_ctrl["boarding_time"])

        return {
            "spearman_r": round(r, 3),
            "p_value": round(p_val, 3),
            "no_spillover": p_val > 0.05,
            "interpretation": (
                "No significant temporal trend in control period; comparison is clean."
                if p_val > 0.05
                else "Significant trend detected; potential spillover contamination."
            ),
        }

    # ------------------------------------------------------------------
    # Basic comparison
    # ------------------------------------------------------------------

    def basic_comparison(self, outcome_col: str = "boarding_time") -> dict:
        """Simple Mann-Whitney U comparison between intervention and control."""
        interv = self.data[self.data["half"] == "Intervention"][outcome_col]
        ctrl = self.data[self.data["half"] == "Control"][outcome_col]

        u_stat, p_val = stats.mannwhitneyu(interv, ctrl, alternative="two-sided")

        return {
            "intervention_mean": round(interv.mean(), 1),
            "control_mean": round(ctrl.mean(), 1),
            "difference": round(interv.mean() - ctrl.mean(), 1),
            "p_value": p_val,
            "u_statistic": u_stat,
            "n_intervention": len(interv),
            "n_control": len(ctrl),
        }

    def run_all(self) -> dict:
        """Execute all analyses and return consolidated results."""
        return {
            "basic_comparison": self.basic_comparison(),
            "psm": self.propensity_score_matching(),
            "its": self.interrupted_time_series(),
            "heterogeneous_effects": self.heterogeneous_effects(),
            "spillover": self.spillover_analysis(),
        }
