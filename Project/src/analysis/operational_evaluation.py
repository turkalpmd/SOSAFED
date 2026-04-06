"""
SO-SAFED Operational Evaluation Module

Evaluates the operational impact of AI-optimized staffing through:
- Census-Staffing alignment (Mean Squared Difference)
- Quality vs Efficiency tradeoff curves
- Staffing deviation / intervention fidelity analysis
- Covariate-controlled regression

Methodology adapted from Hu et al. "Implementing a prediction-driven framework
for emergency department nurse staffing to optimize real-time decisions"
(npj Health Systems, 2025. doi:10.1038/s44401-025-00019-2).
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

matplotlib.use("Agg")
plt.style.use("ggplot")


class OperationalEvaluation:
    """Evaluates census-staffing alignment and operational efficiency."""

    IDEAL_RATIO = 16  # target patients per physician
    HOURLY_RATE = 50  # USD per physician-hour (for visualization scaling)

    def __init__(
        self,
        patient_data: pd.DataFrame,
        shifts_data: pd.DataFrame,
        output_dir: str = "./figures",
        analysis_hours: tuple[int, int] = (16, 23),
    ):
        """
        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient-level data with: arrival (datetime), departure (datetime),
            boarding_time, exam_los, triage_risk.
        shifts_data : pd.DataFrame
            Daily shift data with: date (datetime), doctor (int).
        output_dir
            Directory for saving figures.
        analysis_hours
            Inclusive hour range for the analysis window.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.patients = self._prepare_patients(patient_data, analysis_hours)
        self.shifts = shifts_data.copy()
        if not self.shifts.empty:
            self.shifts["date"] = pd.to_datetime(self.shifts["date"]).dt.normalize()

        self.daily = self._compute_daily_stats()
        self.census_hourly = self._compute_hourly_census()

    @staticmethod
    def _prepare_patients(
        df: pd.DataFrame, hours: tuple[int, int]
    ) -> pd.DataFrame:
        df = df.copy()
        df["arrival"] = pd.to_datetime(df["arrival"])
        df["departure"] = pd.to_datetime(df["departure"])
        df = df[df["arrival"].dt.hour.between(hours[0], hours[1])].copy()
        df["date"] = df["arrival"].dt.date
        df["day_of_month"] = df["arrival"].dt.day
        df["half"] = df["day_of_month"].apply(
            lambda x: "Intervention" if x <= 15 else "Control"
        )
        df["month"] = df["arrival"].dt.month
        df["day_of_week"] = df["arrival"].dt.dayofweek
        return df

    def _compute_daily_stats(self) -> pd.DataFrame:
        daily = (
            self.patients.groupby("date")
            .agg(
                avg_boarding=("boarding_time", "mean"),
                avg_exam_los=("exam_los", "mean"),
                n_patients=("arrival", "count"),
                high_risk_pct=(
                    "triage_risk",
                    lambda x: (x == "High Risk").sum() / len(x) * 100,
                ),
            )
            .reset_index()
        )
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        daily["day_of_month"] = daily["date"].dt.day
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month
        daily["half"] = daily["day_of_month"].apply(
            lambda x: "Intervention" if x <= 15 else "Control"
        )
        daily = daily.merge(
            self.shifts[["date", "doctor"]], on="date", how="left"
        )
        daily["doctor"] = daily["doctor"].fillna(4.0)
        daily["physician_cost"] = daily["doctor"] * 8 * self.HOURLY_RATE
        return daily

    def _compute_hourly_census(self) -> pd.DataFrame:
        data = self.patients.copy()
        data["hour"] = data["arrival"].dt.floor("h")
        census_list = []
        for hour_start in data["hour"].unique():
            in_sys = (
                (data["arrival"] <= hour_start) & (data["departure"] > hour_start)
            ).sum()
            census_list.append({"hour": hour_start, "census": in_sys})
        census = pd.DataFrame(census_list)
        census["date"] = pd.to_datetime(census["hour"]).dt.normalize()
        census["day_of_month"] = census["hour"].dt.day
        census["half"] = census["day_of_month"].apply(
            lambda x: "Intervention" if x <= 15 else "Control"
        )
        census = census.merge(
            self.daily[["date", "doctor"]].drop_duplicates(), on="date", how="left"
        )
        census["doctor"] = census["doctor"].fillna(4.0)
        census["scaled_staffing"] = census["doctor"] * self.IDEAL_RATIO
        census["squared_diff"] = (census["census"] - census["scaled_staffing"]) ** 2
        return census

    # ------------------------------------------------------------------
    # MSD (Mean Squared Difference)
    # ------------------------------------------------------------------

    def compute_msd(self) -> dict:
        """Census-staffing alignment metric."""
        msd_i = self.census_hourly[self.census_hourly["half"] == "Intervention"][
            "squared_diff"
        ].mean()
        msd_c = self.census_hourly[self.census_hourly["half"] == "Control"][
            "squared_diff"
        ].mean()
        reduction = (msd_c - msd_i) / msd_c * 100 if msd_c > 0 else 0
        return {
            "msd_intervention": round(msd_i, 2),
            "msd_control": round(msd_c, 2),
            "reduction_pct": round(reduction, 1),
        }

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def fig_census_staffing(self) -> str:
        """Census-Staffing alignment time series."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, half, title in [
            (axes[0], "Control", "Control (Days 16-31, Fixed 4 Physicians)"),
            (axes[1], "Intervention", "Intervention (Days 1-15, AI-Optimised)"),
        ]:
            sub = self.census_hourly[self.census_hourly["half"] == half].sort_values("hour")
            sub["census_smooth"] = sub["census"].rolling(24, min_periods=1).mean()
            sub["staff_smooth"] = sub["scaled_staffing"].rolling(24, min_periods=1).mean()
            ax.plot(sub["hour"], sub["census_smooth"], color="#1f78b4", lw=2, label="Census (smoothed)")
            ax.plot(sub["hour"], sub["staff_smooth"], color="#e31a1c", lw=2, ls="--", label="Staffing capacity")
            ax.set_xlabel("Date")
            ax.set_ylabel("Patients / Capacity")
            msd_val = self.compute_msd()[f"msd_{'control' if half == 'Control' else 'intervention'}"]
            ax.set_title(f"{title}\nMSD = {msd_val:.2f}", fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)
        plt.suptitle("Census-Staffing Alignment (16:00-24:00)", fontweight="bold", y=1.02)
        plt.tight_layout()
        path = str(self.output_dir / "fig_census_staffing.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path

    def fig_tradeoff(self) -> str:
        """Quality vs Efficiency tradeoff curves."""
        interv = self.daily[self.daily["half"] == "Intervention"]
        ctrl = self.daily[self.daily["half"] == "Control"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, x_col, x_label in [
            (axes[0], "avg_boarding", "Daily Mean Boarding Time (min)"),
            (axes[1], "avg_exam_los", "Daily Mean Exam LOS (min)"),
        ]:
            ax.scatter(ctrl[x_col], ctrl["physician_cost"], c="#e31a1c", s=60, alpha=0.7, label="Control", edgecolors="white", lw=0.5)
            ax.scatter(interv[x_col], interv["physician_cost"], c="#1f78b4", s=60, alpha=0.7, label="Intervention", edgecolors="white", lw=0.5)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Daily Physician Cost (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle("Tradeoff Curves: Quality vs Efficiency", fontweight="bold", y=1.02)
        plt.tight_layout()
        path = str(self.output_dir / "fig_tradeoff.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path

    def fig_deviation(self) -> str:
        """Staffing deviation analysis (intervention fidelity)."""
        daily = self.daily.copy()
        daily["recommended"] = np.ceil(daily["n_patients"] / self.IDEAL_RATIO).clip(3, 6)
        daily["deviation"] = np.where(daily["half"] == "Intervention", 0, 4 - daily["recommended"])

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        interv = daily[daily["half"] == "Intervention"]
        ctrl = daily[daily["half"] == "Control"]

        axes[0].scatter(interv["date"], interv["deviation"], c="#1f78b4", s=40, alpha=0.8, label="Intervention (deviation=0)")
        axes[0].scatter(ctrl["date"], ctrl["deviation"], c="#e31a1c", s=40, alpha=0.8, label="Control (4 - recommended)")
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_ylabel("Deviation (Actual - Recommended)")
        axes[0].set_title("Staffing Deviation Over Time", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(ctrl["deviation"], bins=15, color="#e31a1c", alpha=0.7, edgecolor="white")
        axes[1].axvline(ctrl["deviation"].mean(), color="black", ls="--", lw=2, label=f'Mean = {ctrl["deviation"].mean():.2f}')
        axes[1].set_xlabel("Deviation (Control Period)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Staffing Mismatch (Control)", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Deviation Analysis: Intervention Fidelity", fontweight="bold", y=1.02)
        plt.tight_layout()
        path = str(self.output_dir / "fig_deviation.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path

    # ------------------------------------------------------------------
    # Covariate-Controlled Regression
    # ------------------------------------------------------------------

    def covariate_regression(self) -> pd.DataFrame:
        """OLS regression: avg_boarding ~ intervention + covariates."""
        reg_df = self.daily.copy()
        reg_df["intervention"] = (reg_df["half"] == "Intervention").astype(int)
        for i in range(6):
            reg_df[f"dow_{i}"] = (reg_df["day_of_week"] == i).astype(int)

        formula = (
            "avg_boarding ~ intervention + doctor + n_patients + high_risk_pct"
            " + dow_0 + dow_1 + dow_2 + dow_3 + dow_4 + dow_5"
        )
        model = smf.ols(formula, data=reg_df).fit()

        rows = []
        for name in model.params.index:
            pval = model.pvalues[name]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            rows.append({
                "Covariate": name,
                "Coefficient": f"{model.params[name]:.3f}",
                "SE": f"{model.bse[name]:.3f}",
                "P-value": f"{pval:.4f}{sig}",
            })

        self._regression_model = model
        return pd.DataFrame(rows)

    def run_all(self, save_figures: bool = True) -> dict:
        """Execute all operational evaluation analyses."""
        results = {"msd": self.compute_msd()}

        if save_figures:
            results["fig_census_staffing"] = self.fig_census_staffing()
            results["fig_tradeoff"] = self.fig_tradeoff()
            results["fig_deviation"] = self.fig_deviation()

        results["regression_table"] = self.covariate_regression()
        results["r_squared"] = round(self._regression_model.rsquared, 3)
        results["adj_r_squared"] = round(self._regression_model.rsquared_adj, 3)

        return results
