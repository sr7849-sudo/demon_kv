"""
Verified quality-feedback components for RiskKV.

======================================================================
OVERVIEW
======================================================================

This module implements the proposed replacement for the earlier
quality PI/KMeans feedback mechanisms.

The design has four components:

    1. Shared contextual probability recalibration
    2. Budgeted verified auditing
    3. Inverse-propensity-weighted online updates
    4. Explicit expected violation-budget accounting

The important distinction is:

    Hardware feedback:
        operates on directly measurable T/E/P/M.

    Quality feedback:
        operates on sparse VERIFIED outcomes because true quality
        degradation is generally not directly observable at runtime.

======================================================================
QUALITY PIPELINE
======================================================================

Base RiskKV gives:

    p_hat = P(D(r,p) <= delta)

The online calibrator converts this to:

    p_tilde = calibrated probability of safety.

The calibrator sees:

    - base predictor probability;
    - request context length;
    - prefill NLL;
    - last-logit entropy;
    - last-logit margin.

Unlike KMeans, it is trained DIRECTLY against prediction error.

======================================================================
WHY ONE SHARED CALIBRATOR?
======================================================================

Every verified audit updates the same small calibration model.

This avoids the data fragmentation of:

    cluster 0 controller
    cluster 1 controller
    ...
    cluster K controller

while still allowing request-dependent calibration.

======================================================================
AUDITING
======================================================================

Two audit strategies are supported:

    random
        every selected compressed request has equal audit probability.

    uncertainty
        uncertain selected requests are audited more often,
        while retaining a non-zero random floor.

Because uncertainty-directed auditing changes the observation
probability, online calibration uses inverse-propensity weighting:

    weight = 1 / q_t

where q_t is the known audit probability.

Weights are clipped only for numerical variance control.

======================================================================
VIOLATION BUDGET
======================================================================

For horizon H and desired violation rate epsilon:

    B_V = epsilon * H

is the total EXPECTED quality-risk budget.

For each selected configuration:

    predicted_risk = 1 - p_tilde

is subtracted from the remaining budget.

The next request receives:

    epsilon_t =
        remaining violation budget
        --------------------------
        remaining requests

as its current maximum predicted risk allowance.

Thus the controller directly manages the quantity that RiskKV cares
about instead of moving a generic PI threshold.

======================================================================
IMPORTANT
======================================================================

Benchmark experiments obtain verified quality labels from the frozen
LongBench oracle AFTER selection.

In deployment, this feedback would require a trustworthy delayed
signal such as:

    - task success/failure;
    - a task-specific verifier;
    - reference ground truth when available;
    - human/user outcome;
    - another explicitly justified quality-evaluation mechanism.

The benchmark oracle is therefore used to simulate VERIFIED delayed
feedback, not information available before selection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ======================================================================
# NUMERICAL HELPERS
# ======================================================================

def sigmoid(x):
    x = np.clip(
        x,
        -40.0,
        40.0,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def logit(p):
    p = np.clip(
        p,
        1e-4,
        1.0 - 1e-4,
    )

    return np.log(
        p / (1.0 - p)
    )


# ======================================================================
# SHARED CONTEXTUAL CALIBRATOR
# ======================================================================

class ContextualOnlineCalibrator:
    """
    Small shared probability-calibration model.

    ------------------------------------------------------------
    INPUT
    ------------------------------------------------------------

    Original RiskKV probability:

        p_hat

    plus request context:

        log(context length)
        prefill NLL
        entropy
        margin

    We also include interactions between base confidence and context.

    Therefore the calibrator can learn patterns such as:

        "The base model tends to be overconfident at high probability
         for long, high-NLL requests."

    That is fundamentally different from clustering requests according
    to feature-space geometry.

    ------------------------------------------------------------
    MODEL
    ------------------------------------------------------------

    Logistic calibration:

        p_tilde =
            sigmoid(theta^T phi(r, p_hat))

    where phi contains:

        logit(p_hat)
        logit(p_hat)^2
        contextual features
        probability-context interactions.

    ------------------------------------------------------------
    ONLINE UPDATE
    ------------------------------------------------------------

    When an audited request reveals y_safe:

        y_safe = 1  if D <= delta
               = 0  otherwise

    perform a small inverse-propensity-weighted logistic-gradient
    update.

    The offline fitted theta is retained as an anchor to prevent
    uncontrolled drift.
    """

    CONTEXT_COLUMNS = [
        "context_tokens",
        "prefill_nll",
        "last_logit_entropy",
        "last_logit_margin",
    ]

    def __init__(
        self,
        anchor_strength=1e-4,
        max_ipw_weight=100.0,
        max_gradient_norm=10.0,
    ):
        self.anchor_strength = float(
            anchor_strength
        )

        self.max_ipw_weight = float(
            max_ipw_weight
        )

        self.max_gradient_norm = float(
            max_gradient_norm
        )

        self.context_medians = None
        self.feature_scaler = None

        self.theta = None
        self.theta0 = None

        self.feature_names = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _context_dataframe(
        self,
        df,
        fit_medians=False,
    ):
        out = pd.DataFrame(
            index=df.index
        )

        context_tokens = pd.to_numeric(
            df["context_tokens"],
            errors="coerce",
        )

        prefill_nll = pd.to_numeric(
            df["prefill_nll"],
            errors="coerce",
        )

        entropy = pd.to_numeric(
            df["last_logit_entropy"],
            errors="coerce",
        )

        margin = pd.to_numeric(
            df["last_logit_margin"],
            errors="coerce",
        )

        raw = pd.DataFrame(
            {
                "log_context_tokens": (
                    np.log1p(
                        context_tokens
                    )
                ),
                "prefill_nll": (
                    prefill_nll
                ),
                "entropy": (
                    entropy
                ),
                "margin": (
                    margin
                ),
            },
            index=df.index,
        )

        raw = raw.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        if fit_medians:

            self.context_medians = (
                raw.median()
            )

        raw = raw.fillna(
            self.context_medians
        )

        return raw

    def _raw_features(
        self,
        df,
        fit_medians=False,
    ):
        """
        Build features that are directly related to calibration error.

        We intentionally DO NOT include task labels.

        The base predictor has already seen policy identity, so this
        calibration layer does not need a separate policy architecture.
        """

        context = (
            self._context_dataframe(
                df,
                fit_medians=(
                    fit_medians
                ),
            )
        )

        p = pd.to_numeric(
            df["prob_safe_mean"],
            errors="coerce",
        ).fillna(
            0.5
        )

        p = np.clip(
            p.to_numpy(
                dtype=float
            ),
            1e-4,
            1.0 - 1e-4,
        )

        lp = logit(p)

        result = pd.DataFrame(
            index=df.index
        )

        result[
            "base_logit"
        ] = lp

        result[
            "base_logit_sq"
        ] = lp ** 2

        for col in (
            context.columns
        ):

            values = (
                context[col]
                .to_numpy(
                    dtype=float
                )
            )

            result[
                col
            ] = values

            result[
                f"logit_x_{col}"
            ] = (
                lp * values
            )

        return result

    # ------------------------------------------------------------------
    # Offline fit
    # ------------------------------------------------------------------

    def fit(
        self,
        calibration_df,
    ):
        """
        Fit the initial contextual calibration model using SOURCE
        calibration data only.
        """

        X_raw = self._raw_features(
            calibration_df,
            fit_medians=True,
        )

        self.feature_names = list(
            X_raw.columns
        )

        self.feature_scaler = (
            StandardScaler()
        )

        X = (
            self.feature_scaler
            .fit_transform(
                X_raw
            )
        )

        y = (
            calibration_df[
                "safe"
            ]
            .astype(int)
            .to_numpy()
        )

        if len(
            np.unique(y)
        ) < 2:

            raise RuntimeError(
                "Contextual calibrator received "
                "only one safety class."
            )

        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )

        model.fit(
            X,
            y,
        )

        self.theta = np.concatenate(
            [
                model.intercept_,
                model.coef_[0],
            ]
        ).astype(float)

        self.theta0 = (
            self.theta.copy()
        )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _design_matrix(
        self,
        df,
    ):
        X_raw = self._raw_features(
            df,
            fit_medians=False,
        )

        X_raw = X_raw.reindex(
            columns=self.feature_names
        )

        X = (
            self.feature_scaler
            .transform(
                X_raw
            )
        )

        ones = np.ones(
            (
                len(X),
                1,
            )
        )

        return np.concatenate(
            [
                ones,
                X,
            ],
            axis=1,
        )

    def predict_proba(
        self,
        df,
    ):
        Phi = (
            self._design_matrix(
                df
            )
        )

        logits = (
            Phi
            @ self.theta
        )

        return sigmoid(
            logits
        )

    # ------------------------------------------------------------------
    # Online audited update
    # ------------------------------------------------------------------

    def update(
        self,
        row,
        y_safe,
        audit_probability,
        learning_rate,
    ):
        """
        One online verified-feedback update.

        Inverse propensity weighting corrects the fact that uncertain
        requests may be intentionally audited more frequently.

            w = 1 / q

        A finite cap controls extreme gradient variance at very small
        q.

        This is a calibration update, NOT full risk-model retraining.
        """

        if audit_probability <= 0:
            return {
                "ipw_weight": 0.0,
                "gradient_norm": 0.0,
                "preupdate_probability": np.nan,
            }

        df = pd.DataFrame(
            [row]
        )

        Phi = (
            self._design_matrix(
                df
            )[0]
        )

        p = float(
            sigmoid(
                np.dot(
                    Phi,
                    self.theta,
                )
            )
        )

        weight = min(
            1.0
            / max(
                float(
                    audit_probability
                ),
                1e-8,
            ),
            self.max_ipw_weight,
        )

        gradient = (
            weight
            * (
                p
                - float(
                    y_safe
                )
            )
            * Phi
        )

        # Keep the online model weakly anchored to the offline
        # calibration model. Do not regularize the intercept.
        anchor = np.zeros_like(
            self.theta
        )

        anchor[1:] = (
            self.anchor_strength
            * (
                self.theta[1:]
                - self.theta0[1:]
            )
        )

        gradient += anchor

        norm = float(
            np.linalg.norm(
                gradient
            )
        )

        if norm > (
            self.max_gradient_norm
        ):

            gradient *= (
                self.max_gradient_norm
                / norm
            )

        self.theta -= (
            float(
                learning_rate
            )
            * gradient
        )

        return {
            "ipw_weight": (
                weight
            ),
            "gradient_norm": (
                min(
                    norm,
                    self.max_gradient_norm,
                )
            ),
            "preupdate_probability": (
                p
            ),
        }

    def clone(self):
        return copy.deepcopy(
            self
        )


# ======================================================================
# AUDIT POLICY
# ======================================================================

@dataclass
class AuditPolicy:
    """
    Budgeted verified-feedback acquisition.

    random:
        q_t = target audit rate

    uncertainty:
        q_t increases smoothly with ensemble uncertainty while
        retaining a random floor.

    The normalization statistics come from SOURCE calibration data,
    not the held-out deployment workload.
    """

    target_rate: float
    strategy: str

    uncertainty_low: float
    uncertainty_high: float
    mean_priority: float

    @classmethod
    def from_reference(
        cls,
        uncertainties,
        target_rate,
        strategy,
    ):
        values = np.asarray(
            uncertainties,
            dtype=float,
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) == 0:

            values = np.asarray(
                [0.0]
            )

        low = float(
            np.quantile(
                values,
                0.10,
            )
        )

        high = float(
            np.quantile(
                values,
                0.90,
            )
        )

        denom = max(
            high - low,
            1e-8,
        )

        normalized = np.clip(
            (
                values - low
            )
            / denom,
            0.0,
            1.0,
        )

        # Priority range [0.5, 1.5].
        # Therefore every request retains a non-zero random audit floor.
        priority = (
            0.5
            + normalized
        )

        return cls(
            target_rate=float(
                target_rate
            ),
            strategy=str(
                strategy
            ),
            uncertainty_low=low,
            uncertainty_high=high,
            mean_priority=float(
                np.mean(
                    priority
                )
            ),
        )

    def probability(
        self,
        uncertainty,
    ):
        if self.strategy == "random":

            return float(
                np.clip(
                    self.target_rate,
                    0.0,
                    1.0,
                )
            )

        if self.strategy != "uncertainty":

            raise ValueError(
                "Unknown audit strategy: "
                + self.strategy
            )

        u = float(
            uncertainty
        )

        denom = max(
            self.uncertainty_high
            - self.uncertainty_low,
            1e-8,
        )

        u_norm = np.clip(
            (
                u
                - self.uncertainty_low
            )
            / denom,
            0.0,
            1.0,
        )

        priority = (
            0.5
            + u_norm
        )

        q = (
            self.target_rate
            * priority
            / max(
                self.mean_priority,
                1e-8,
            )
        )

        # Maintain a random floor and bound excessive oversampling.
        q_min = (
            0.25
            * self.target_rate
        )

        q_max = min(
            1.0,
            3.0
            * self.target_rate,
        )

        return float(
            np.clip(
                q,
                q_min,
                q_max,
            )
        )


# ======================================================================
# STRONG GLOBAL FEEDBACK BASELINE
# ======================================================================

class GlobalEWMAThresholdController:
    """
    Stronger global baseline than the previous cumulative-mean PI.

    The old controller effectively integrated a statistic that already
    contained the entire history.

    This implementation instead uses an exponentially weighted,
    inverse-propensity-normalized audit estimate.

    It still has the fundamental limitation that ONE scalar threshold
    correction is applied to every future request.

    This is therefore a stronger baseline, not the proposed controller.
    """

    def __init__(
        self,
        epsilon,
        kp,
        ki,
        beta,
        output_min=-0.40,
        output_max=0.40,
    ):
        self.epsilon = float(
            epsilon
        )

        self.kp = float(kp)
        self.ki = float(ki)
        self.beta = float(beta)

        self.output_min = float(
            output_min
        )

        self.output_max = float(
            output_max
        )

        # Self-normalized IPW state.
        self.weighted_violations = (
            self.epsilon
        )

        self.weight_mass = 1.0

        self.integral = 0.0
        self.correction = 0.0
        self.estimated_rate = (
            self.epsilon
        )

    def update(
        self,
        violation,
        audit_probability,
    ):
        """
        Update only when verified feedback actually arrives.
        """

        q = max(
            float(
                audit_probability
            ),
            1e-8,
        )

        w = 1.0 / q

        self.weighted_violations = (
            (
                1.0
                - self.beta
            )
            * self.weighted_violations
            + self.beta
            * w
            * float(
                violation
            )
        )

        self.weight_mass = (
            (
                1.0
                - self.beta
            )
            * self.weight_mass
            + self.beta
            * w
        )

        self.estimated_rate = (
            self.weighted_violations
            / max(
                self.weight_mass,
                1e-8,
            )
        )

        error = (
            self.estimated_rate
            - self.epsilon
        )

        self.integral += error

        self.integral = float(
            np.clip(
                self.integral,
                -5.0,
                5.0,
            )
        )

        raw = (
            self.kp
            * error
            + self.ki
            * self.integral
        )

        self.correction = float(
            np.clip(
                raw,
                self.output_min,
                self.output_max,
            )
        )


# ======================================================================
# EXPLICIT EXPECTED VIOLATION-BUDGET MANAGER
# ======================================================================

class ViolationBudgetManager:
    """
    Explicit long-horizon risk-budget accounting.

    Initial budget:

        B_0 = epsilon * H

    where H is the deployment/control horizon.

    Every selected compressed configuration consumes its calibrated
    expected violation probability:

        r_t = 1 - p_tilde

    Therefore:

        B_{t+1}
            =
        B_t - r_t

    The allowable risk for the next request is:

        epsilon_t
            =
        B_t
        -------------------
        remaining requests

    This is the central replacement for generic PI threshold control.
    """

    def __init__(
        self,
        epsilon,
        horizon,
    ):
        self.epsilon = float(
            epsilon
        )

        self.horizon = int(
            horizon
        )

        self.initial_budget = (
            self.epsilon
            * self.horizon
        )

        self.remaining_budget = (
            self.initial_budget
        )

        self.consumed_expected_risk = (
            0.0
        )

    def allowance(
        self,
        step,
    ):
        remaining_requests = (
            self.horizon
            - int(step)
        )

        if remaining_requests <= 0:
            return 0.0

        if self.remaining_budget <= 0:
            return 0.0

        return float(
            np.clip(
                self.remaining_budget
                / remaining_requests,
                0.0,
                1.0,
            )
        )

    def consume(
        self,
        predicted_risk,
    ):
        r = float(
            np.clip(
                predicted_risk,
                0.0,
                1.0,
            )
        )

        self.consumed_expected_risk += r

        self.remaining_budget = max(
            0.0,
            self.remaining_budget
            - r,
        )


# ======================================================================
# CALIBRATION DIAGNOSTICS
# ======================================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):
    """
    Simple equal-width ECE diagnostic.

    Lower is better.
    """

    y = np.asarray(
        y_true,
        dtype=float,
    )

    p = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(
        n_bins
    ):

        if i == (
            n_bins - 1
        ):

            mask = (
                (p >= edges[i])
                & (p <= edges[i + 1])
            )

        else:

            mask = (
                (p >= edges[i])
                & (p < edges[i + 1])
            )

        if not mask.any():
            continue

        confidence = float(
            p[mask].mean()
        )

        accuracy = float(
            y[mask].mean()
        )

        ece += (
            mask.mean()
            * abs(
                confidence
                - accuracy
            )
        )

    return float(ece)
