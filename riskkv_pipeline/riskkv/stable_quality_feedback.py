"""
Stable verified quality feedback for RiskKV.

This replaces the high-variance DR budget controller used in 09B.

Core principle
--------------
Every request initially consumes its predicted violation risk:

    consumed += r_hat

If that request is later audited, replace the prediction by the
verified outcome:

    consumed += z - r_hat

Therefore:

    audited request   -> contributes exactly z
    unaudited request -> contributes predicted risk

Verified residuals are additionally used to correct FUTURE predicted
risks. They are never retroactively multiplied over previous requests.

Random auditing only is used in this experiment. Because every selected
compressed request has the same audit probability, no IPW correction is
needed for the residual mean.
"""

from __future__ import annotations

import numpy as np


class VerifiedReplacementBudget:
    """
    Stable violation-budget accounting.

    Budget:
        epsilon * horizon

    Audited requests contribute their actual Bernoulli outcome.
    Unaudited requests contribute their risk prediction.
    """

    def __init__(
        self,
        epsilon: float,
        horizon: int,
    ):
        self.epsilon = float(epsilon)
        self.horizon = int(horizon)

        self.initial_budget = (
            self.epsilon
            * self.horizon
        )

        self.consumed = 0.0
        self.steps = 0

        self.audit_count = 0
        self.audit_residual_sum = 0.0

    def allowance(
        self,
        step: int,
    ):
        remaining_requests = (
            self.horizon
            - int(step)
        )

        if remaining_requests <= 0:
            return 0.0

        remaining_budget = max(
            0.0,
            self.initial_budget
            - self.consumed,
        )

        return float(
            np.clip(
                remaining_budget
                / remaining_requests,
                0.0,
                1.0,
            )
        )

    def consume_prediction(
        self,
        predicted_risk: float,
    ):
        r = float(
            np.clip(
                predicted_risk,
                0.0,
                1.0,
            )
        )

        self.consumed += r
        self.steps += 1

    def replace_with_verified(
        self,
        predicted_risk: float,
        violation: bool,
    ):
        """
        Prediction for this request has already been consumed.

        Replace:
            predicted risk
        with:
            actual 0/1 outcome.
        """

        r = float(
            np.clip(
                predicted_risk,
                0.0,
                1.0,
            )
        )

        z = float(
            bool(violation)
        )

        residual = (
            z - r
        )

        self.consumed += residual

        self.audit_count += 1
        self.audit_residual_sum += residual

    @property
    def remaining_budget(self):
        return float(
            max(
                0.0,
                self.initial_budget
                - self.consumed,
            )
        )

    @property
    def estimated_violation_count(self):
        return float(
            max(
                0.0,
                self.consumed,
            )
        )

    @property
    def mean_audit_residual(self):
        if self.audit_count == 0:
            return 0.0

        return float(
            self.audit_residual_sum
            / self.audit_count
        )


class OnlineResidualBias:
    """
    Corrects FUTURE risk estimates using verified calibration residuals.

        residual = actual_violation - predicted_risk

    Positive bias:
        predictor has recently been too optimistic.

    Negative bias:
        predictor has recently been too conservative.

    This is a calibration-drift estimate, not a PI controller.
    """

    def __init__(
        self,
        beta: float = 0.10,
        min_audits: int = 3,
        max_abs_bias: float = 0.20,
    ):
        self.beta = float(beta)
        self.min_audits = int(
            min_audits
        )

        self.max_abs_bias = float(
            max_abs_bias
        )

        self.ewma = 0.0
        self.audit_count = 0

    def update(
        self,
        predicted_risk: float,
        violation: bool,
    ):
        r = float(
            np.clip(
                predicted_risk,
                0.0,
                1.0,
            )
        )

        z = float(
            bool(violation)
        )

        residual = (
            z - r
        )

        if self.audit_count == 0:
            self.ewma = residual

        else:
            self.ewma = (
                (
                    1.0
                    - self.beta
                )
                * self.ewma
                +
                self.beta
                * residual
            )

        self.audit_count += 1

    @property
    def bias(self):
        if (
            self.audit_count
            < self.min_audits
        ):
            return 0.0

        return float(
            np.clip(
                self.ewma,
                -self.max_abs_bias,
                self.max_abs_bias,
            )
        )

    def correct(
        self,
        risk,
    ):
        return np.clip(
            np.asarray(
                risk,
                dtype=float,
            )
            + self.bias,
            0.0,
            1.0,
        )
