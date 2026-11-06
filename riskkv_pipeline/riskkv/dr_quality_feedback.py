"""
Corrected verified quality feedback for RiskKV.

======================================================================
PURPOSE
======================================================================

The previous "calibrated budget" controller accounted for:

    predicted_risk = 1 - calibrated_P_safe

but DID NOT directly feed verified runtime violations into the budget.

Therefore under distribution shift the controller could believe that it
had consumed little risk even while actual violations were large.

This module fixes that.

======================================================================
CORE IDEA
======================================================================

For request t:

    r_hat_t = predicted violation probability

If the request is audited:

    z_t = 1   if D(r_t,p_t) > delta
          0   otherwise

and q_t is the probability with which that request was audited.

The canonical model + inverse-propensity residual correction is:

    r_hat_t + A_t/q_t * (z_t - r_hat_t)

where:

    A_t = 1 if audited, else 0.

The raw cumulative form is unbiased under the usual propensity
assumptions but can have very high variance when q_t is small.

For runtime CONTROL we therefore additionally maintain a stabilized,
self-normalized residual estimate:

                          sum w_i (z_i - r_hat_i)
    residual_bias = --------------------------------
                              sum w_i

where:

    w_i = 1 / q_i.

Then:

    estimated cumulative violations
        =
    cumulative predicted risk
        +
    number of executed requests * residual_bias.

The raw Horvitz-Thompson/AIPW-style estimate is still recorded for
diagnostics.

======================================================================
SECOND COMPONENT
======================================================================

The optional OnlineResidualRiskCorrector directly corrects future
predicted risks:

    r_corrected
        =
    clip(r_hat + residual_bias, 0, 1).

Unlike the old PI controller, this state has a direct statistical
meaning:

    "How much has actual violation probability recently differed from
     predicted violation probability?"

It is NOT an integral controller over a cumulative violation rate.
"""

from __future__ import annotations

import numpy as np


# ======================================================================
# VERIFIED FEEDBACK-CORRECTED VIOLATION BUDGET
# ======================================================================

class VerifiedDRBudgetManager:
    """
    Stabilized feedback-corrected violation-budget manager.

    Initial long-horizon budget:

        B = epsilon * H

    The budget state combines:

        model-predicted risk
            +
        verified audit residuals.

    No quality label from request t is visible until AFTER the policy
    for request t has already been selected and executed.
    """

    def __init__(
        self,
        epsilon: float,
        horizon: int,
        max_ipw_weight: float = 100.0,
        max_abs_residual_bias: float = 0.50,
    ):
        self.epsilon = float(epsilon)
        self.horizon = int(horizon)

        self.initial_budget = (
            self.epsilon
            * self.horizon
        )

        self.max_ipw_weight = float(
            max_ipw_weight
        )

        self.max_abs_residual_bias = float(
            max_abs_residual_bias
        )

        # Sum of model-estimated risk for all executed requests.
        self.predicted_risk_sum = 0.0

        # Audit residual statistics.
        self.weighted_residual_sum = 0.0
        self.weight_mass = 0.0

        # Number of requests whose risk has been consumed.
        self.steps_consumed = 0

        self.audit_count = 0

    # ------------------------------------------------------------------
    # Audited residual
    # ------------------------------------------------------------------

    def update_audit(
        self,
        predicted_risk: float,
        violation: bool,
        audit_probability: float,
    ):
        """
        Feed one VERIFIED outcome into the budget state.

        residual:
            actual Bernoulli violation
                -
            predicted violation probability

        Positive:
            model was too optimistic.

        Negative:
            model was too conservative.
        """

        q = max(
            float(audit_probability),
            1e-8,
        )

        w = min(
            1.0 / q,
            self.max_ipw_weight,
        )

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

        self.weighted_residual_sum += (
            w * residual
        )

        self.weight_mass += w

        self.audit_count += 1

    # ------------------------------------------------------------------
    # Risk consumed by current request
    # ------------------------------------------------------------------

    def consume_prediction(
        self,
        predicted_risk: float,
    ):
        """
        Every executed request first consumes its current model-estimated
        risk.

        Verified audits subsequently correct that estimate through the
        residual state.
        """

        r = float(
            np.clip(
                predicted_risk,
                0.0,
                1.0,
            )
        )

        self.predicted_risk_sum += r

        self.steps_consumed += 1

    # ------------------------------------------------------------------
    # Stabilized residual estimate
    # ------------------------------------------------------------------

    @property
    def residual_bias(self):
        if self.weight_mass <= 0:
            return 0.0

        bias = (
            self.weighted_residual_sum
            / self.weight_mass
        )

        return float(
            np.clip(
                bias,
                -self.max_abs_residual_bias,
                self.max_abs_residual_bias,
            )
        )

    # ------------------------------------------------------------------
    # Raw DR diagnostic
    # ------------------------------------------------------------------

    @property
    def raw_dr_estimate(self):
        """
        Raw cumulative model + IPW residual estimate.

        This is useful diagnostically but can be extremely noisy at
        1% auditing, so the runtime controller uses the stabilized
        self-normalized estimate below.
        """

        estimate = (
            self.predicted_risk_sum
            + self.weighted_residual_sum
        )

        return float(
            np.clip(
                estimate,
                0.0,
                max(
                    self.steps_consumed,
                    0,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Stabilized cumulative violation estimate
    # ------------------------------------------------------------------

    @property
    def estimated_violation_count(self):
        """
        Cumulative predicted risk corrected by the observed weighted
        residual bias.

        If verified outcomes show that actual violations have been
        systematically larger than predicted risk, this estimate rises
        and the remaining risk allowance contracts.
        """

        estimate = (
            self.predicted_risk_sum
            + self.steps_consumed
            * self.residual_bias
        )

        return float(
            np.clip(
                estimate,
                0.0,
                max(
                    self.steps_consumed,
                    0,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Remaining risk budget
    # ------------------------------------------------------------------

    @property
    def remaining_budget(self):
        return float(
            max(
                0.0,
                self.initial_budget
                - self.estimated_violation_count,
            )
        )

    def allowance(
        self,
        step: int,
    ):
        """
        Risk allowance for the NEXT policy decision:

                  remaining estimated violation budget
            -----------------------------------------------
                      remaining request count
        """

        remaining_requests = (
            self.horizon
            - int(step)
        )

        if remaining_requests <= 0:
            return 0.0

        return float(
            np.clip(
                self.remaining_budget
                / remaining_requests,
                0.0,
                1.0,
            )
        )


# ======================================================================
# ONLINE VERIFIED RESIDUAL RISK CORRECTOR
# ======================================================================

class OnlineResidualRiskCorrector:
    """
    Online correction of future predicted violation probabilities.

    Instead of moving a generic probability threshold, this tracks:

        actual violation
            -
        predicted violation probability.

    The state therefore directly estimates calibration drift.

    A small number of audits is required before correction is activated
    so that one very early sparse audit does not completely change the
    policy selector.
    """

    def __init__(
        self,
        beta: float = 0.10,
        max_ipw_weight: float = 100.0,
        max_abs_bias: float = 0.25,
        min_audits: int = 3,
    ):
        self.beta = float(beta)

        self.max_ipw_weight = float(
            max_ipw_weight
        )

        self.max_abs_bias = float(
            max_abs_bias
        )

        self.min_audits = int(
            min_audits
        )

        self.weighted_residual = 0.0
        self.weight_mass = 0.0

        self.audit_count = 0

    def update(
        self,
        predicted_risk: float,
        violation: bool,
        audit_probability: float,
    ):
        q = max(
            float(audit_probability),
            1e-8,
        )

        w = min(
            1.0 / q,
            self.max_ipw_weight,
        )

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

        # Exponentially weighted, propensity-adjusted residual.
        self.weighted_residual = (
            (
                1.0
                - self.beta
            )
            * self.weighted_residual
            +
            self.beta
            * w
            * residual
        )

        self.weight_mass = (
            (
                1.0
                - self.beta
            )
            * self.weight_mass
            +
            self.beta
            * w
        )

        self.audit_count += 1

    @property
    def bias(self):
        if (
            self.audit_count
            < self.min_audits
            or self.weight_mass <= 0
        ):
            return 0.0

        value = (
            self.weighted_residual
            / self.weight_mass
        )

        return float(
            np.clip(
                value,
                -self.max_abs_bias,
                self.max_abs_bias,
            )
        )

    def correct(
        self,
        predicted_risk,
    ):
        """
        Apply current verified residual correction to one or many
        candidate risk estimates.
        """

        risk = np.asarray(
            predicted_risk,
            dtype=float,
        )

        return np.clip(
            risk + self.bias,
            0.0,
            1.0,
        )
