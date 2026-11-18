"""
Physics-Informed Loss Function
================================
Hybrid loss: L = L_data + lambda * L_physics

Physics constraint used:
  SAME-ENGINE PAIR MONOTONICITY

  For two sequences from the SAME engine at times t and t+k:
    pred(t) > pred(t+k)
  i.e. predicted RUL at a later time must be LOWER.

  This is the fundamental physical law: remaining life can only decrease
  for a degrading system that receives no maintenance.

  WHY THIS CONSTRAINT MATTERS:
  A standard LSTM is trained on shuffled batches of independent sequences.
  It never explicitly sees the relationship between consecutive states of
  the same engine. As a result, it CAN produce non-monotonic predictions
  at inference time — predicting higher RUL at cycle 80 than at cycle 60
  for the same engine. This violates basic physics.

  The pair monotonicity loss directly penalises these violations during
  training, enforcing temporal consistency that the data-only loss cannot
  guarantee.

  L_physics = mean( relu( pred(t+k) - pred(t) )^2 )
            = mean of squared violations (0 if already monotonic)

  L_total = L_data + lambda * L_physics

  ADVANTAGE OVER GRADIENT PENALTY:
  The gradient-based approach (d(RUL)/d(cycle)) is always near zero
  because the LSTM quickly learns the general direction from data.
  The pair approach penalises specific per-engine violations that
  persist even in well-trained models.

Reference: Raissi, M. et al. (2019). Physics-informed neural networks.
           Journal of Computational Physics, 378, 686-707.
"""

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """Hybrid loss: L = L_data + lambda_physics * L_physics

    Args:
        lambda_physics : Weight of the physics constraint term.
                         Set to 0.0 for baseline LSTM (no constraint).
        use_pair_loss  : True for PINN-LSTM, False for baseline.
    """

    def __init__(self,
                 lambda_physics: float = 0.1,
                 use_pair_loss: bool = True):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.use_pair_loss = use_pair_loss
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------------
    # Physics constraint: same-engine pair monotonicity
    # ------------------------------------------------------------------
    def pair_monotonicity_loss(self,
                               pred_early: torch.Tensor,
                               pred_late: torch.Tensor) -> torch.Tensor:
        """Penalise when predicted RUL increases between two states of the same engine.

        For each same-engine pair (early sequence, late sequence):
          - pred_early: predicted RUL at earlier time t
          - pred_late : predicted RUL at later  time t+k

        Physical law: pred_late <= pred_early
        Penalty: relu(pred_late - pred_early)^2

        When predictions are already monotonic (pred_late < pred_early),
        relu returns 0 and no penalty is added.

        Args:
            pred_early: shape (n_pairs, 1) — RUL prediction at time t
            pred_late : shape (n_pairs, 1) — RUL prediction at time t+k

        Returns:
            Scalar mean violation penalty
        """
        diff = pred_late.squeeze() - pred_early.squeeze()   # should be <= 0
        violations = torch.relu(diff)                        # positive = violation
        return violations.pow(2).mean()

    # ------------------------------------------------------------------
    # Forward — single-sequence mode (standard batched training)
    # ------------------------------------------------------------------
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                pred_early: torch.Tensor = None,
                pred_late: torch.Tensor = None) -> dict:
        """Compute hybrid loss.

        Args:
            predictions : model output for main batch, shape (batch, 1)
            targets     : true normalised RUL, shape (batch,)
            pred_early  : RUL predictions for early sequences (pairs), shape (n_pairs, 1)
            pred_late   : RUL predictions for late  sequences (pairs), shape (n_pairs, 1)

        Returns:
            dict with keys:
              'total'     — combined loss (used for backprop)
              'l_data'    — MSE on main batch
              'l_physics' — pair monotonicity penalty (before lambda scaling)
        """
        l_data = self.mse(predictions.view(-1), targets.view(-1))
        l_physics = torch.tensor(0.0, device=predictions.device)

        if self.use_pair_loss and pred_early is not None and pred_late is not None:
            l_physics = self.pair_monotonicity_loss(pred_early, pred_late)

        total = l_data + self.lambda_physics * l_physics

        return {
            "total": total,
            "l_data": l_data.detach(),
            "l_physics": l_physics.detach() if isinstance(l_physics, torch.Tensor)
                         else l_physics,
        }
