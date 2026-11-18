"""
Training loop for LSTM/CNN-LSTM models with optional physics-informed loss.

Two modes:
  use_pair_loss=False → Baseline LSTM  — trains on MSE only (lambda = 0)
  use_pair_loss=True  → PINN-LSTM      — adds same-engine pair monotonicity penalty

The pair loss requires a second DataLoader (train_pairs) that yields
(X_early, X_late) from the same engine. Each training step:
  1. Sample main batch (X, y)       → forward pass → L_data  (MSE)
  2. Sample pair batch (Xe, Xl)     → two forward passes → L_physics (monotonicity)
  3. total = L_data + lambda * L_physics → backprop
"""

import torch
import itertools
from ..models.physics_loss import PhysicsInformedLoss


class Trainer:
    """Handles model training, validation, and checkpointing."""

    def __init__(self, model, config: dict, device: str = None,
                 use_pair_loss: bool = True):
        """
        Args:
            model         : LSTMPredictor or CNNLSTMPredictor
            config        : full config dict (training section used)
            device        : 'cuda' or 'cpu' (auto-detected if None)
            use_pair_loss : True → PINN-LSTM, False → baseline LSTM
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_pair_loss = use_pair_loss

        train_cfg = config["training"]
        lambda_p = train_cfg["physics_loss_weight"] if use_pair_loss else 0.0

        self.criterion = PhysicsInformedLoss(
            lambda_physics=lambda_p,
            use_pair_loss=use_pair_loss,
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_cfg["scheduler_step_size"],
            gamma=train_cfg["scheduler_gamma"],
        )
        self.epochs = train_cfg["epochs"]
        self.patience = train_cfg["early_stopping_patience"]

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader, pair_loader=None) -> dict:
        """Train for one epoch.

        Args:
            dataloader  : main DataLoader yielding (X, y)
            pair_loader : optional DataLoader yielding (X_early, X_late)
                          used only when use_pair_loss=True
        """
        self.model.train()
        total_loss = total_l_data = total_l_phys = 0.0
        n_batches = 0

        # Cycle pair_loader infinitely so it stays in sync with main loader
        pair_iter = itertools.cycle(pair_loader) if pair_loader is not None else None

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            # Main forward pass
            predictions = self.model(X_batch)

            # Pair forward passes (PINN only)
            pred_early = pred_late = None
            if self.use_pair_loss and pair_iter is not None:
                X_early, X_late = next(pair_iter)
                X_early = X_early.to(self.device)
                X_late  = X_late.to(self.device)
                with torch.set_grad_enabled(True):
                    pred_early = self.model(X_early)
                    pred_late  = self.model(X_late)

            loss_dict = self.criterion(
                predictions, y_batch,
                pred_early=pred_early,
                pred_late=pred_late,
            )

            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss   += loss_dict["total"].item()
            total_l_data += loss_dict["l_data"].item()
            lp = loss_dict["l_physics"]
            total_l_phys += lp.item() if isinstance(lp, torch.Tensor) else float(lp)
            n_batches += 1

        return {
            "loss":      total_loss   / n_batches,
            "l_data":    total_l_data / n_batches,
            "l_physics": total_l_phys / n_batches,
        }

    # ------------------------------------------------------------------
    def validate(self, dataloader) -> dict:
        """Validate on MSE only (no physics penalty during eval)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = torch.nn.functional.mse_loss(preds.view(-1), y_batch.view(-1))
                total_loss += loss.item()
                n_batches  += 1
        return {"loss": total_loss / n_batches}

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader,
            pair_loader=None, save_path: str = None) -> dict:
        """Full training loop with early stopping.

        Args:
            train_loader : main training DataLoader
            val_loader   : validation DataLoader
            pair_loader  : same-engine pair DataLoader (PINN only)
            save_path    : where to save the best checkpoint

        Returns:
            history dict with loss curves
        """
        history = {"train_loss": [], "val_loss": [], "l_data": [], "l_physics": []}
        best_val  = float("inf")
        patience  = 0
        mode_str  = "PINN-LSTM" if self.use_pair_loss else "LSTM (baseline)"

        print(f"  Mode            : {mode_str}")
        print(f"  lambda_physics  : {self.criterion.lambda_physics:.3f}")
        if self.use_pair_loss and pair_loader:
            print(f"  Pair batches/ep : {len(pair_loader):,}")

        for epoch in range(self.epochs):
            tr = self.train_epoch(train_loader,
                                  pair_loader if self.use_pair_loss else None)
            va = self.validate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(va["loss"])
            history["l_data"].append(tr["l_data"])
            history["l_physics"].append(tr["l_physics"])

            if (epoch + 1) % 10 == 0:
                print(f"  Ep {epoch+1:>3}/{self.epochs} | "
                      f"Train {tr['loss']:.5f} "
                      f"(data={tr['l_data']:.5f}, phys={tr['l_physics']:.5f}) | "
                      f"Val {va['loss']:.5f}")

            if va["loss"] < best_val:
                best_val = va["loss"]
                patience = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience += 1
                if patience >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(best val: {best_val:.5f})")
                    break

        return history
