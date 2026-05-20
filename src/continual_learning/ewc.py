# src/continual_learning/ewc.py
#
# PURPOSE:
#   Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
#   Constrains important weights from changing too much during fine-tuning.
#
# REFERENCE:
#   Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", 2017
#   https://arxiv.org/abs/1612.00220

import os
import sys
import copy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


class EWC:
    """
    Elastic Weight Consolidation.

    After training on Task A, compute the Fisher Information Matrix (diagonal
    approximation) to identify which weights are most important. During
    training on Task B, add a penalty that discourages changing those weights.

    Usage:
        # After training Task A:
        ewc = EWC(model, dataloader)

        # During Task B training:
        loss = task_b_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module = None,
        dataloader=None,
        lambda_: float = None,
        n_samples: int = None,
    ):
        self.lambda_ = lambda_ or cfg.EWC_LAMBDA
        self.n_samples = n_samples or cfg.EWC_N_SAMPLES
        self._fisher: Dict[str, torch.Tensor] = {}
        self._params_star: Dict[str, torch.Tensor] = {}

        if model is not None and dataloader is not None:
            self.compute_fisher(model, dataloader)

    def compute_fisher(self, model: nn.Module, dataloader):
        """
        Compute diagonal Fisher Information Matrix.

        This estimates how important each parameter is for the current task.
        Parameters with high Fisher values are "important" and should be
        preserved during future training.

        Args:
            model: the trained model
            dataloader: DataLoader of the current task's training data
        """
        model.eval()
        fisher = {}
        params_star = {}

        # Store current parameter values (θ*)
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_star[name] = param.data.clone()
                fisher[name] = torch.zeros_like(param.data)

        # Estimate Fisher using empirical samples
        n = 0
        for batch in dataloader:
            if n >= self.n_samples:
                break

            model.zero_grad()

            # Forward pass — use the model's own loss if available
            if hasattr(batch, '__len__') and len(batch) == 2:
                inputs, targets = batch
                outputs = model(inputs)
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                else:
                    # Generic cross-entropy fallback
                    loss = nn.functional.cross_entropy(
                        outputs if isinstance(outputs, torch.Tensor) else outputs[0],
                        targets
                    )
            else:
                # For YOLO-style batches, try direct forward
                try:
                    loss = model(batch)
                    if isinstance(loss, dict):
                        loss = sum(loss.values())
                except Exception:
                    continue

            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            n += 1

        # Average Fisher values
        for name in fisher:
            fisher[name] /= max(n, 1)

        self._fisher = fisher
        self._params_star = params_star

        print(f"[EWC] Fisher computed over {n} samples, "
              f"{len(fisher)} parameter groups tracked.")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty to add to the training loss.

        penalty = (λ/2) Σ_i F_i (θ_i - θ*_i)²

        Args:
            model: the model being trained on the new task

        Returns:
            Scalar tensor — the EWC penalty loss
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self._fisher and name in self._params_star:
                fisher = self._fisher[name].to(param.device)
                params_star = self._params_star[name].to(param.device)
                loss += (fisher * (param - params_star).pow(2)).sum()

        return (self.lambda_ / 2) * loss

    def save(self, path: str = None):
        """Save Fisher matrix and reference parameters."""
        path = path or os.path.join(cfg.MODELS_DIR, "ewc_state.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "fisher": self._fisher,
            "params_star": self._params_star,
            "lambda": self.lambda_,
        }, path)
        print(f"[EWC] State saved to {path}")

    def load(self, path: str = None):
        """Load previously computed Fisher matrix."""
        path = path or os.path.join(cfg.MODELS_DIR, "ewc_state.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=False)
            self._fisher = state["fisher"]
            self._params_star = state["params_star"]
            self.lambda_ = state.get("lambda", self.lambda_)
            print(f"[EWC] State loaded from {path}")
            return True
        return False

    @property
    def is_ready(self) -> bool:
        """Whether Fisher information has been computed."""
        return len(self._fisher) > 0
