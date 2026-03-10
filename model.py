import math
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class _MNISTNet(nn.Module):
    """Feed-forward network: 784 -> 256 -> 128 -> 10 with ReLU."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DPNeuralNetwork:
    """Neural network trainer with selectable backpropagation strategy.

    Parameters
    ----------
    backprop_type : "standard" or "dp"
        "standard" uses vanilla SGD.
        "dp" uses differentially-private SGD (Algorithm 1 from Abadi et al.).
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Lot / group size *L* (used for both standard and DP training).
    noise_scale : float
        Noise multiplier *sigma* for DP-SGD. Ignored when backprop_type="standard".
    clip_bound : float
        Per-example gradient norm bound *C*. Ignored when backprop_type="standard".
    delta : float
        Target delta for (epsilon, delta)-DP accounting.
    device : str or None
        "cpu", "cuda", etc.  Auto-detected if None.
    """

    def __init__(
        self,
        backprop_type: Literal["standard", "dp"] = "standard",
        lr: float = 0.1,
        epochs: int = 5,
        batch_size: int = 256,
        noise_scale: float = 1.0,
        clip_bound: float = 1.0,
        delta: float = 1e-5,
        device: str | None = None,
    ):
        self.backprop_type = backprop_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.clip_bound = clip_bound
        self.delta = delta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = _MNISTNet().to(self.device)
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self._steps_taken = 0
        self._dataset_size = 0

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(self, data_root: str = "./data") -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

        self._dataset_size = len(train_ds)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Train the model for ``self.epochs`` epochs.

        Returns a list of per-epoch metric dicts:
            {"epoch", "train_loss", "test_accuracy", "elapsed_sec"}
        """
        if self.train_loader is None:
            raise RuntimeError("Call load_data() before train().")

        history: list[dict] = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            if self.backprop_type == "standard":
                avg_loss = self._train_epoch_standard()
            else:
                avg_loss = self._train_epoch_dp()
            elapsed = time.time() - t0

            test_acc = self.evaluate()
            record = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "test_accuracy": test_acc,
                "elapsed_sec": elapsed,
            }
            history.append(record)
            print(
                f"[{self.backprop_type:>8}] Epoch {epoch}/{self.epochs}  "
                f"loss={avg_loss:.4f}  acc={test_acc:.4f}  time={elapsed:.1f}s"
            )

        return history

    # -- standard SGD --------------------------------------------------

    def _train_epoch_standard(self) -> float:
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        total_loss = 0.0
        n_batches = 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    # -- DP-SGD (Algorithm 1) -----------------------------------------

    def _train_epoch_dp(self) -> float:
        """One epoch of DP-SGD following Algorithm 1 from the paper."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            batch_loss = self._dp_step(data, target)
            total_loss += batch_loss
            n_batches += 1
            self._steps_taken += 1

        return total_loss / n_batches

    def _dp_step(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """Single DP-SGD update step.

        1. Compute per-example gradients
        2. Clip each to norm C
        3. Sum + add Gaussian noise scaled to sigma*C
        4. Average and apply update
        """
        L = data.size(0)  # lot size

        # Accumulate clipped gradients
        clipped_grads: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param) for name, param in self.model.named_parameters()
        }
        batch_loss = 0.0

        for i in range(L):
            self.model.zero_grad()
            xi = data[i : i + 1]
            yi = target[i : i + 1]
            loss_i = F.cross_entropy(self.model(xi), yi)
            loss_i.backward()
            batch_loss += loss_i.item()

            # Collect this example's gradients and compute its total norm
            grads = {}
            total_norm_sq = 0.0
            for name, param in self.model.named_parameters():
                g = param.grad.clone()
                grads[name] = g
                total_norm_sq += g.norm(2).item() ** 2
            total_norm = math.sqrt(total_norm_sq)

            # Clip: g_bar = g / max(1, ||g||_2 / C)
            clip_factor = max(1.0, total_norm / self.clip_bound)
            for name in grads:
                clipped_grads[name] += grads[name] / clip_factor

        # Add Gaussian noise and average
        for name, param in self.model.named_parameters():
            noise = torch.normal(
                mean=0.0,
                std=self.noise_scale * self.clip_bound,
                size=param.shape,
                device=self.device,
            )
            noisy_grad = (clipped_grads[name] + noise) / L
            param.data -= self.lr * noisy_grad

        return batch_loss / L

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        """Return test-set accuracy as a float in [0, 1]."""
        if self.test_loader is None:
            raise RuntimeError("Call load_data() before evaluate().")

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.model(data).argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        return correct / total

    # ------------------------------------------------------------------
    # Privacy accounting  (RDP-based)
    # ------------------------------------------------------------------

    def compute_privacy_cost(self) -> tuple[float, float]:
        """Compute (epsilon, delta) privacy cost using RDP accounting.

        Uses the Rényi Differential Privacy framework (Mironov, 2017) which
        generalises the moments accountant from the original paper.

        Returns (epsilon, self.delta).
        """
        if self._steps_taken == 0 or self._dataset_size == 0:
            return (0.0, self.delta)

        q = self.batch_size / self._dataset_size  # sampling probability
        sigma = self.noise_scale
        T = self._steps_taken

        # Evaluate RDP at a grid of integer orders alpha >= 2
        alphas = list(range(2, 256))
        rdp_values = [self._compute_rdp_single_alpha(q, sigma, alpha) * T for alpha in alphas]

        # Convert RDP to (eps, delta)-DP:  eps = min over alpha of  rdp - log(delta)/(alpha-1)
        eps_values = [
            rdp - math.log(self.delta) / (alpha - 1)
            for alpha, rdp in zip(alphas, rdp_values)
        ]
        epsilon = min(eps_values)
        return (epsilon, self.delta)

    @staticmethod
    def _compute_rdp_single_alpha(q: float, sigma: float, alpha: int) -> float:
        """RDP of the sampled Gaussian mechanism at a single integer order.

        Uses the bound from Mironov et al. (2019), Proposition 3.
        For the subsampled Gaussian mechanism with sampling rate q,
        noise multiplier sigma, at integer order alpha:

            RDP(alpha) <= (1/(alpha-1)) * log( sum_{k=0}^{alpha} C(alpha,k)
                            * (1-q)^{alpha-k} * q^k * exp(k*(k-1)/(2*sigma^2)) )
        """
        if sigma == 0:
            return float("inf")

        log_terms = []
        for k in range(alpha + 1):
            log_comb = (
                math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
            )
            log_q_part = k * math.log(q) if q > 0 else (-float("inf") if k > 0 else 0.0)
            log_1mq_part = (alpha - k) * math.log(1 - q) if q < 1 else (-float("inf") if k < alpha else 0.0)
            log_exp_part = k * (k - 1) / (2.0 * sigma ** 2)
            log_terms.append(log_comb + log_q_part + log_1mq_part + log_exp_part)

        # log-sum-exp for numerical stability
        max_log = max(log_terms)
        sum_exp = sum(math.exp(lt - max_log) for lt in log_terms)
        log_sum = max_log + math.log(sum_exp)

        return log_sum / (alpha - 1)
