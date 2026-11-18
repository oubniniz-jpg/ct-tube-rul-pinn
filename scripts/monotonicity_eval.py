"""
Monotonicity-violation eval: does PINN-LSTM produce physically-valid
(monotonically decreasing) RUL where the plain LSTM does not?

For each held-out test tube we run the model on its ordered windows and
count how often the predicted RUL INCREASES from one window to the next
(a physically impossible event). Lower is better; a perfect predictor = 0%.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, numpy as np, torch, torch.nn as nn
from src.preprocessing.ct_loader import CTLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class LSTMPredictor(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.lstm = nn.LSTM(input_size=m["input_features"], hidden_size=m["hidden_size"],
                            num_layers=m["num_layers"],
                            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(m["hidden_size"], 64), nn.ReLU(), nn.Dropout(m["dropout"]),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, m["output_size"]), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

loader = CTLoader(csv_path="data/synthetic/ct_tubes_small.csv",
                  seq_len=config["data"]["sequence_length"],
                  batch_size=32, max_tubes=200, stride=100)
dls = loader.prepare()
tpt = dls["test_per_tube"]
m = dict(config["model"])
m["input_features"] = dls["n_features"]
m.setdefault("output_size", 1)


def load(path):
    net = LSTMPredictor(m)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def violations(net, step):
    """Fraction of (i, i+step) pairs where predicted RUL INCREASES (violation)."""
    vv = tt = 0
    with torch.no_grad():
        for seqs in tpt:
            if len(seqs) < step + 1:
                continue
            preds = net(torch.FloatTensor(np.array(seqs))).numpy().flatten()
            d = preds[step:] - preds[:-step]   # should be <= 0 (RUL decreasing)
            vv += int((d > 0).sum()); tt += len(d)
    return vv / tt * 100 if tt else float("nan"), tt


print(f"test tubes with >=2 windows: {sum(1 for s in tpt if len(s) >= 2)}")
nets = {}
for frac in ["f20", "f100"]:
    nets[("LSTM", frac)] = load(f"results/ct_tubes/LSTM_{frac}.pt")
    nets[("PINN-LSTM", frac)] = load(f"results/ct_tubes/PINN-LSTM_ours_{frac}.pt")

for frac in ["f20", "f100"]:
    print(f"\n=== {frac} (data fraction) -- % of pairs where RUL increases (lower=better) ===")
    print(f"  {'step':<6}{'LSTM':>10}{'PINN-LSTM':>12}")
    for step in [1, 5, 10, 20]:
        lr, _ = violations(nets[("LSTM", frac)], step)
        pr, _ = violations(nets[("PINN-LSTM", frac)], step)
        print(f"  {step:<6}{lr:>9.1f}%{pr:>11.1f}%")
