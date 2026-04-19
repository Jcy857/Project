import torch
import numpy as np
import pandas as pd

# Import the model class from train.py
from train import StochasticControlModel, set_seed

# ========================= CONFIG =========================
T = 10
TARGET_SHARES = 1000.0
IMPACT_FACTOR = 0.00001
state_dim = 20
control_dim = state_dim
hidden_dim = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    set_seed(1234)

    model = StochasticControlModel(T=T, 
                               state_dim=state_dim, 
                               control_dim=control_dim, 
                               hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load("sde_model.pt", map_location=DEVICE))
    model.eval()
    print("Model loaded successfully for evaluation.\n")

    # Load test data
    df = pd.read_csv("test.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.dropna(axis=1)
    price_data = torch.tensor(df.values[:, :state_dim], dtype=torch.float32).to(DEVICE)
    s0_single = torch.full((control_dim,), TARGET_SHARES, device=DEVICE)
    optimal_a = model.get_optimal_controls(s0_single)
    remaining = s0_single.clone()
    total_execution_cost = torch.zeros(control_dim, device=DEVICE)

    for t, a_t in enumerate(optimal_a):
        # Update remaining shares
        a_t  = torch.as_tensor(a_t, device=DEVICE).flatten()
        remaining -= a_t
        p_t = price_data[t]

        # Linear impact cost for this step
        step_cost = p_t * a_t * (1 + IMPACT_FACTOR * a_t)
        total_execution_cost += step_cost

    total_execution_cost = np.sum(total_execution_cost.cpu().numpy())
    print(f"Approximate total execution cost   : {total_execution_cost:,.2f}")

    # Instant purchase baseline
    instant_cost = np.sum(price_data[0].cpu().numpy() * TARGET_SHARES * (1 + IMPACT_FACTOR * TARGET_SHARES))
    print(f"Instant purchase cost (all at t=0) : {instant_cost:,.2f}")

    # Naive equal split baseline
    equal_split_cost = torch.zeros(control_dim, device=DEVICE)
    for t in range(T):
        p_t = price_data[t]
        a_t = TARGET_SHARES / T
        step_cost = p_t * a_t * (1 + IMPACT_FACTOR * a_t)
        equal_split_cost += step_cost
    equal_split_cost = np.sum(equal_split_cost.cpu().numpy())
    print(f"Equal split cost (naive)           : {equal_split_cost:,.2f}")

