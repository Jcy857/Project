import torch
import pandas as pd

# Import the model class from train.py
from train import StochasticControlModel

# ========================= CONFIG =========================
T = 20
TARGET_SHARES = 1000.0
IMPACT_FACTOR = 0.0001
STATE_DIM = 10
CONTROL_DIM = 10
HIDDEN_DIM = 100
IMPACT_DIM = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_model(model_path, prices, A_mat, B_mat, C_mat):
    model = StochasticControlModel(
        T=T, 
        state_dim=STATE_DIM, 
        control_dim=CONTROL_DIM, 
        hidden_dim=HIDDEN_DIM,
        impact_dim=IMPACT_DIM,
        impact_factor=IMPACT_FACTOR
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"--- Model loaded from {model_path} ---")

    
    s_t = torch.full((1, STATE_DIM), TARGET_SHARES, device=DEVICE)
    x_t = torch.zeros(1, IMPACT_DIM, device=DEVICE).unsqueeze(-1)

    total_model_cost: torch.Tensor = torch.zeros(1, device=DEVICE)

    print(f"\n{'Step':<5} | {'Action (Avg)':<12} | {'Remaining (Avg)':<15} | {'Step Cost':<10}")
    print("-" * 60)

    with torch.no_grad():
        prices = prices[:T].unsqueeze(0).to(DEVICE)
        _, actions = model.compute_loss(s_t, prices, return_actions=True)
        if actions is not None:
            for t in range(T):
                a_t = actions.T[t].unsqueeze(-1)
                p_t = prices[:, t].to(torch.float32).unsqueeze(-1)
                P_t_tilde = torch.diag_embed(p_t.squeeze(-1))

                impact_price = P_t_tilde @ (A_mat @ P_t_tilde @ a_t + B_mat @ x_t)
                execution_price = p_t + impact_price

                step_cost = torch.sum(execution_price * a_t, dim=1).flatten()
                total_model_cost += step_cost

                s_t = s_t - a_t
                x_t = C_mat @ x_t + torch.randn(IMPACT_DIM, 1, device=DEVICE)

                print(f"{t:<5} | {a_t.mean().item():<12.2f} | {s_t.mean().item():<15.2f} | {step_cost.mean().item():,.2f}")
    print("-" * 60)
    return total_model_cost.mean()

def run_instant(A_mat, B_mat, C_mat):
    x_t = torch.zeros(1, IMPACT_DIM, 1, device=DEVICE)
    instant_qty = TARGET_SHARES
    p_t = prices[0].view(STATE_DIM, 1)
    a_instant = torch.full((STATE_DIM, 1), instant_qty, device=DEVICE)
    P_t_tilde = torch.diag_embed(p_t.squeeze(-1))
    exec_p = p_t + (P_t_tilde @ (A_mat @ P_t_tilde @ a_instant + B_mat @ x_t))
    total_instant_cost = torch.sum(exec_p * a_instant)
    return total_instant_cost

def run_equal_split(A_mat, B_mat, C_mat):
    x_t = torch.zeros(1, IMPACT_DIM, 1, device=DEVICE)
    equal_qty = TARGET_SHARES / T
    total_equal_cost = 0.0
    for t in range(T):
        p_t = prices[t].view(STATE_DIM, 1)
        a_eq = torch.full((STATE_DIM, 1), equal_qty, device=DEVICE)
        P_t_tilde = torch.diag_embed(p_t.squeeze(-1))
        exec_p = p_t + (P_t_tilde @ (A_mat @ P_t_tilde @ a_eq + B_mat @ x_t))
        total_equal_cost += torch.sum(exec_p * a_eq).item()
    return total_equal_cost

def solve_analytical_recursion(T, A_mat, B_mat, C_mat):
    """
    Implements the recursive equations for Ak and Bk from the provided image.
    k represents steps remaining.
    """
    A_list:torch.Tensor = [None] * (T + 1)
    B_list:torch.Tensor = [None] * (T + 1)

    # Base cases: A0 = A, B0 = B'
    A_list[0] = A_mat
    B_list[0] = B_mat.t()

    for k in range(1, T + 1):
        # Ak = A - 1/4 * A * inv(Ak-1) * A'
        inv_prev_A = torch.linalg.inv(A_list[k-1])
        A_list[k] = A_mat - 0.25 * (A_mat @ inv_prev_A @ A_mat.t())

        # Bk = 1/2 * C' * Bk-1 * inv(Ak-1') * A' + B'
        B_list[k] = (0.5 * (C_mat.t() @ B_list[k-1]) @ torch.linalg.inv(A_list[k-1].t()) @ A_mat.t()) + B_mat.t()

    return A_list, B_list

def run_analytic(A_mat, B_mat, C_mat):
    A_list, B_list = solve_analytical_recursion(T, A_mat, B_mat, C_mat)
    s_t = torch.full((STATE_DIM, 1), TARGET_SHARES, device=DEVICE)
    x_t = torch.zeros(IMPACT_DIM, 1, device=DEVICE)
    total_analytical_cost = 0.0
    
    print(f"\n{'Step':<5} | {'Action (Avg)':<12} | {'Remaining (Avg)':<15} | {'Step Cost':<10}")
    print("-" * 60)

    for t in range(T):
        k = T - t - 1
        idx = k - 1 
        inv_Ak = torch.linalg.inv(A_list[idx])
        
        # Eq (10): s*_T-k = (I - 1/2 * inv(Ak-1) * A') * w_T-k + 1/2 * inv(Ak-1) * Bk-1' * C * x_T-k
        I_mat = torch.eye(STATE_DIM, device=DEVICE)
        a_t = (I_mat - 0.5 * (inv_Ak @ A_mat.t())) @ s_t + 0.5 * (inv_Ak @ B_list[idx].t() @ C_mat @ x_t)
        # Ensure feasibility (Hard constraint for final step)
        if t == T - 1:
            a_t = s_t
        else:
            a_t = torch.minimum(a_t, s_t)

        # Cost Calculation (Market Impact)
        p_t = prices[t].view(STATE_DIM, 1)
        P_t_tilde = torch.diag_embed(p_t.squeeze(-1))
        
        # Impact model: delta = P_t_tilde * (A * P_t_tilde * a + B * x)
        impact_price = P_t_tilde @ (A_mat @ P_t_tilde @ a_t + B_mat @ x_t)
        execution_price = p_t + impact_price
        
        step_cost = torch.sum(execution_price * a_t).item()
        total_analytical_cost += step_cost
        
        s_t = s_t - a_t
        x_t = C_mat @ x_t + torch.randn(IMPACT_DIM, 1, device=DEVICE)
        print(f"{t:<5} | {a_t.mean().item():<12.2f} | {s_t.mean().item():<15.2f} | {step_cost:,.2f}")


    print("-" * 60)
    return total_analytical_cost

if __name__ == "__main__":
    # Load and Prepare Test Data
    df = pd.read_csv("test.csv")
    if "Date" in df.columns:
        df = df.set_index("Date")
    df = df.dropna(axis=1)
    prices = torch.tensor(df.values[:, :STATE_DIM], dtype=torch.float32, device=DEVICE)

    # Pre-define matrices (Matching training logic)
    A_mat = torch.eye(STATE_DIM, device=DEVICE) * IMPACT_FACTOR # Assume no cross stock impact for simplicity
    B_mat = torch.zeros(STATE_DIM, IMPACT_DIM, device=DEVICE)   # Set to 0 for simplicity
    C_mat = torch.zeros(IMPACT_DIM, IMPACT_DIM, device=DEVICE)  # Set to 0 for simplicity
    
    # Final Summary Table
    print("\n" + "="*60)
    total_model_cost = run_model("sde_model.pt", prices, A_mat, B_mat, C_mat)
    total_equal_cost = run_equal_split(A_mat, B_mat, C_mat)
    total_instant_cost = run_instant(A_mat, B_mat, C_mat)
    total_analytical_cost = run_analytic(A_mat, B_mat, C_mat)

    print(f"{'STRATEGY':<30} | {'TOTAL COST':<20}")
    print("-" * 60)
    print(f"{'Stochastic Model (AI)':<30} | {total_model_cost:,.2f}")
    print(f"{'Analytical DP Cost :':<30} | {total_analytical_cost:,.2f}")
    print(f"{'Equal Split (Naive)':<30} | {total_equal_cost:,.2f}")
    print(f"{'Instant Purchase (t=0)':<30} | {total_instant_cost:,.2f}")
    print("="*60)
