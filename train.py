import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from typing import Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class ControlSubNet(nn.Module):
    """
    Sub-network for approximating the control a_t(s_t | θ_t).
    - 1 input layer (state_dim) + 2 hidden layers (hidden_dim neurons each) + 1 output layer (control_dim).
    - ReLU activation on hidden layers only (no activation on final layer by default).
    - Batch normalization after each linear transformation and before activation (for hidden layers).
    - Weights initialized from normal distribution (mean=0, std=1).
    """
    def __init__(self, 
                 state_dim: int, 
                 control_dim: int, 
                 hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim) # Oringial paper use batchnorm1d
        self.fc_out = nn.Linear(hidden_dim, control_dim)

        # Normal initialization for all weights
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.normal_(layer.weight, mean=0, std=1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(s)))
        x = torch.relu(self.bn2(self.fc2(x)))
        a = self.fc_out(x)
        return a
        
class StochasticControlModel(nn.Module):
    """
    Full model for the stochastic control approximation algorithm from the paper.
    - Holds a separate sub-network (different θ_t) for each time step t = 0 to T-1.
    - Unrolls the entire trajectory as a single computation graph.
    - Computes the total cost C_T (including penalties) via Monte-Carlo sampling of noise paths {ξ_t}.
    - Backpropagation through the unrolled graph is performed automatically by PyTorch.
    """
    def __init__(self, T: int, 
                 state_dim: int, 
                 control_dim: int, 
                 hidden_dim: int,
                 impact_dim: int,
                 impact_factor: float):
        super().__init__()
        self.T = T
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.impact_dim = impact_dim
        self.impact_factor = impact_factor
        
        self.policies = nn.ModuleList([
            ControlSubNet(state_dim, control_dim, hidden_dim)
            for _ in range(T)
        ])

    def compute_loss(self,
                     s0: torch.Tensor,
                     prices: torch.Tensor, 
                     return_actions=False) -> Tuple[torch.Tensor, torch.Tensor|None]:
        """
        Forward pass that unrolls the trajectory and computes the expected total cost (Monte-Carlo approximation).
        
        Args:
            s0: initial remaining shares (batch_size, state_dim)
            price_path: (batch_size, T, n_stocks)
        
        Returns:
            Scalar loss = E[C_T] approximated by batch mean.
        """
        
        state_dim = self.state_dim
        impact_dim = self.impact_dim
        batch_size = s0.shape[0]
        device = s0.device
        impact_factor = self.impact_factor
        actions_history = []

        # Monte-Carlo estimate of total cost for 64 trajectories in the batch
        s_t = s0.clone().unsqueeze(-1).to(device)  # (batch_size, state_dim, 1)
        total_cost: torch.Tensor = torch.zeros(batch_size, device=device)
        A_mat = torch.eye(state_dim, device=device) * impact_factor # Assume no cross stock impact for simplicity
        B_mat = torch.zeros(state_dim, impact_dim, device=device)   # Set to 0 for simplicity
        C_mat = torch.zeros(impact_dim, impact_dim, device=device)  # Set to 0 for simplicity

        # Potential influence of market conditions
        x_t = torch.zeros(batch_size, impact_dim, device=device).unsqueeze(-1)
        for t in range(self.T):
            # Hard constraint
            if t < self.T - 1:
                a_t = self.policies[t](s_t.squeeze(-1)) 
                a_t = torch.minimum(a_t, s_t.squeeze(-1)).unsqueeze(-1)
            else:
                a_t = s_t

            if return_actions:
                actions_history.append(a_t)

            # Market Dynamics
            p_t = prices[:, t].to(torch.float32).to(device).unsqueeze(-1)
            P_t_tilde = torch.diag_embed(p_t.squeeze(-1))


            # Step Cost: Sum over the assets dimension
            impact_price = P_t_tilde @ (A_mat @ P_t_tilde @ a_t + B_mat @ x_t)
            execution_price = p_t + impact_price

            step_cost = torch.sum(execution_price * a_t, dim=1).flatten()
            total_cost += step_cost

            # Update remaining shares
            s_t = s_t - a_t
            x_t = C_mat @ x_t + torch.randn(batch_size, impact_dim, 1, device=device)
            
        if return_actions:
            actions = torch.stack(actions_history)
            actions = actions.squeeze(-1)
            avg_actions = actions.mean(dim=1)
            avg_trajectory = avg_actions.T
            # Stack into (T, batch_size, state_dim)
            return total_cost.mean(), avg_trajectory
    
        # Return Monte-Carlo estimate of E[C_T]
        return total_cost.mean(), None
    
        
# Training loop
def train_portfolio(prices: torch.Tensor,
                    T: int, 
                    state_dim: int, 
                    control_dim: int,
                    hidden_dim: int, 
                    impact_dim: int,
                    impact_factor: float,
                    batch_size: int,
                    num_iterations: int, 
                    lr: float):

    model = StochasticControlModel(T=T, 
                                   state_dim=state_dim, 
                                   control_dim=control_dim,
                                   hidden_dim=hidden_dim,
                                   impact_dim = impact_dim,
                                   impact_factor=impact_factor).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training started on {DEVICE}:".ljust(25)+ f" | T={T} | Iterations={num_iterations}")

    for iter_idx in range(num_iterations):
        s0 = torch.full((batch_size, state_dim), TARGET_SHARES, device=DEVICE)

        # Get price path for this batch
        import numpy as np
        start_idx = np.random.randint(0, prices.shape[0] - T, batch_size)
        batch_prices = torch.stack([prices[i:i+T] for i in start_idx]).to(DEVICE)
        #batch_prices = prices[:T].unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE)

        loss, _ = model.compute_loss(s0, batch_prices) 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter_idx % 100 == 0 or iter_idx == num_iterations - 1:
            print(f"Iter: {iter_idx:5d}".ljust(25) + f" | Avg Execution Cost: {loss.item():,.2f}")
        
    print("Training completed!")
    return model

if __name__ == "__main__":
    #torch.manual_seed(42)
    # Load data
    df = pd.read_csv("train.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.dropna(axis=1)

    # Hyper-parameters matching the paper
    T = 20
    TARGET_SHARES = 1000.0
    IMPACT_FACTOR = 0.0001
    STATE_DIM = 10
    CONTROL_DIM = 10
    HIDDEN_DIM = 100 
    IMPACT_DIM = 3
    BATCH_SIZE = 64 
    NUM_ITERATIONS = 1000 #15000
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Convert to PyTorch tensor
    prices = torch.tensor(df.values[:, :STATE_DIM], dtype=torch.float32, device=DEVICE)

    # Train the model
    model = train_portfolio(
        prices=prices,
        T=T,
        state_dim=STATE_DIM,
        control_dim=CONTROL_DIM,
        hidden_dim=HIDDEN_DIM,
        impact_dim=IMPACT_DIM,
        impact_factor=IMPACT_FACTOR,
        batch_size=BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        lr=LR
    )    

    # Save model
    torch.save(model.state_dict(), "sde_model.pt")
    print("Model saved as 'sde_model.pt'")