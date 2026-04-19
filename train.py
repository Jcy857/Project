import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class ControlSubNet(nn.Module):
    """
    Sub-network for approximating the control a_t(s_t | θ_t).
    - 1 input layer (state_dim) + 2 hidden layers (hidden_dim neurons each) + 1 output layer (control_dim).
    - ReLU activation on hidden layers only (no activation on final layer by default).
    - Batch normalization after each linear transformation and before activation (for hidden layers).
    - Weights initialized from normal distribution (mean=0, std=0.1).
    """
    def __init__(self, 
                 state_dim: int, 
                 control_dim: int, 
                 hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, control_dim)

        # Normal initialization for all weights
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
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
                 hidden_dim: int):
        super().__init__()
        self.T = T
        self.policies = nn.ModuleList([
            ControlSubNet(state_dim, control_dim, hidden_dim)
            for _ in range(T)
        ])

    def compute_loss(self,
                     s0: torch.Tensor,
                     price_path: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that unrolls the trajectory and computes the expected total cost (Monte-Carlo approximation).
        
        Args:
            s0: initial remaining shares (batch_size, state_dim)
            price_path: (batch_size, T, n_stocks)
        
        Returns:
            Scalar loss = E[C_T] approximated by batch mean.
        """
        batch_size = s0.shape[0]
        device = s0.device
        s = s0.clone()
        total_cost = torch.zeros(batch_size, device=device)

        for t in range(self.T):
            # Control from sub-network at time t
            a = self.policies[t](s)  # (batch_size, control_dim)
            a = torch.clamp(a, min=0.0)
            a = torch.minimum(a, s)
            

            # Linear percentage price impact model
            p_t = price_path[:, t].to(torch.float32)
            impact_price = p_t * (1.0 + IMPACT_FACTOR * a)
            execution_cost = torch.sum(impact_price * a, dim=1)

            total_cost += execution_cost

            # Update remaining shares
            s = torch.clamp(s - a, min=0.0)


        # Terminal cost: Buy the remaining shares at the final price
        p_T = price_path[:, -1]  # price at final time step
        terminal_cost = torch.sum(p_T * s * (1.0 + IMPACT_FACTOR * s), dim=1)
        total_cost += terminal_cost
        # Return Monte-Carlo estimate of E[C_T]
        return total_cost.mean()
    
    def get_optimal_controls(self, s0: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns the optimal control a_t for each time step t.
        """
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        
        s = s0.clone().to(torch.float32)
        controls = []
        
        with torch.no_grad():
            for t in range(self.T):
                a = self.policies[t](s)
                a = torch.clamp(a, min=0.0)
                a = torch.minimum(a, s)
                controls.append(a)
                s = torch.clamp(s - a, min=0.0)
        
        return controls

# Training loop
def train_portfolio(price_data: torch.Tensor,
                    T: int, 
                    state_dim: int, 
                    control_dim: int,
                    hidden_dim: int, 
                    batch_size: int,
                    num_iterations: int, 
                    lr: float):


    model = StochasticControlModel(T=T, 
                                   state_dim=state_dim, 
                                   control_dim=control_dim,
                                   hidden_dim=hidden_dim).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training started on {DEVICE}:".ljust(25)+ f" | T={T} | Iterations={num_iterations}")

    for iter_idx in range(num_iterations):
        # Sample random starting time indices to avoid overfitting to the same trajectories
        start_idx = np.random.randint(0, price_data.shape[0] - T, batch_size)

        s0 = torch.full((batch_size, control_dim), TARGET_SHARES, device=DEVICE)

        # Get price path for this batch
        batch_prices = torch.stack([price_data[i:i+T] for i in start_idx])
        batch_prices = batch_prices.to(DEVICE)

        loss = model.compute_loss(s0, batch_prices)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter_idx % 500 == 0:
            print(f"Iter: {iter_idx:5d}".ljust(25) + f" | Avg Execution Cost: {loss.item():.2f}")
    print("Training completed!")
    return model

def set_seed(seed=42):
    random.seed(seed)                # Python random
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # PyTorch GPU

if __name__ == "__main__":
    set_seed(1234)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = pd.read_csv("train.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.dropna(axis=1)

    # Hyper-parameters matching the paper
    T = 10
    TARGET_SHARES = 1000.0
    IMPACT_FACTOR = 0.00001
    state_dim = 20
    control_dim = state_dim
    hidden_dim = 100
    batch_size = 64                
    num_iterations = 15000
    lr = 0.001

    # Convert to PyTorch tensor
    price_data = torch.tensor(df.values[:, :state_dim], dtype=torch.float32).to(DEVICE)
    print(f"Initial shares per stock : {TARGET_SHARES:,.0f}")
    print(f"Number of stocks         : {control_dim}")


    # Train the model
    model = train_portfolio(
        price_data=price_data,
        T=T,
        state_dim=state_dim,
        control_dim=control_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        num_iterations=num_iterations,
        lr=lr
    )    

    # Save model
    torch.save(model.state_dict(), "sde_model.pt")
    print("Model saved as 'sde_model.pt'")
