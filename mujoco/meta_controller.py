import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MetaController(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        lr: float = 3e-4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define possible beta values
        self.beta_values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network to predict beta class
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(self.beta_values))  # Output logits for each beta class
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def predict_beta(self, state: torch.Tensor, action: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(state, action)
            probs = F.softmax(logits, dim=-1)
            # Sample from the probability distribution
            beta_idx = torch.multinomial(probs, 1).item()
            return self.beta_values[beta_idx].item()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        old_betas: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update the meta-controller using the returns as the meta-objective
        
        Args:
            states: Batch of states
            actions: Batch of actions
            returns: Batch of returns achieved with old_betas
            old_betas: Batch of beta values used to achieve the returns
            
        Returns:
            Tuple of (meta_loss, mean_beta)
        """
        # Get logits for all beta classes
        logits = self.forward(states, actions)  # [batch_size, num_betas]
        
        # Compute probabilities for each beta class
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_betas]
        
        # Compute expected returns for each beta class
        expected_returns = torch.matmul(probs, self.beta_values)  # [batch_size]
        
        # Compute meta-loss: encourage higher returns
        # We want to predict betas that would have led to higher returns
        meta_loss = -torch.mean(returns * expected_returns)
        
        # Add entropy regularization to encourage exploration
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        meta_loss -= 0.01 * entropy  # Small entropy bonus
        
        # Update meta-controller
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        # Get mean predicted beta for logging
        mean_beta = torch.mean(expected_returns).item()
        
        return meta_loss.item(), mean_beta 