import torch
from torch import nn as nn
from torch.nn import functional as F


class IntrinsicRewardModule(nn.Module):
    """The base class for an intrinsic reward method."""

    def calculate_reward(self, obs, next_obs, actions):
        return NotImplemented

    def calculate_loss(self, obs, next_obs, actions):
        return NotImplemented


class DummyIntrinsicRewardModule(IntrinsicRewardModule):
    """Used as a dummy for vanilla DQN."""

    def calculate_reward(self, obs, next_obs, actions):
        return torch.Tensor([0.0]).unsqueeze(0)


class RNDNetwork(IntrinsicRewardModule):
    """Implementation of Random Network Distillation (RND)"""

    def __init__(self, num_obs, num_out, alpha=1.0):
        super().__init__()

        
        self.target = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), 
            nn.Linear(128, num_out) 
        )
        
        
        self.predictor = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), 
            nn.Linear(128, num_out)
        )
        
        # Congelamos la red target explícitamente
        for param in self.target.parameters():
            param.requires_grad = False

        self.alpha = alpha

    def calculate_loss(self, obs, next_obs, actions):
        with torch.no_grad():
            target_out = self.target(next_obs)
        
        predictor_out = self.predictor(next_obs)
        
        # El MSE es el estándar para RND
        loss = F.mse_loss(predictor_out, target_out)
        return loss

    def calculate_reward(self, obs, next_obs, actions):
        with torch.no_grad():
            target_out = self.target(next_obs)
            predictor_out = self.predictor(next_obs)
        
        
        reward = (target_out - predictor_out).pow(2).sum(dim=1)
        
        # Escalamos y quitamos el clamp si es necesario, 
        # o asegúrate de que alpha no sea demasiado pequeño.
        reward = reward * self.alpha
        
        return reward

class ICMNetwork(IntrinsicRewardModule):
    """Implementation of Intrinsic Curiosity Module (ICM)"""

    def __init__(self, num_obs, num_feature, num_act, alpha=10.0, beta=0.5):
        super().__init__()

        self.feature = nn.Sequential(nn.Linear(num_obs, num_feature), nn.ReLU(), )

        self.inverse_dynamics = nn.Sequential(
            nn.Linear(num_feature * 2, num_act)
        )

        self.forward_dynamics = nn.Sequential(
            nn.Linear(num_feature + num_act, num_feature),
        )

        self.alpha = alpha
        self.beta = beta
        self.num_actions = num_act
        self.num_feat = num_feature

    def calculate_loss(self, obs, next_obs, actions):
        # Ground-truth actions en formato one-hot
        actions_target = torch.zeros(obs.size()[0], self.num_actions).to(obs.device)
        for i, a in enumerate(actions):
            actions_target[i, int(a)] = 1.0

        # 1. Extraer características de los estados
        phi_curr = self.feature(obs)
        phi_next = self.feature(next_obs)

        # 2. Inverse dynamics loss: predecir la acción realizada dado phi_curr y phi_next
        # Concatenamos las características: dim=1 para el batch
        pred_action_logits = self.inverse_dynamics(torch.cat([phi_curr, phi_next], dim=1))
        # Usamos CrossEntropy entre los logits y los índices de las acciones
        inverse_dynamics_loss = F.cross_entropy(pred_action_logits, actions.long())

        # 3. Forward dynamics loss: predecir phi_next dado phi_curr y la acción
        # Combinamos el estado actual con la acción (one-hot)
        forward_input = torch.cat([phi_curr, actions_target], dim=1)
        pred_phi_next = self.forward_dynamics(forward_input)
        
        # MSE entre la predicción y el valor real de phi_next, multiplicado por 0.5 según enunciado
        forward_dynamics_loss = 0.5 * F.mse_loss(pred_phi_next, phi_next.detach())

        # Suma ponderada de ambas pérdidas
        loss = (1.0 - self.beta) * inverse_dynamics_loss + self.beta * forward_dynamics_loss
        return loss

    def calculate_reward(self, obs, next_obs, actions):
        # One-hot encoding de las acciones
        actions_one_hot = torch.zeros((obs.size()[0], self.num_actions)).to(obs.device)
        for i, a in enumerate(actions):
            actions_one_hot[i, int(a)] = 1.0

        # Calculamos la recompensa intrínseca sin gradientes
        with torch.no_grad():
            phi_curr = self.feature(obs)
            phi_next = self.feature(next_obs)
            
            forward_input = torch.cat([phi_curr, actions_one_hot], dim=1)
            pred_phi_next = self.forward_dynamics(forward_input)
            
            # La recompensa es el MAE (L1) entre la predicción y la realidad
            # Calculamos el error por cada elemento del batch (dim=1)
            reward = F.l1_loss(pred_phi_next, phi_next, reduction='none').mean(dim=1)
            
            # Escalamos por alpha
            reward = reward * self.alpha
        
        return reward