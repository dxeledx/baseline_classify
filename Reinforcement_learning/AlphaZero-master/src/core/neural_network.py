"""
Neural Network Architecture for AlphaZero

This module implements the neural network with policy and value heads
used in the AlphaZero algorithm. The architecture follows the design
from the DeepMind paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with batch normalization"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero Neural Network with Policy and Value Heads
    
    Architecture:
    - Convolutional input block
    - Multiple residual blocks
    - Policy head: outputs move probabilities
    - Value head: outputs position evaluation [-1, 1]
    """
    
    def __init__(self, game, num_channels=256, num_res_blocks=19, dropout=0.3):
        """
        Args:
            game: Game instance with board dimensions and action space
            num_channels: Number of channels in convolutional layers
            num_res_blocks: Number of residual blocks
            dropout: Dropout rate for regularization
        """
        super(AlphaZeroNetwork, self).__init__()
        
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        
        # Initial convolutional block
        self.conv_input = nn.Conv2d(
            in_channels=game.get_num_planes(),
            out_channels=num_channels,
            kernel_size=3,
            padding=1
        )
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_x * self.board_y, self.action_size)
        self.policy_dropout = nn.Dropout(dropout)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, planes, board_x, board_y)
            
        Returns:
            policy: Log probabilities of moves (batch_size, action_size)
            value: Position evaluation (batch_size, 1)
        """
        # Initial convolution
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = self.value_dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board):
        """
        Predict policy and value for a single board state
        
        Args:
            board: Numpy array of board state
            
        Returns:
            policy: Numpy array of move probabilities
            value: Scalar evaluation
        """
        self.eval()
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                board_tensor = board_tensor.cuda()
            
            policy, value = self.forward(board_tensor)
            policy = torch.exp(policy)  # Convert log probs to probs
            
        return policy.cpu().numpy()[0], value.cpu().item()


class AlphaZeroLoss(nn.Module):
    """Combined loss function for AlphaZero training"""
    
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        
    def forward(self, policy_pred, policy_target, value_pred, value_target):
        """
        Calculate combined loss
        
        Args:
            policy_pred: Predicted log probabilities
            policy_target: Target probability distribution
            value_pred: Predicted value
            value_target: Target value
            
        Returns:
            total_loss: Combined loss
            policy_loss: Policy loss component
            value_loss: Value loss component
        """
        # Policy loss: cross-entropy
        policy_loss = -torch.sum(policy_target * policy_pred) / policy_target.size(0)
        
        # Value loss: mean squared error
        value_loss = F.mse_loss(value_pred.view(-1), value_target)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss

