"""
Reinforcement Learning-based Adaptive Attack Engine for Neural Operator Cryptanalysis.

This module implements a Deep Q-Network (DQN) approach to automatically optimize
attack parameters and discover novel attack strategies autonomously.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json

from .core import NeuralSCA, TraceData
from .utils.config import Config
from .utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Experience tuple for DQN replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class AttackState:
    """Represents the current state of a side-channel attack."""
    snr: float = 0.0
    success_rate: float = 0.0
    traces_used: int = 0
    confidence: float = 0.0
    preprocessing_method: str = "standardize"
    window_size: int = 1000
    window_offset: int = 0
    n_pois: int = 50
    poi_method: str = "mutual_information"
    current_step: int = 0
    target_complexity: float = 1.0
    noise_level: float = 0.1
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for RL agent."""
        categorical_encoding = {
            'standardize': 0, 'normalize': 1, 'filtering': 2, 'none': 3,
            'mutual_information': 0, 'correlation': 1, 'variance': 2, 'sosd': 3
        }
        
        return np.array([
            self.snr,
            self.success_rate, 
            self.traces_used / 10000.0,  # Normalize
            self.confidence,
            categorical_encoding.get(self.preprocessing_method, 0),
            self.window_size / 10000.0,  # Normalize
            self.window_offset / 10000.0,  # Normalize
            self.n_pois / 1000.0,  # Normalize
            categorical_encoding.get(self.poi_method, 0),
            self.current_step / 100.0,  # Normalize
            self.target_complexity,
            self.noise_level
        ], dtype=np.float32)

@dataclass
class AttackAction:
    """Represents an action the RL agent can take."""
    action_type: str  # 'adjust_window', 'change_preprocessing', 'modify_pois', 'add_traces'
    parameter: str    # Which parameter to modify
    value: float      # New value or adjustment amount
    
    @classmethod
    def from_action_id(cls, action_id: int) -> 'AttackAction':
        """Convert discrete action ID to AttackAction."""
        actions = {
            0: ('adjust_window', 'window_size', 1.2),      # Increase window size
            1: ('adjust_window', 'window_size', 0.8),      # Decrease window size
            2: ('adjust_window', 'window_offset', 100),    # Shift window forward
            3: ('adjust_window', 'window_offset', -100),   # Shift window backward
            4: ('change_preprocessing', 'method', 0),      # Standardize
            5: ('change_preprocessing', 'method', 1),      # Normalize
            6: ('change_preprocessing', 'method', 2),      # Filtering
            7: ('modify_pois', 'n_pois', 1.5),            # Increase POIs
            8: ('modify_pois', 'n_pois', 0.7),            # Decrease POIs
            9: ('modify_pois', 'poi_method', 0),          # Mutual information
            10: ('modify_pois', 'poi_method', 1),         # Correlation
            11: ('add_traces', 'traces', 1000),           # Add more traces
            12: ('add_traces', 'traces', 500),            # Add fewer traces
        }
        
        action_type, parameter, value = actions.get(action_id, ('add_traces', 'traces', 500))
        return cls(action_type, parameter, value)

class DQNNetwork(nn.Module):
    """Deep Q-Network for attack parameter optimization."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 64]
        
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class AdaptiveAttackEngine:
    """
    Reinforcement Learning-based engine for autonomous attack optimization.
    
    Uses Deep Q-Learning to automatically adjust attack parameters and discover
    optimal strategies for different targets and noise conditions.
    """
    
    def __init__(self, 
                 neural_sca: NeuralSCA,
                 state_dim: int = 12,
                 action_dim: int = 13,
                 learning_rate: float = 1e-4,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 device: str = 'cpu'):
        
        self.neural_sca = neural_sca
        self.device = torch.device(device)
        
        # DQN parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size)
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []
        
        # Attack history for learning
        self.attack_history = []
        
        logger.info(f"Initialized AdaptiveAttackEngine with {action_dim} actions")
    
    def select_action(self, state: AttackState, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def compute_reward(self, old_state: AttackState, new_state: AttackState, 
                      action: AttackAction) -> float:
        """Compute reward for state transition."""
        reward = 0.0
        
        # Primary reward: improvement in success rate
        success_improvement = new_state.success_rate - old_state.success_rate
        reward += success_improvement * 100.0
        
        # Secondary rewards
        confidence_improvement = new_state.confidence - old_state.confidence
        reward += confidence_improvement * 10.0
        
        # SNR improvement bonus
        snr_improvement = new_state.snr - old_state.snr
        reward += snr_improvement * 5.0
        
        # Efficiency penalties
        trace_penalty = (new_state.traces_used - old_state.traces_used) / 1000.0
        reward -= trace_penalty * 2.0  # Penalize using too many traces
        
        # Encourage exploration early, exploitation later
        exploration_bonus = 0.1 if old_state.current_step < 10 else 0.0
        reward += exploration_bonus
        
        # Large bonus for achieving high success rate
        if new_state.success_rate > 0.9:
            reward += 50.0
        elif new_state.success_rate > 0.7:
            reward += 20.0
        
        # Penalty for no improvement after many steps
        if old_state.current_step > 20 and new_state.success_rate < 0.3:
            reward -= 10.0
        
        return float(reward)
    
    def apply_action(self, state: AttackState, action: AttackAction) -> AttackState:
        """Apply action to current state to get new state."""
        new_state = AttackState(**state.__dict__)
        
        if action.action_type == 'adjust_window':
            if action.parameter == 'window_size':
                new_state.window_size = max(100, int(new_state.window_size * action.value))
                new_state.window_size = min(10000, new_state.window_size)
            elif action.parameter == 'window_offset':
                new_state.window_offset = max(0, int(new_state.window_offset + action.value))
                new_state.window_offset = min(5000, new_state.window_offset)
        
        elif action.action_type == 'change_preprocessing':
            methods = ['standardize', 'normalize', 'filtering', 'none']
            method_idx = int(action.value) % len(methods)
            new_state.preprocessing_method = methods[method_idx]
        
        elif action.action_type == 'modify_pois':
            if action.parameter == 'n_pois':
                new_state.n_pois = max(10, int(new_state.n_pois * action.value))
                new_state.n_pois = min(1000, new_state.n_pois)
            elif action.parameter == 'poi_method':
                methods = ['mutual_information', 'correlation', 'variance', 'sosd']
                method_idx = int(action.value) % len(methods)
                new_state.poi_method = methods[method_idx]
        
        elif action.action_type == 'add_traces':
            new_state.traces_used += int(action.value)
            new_state.traces_used = min(50000, new_state.traces_used)  # Cap at 50k traces
        
        new_state.current_step += 1
        return new_state
    
    def evaluate_attack_performance(self, state: AttackState, 
                                  traces: TraceData) -> Tuple[float, float, float]:
        """Evaluate attack performance for current state parameters."""
        try:
            # Configure neural SCA with current parameters
            self.neural_sca.config['analysis'].update({
                'preprocessing': [state.preprocessing_method],
                'poi_method': state.poi_method,
                'n_pois': state.n_pois,
                'window_size': state.window_size,
                'window_offset': state.window_offset
            })
            
            # Use subset of traces based on current state
            n_traces = min(state.traces_used, len(traces))
            subset_traces = traces[:n_traces] if n_traces > 0 else traces[:1000]
            
            # Perform attack
            results = self.neural_sca.attack(subset_traces, strategy='direct')
            
            # Calculate metrics
            success_rate = results.get('success', 0.0)
            confidence = results.get('avg_confidence', 0.0)
            
            # Estimate SNR (simplified)
            trace_data = subset_traces.traces if hasattr(subset_traces, 'traces') else subset_traces
            if len(trace_data) > 0:
                snr = float(np.var(np.mean(trace_data, axis=0)) / np.mean(np.var(trace_data, axis=1)))
            else:
                snr = 0.0
            
            return success_rate, confidence, snr
            
        except Exception as e:
            logger.warning(f"Attack evaluation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def train_episode(self, traces: TraceData, max_steps: int = 50) -> float:
        """Train one episode of attack optimization."""
        # Initialize state
        state = AttackState()
        total_reward = 0.0
        
        for step in range(max_steps):
            # Select action
            action_id = self.select_action(state)
            action = AttackAction.from_action_id(action_id)
            
            # Apply action
            new_state = self.apply_action(state, action)
            
            # Evaluate performance
            success_rate, confidence, snr = self.evaluate_attack_performance(new_state, traces)
            new_state.success_rate = success_rate
            new_state.confidence = confidence
            new_state.snr = snr
            
            # Compute reward
            reward = self.compute_reward(state, new_state, action)
            total_reward += reward
            
            # Check if episode is done
            done = (step >= max_steps - 1 or 
                   new_state.success_rate > 0.95 or
                   new_state.current_step > max_steps)
            
            # Store experience
            experience = Experience(
                state.to_vector(),
                action_id,
                reward,
                new_state.to_vector(),
                done
            )
            self.memory.push(experience)
            
            # Update state
            state = new_state
            
            if done:
                break
        
        # Train DQN if enough experiences
        if len(self.memory) > self.batch_size:
            loss = self._train_step()
            self.episode_losses.append(loss)
        
        self.episode_rewards.append(total_reward)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_step += 1
        
        logger.info(f"Episode completed: reward={total_reward:.2f}, "
                   f"final_success={state.success_rate:.3f}, epsilon={self.epsilon:.3f}")
        
        return total_reward
    
    def _train_step(self) -> float:
        """Perform one training step on the DQN."""
        experiences = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def autonomous_attack(self, traces: TraceData, 
                         target_success_rate: float = 0.9,
                         max_episodes: int = 100,
                         patience: int = 10) -> Dict[str, Any]:
        """
        Perform fully autonomous attack optimization.
        
        The RL agent will automatically discover optimal attack parameters
        without human intervention.
        """
        logger.info(f"Starting autonomous attack optimization")
        logger.info(f"Target: {target_success_rate:.1%} success rate")
        
        best_reward = float('-inf')
        best_state = None
        patience_counter = 0
        
        for episode in range(max_episodes):
            reward = self.train_episode(traces)
            
            # Check if we found a better solution
            if reward > best_reward:
                best_reward = reward
                best_state = self.get_best_state_from_history()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at episode {episode}")
                break
            
            # Check if target achieved
            if (best_state and 
                best_state.success_rate >= target_success_rate):
                logger.info(f"Target success rate achieved at episode {episode}")
                break
        
        # Final evaluation with best parameters
        if best_state:
            final_success, final_confidence, final_snr = self.evaluate_attack_performance(
                best_state, traces
            )
        else:
            final_success, final_confidence, final_snr = 0.0, 0.0, 0.0
            best_state = AttackState()
        
        results = {
            'success_rate': final_success,
            'confidence': final_confidence,
            'snr': final_snr,
            'optimal_parameters': {
                'preprocessing_method': best_state.preprocessing_method,
                'window_size': best_state.window_size,
                'window_offset': best_state.window_offset,
                'n_pois': best_state.n_pois,
                'poi_method': best_state.poi_method,
                'traces_used': best_state.traces_used
            },
            'training_episodes': len(self.episode_rewards),
            'final_reward': best_reward,
            'convergence_episode': len(self.episode_rewards) - patience_counter
        }
        
        logger.info(f"Autonomous optimization complete: "
                   f"success_rate={final_success:.3f}, "
                   f"reward={best_reward:.2f}")
        
        return results
    
    def get_best_state_from_history(self) -> AttackState:
        """Get the best performing state from training history."""
        if not self.attack_history:
            return AttackState()
        
        best_state = max(self.attack_history, 
                        key=lambda s: s.success_rate + s.confidence)
        return best_state
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'epsilon': self.epsilon
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.epsilon = checkpoint['epsilon']
        
        logger.info(f"Model loaded from {path}")

class MetaLearningAdaptiveEngine(AdaptiveAttackEngine):
    """
    Enhanced adaptive engine with meta-learning capabilities for rapid adaptation
    to new targets and implementation variants.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Meta-learning components
        self.task_embeddings = {}  # Store embeddings for different targets
        self.adaptation_history = []
        
        # Meta-network for quick adaptation
        self.meta_network = nn.Sequential(
            nn.Linear(self.state_dim + 32, 128),  # +32 for task embedding
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        ).to(self.device)
        
        self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=1e-4)
        
        logger.info("Initialized MetaLearningAdaptiveEngine")
    
    def extract_target_embedding(self, traces: TraceData) -> torch.Tensor:
        """Extract embedding representing the target characteristics."""
        # Simple target characterization based on trace statistics
        if hasattr(traces, 'traces') and len(traces.traces) > 0:
            trace_data = traces.traces[:100]  # Use first 100 traces
            
            features = [
                np.mean(trace_data),
                np.std(trace_data),
                np.var(np.mean(trace_data, axis=0)),  # Signal variance
                np.mean(np.var(trace_data, axis=1)),  # Noise variance
                len(trace_data[0]) if len(trace_data) > 0 else 0,  # Trace length
            ]
            
            # Pad or truncate to fixed size
            features = features[:32] + [0.0] * max(0, 32 - len(features))
            return torch.FloatTensor(features).to(self.device)
        else:
            return torch.zeros(32).to(self.device)
    
    def rapid_adaptation(self, traces: TraceData, 
                        adaptation_steps: int = 5) -> Dict[str, Any]:
        """
        Rapidly adapt to new target using meta-learning.
        
        This method uses few-shot learning to quickly optimize attack parameters
        for previously unseen targets.
        """
        target_embedding = self.extract_target_embedding(traces)
        
        # Initialize with meta-network predictions
        state = AttackState()
        state_vector = torch.cat([
            torch.FloatTensor(state.to_vector()).to(self.device),
            target_embedding
        ])
        
        # Get initial action suggestions from meta-network
        with torch.no_grad():
            action_probs = F.softmax(self.meta_network(state_vector), dim=0)
            initial_action = torch.multinomial(action_probs, 1).item()
        
        # Perform rapid adaptation
        adaptation_rewards = []
        best_state = state
        best_performance = 0.0
        
        for step in range(adaptation_steps):
            action = AttackAction.from_action_id(initial_action if step == 0 
                                               else self.select_action(state, training=False))
            
            new_state = self.apply_action(state, action)
            
            # Quick evaluation
            success_rate, confidence, snr = self.evaluate_attack_performance(
                new_state, traces
            )
            new_state.success_rate = success_rate
            new_state.confidence = confidence  
            new_state.snr = snr
            
            performance = success_rate + confidence
            if performance > best_performance:
                best_performance = performance
                best_state = new_state
            
            reward = self.compute_reward(state, new_state, action)
            adaptation_rewards.append(reward)
            
            state = new_state
        
        results = {
            'adapted_success_rate': best_state.success_rate,
            'adapted_confidence': best_state.confidence,
            'adaptation_steps': adaptation_steps,
            'adaptation_rewards': adaptation_rewards,
            'optimal_parameters': {
                'preprocessing_method': best_state.preprocessing_method,
                'window_size': best_state.window_size,
                'window_offset': best_state.window_offset,
                'n_pois': best_state.n_pois,
                'poi_method': best_state.poi_method
            }
        }
        
        logger.info(f"Rapid adaptation complete: "
                   f"success_rate={best_state.success_rate:.3f} "
                   f"in {adaptation_steps} steps")
        
        return results

# Factory function for easy instantiation
def create_adaptive_engine(neural_sca: NeuralSCA, 
                          engine_type: str = 'standard',
                          **kwargs) -> AdaptiveAttackEngine:
    """Create adaptive attack engine."""
    if engine_type == 'meta_learning':
        return MetaLearningAdaptiveEngine(neural_sca, **kwargs)
    else:
        return AdaptiveAttackEngine(neural_sca, **kwargs)