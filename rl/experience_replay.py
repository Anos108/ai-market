"""
Experience Replay Buffer for Reinforcement Learning

This module provides experience replay functionality for training RL agents,
including prioritized experience replay and multi-step learning.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import random
import logging
from dataclasses import dataclass
from collections import deque
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Represents a single experience for replay."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0
    timestamp: float = 0.0


@dataclass
class MultiStepExperience:
    """Represents a multi-step experience."""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    next_states: List[np.ndarray]
    dones: List[bool]
    n_steps: int
    gamma: float
    priority: float = 1.0


class ExperienceReplay:
    """
    Experience replay buffer for RL training.
    
    This class provides various experience replay strategies including:
    - Uniform random sampling
    - Prioritized experience replay (PER)
    - Multi-step learning
    - Experience augmentation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experience replay buffer.
        
        Args:
            config: Configuration dictionary for the replay buffer
        """
        default_config = {
            'buffer_size': 100000,
            'batch_size': 64,
            'prioritized_replay': True,
            'alpha': 0.6,  # Prioritization exponent
            'beta': 0.4,   # Importance sampling exponent
            'beta_increment': 0.001,
            'epsilon': 1e-6,  # Small constant to avoid zero priorities
            'multi_step': False,
            'n_steps': 3,
            'gamma': 0.99,
            'augmentation': False,
            'noise_std': 0.01,
            'save_buffer': False,
            'buffer_path': 'buffers/experience_replay.pkl'
        }
        
        self.config = config or default_config
        
        # Buffer storage
        self.buffer = deque(maxlen=self.config['buffer_size'])
        self.priorities = deque(maxlen=self.config['buffer_size'])
        
        # Prioritized replay parameters
        self.alpha = self.config['alpha']
        self.beta = self.config['beta']
        self.beta_increment = self.config['beta_increment']
        self.epsilon = self.config['epsilon']
        
        # Multi-step parameters
        self.multi_step = self.config['multi_step']
        self.n_steps = self.config['n_steps']
        self.gamma = self.config['gamma']
        
        # Statistics
        self.total_experiences = 0
        self.sampled_experiences = 0
        
        # Buffer path
        self.buffer_path = Path(self.config['buffer_path'])
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ExperienceReplay with buffer size: {self.config['buffer_size']}")
    
    def add_experience(self, 
                      state: np.ndarray,
                      action: int,
                      reward: float,
                      next_state: np.ndarray,
                      done: bool,
                      priority: Optional[float] = None) -> None:
        """
        Add an experience to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority for prioritized replay (optional)
        """
        try:
            # Calculate priority if not provided
            if priority is None:
                if self.config['prioritized_replay']:
                    # Use TD error as priority (will be updated later)
                    priority = max(self.priorities) if self.priorities else 1.0
                else:
                    priority = 1.0
            
            # Create experience
            experience = Experience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                priority=priority,
                timestamp=len(self.buffer)
            )
            
            # Add to buffer
            self.buffer.append(experience)
            
            # Add priority for prioritized replay
            if self.config['prioritized_replay']:
                self.priorities.append(priority)
            
            self.total_experiences += 1
            
        except Exception as e:
            logger.error(f"Failed to add experience: {e}")
    
    def sample_batch(self, batch_size: Optional[int] = None) -> Tuple[List[Experience], List[float], List[int]]:
        """
        Sample a batch of experiences from the replay buffer.
        
        Args:
            batch_size: Size of batch to sample (optional)
            
        Returns:
            Tuple of (experiences, importance_weights, indices)
        """
        try:
            if len(self.buffer) == 0:
                return [], [], []
            
            batch_size = batch_size or self.config['batch_size']
            batch_size = min(batch_size, len(self.buffer))
            
            if self.config['prioritized_replay']:
                return self._sample_prioritized_batch(batch_size)
            else:
                return self._sample_uniform_batch(batch_size)
                
        except Exception as e:
            logger.error(f"Failed to sample batch: {e}")
            return [], [], []
    
    def _sample_uniform_batch(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample batch uniformly from buffer."""
        try:
            indices = random.sample(range(len(self.buffer)), batch_size)
            experiences = [self.buffer[i] for i in indices]
            weights = [1.0] * batch_size  # Uniform weights
            
            self.sampled_experiences += batch_size
            
            return experiences, weights, indices
            
        except Exception as e:
            logger.error(f"Uniform batch sampling failed: {e}")
            return [], [], []
    
    def _sample_prioritized_batch(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample batch using prioritized experience replay."""
        try:
            # Calculate sampling probabilities
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities = probabilities / np.sum(probabilities)
            
            # Sample indices based on probabilities
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=probabilities
            )
            
            # Get experiences
            experiences = [self.buffer[i] for i in indices]
            
            # Calculate importance sampling weights
            weights = []
            for i in indices:
                weight = (len(self.buffer) * probabilities[i]) ** (-self.beta)
                weights.append(weight)
            
            # Normalize weights
            max_weight = max(weights) if weights else 1.0
            weights = [w / max_weight for w in weights]
            
            # Update beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            self.sampled_experiences += batch_size
            
            return experiences, weights, indices.tolist()
            
        except Exception as e:
            logger.error(f"Prioritized batch sampling failed: {e}")
            return [], [], []
    
    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """
        Update priorities for experiences based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for each experience
        """
        try:
            if not self.config['prioritized_replay']:
                return
            
            for idx, td_error in zip(indices, td_errors):
                if 0 <= idx < len(self.priorities):
                    # Update priority based on TD error
                    priority = (abs(td_error) + self.epsilon) ** self.alpha
                    self.priorities[idx] = priority
                    
        except Exception as e:
            logger.error(f"Failed to update priorities: {e}")
    
    def get_multi_step_experience(self, start_idx: int) -> Optional[MultiStepExperience]:
        """
        Get multi-step experience starting from given index.
        
        Args:
            start_idx: Starting index in buffer
            
        Returns:
            Multi-step experience or None
        """
        try:
            if not self.multi_step or start_idx + self.n_steps >= len(self.buffer):
                return None
            
            # Collect experiences for n-steps
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            total_reward = 0.0
            gamma_power = 1.0
            
            for i in range(self.n_steps):
                exp = self.buffer[start_idx + i]
                
                states.append(exp.state)
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                dones.append(exp.done)
                
                # Calculate discounted reward
                total_reward += exp.reward * gamma_power
                gamma_power *= self.gamma
                
                # Stop if episode ends
                if exp.done:
                    break
            
            # Get final next state
            final_next_state = self.buffer[start_idx + self.n_steps - 1].next_state
            
            return MultiStepExperience(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                n_steps=len(states),
                gamma=self.gamma,
                priority=1.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get multi-step experience: {e}")
            return None
    
    def augment_experience(self, experience: Experience) -> List[Experience]:
        """
        Augment experience with noise for data augmentation.
        
        Args:
            experience: Original experience
            
        Returns:
            List of augmented experiences
        """
        try:
            if not self.config['augmentation']:
                return [experience]
            
            augmented_experiences = [experience]
            noise_std = self.config['noise_std']
            
            # Create augmented versions with noise
            for _ in range(2):  # Create 2 augmented versions
                # Add noise to state
                noisy_state = experience.state + np.random.normal(0, noise_std, experience.state.shape)
                noisy_next_state = experience.next_state + np.random.normal(0, noise_std, experience.next_state.shape)
                
                # Create augmented experience
                aug_exp = Experience(
                    state=noisy_state,
                    action=experience.action,
                    reward=experience.reward,
                    next_state=noisy_next_state,
                    done=experience.done,
                    priority=experience.priority,
                    timestamp=experience.timestamp
                )
                
                augmented_experiences.append(aug_exp)
            
            return augmented_experiences
            
        except Exception as e:
            logger.error(f"Experience augmentation failed: {e}")
            return [experience]
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        try:
            if not self.buffer:
                return {
                    'buffer_size': 0,
                    'total_experiences': 0,
                    'sampled_experiences': 0,
                    'avg_priority': 0.0,
                    'min_priority': 0.0,
                    'max_priority': 0.0,
                    'utilization': 0.0
                }
            
            priorities = list(self.priorities) if self.priorities else [1.0]
            
            return {
                'buffer_size': len(self.buffer),
                'total_experiences': self.total_experiences,
                'sampled_experiences': self.sampled_experiences,
                'avg_priority': np.mean(priorities),
                'min_priority': np.min(priorities),
                'max_priority': np.max(priorities),
                'utilization': len(self.buffer) / self.config['buffer_size']
            }
            
        except Exception as e:
            logger.error(f"Failed to get buffer stats: {e}")
            return {}
    
    def clear_buffer(self) -> None:
        """Clear the replay buffer."""
        try:
            self.buffer.clear()
            self.priorities.clear()
            self.total_experiences = 0
            self.sampled_experiences = 0
            
            logger.info("Experience replay buffer cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear buffer: {e}")
    
    def save_buffer(self) -> bool:
        """Save buffer to disk."""
        try:
            if not self.config['save_buffer']:
                return False
            
            buffer_data = {
                'buffer': list(self.buffer),
                'priorities': list(self.priorities),
                'total_experiences': self.total_experiences,
                'sampled_experiences': self.sampled_experiences,
                'config': self.config
            }
            
            with open(self.buffer_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            
            logger.info(f"Buffer saved to {self.buffer_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save buffer: {e}")
            return False
    
    def load_buffer(self) -> bool:
        """Load buffer from disk."""
        try:
            if not self.buffer_path.exists():
                logger.info(f"No saved buffer found at {self.buffer_path}")
                return False
            
            with open(self.buffer_path, 'rb') as f:
                buffer_data = pickle.load(f)
            
            self.buffer = deque(buffer_data['buffer'], maxlen=self.config['buffer_size'])
            self.priorities = deque(buffer_data['priorities'], maxlen=self.config['buffer_size'])
            self.total_experiences = buffer_data['total_experiences']
            self.sampled_experiences = buffer_data['sampled_experiences']
            
            logger.info(f"Buffer loaded from {self.buffer_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")
            return False
    
    def get_recent_experiences(self, n: int = 100) -> List[Experience]:
        """Get the most recent n experiences."""
        try:
            if len(self.buffer) == 0:
                return []
            
            n = min(n, len(self.buffer))
            return list(self.buffer)[-n:]
            
        except Exception as e:
            logger.error(f"Failed to get recent experiences: {e}")
            return []
    
    def get_experience_by_index(self, index: int) -> Optional[Experience]:
        """Get experience by index."""
        try:
            if 0 <= index < len(self.buffer):
                return self.buffer[index]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get experience by index: {e}")
            return None
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        try:
            self.config.update(new_config)
            
            # Update parameters
            self.alpha = self.config['alpha']
            self.beta = self.config['beta']
            self.beta_increment = self.config['beta_increment']
            self.epsilon = self.config['epsilon']
            self.multi_step = self.config['multi_step']
            self.n_steps = self.config['n_steps']
            self.gamma = self.config['gamma']
            
            logger.info(f"Configuration updated: {new_config}")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
