"""
Training Utilities for Reinforcement Learning

This module provides utilities for training RL agents including:
- Training loops and episode management
- Model evaluation and testing
- Hyperparameter optimization
- Performance monitoring and logging
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TrainingUtils:
    """
    Training utilities for RL agents.
    
    This class provides comprehensive training utilities including:
    - Training loop management
    - Performance evaluation
    - Hyperparameter optimization
    - Model checkpointing
    - Visualization and logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize training utilities.
        
        Args:
            config: Configuration dictionary for training
        """
        default_config = {
            'max_episodes': 1000,
            'max_steps_per_episode': 1000,
            'eval_frequency': 100,
            'save_frequency': 200,
            'log_frequency': 10,
            'early_stopping_patience': 50,
            'target_score': 200.0,
            'learning_rate_schedule': 'linear',
            'exploration_schedule': 'epsilon_greedy',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'plot_dir': 'plots',
            'save_best_model': True,
            'save_final_model': True,
            'verbose': True
        }
        
        self.config = config or default_config
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_scores = []
        self.training_losses = []
        self.exploration_rates = []
        
        # Performance tracking
        self.best_score = -float('inf')
        self.best_episode = 0
        self.episodes_without_improvement = 0
        
        # Create directories
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.log_dir = Path(self.config['log_dir'])
        self.plot_dir = Path(self.config['plot_dir'])
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TrainingUtils with config: {self.config}")
    
    def train_agent(self, 
                   agent,
                   environment,
                   reward_function,
                   experience_replay,
                   eval_environment: Optional[Any] = None) -> Dict[str, Any]:
        """
        Train an RL agent.
        
        Args:
            agent: RL agent to train
            environment: Training environment
            reward_function: Reward function
            experience_replay: Experience replay buffer
            eval_environment: Evaluation environment (optional)
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting RL agent training")
            
            training_start_time = datetime.now()
            
            for episode in range(self.config['max_episodes']):
                # Train one episode
                episode_result = self._train_episode(
                    agent, environment, reward_function, experience_replay, episode
                )
                
                # Update training state
                self.episode_rewards.append(episode_result['total_reward'])
                self.episode_lengths.append(episode_result['episode_length'])
                self.training_losses.append(episode_result.get('training_loss', 0.0))
                self.exploration_rates.append(episode_result.get('exploration_rate', 0.0))
                
                # Log progress
                if episode % self.config['log_frequency'] == 0:
                    self._log_training_progress(episode, episode_result)
                
                # Evaluate agent
                if episode % self.config['eval_frequency'] == 0:
                    eval_score = self._evaluate_agent(agent, eval_environment or environment)
                    self.eval_scores.append(eval_score)
                    
                    # Check for improvement
                    if eval_score > self.best_score:
                        self.best_score = eval_score
                        self.best_episode = episode
                        self.episodes_without_improvement = 0
                        
                        # Save best model
                        if self.config['save_best_model']:
                            self._save_checkpoint(agent, episode, eval_score, is_best=True)
                    else:
                        self.episodes_without_improvement += 1
                
                # Save checkpoint
                if episode % self.config['save_frequency'] == 0:
                    self._save_checkpoint(agent, episode, self.episode_rewards[-1])
                
                # Early stopping
                if self._should_early_stop():
                    logger.info(f"Early stopping at episode {episode}")
                    break
                
                # Check if target score reached
                if self.episode_rewards[-1] >= self.config['target_score']:
                    logger.info(f"Target score reached at episode {episode}")
                    break
            
            # Training completed
            training_end_time = datetime.now()
            training_duration = training_end_time - training_start_time
            
            # Save final model
            if self.config['save_final_model']:
                self._save_checkpoint(agent, episode, self.episode_rewards[-1], is_final=True)
            
            # Generate training report
            training_results = self._generate_training_report(training_duration)
            
            logger.info("RL agent training completed")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _train_episode(self, 
                      agent,
                      environment,
                      reward_function,
                      experience_replay,
                      episode: int) -> Dict[str, Any]:
        """Train one episode."""
        try:
            # Reset environment
            state = environment.reset()
            episode_reward = 0.0
            episode_length = 0
            training_loss = 0.0
            
            # Get exploration rate
            exploration_rate = self._get_exploration_rate(episode)
            
            for step in range(self.config['max_steps_per_episode']):
                # Get action from agent
                action = agent.get_action(state, exploration_rate=exploration_rate)
                
                # Execute action in environment
                next_state, reward, done, info = environment.step(action)
                
                # Calculate reward using reward function
                reward_components = reward_function.calculate_reward(
                    current_portfolio_value=info.get('portfolio_value', 0),
                    previous_portfolio_value=state.portfolio_value,
                    action=str(action),
                    market_data=info,
                    trade_result=info.get('trade_result')
                )
                
                final_reward = reward_components.total_reward
                
                # Store experience
                experience_replay.add_experience(
                    state=self._state_to_array(state),
                    action=action,
                    reward=final_reward,
                    next_state=self._state_to_array(next_state),
                    done=done
                )
                
                # Train agent if enough experiences
                if len(experience_replay.buffer) >= experience_replay.config['batch_size']:
                    loss = agent.update_model(experience_replay)
                    training_loss += loss if loss is not None else 0.0
                
                # Update state and reward
                state = next_state
                episode_reward += final_reward
                episode_length += 1
                
                # Check if episode is done
                if done:
                    break
            
            return {
                'total_reward': episode_reward,
                'episode_length': episode_length,
                'training_loss': training_loss / max(episode_length, 1),
                'exploration_rate': exploration_rate
            }
            
        except Exception as e:
            logger.error(f"Episode training failed: {e}")
            return {
                'total_reward': 0.0,
                'episode_length': 0,
                'training_loss': 0.0,
                'exploration_rate': 0.0
            }
    
    def _evaluate_agent(self, agent, environment, num_episodes: int = 5) -> float:
        """Evaluate agent performance."""
        try:
            eval_scores = []
            
            for _ in range(num_episodes):
                state = environment.reset()
                episode_reward = 0.0
                
                for step in range(self.config['max_steps_per_episode']):
                    # Get action without exploration
                    action = agent.get_action(state, exploration_rate=0.0)
                    
                    # Execute action
                    next_state, reward, done, info = environment.step(action)
                    
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                eval_scores.append(episode_reward)
            
            return np.mean(eval_scores)
            
        except Exception as e:
            logger.error(f"Agent evaluation failed: {e}")
            return 0.0
    
    def _get_exploration_rate(self, episode: int) -> float:
        """Get exploration rate for current episode."""
        try:
            if self.config['exploration_schedule'] == 'epsilon_greedy':
                # Linear decay
                start_epsilon = 1.0
                end_epsilon = 0.01
                decay_episodes = self.config['max_episodes'] // 2
                
                if episode < decay_episodes:
                    return start_epsilon - (start_epsilon - end_epsilon) * (episode / decay_episodes)
                else:
                    return end_epsilon
            
            elif self.config['exploration_schedule'] == 'exponential':
                # Exponential decay
                return 1.0 * (0.99 ** episode)
            
            else:
                return 0.1  # Fixed exploration rate
            
        except Exception as e:
            logger.error(f"Exploration rate calculation failed: {e}")
            return 0.1
    
    def _state_to_array(self, state) -> np.ndarray:
        """Convert state to numpy array."""
        try:
            # This is a simplified conversion
            # In practice, you'd need to handle different state types
            if hasattr(state, 'technical_indicators'):
                indicators = state.technical_indicators
                return np.array(list(indicators.values()))
            else:
                return np.array([state.current_price, state.portfolio_value, state.cash])
                
        except Exception as e:
            logger.error(f"State to array conversion failed: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _log_training_progress(self, episode: int, episode_result: Dict[str, Any]) -> None:
        """Log training progress."""
        try:
            if not self.config['verbose']:
                return
            
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0.0
            
            logger.info(
                f"Episode {episode}: "
                f"Reward={episode_result['total_reward']:.2f}, "
                f"Avg Reward={avg_reward:.2f}, "
                f"Length={episode_result['episode_length']}, "
                f"Avg Length={avg_length:.1f}, "
                f"Exploration={episode_result.get('exploration_rate', 0.0):.3f}"
            )
            
        except Exception as e:
            logger.error(f"Progress logging failed: {e}")
    
    def _should_early_stop(self) -> bool:
        """Check if training should stop early."""
        try:
            return self.episodes_without_improvement >= self.config['early_stopping_patience']
            
        except Exception as e:
            logger.error(f"Early stopping check failed: {e}")
            return False
    
    def _save_checkpoint(self, agent, episode: int, score: float, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        try:
            checkpoint_name = f"checkpoint_episode_{episode}_score_{score:.2f}.pkl"
            if is_best:
                checkpoint_name = f"best_model_episode_{episode}_score_{score:.2f}.pkl"
            elif is_final:
                checkpoint_name = f"final_model_episode_{episode}_score_{score:.2f}.pkl"
            
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            checkpoint_data = {
                'episode': episode,
                'score': score,
                'agent_state': agent.get_state_dict() if hasattr(agent, 'get_state_dict') else {},
                'training_state': {
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths,
                    'eval_scores': self.eval_scores,
                    'training_losses': self.training_losses,
                    'exploration_rates': self.exploration_rates
                },
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Checkpoint saving failed: {e}")
    
    def _generate_training_report(self, training_duration) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        try:
            # Calculate statistics
            total_episodes = len(self.episode_rewards)
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            max_reward = np.max(self.episode_rewards) if self.episode_rewards else 0.0
            min_reward = np.min(self.episode_rewards) if self.episode_rewards else 0.0
            std_reward = np.std(self.episode_rewards) if self.episode_rewards else 0.0
            
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
            avg_eval_score = np.mean(self.eval_scores) if self.eval_scores else 0.0
            
            # Calculate convergence metrics
            convergence_episode = self._calculate_convergence_episode()
            
            # Generate plots
            self._generate_training_plots()
            
            report = {
                'training_summary': {
                    'total_episodes': total_episodes,
                    'training_duration': str(training_duration),
                    'best_score': self.best_score,
                    'best_episode': self.best_episode,
                    'convergence_episode': convergence_episode
                },
                'performance_metrics': {
                    'avg_reward': avg_reward,
                    'max_reward': max_reward,
                    'min_reward': min_reward,
                    'std_reward': std_reward,
                    'avg_episode_length': avg_length,
                    'avg_eval_score': avg_eval_score
                },
                'training_curves': {
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths,
                    'eval_scores': self.eval_scores,
                    'training_losses': self.training_losses,
                    'exploration_rates': self.exploration_rates
                },
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save report
            report_path = self.log_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Training report saved: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Training report generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_convergence_episode(self) -> Optional[int]:
        """Calculate episode where training converged."""
        try:
            if len(self.episode_rewards) < 50:
                return None
            
            # Look for convergence in the last 20% of episodes
            window_size = max(20, len(self.episode_rewards) // 5)
            recent_rewards = self.episode_rewards[-window_size:]
            
            # Check if variance is low (converged)
            if np.std(recent_rewards) < 0.1 * np.mean(recent_rewards):
                return len(self.episode_rewards) - window_size
            
            return None
            
        except Exception as e:
            logger.error(f"Convergence calculation failed: {e}")
            return None
    
    def _generate_training_plots(self) -> None:
        """Generate training visualization plots."""
        try:
            if not self.episode_rewards:
                return
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('RL Training Progress', fontsize=16)
            
            # Episode rewards
            axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            if len(self.episode_rewards) > 10:
                # Moving average
                window = min(50, len(self.episode_rewards) // 10)
                moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
                axes[0, 0].plot(moving_avg, label=f'Moving Average ({window})', linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Episode lengths
            axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='orange')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True)
            
            # Evaluation scores
            if self.eval_scores:
                eval_episodes = np.arange(0, len(self.eval_scores)) * self.config['eval_frequency']
                axes[1, 0].plot(eval_episodes, self.eval_scores, 'o-', color='green')
                axes[1, 0].set_title('Evaluation Scores')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].grid(True)
            
            # Training losses
            if self.training_losses:
                axes[1, 1].plot(self.training_losses, alpha=0.6, color='red')
                axes[1, 1].set_title('Training Losses')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plot_dir / f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore training state
            self.episode_rewards = checkpoint_data['training_state']['episode_rewards']
            self.episode_lengths = checkpoint_data['training_state']['episode_lengths']
            self.eval_scores = checkpoint_data['training_state']['eval_scores']
            self.training_losses = checkpoint_data['training_state']['training_losses']
            self.exploration_rates = checkpoint_data['training_state']['exploration_rates']
            
            self.best_score = max(self.eval_scores) if self.eval_scores else -float('inf')
            self.best_episode = checkpoint_data['episode']
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            return {}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        try:
            return {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'eval_scores': self.eval_scores,
                'training_losses': self.training_losses,
                'exploration_rates': self.exploration_rates,
                'best_score': self.best_score,
                'best_episode': self.best_episode,
                'episodes_without_improvement': self.episodes_without_improvement
            }
            
        except Exception as e:
            logger.error(f"Training stats retrieval failed: {e}")
            return {}
    
    def reset_training_state(self) -> None:
        """Reset training state."""
        try:
            self.episode_rewards = []
            self.episode_lengths = []
            self.eval_scores = []
            self.training_losses = []
            self.exploration_rates = []
            
            self.best_score = -float('inf')
            self.best_episode = 0
            self.episodes_without_improvement = 0
            
            logger.info("Training state reset")
            
        except Exception as e:
            logger.error(f"Training state reset failed: {e}")
