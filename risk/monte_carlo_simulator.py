"""
Monte Carlo Risk Simulation System

This module provides Monte Carlo simulation capabilities for
risk modeling and portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """Types of Monte Carlo simulations."""
    PRICE_SIMULATION = "price_simulation"
    PORTFOLIO_SIMULATION = "portfolio_simulation"
    VAR_SIMULATION = "var_simulation"
    STRESS_TEST = "stress_test"
    SCENARIO_ANALYSIS = "scenario_analysis"


class RiskMetric(Enum):
    """Risk metrics."""
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"


@dataclass
class SimulationResult:
    """Monte Carlo simulation result."""
    simulation_type: SimulationType
    n_simulations: int
    time_horizon: int
    confidence_levels: List[float]
    var_results: Dict[float, float]
    cvar_results: Dict[float, float]
    expected_returns: List[float]
    final_values: List[float]
    max_drawdowns: List[float]
    statistics: Dict[str, float]
    scenarios: Optional[pd.DataFrame] = None


@dataclass
class RiskScenario:
    """Risk scenario definition."""
    name: str
    probability: float
    market_shock: float
    volatility_multiplier: float
    correlation_change: float
    description: str


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for risk modeling.
    
    This class provides:
    - Price path simulation using various models
    - Portfolio risk simulation
    - Value at Risk (VaR) calculation
    - Stress testing scenarios
    - Scenario analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Monte Carlo Simulator.
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            'n_simulations': 10000,
            'time_horizon': 252,  # 1 year
            'confidence_levels': [0.95, 0.99, 0.999],
            'risk_free_rate': 0.02,
            'random_seed': 42,
            'models': {
                'geometric_brownian_motion': True,
                'jump_diffusion': True,
                'garch': True,
                'regime_switching': True
            },
            'scenarios': {
                'market_crash': RiskScenario('Market Crash', 0.05, -0.3, 2.0, 0.8, '30% market decline'),
                'volatility_spike': RiskScenario('Volatility Spike', 0.1, 0.0, 3.0, 0.9, '3x volatility increase'),
                'correlation_breakdown': RiskScenario('Correlation Breakdown', 0.15, 0.0, 1.5, 0.3, 'Correlation breakdown'),
                'normal_market': RiskScenario('Normal Market', 0.7, 0.0, 1.0, 1.0, 'Normal market conditions')
            }
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        np.random.seed(self.config['random_seed'])
        
        logger.info(f"Initialized MonteCarloSimulator with config: {self.config}")
    
    def simulate_price_paths(self, 
                           initial_price: float,
                           expected_return: float,
                           volatility: float,
                           time_horizon: Optional[int] = None,
                           n_simulations: Optional[int] = None,
                           model: str = 'geometric_brownian_motion') -> SimulationResult:
        """
        Simulate price paths using Monte Carlo methods.
        
        Args:
            initial_price: Initial price
            expected_return: Expected annual return
            volatility: Annual volatility
            time_horizon: Number of time steps
            n_simulations: Number of simulations
            model: Simulation model to use
            
        Returns:
            Simulation results
        """
        try:
            time_horizon = time_horizon or self.config['time_horizon']
            n_simulations = n_simulations or self.config['n_simulations']
            
            logger.info(f"Running {n_simulations} price simulations over {time_horizon} periods")
            
            if model == 'geometric_brownian_motion':
                scenarios = self._simulate_gbm(initial_price, expected_return, volatility, time_horizon, n_simulations)
            elif model == 'jump_diffusion':
                scenarios = self._simulate_jump_diffusion(initial_price, expected_return, volatility, time_horizon, n_simulations)
            elif model == 'garch':
                scenarios = self._simulate_garch(initial_price, expected_return, volatility, time_horizon, n_simulations)
            else:
                scenarios = self._simulate_gbm(initial_price, expected_return, volatility, time_horizon, n_simulations)
            
            # Calculate results
            result = self._calculate_simulation_results(scenarios, SimulationType.PRICE_SIMULATION, time_horizon, n_simulations)
            
            return result
            
        except Exception as e:
            logger.error(f"Price path simulation failed: {e}")
            return self._create_empty_result(SimulationType.PRICE_SIMULATION)
    
    def simulate_portfolio_risk(self,
                              portfolio_weights: Dict[str, float],
                              expected_returns: Dict[str, float],
                              covariance_matrix: pd.DataFrame,
                              initial_value: float = 100000,
                              time_horizon: Optional[int] = None,
                              n_simulations: Optional[int] = None) -> SimulationResult:
        """
        Simulate portfolio risk using Monte Carlo methods.
        
        Args:
            portfolio_weights: Portfolio weights for each asset
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            initial_value: Initial portfolio value
            time_horizon: Number of time steps
            n_simulations: Number of simulations
            
        Returns:
            Portfolio simulation results
        """
        try:
            time_horizon = time_horizon or self.config['time_horizon']
            n_simulations = n_simulations or self.config['n_simulations']
            
            logger.info(f"Running {n_simulations} portfolio simulations over {time_horizon} periods")
            
            # Generate correlated returns
            assets = list(portfolio_weights.keys())
            n_assets = len(assets)
            
            # Convert to numpy arrays
            weights = np.array([portfolio_weights[asset] for asset in assets])
            expected_returns_array = np.array([expected_returns[asset] for asset in assets])
            cov_matrix = covariance_matrix.values
            
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                expected_returns_array / 252,  # Daily returns
                cov_matrix / 252,  # Daily covariance
                (n_simulations, time_horizon)
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(random_returns * weights, axis=2)
            
            # Calculate portfolio values
            portfolio_values = initial_value * np.cumprod(1 + portfolio_returns, axis=1)
            
            # Create scenarios DataFrame
            scenarios = pd.DataFrame(portfolio_values)
            scenarios.columns = [f'day_{i}' for i in range(time_horizon)]
            scenarios['simulation'] = range(n_simulations)
            
            # Calculate results
            result = self._calculate_simulation_results(scenarios, SimulationType.PORTFOLIO_SIMULATION, time_horizon, n_simulations)
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio risk simulation failed: {e}")
            return self._create_empty_result(SimulationType.PORTFOLIO_SIMULATION)
    
    def calculate_var(self,
                     returns: List[float],
                     confidence_levels: Optional[List[float]] = None,
                     method: str = 'historical') -> Dict[float, float]:
        """
        Calculate Value at Risk (VaR) using various methods.
        
        Args:
            returns: Historical returns
            confidence_levels: Confidence levels for VaR calculation
            method: VaR calculation method
            
        Returns:
            VaR results for each confidence level
        """
        try:
            confidence_levels = confidence_levels or self.config['confidence_levels']
            returns_array = np.array(returns)
            
            var_results = {}
            
            for confidence_level in confidence_levels:
                if method == 'historical':
                    # Historical simulation
                    var = np.percentile(returns_array, (1 - confidence_level) * 100)
                elif method == 'parametric':
                    # Parametric method (assuming normal distribution)
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    z_score = self._get_z_score(confidence_level)
                    var = mean_return + z_score * std_return
                elif method == 'monte_carlo':
                    # Monte Carlo simulation
                    n_simulations = 10000
                    simulated_returns = np.random.normal(
                        np.mean(returns_array),
                        np.std(returns_array),
                        n_simulations
                    )
                    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                else:
                    var = np.percentile(returns_array, (1 - confidence_level) * 100)
                
                var_results[confidence_level] = var
            
            return var_results
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return {}
    
    def calculate_cvar(self,
                      returns: List[float],
                      confidence_levels: Optional[List[float]] = None) -> Dict[float, float]:
        """
        Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.
        
        Args:
            returns: Historical returns
            confidence_levels: Confidence levels for CVaR calculation
            
        Returns:
            CVaR results for each confidence level
        """
        try:
            confidence_levels = confidence_levels or self.config['confidence_levels']
            returns_array = np.array(returns)
            
            cvar_results = {}
            
            for confidence_level in confidence_levels:
                # Calculate VaR threshold
                var_threshold = np.percentile(returns_array, (1 - confidence_level) * 100)
                
                # Calculate CVaR (average of returns below VaR threshold)
                tail_returns = returns_array[returns_array <= var_threshold]
                cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
                
                cvar_results[confidence_level] = cvar
            
            return cvar_results
            
        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            return {}
    
    def run_stress_test(self,
                       portfolio_weights: Dict[str, float],
                       base_scenarios: Optional[Dict[str, RiskScenario]] = None) -> Dict[str, SimulationResult]:
        """
        Run stress tests on portfolio under various scenarios.
        
        Args:
            portfolio_weights: Portfolio weights
            base_scenarios: Stress test scenarios
            
        Returns:
            Stress test results for each scenario
        """
        try:
            scenarios = base_scenarios or self.config['scenarios']
            stress_results = {}
            
            logger.info(f"Running stress tests for {len(scenarios)} scenarios")
            
            for scenario_name, scenario in scenarios.items():
                logger.info(f"Running stress test: {scenario_name}")
                
                # Apply scenario shocks
                shocked_weights = self._apply_scenario_shocks(portfolio_weights, scenario)
                
                # Run simulation with shocked parameters
                result = self._simulate_stress_scenario(shocked_weights, scenario)
                stress_results[scenario_name] = result
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return {}
    
    def _simulate_gbm(self, initial_price: float, expected_return: float, volatility: float, 
                     time_horizon: int, n_simulations: int) -> pd.DataFrame:
        """Simulate using Geometric Brownian Motion."""
        try:
            dt = 1 / 252  # Daily time step
            drift = (expected_return - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt)
            
            # Generate random shocks
            random_shocks = np.random.normal(0, 1, (n_simulations, time_horizon))
            
            # Calculate price paths
            price_paths = np.zeros((n_simulations, time_horizon + 1))
            price_paths[:, 0] = initial_price
            
            for t in range(time_horizon):
                price_paths[:, t + 1] = price_paths[:, t] * np.exp(drift + diffusion * random_shocks[:, t])
            
            # Create DataFrame
            scenarios = pd.DataFrame(price_paths)
            scenarios.columns = [f'day_{i}' for i in range(time_horizon + 1)]
            scenarios['simulation'] = range(n_simulations)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"GBM simulation failed: {e}")
            return pd.DataFrame()
    
    def _simulate_jump_diffusion(self, initial_price: float, expected_return: float, volatility: float,
                                time_horizon: int, n_simulations: int) -> pd.DataFrame:
        """Simulate using Jump Diffusion model."""
        try:
            dt = 1 / 252
            drift = (expected_return - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt)
            
            # Jump parameters
            jump_intensity = 0.1  # Average number of jumps per year
            jump_mean = 0.0  # Mean jump size
            jump_std = 0.02  # Jump volatility
            
            price_paths = np.zeros((n_simulations, time_horizon + 1))
            price_paths[:, 0] = initial_price
            
            for t in range(time_horizon):
                # Brownian motion component
                brownian_shock = np.random.normal(0, 1, n_simulations)
                
                # Jump component
                jump_events = np.random.poisson(jump_intensity * dt, n_simulations)
                jump_shocks = np.random.normal(jump_mean, jump_std, n_simulations)
                jump_component = jump_events * jump_shocks
                
                # Update prices
                price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                    drift + diffusion * brownian_shock + jump_component
                )
            
            # Create DataFrame
            scenarios = pd.DataFrame(price_paths)
            scenarios.columns = [f'day_{i}' for i in range(time_horizon + 1)]
            scenarios['simulation'] = range(n_simulations)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Jump diffusion simulation failed: {e}")
            return pd.DataFrame()
    
    def _simulate_garch(self, initial_price: float, expected_return: float, volatility: float,
                       time_horizon: int, n_simulations: int) -> pd.DataFrame:
        """Simulate using GARCH model."""
        try:
            # Simplified GARCH(1,1) simulation
            dt = 1 / 252
            alpha = 0.1  # ARCH coefficient
            beta = 0.85  # GARCH coefficient
            omega = volatility**2 * (1 - alpha - beta)  # Long-term variance
            
            price_paths = np.zeros((n_simulations, time_horizon + 1))
            variance_paths = np.zeros((n_simulations, time_horizon + 1))
            
            price_paths[:, 0] = initial_price
            variance_paths[:, 0] = volatility**2
            
            for t in range(time_horizon):
                # Update variance
                if t > 0:
                    variance_paths[:, t] = omega + alpha * (price_paths[:, t] * np.log(price_paths[:, t] / price_paths[:, t-1]))**2 + beta * variance_paths[:, t-1]
                
                # Generate returns
                current_volatility = np.sqrt(variance_paths[:, t])
                returns = expected_return * dt + current_volatility * np.sqrt(dt) * np.random.normal(0, 1, n_simulations)
                
                # Update prices
                price_paths[:, t + 1] = price_paths[:, t] * np.exp(returns)
            
            # Create DataFrame
            scenarios = pd.DataFrame(price_paths)
            scenarios.columns = [f'day_{i}' for i in range(time_horizon + 1)]
            scenarios['simulation'] = range(n_simulations)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"GARCH simulation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_simulation_results(self, scenarios: pd.DataFrame, simulation_type: SimulationType,
                                    time_horizon: int, n_simulations: int) -> SimulationResult:
        """Calculate simulation results and statistics."""
        try:
            if scenarios.empty:
                return self._create_empty_result(simulation_type)
            
            # Extract final values
            final_values = scenarios.iloc[:, -2].values  # Second to last column (before simulation ID)
            
            # Calculate returns
            if simulation_type == SimulationType.PRICE_SIMULATION:
                initial_value = scenarios.iloc[0, 0]
                expected_returns = (final_values - initial_value) / initial_value
            else:
                initial_value = scenarios.iloc[0, 0]
                expected_returns = (final_values - initial_value) / initial_value
            
            # Calculate VaR and CVaR
            var_results = self.calculate_var(expected_returns.tolist())
            cvar_results = self.calculate_cvar(expected_returns.tolist())
            
            # Calculate max drawdowns
            max_drawdowns = []
            for i in range(n_simulations):
                path = scenarios.iloc[i, :-1].values  # Exclude simulation ID
                peak = path[0]
                max_dd = 0
                
                for value in path:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_dd = max(max_dd, drawdown)
                
                max_drawdowns.append(max_dd)
            
            # Calculate statistics
            statistics = {
                'mean_return': np.mean(expected_returns),
                'std_return': np.std(expected_returns),
                'min_return': np.min(expected_returns),
                'max_return': np.max(expected_returns),
                'median_return': np.median(expected_returns),
                'mean_final_value': np.mean(final_values),
                'std_final_value': np.std(final_values),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'max_max_drawdown': np.max(max_drawdowns)
            }
            
            return SimulationResult(
                simulation_type=simulation_type,
                n_simulations=n_simulations,
                time_horizon=time_horizon,
                confidence_levels=self.config['confidence_levels'],
                var_results=var_results,
                cvar_results=cvar_results,
                expected_returns=expected_returns.tolist(),
                final_values=final_values.tolist(),
                max_drawdowns=max_drawdowns,
                statistics=statistics,
                scenarios=scenarios
            )
            
        except Exception as e:
            logger.error(f"Simulation results calculation failed: {e}")
            return self._create_empty_result(simulation_type)
    
    def _apply_scenario_shocks(self, portfolio_weights: Dict[str, float], scenario: RiskScenario) -> Dict[str, float]:
        """Apply scenario shocks to portfolio weights."""
        try:
            shocked_weights = portfolio_weights.copy()
            
            # Apply market shock (simplified)
            for asset in shocked_weights:
                if scenario.market_shock != 0:
                    # Adjust weight based on market shock
                    shocked_weights[asset] *= (1 + scenario.market_shock * 0.1)
            
            # Normalize weights
            total_weight = sum(shocked_weights.values())
            if total_weight > 0:
                shocked_weights = {k: v / total_weight for k, v in shocked_weights.items()}
            
            return shocked_weights
            
        except Exception as e:
            logger.error(f"Scenario shock application failed: {e}")
            return portfolio_weights
    
    def _simulate_stress_scenario(self, weights: Dict[str, float], scenario: RiskScenario) -> SimulationResult:
        """Simulate a stress test scenario."""
        try:
            # Simplified stress test simulation
            n_simulations = 1000
            time_horizon = 30  # 30 days
            
            # Generate stressed returns
            stressed_returns = np.random.normal(
                scenario.market_shock / 252,  # Daily stressed return
                scenario.volatility_multiplier * 0.02,  # Stressed volatility
                (n_simulations, time_horizon)
            )
            
            # Calculate portfolio values
            initial_value = 100000
            portfolio_values = initial_value * np.cumprod(1 + stressed_returns, axis=1)
            
            # Create scenarios DataFrame
            scenarios = pd.DataFrame(portfolio_values)
            scenarios.columns = [f'day_{i}' for i in range(time_horizon)]
            scenarios['simulation'] = range(n_simulations)
            
            # Calculate results
            result = self._calculate_simulation_results(scenarios, SimulationType.STRESS_TEST, time_horizon, n_simulations)
            
            return result
            
        except Exception as e:
            logger.error(f"Stress scenario simulation failed: {e}")
            return self._create_empty_result(SimulationType.STRESS_TEST)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get Z-score for given confidence level."""
        from scipy.stats import norm
        return norm.ppf(1 - confidence_level)
    
    def _create_empty_result(self, simulation_type: SimulationType) -> SimulationResult:
        """Create empty result for failed simulations."""
        return SimulationResult(
            simulation_type=simulation_type,
            n_simulations=0,
            time_horizon=0,
            confidence_levels=self.config['confidence_levels'],
            var_results={},
            cvar_results={},
            expected_returns=[],
            final_values=[],
            max_drawdowns=[],
            statistics={}
        )
