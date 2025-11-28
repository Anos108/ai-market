"""
TickerRankerAgent - Market Opportunity Ranking Engine

This agent ranks ticker opportunities based on:
- Sharpe ratio and risk-adjusted returns
- Confidence metrics from multiple sources
- Risk metrics (volatility, drawdown, correlation)
- Technical strength indicators
- Fundamental factors
- Market regime alignment
- Sector rotation patterns

Part of Sprint 5-6: Market Ticker Discovery Engine
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from services.real_data_service import RealDataService
from agents.ticker_scanner_agent import ScanResult, ScanTrigger, ScanPriority

class RankingCriteria(Enum):
    """Ranking criteria types."""
    SHARPE_RATIO = "sharpe_ratio"
    CONFIDENCE = "confidence"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    TECHNICAL_STRENGTH = "technical_strength"
    FUNDAMENTAL_SCORE = "fundamental_score"
    SECTOR_MOMENTUM = "sector_momentum"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"

class RankingWeight(Enum):
    """Ranking weight categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RankingScore:
    """Individual ranking score component."""
    criteria: RankingCriteria
    score: float
    weight: float
    description: str
    metadata: Dict

@dataclass
class RankedOpportunity:
    """Ranked ticker opportunity."""
    symbol: str
    overall_score: float
    rank: int
    ranking_scores: List[RankingScore]
    scan_result: ScanResult
    recommendation: str
    risk_level: str
    expected_return: float
    max_drawdown: float
    sharpe_ratio: float
    confidence: float
    timestamp: datetime

@dataclass
class TickerRankerSummary:
    """Summary of ticker ranker performance."""
    total_ranked: int
    top_opportunities: List[RankedOpportunity]
    sector_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    avg_expected_return: float
    avg_sharpe_ratio: float
    avg_confidence: float
    ranking_duration: float
    last_ranking: datetime
    criteria_weights: Dict[str, float]

class TickerRankerAgent:
    """Agent for ranking ticker opportunities based on multiple criteria."""
    
    def __init__(self, real_data_service: Optional[RealDataService] = None):
        """Initialize the ticker ranker agent."""
        self.real_data_service = real_data_service
        self.ranked_opportunities: List[RankedOpportunity] = []
        self.ranking_history: List[RankedOpportunity] = []
        
        # Ranking criteria weights
        self.criteria_weights = {
            RankingCriteria.SHARPE_RATIO: 0.25,
            RankingCriteria.CONFIDENCE: 0.20,
            RankingCriteria.RISK_ADJUSTED_RETURN: 0.20,
            RankingCriteria.TECHNICAL_STRENGTH: 0.15,
            RankingCriteria.FUNDAMENTAL_SCORE: 0.10,
            RankingCriteria.SECTOR_MOMENTUM: 0.05,
            RankingCriteria.VOLATILITY: 0.03,
            RankingCriteria.LIQUIDITY: 0.02
        }
        
        # Risk thresholds
        self.low_risk_threshold = 0.3
        self.medium_risk_threshold = 0.6
        self.high_risk_threshold = 0.8
        
        # Expected return thresholds
        self.low_return_threshold = 0.05
        self.medium_return_threshold = 0.10
        self.high_return_threshold = 0.15
        
        logger.info("TickerRankerAgent initialized with %d ranking criteria", len(self.criteria_weights))
    
    async def rank_opportunities(self, scan_results: List[ScanResult]) -> List[RankedOpportunity]:
        """Rank ticker opportunities based on multiple criteria."""
        start_time = datetime.now()
        logger.info("Starting opportunity ranking for %d tickers", len(scan_results))
        
        ranked_opportunities = []
        
        for i, scan_result in enumerate(scan_results):
            try:
                ranked_opportunity = await self._rank_opportunity(scan_result, i + 1)
                if ranked_opportunity:
                    ranked_opportunities.append(ranked_opportunity)
            except Exception as e:
                logger.warning("Error ranking opportunity %s: %s", scan_result.symbol, e)
                continue
        
        # Sort by overall score
        ranked_opportunities.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Update ranks
        for i, opportunity in enumerate(ranked_opportunities):
            opportunity.rank = i + 1
        
        # Store results
        self.ranked_opportunities = ranked_opportunities
        self.ranking_history.extend(ranked_opportunities)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(days=7)
        self.ranking_history = [r for r in self.ranking_history if r.timestamp > cutoff_time]
        
        ranking_duration = (datetime.now() - start_time).total_seconds()
        logger.info("Opportunity ranking completed in %.2fs, ranked %d opportunities", 
                   ranking_duration, len(ranked_opportunities))
        
        return ranked_opportunities
    
    async def _rank_opportunity(self, scan_result: ScanResult, initial_rank: int) -> Optional[RankedOpportunity]:
        """Rank a single opportunity."""
        try:
            symbol = scan_result.symbol
            
            # Get market data for analysis
            data = await self._get_market_data(symbol, periods=100)
            if data is None or len(data) < 50:
                logger.warning("Insufficient data for ranking %s: %d", symbol, len(data) if data is not None else 0)
                return None
            
            # Calculate ranking scores
            ranking_scores = []
            
            # Sharpe ratio score
            sharpe_score = self._calculate_sharpe_score(symbol, data)
            ranking_scores.append(sharpe_score)
            
            # Confidence score
            confidence_score = self._calculate_confidence_score(scan_result)
            ranking_scores.append(confidence_score)
            
            # Risk-adjusted return score
            risk_adjusted_score = self._calculate_risk_adjusted_return_score(symbol, data)
            ranking_scores.append(risk_adjusted_score)
            
            # Technical strength score
            technical_score = self._calculate_technical_strength_score(symbol, data)
            ranking_scores.append(technical_score)
            
            # Fundamental score
            fundamental_score = self._calculate_fundamental_score(symbol, data)
            ranking_scores.append(fundamental_score)
            
            # Sector momentum score
            sector_score = self._calculate_sector_momentum_score(symbol, data)
            ranking_scores.append(sector_score)
            
            # Volatility score
            volatility_score = self._calculate_volatility_score(symbol, data)
            ranking_scores.append(volatility_score)
            
            # Liquidity score
            liquidity_score = self._calculate_liquidity_score(symbol, data)
            ranking_scores.append(liquidity_score)
            
            # Calculate overall score
            overall_score = sum(score.score * score.weight for score in ranking_scores)
            
            # Calculate risk metrics
            expected_return = self._calculate_expected_return(symbol, data)
            max_drawdown = self._calculate_max_drawdown(symbol, data)
            sharpe_ratio = self._calculate_sharpe_ratio(symbol, data)
            confidence = np.mean([score.score for score in ranking_scores])
            
            # Determine risk level
            risk_level = self._determine_risk_level(volatility_score.score, max_drawdown)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(overall_score, risk_level, expected_return)
            
            return RankedOpportunity(
                symbol=symbol,
                overall_score=overall_score,
                rank=initial_rank,
                ranking_scores=ranking_scores,
                scan_result=scan_result,
                recommendation=recommendation,
                risk_level=risk_level,
                expected_return=expected_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error("Error ranking opportunity %s: %s", scan_result.symbol, e)
            return None
    
    def _calculate_sharpe_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate Sharpe ratio score."""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return RankingScore(
                    criteria=RankingCriteria.SHARPE_RATIO,
                    score=0.5,
                    weight=self.criteria_weights[RankingCriteria.SHARPE_RATIO],
                    description="Insufficient data for Sharpe calculation",
                    metadata={'sharpe_ratio': 0.0, 'returns_count': len(returns)}
                )
            
            # Calculate Sharpe ratio
            mean_return = returns.mean() * 252  # Annualized
            std_return = returns.std() * np.sqrt(252)  # Annualized
            risk_free_rate = 0.02  # 2% risk-free rate
            
            sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
            
            # Normalize score (0-1 scale)
            # Good Sharpe ratio is > 1, excellent is > 2
            score = min(1.0, max(0.0, (sharpe_ratio + 1) / 3))  # Map -1 to 2 range to 0-1
            
            return RankingScore(
                criteria=RankingCriteria.SHARPE_RATIO,
                score=score,
                weight=self.criteria_weights[RankingCriteria.SHARPE_RATIO],
                description=f"Sharpe ratio: {sharpe_ratio:.2f}",
                metadata={'sharpe_ratio': sharpe_ratio, 'mean_return': mean_return, 'std_return': std_return}
            )
            
        except Exception as e:
            logger.warning("Error calculating Sharpe score for %s: %s", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.SHARPE_RATIO,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.SHARPE_RATIO],
                description="Error calculating Sharpe ratio",
                metadata={'error': str(e)}
            )
    
    def _calculate_confidence_score(self, scan_result: ScanResult) -> RankingScore:
        """Calculate confidence score from scan result."""
        try:
            # Use scan result confidence as base
            base_confidence = scan_result.confidence
            
            # Adjust based on priority
            priority_multiplier = {
                ScanPriority.HIGH: 1.2,
                ScanPriority.MEDIUM: 1.0,
                ScanPriority.LOW: 0.8
            }
            
            adjusted_confidence = base_confidence * priority_multiplier.get(scan_result.priority, 1.0)
            score = min(1.0, adjusted_confidence)
            
            return RankingScore(
                criteria=RankingCriteria.CONFIDENCE,
                score=score,
                weight=self.criteria_weights[RankingCriteria.CONFIDENCE],
                description=f"Confidence: {score:.2f} (priority: {scan_result.priority.value})",
                metadata={'base_confidence': base_confidence, 'priority': scan_result.priority.value}
            )
            
        except Exception as e:
            logger.warning("Error calculating confidence score: %s", e)
            return RankingScore(
                criteria=RankingCriteria.CONFIDENCE,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.CONFIDENCE],
                description="Error calculating confidence",
                metadata={'error': str(e)}
            )
    
    def _calculate_risk_adjusted_return_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate risk-adjusted return score."""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return RankingScore(
                    criteria=RankingCriteria.RISK_ADJUSTED_RETURN,
                    score=0.5,
                    weight=self.criteria_weights[RankingCriteria.RISK_ADJUSTED_RETURN],
                    description="Insufficient data for risk-adjusted return",
                    metadata={'returns_count': len(returns)}
                )
            
            # Calculate risk-adjusted return (return / volatility)
            mean_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            risk_adjusted_return = mean_return / volatility if volatility > 0 else 0
            
            # Normalize score (0-1 scale)
            # Good risk-adjusted return is > 0.5, excellent is > 1.0
            score = min(1.0, max(0.0, risk_adjusted_return / 2.0))
            
            return RankingScore(
                criteria=RankingCriteria.RISK_ADJUSTED_RETURN,
                score=score,
                weight=self.criteria_weights[RankingCriteria.RISK_ADJUSTED_RETURN],
                description=f"Risk-adjusted return: {risk_adjusted_return:.2f}",
                metadata={'risk_adjusted_return': risk_adjusted_return, 'mean_return': mean_return, 'volatility': volatility}
            )
            
        except Exception as e:
            logger.warning("Error calculating risk-adjusted return score for {}: {}", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.RISK_ADJUSTED_RETURN,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.RISK_ADJUSTED_RETURN],
                description="Error calculating risk-adjusted return",
                metadata={'error': str(e)}
            )
    
    def _calculate_technical_strength_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate technical strength score."""
        try:
            if len(data) < 50:
                return RankingScore(
                    criteria=RankingCriteria.TECHNICAL_STRENGTH,
                    score=0.5,
                    weight=self.criteria_weights[RankingCriteria.TECHNICAL_STRENGTH],
                    description="Insufficient data for technical analysis",
                    metadata={'data_length': len(data)}
                )
            
            # Calculate technical indicators
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            current_price = data['close'].iloc[-1]
            
            # Price above moving averages
            price_above_sma20 = current_price > sma_20.iloc[-1]
            price_above_sma50 = current_price > sma_50.iloc[-1]
            
            # RSI calculation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # MACD calculation
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            macd_histogram = macd - signal
            
            # Technical score components
            ma_score = 0.4 if price_above_sma20 else 0.2
            ma_score += 0.3 if price_above_sma50 else 0.1
            
            rsi_score = 0.3 if 30 < current_rsi < 70 else 0.1  # Good RSI range
            
            macd_score = 0.3 if macd_histogram.iloc[-1] > 0 else 0.1  # Positive MACD histogram
            
            total_score = ma_score + rsi_score + macd_score
            
            return RankingScore(
                criteria=RankingCriteria.TECHNICAL_STRENGTH,
                score=total_score,
                weight=self.criteria_weights[RankingCriteria.TECHNICAL_STRENGTH],
                description=f"Technical strength: {total_score:.2f} (RSI: {current_rsi:.1f})",
                metadata={'rsi': current_rsi, 'price_above_sma20': price_above_sma20, 'price_above_sma50': price_above_sma50}
            )
            
        except Exception as e:
            logger.warning("Error calculating technical strength score for {}: {}", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.TECHNICAL_STRENGTH,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.TECHNICAL_STRENGTH],
                description="Error calculating technical strength",
                metadata={'error': str(e)}
            )
    
    def _calculate_fundamental_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate fundamental score (mock implementation)."""
        try:
            # Mock fundamental analysis
            # In a real implementation, this would integrate with fundamental data APIs
            
            # Simulate fundamental metrics
            np.random.seed(hash(symbol) % 2**32)
            
            # Mock P/E ratio, growth rate, etc.
            pe_ratio = np.random.uniform(10, 30)
            growth_rate = np.random.uniform(-0.1, 0.3)
            debt_ratio = np.random.uniform(0.1, 0.6)
            
            # Calculate fundamental score
            pe_score = 0.4 if 15 < pe_ratio < 25 else 0.2  # Good P/E range
            growth_score = 0.3 if growth_rate > 0.1 else 0.1  # Good growth
            debt_score = 0.3 if debt_ratio < 0.4 else 0.1  # Low debt
            
            total_score = pe_score + growth_score + debt_score
            
            return RankingScore(
                criteria=RankingCriteria.FUNDAMENTAL_SCORE,
                score=total_score,
                weight=self.criteria_weights[RankingCriteria.FUNDAMENTAL_SCORE],
                description=f"Fundamental score: {total_score:.2f} (P/E: {pe_ratio:.1f})",
                metadata={'pe_ratio': pe_ratio, 'growth_rate': growth_rate, 'debt_ratio': debt_ratio}
            )
            
        except Exception as e:
            logger.warning("Error calculating fundamental score for {}: {}", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.FUNDAMENTAL_SCORE,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.FUNDAMENTAL_SCORE],
                description="Error calculating fundamental score",
                metadata={'error': str(e)}
            )
    
    def _calculate_sector_momentum_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate sector momentum score (mock implementation)."""
        try:
            # Mock sector momentum analysis
            # In a real implementation, this would analyze sector rotation patterns
            
            np.random.seed(hash(symbol) % 2**32)
            
            # Simulate sector momentum
            sector_momentum = np.random.uniform(-0.2, 0.3)
            
            # Normalize score
            score = min(1.0, max(0.0, (sector_momentum + 0.2) / 0.5))
            
            return RankingScore(
                criteria=RankingCriteria.SECTOR_MOMENTUM,
                score=score,
                weight=self.criteria_weights[RankingCriteria.SECTOR_MOMENTUM],
                description=f"Sector momentum: {sector_momentum:.2f}",
                metadata={'sector_momentum': sector_momentum}
            )
            
        except Exception as e:
            logger.warning("Error calculating sector momentum score for {}: {}", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.SECTOR_MOMENTUM,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.SECTOR_MOMENTUM],
                description="Error calculating sector momentum",
                metadata={'error': str(e)}
            )
    
    def _calculate_volatility_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate volatility score (lower is better for this criteria)."""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return RankingScore(
                    criteria=RankingCriteria.VOLATILITY,
                    score=0.5,
                    weight=self.criteria_weights[RankingCriteria.VOLATILITY],
                    description="Insufficient data for volatility calculation",
                    metadata={'returns_count': len(returns)}
                )
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Lower volatility is better, so invert the score
            # Good volatility is < 20%, excellent is < 15%
            score = max(0.0, 1.0 - (volatility - 0.15) / 0.15) if volatility > 0.15 else 1.0
            score = min(1.0, score)
            
            return RankingScore(
                criteria=RankingCriteria.VOLATILITY,
                score=score,
                weight=self.criteria_weights[RankingCriteria.VOLATILITY],
                description=f"Volatility: {volatility:.2%}",
                metadata={'volatility': volatility}
            )
            
        except Exception as e:
            logger.warning("Error calculating volatility score for {}: {}", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.VOLATILITY,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.VOLATILITY],
                description="Error calculating volatility",
                metadata={'error': str(e)}
            )
    
    def _calculate_liquidity_score(self, symbol: str, data: pd.DataFrame) -> RankingScore:
        """Calculate liquidity score."""
        try:
            if len(data) < 20:
                return RankingScore(
                    criteria=RankingCriteria.LIQUIDITY,
                    score=0.5,
                    weight=self.criteria_weights[RankingCriteria.LIQUIDITY],
                    description="Insufficient data for liquidity calculation",
                    metadata={'data_length': len(data)}
                )
            
            # Calculate average volume
            avg_volume = data['volume'].mean()
            
            # Higher volume is better
            # Good liquidity is > 1M shares, excellent is > 5M shares
            score = min(1.0, avg_volume / 5000000)  # Normalize to 5M shares
            
            return RankingScore(
                criteria=RankingCriteria.LIQUIDITY,
                score=score,
                weight=self.criteria_weights[RankingCriteria.LIQUIDITY],
                description=f"Avg volume: {avg_volume:,.0f}",
                metadata={'avg_volume': avg_volume}
            )
            
        except Exception as e:
            logger.warning("Error calculating liquidity score for {}: {}", symbol, e)
            return RankingScore(
                criteria=RankingCriteria.LIQUIDITY,
                score=0.5,
                weight=self.criteria_weights[RankingCriteria.LIQUIDITY],
                description="Error calculating liquidity",
                metadata={'error': str(e)}
            )
    
    def _calculate_expected_return(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate expected return."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < 20:
                return 0.05  # Default 5% return
            
            # Annualized expected return
            expected_return = returns.mean() * 252
            return max(0.0, expected_return)  # Ensure non-negative
            
        except Exception as e:
            logger.warning("Error calculating expected return for {}: {}", symbol, e)
            return 0.05
    
    def _calculate_max_drawdown(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < 20:
                return 0.1  # Default 10% drawdown
            
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = abs(drawdown.min())
            return max_drawdown
            
        except Exception as e:
            logger.warning("Error calculating max drawdown for {}: {}", symbol, e)
            return 0.1
    
    def _calculate_sharpe_ratio(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio."""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < 20:
                return 0.5  # Default Sharpe ratio
            
            mean_return = returns.mean() * 252
            std_return = returns.std() * np.sqrt(252)
            risk_free_rate = 0.02
            
            sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
            return sharpe_ratio
            
        except Exception as e:
            logger.warning("Error calculating Sharpe ratio for {}: {}", symbol, e)
            return 0.5
    
    def _determine_risk_level(self, volatility_score: float, max_drawdown: float) -> str:
        """Determine risk level based on volatility and drawdown."""
        try:
            # Combine volatility and drawdown for risk assessment
            risk_score = (1 - volatility_score) * 0.6 + max_drawdown * 0.4
            
            if risk_score < self.low_risk_threshold:
                return "Low"
            elif risk_score < self.medium_risk_threshold:
                return "Medium"
            else:
                return "High"
                
        except Exception as e:
            logger.warning("Error determining risk level: {}", e)
            return "Medium"
    
    def _generate_recommendation(self, overall_score: float, risk_level: str, expected_return: float) -> str:
        """Generate investment recommendation."""
        try:
            if overall_score > 0.8 and risk_level == "Low":
                return "Strong Buy"
            elif overall_score > 0.7 and risk_level in ["Low", "Medium"]:
                return "Buy"
            elif overall_score > 0.6 and expected_return > 0.1:
                return "Buy"
            elif overall_score > 0.5:
                return "Hold"
            elif overall_score > 0.3:
                return "Watch"
            else:
                return "Avoid"
                
        except Exception as e:
            logger.warning("Error generating recommendation: {}", e)
            return "Hold"
    
    async def _get_market_data(self, symbol: str, periods: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for analysis."""
        try:
            if not self.real_data_service:
                logger.warning("Real data service not available, using mock data")
                return self._generate_mock_data(symbol, periods)
            
            # Get historical data
            data = self.real_data_service.get_price_history(symbol, days=periods)
            if data is None or data.empty:
                logger.warning(f"No real data available for {symbol}, using mock data")
                return self._generate_mock_data(symbol, periods)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            logger.warning(f"Using mock data for {symbol}")
            return self._generate_mock_data(symbol, periods)
    
    def _generate_mock_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate mock market data for testing."""
        try:
            # Base prices for different symbols
            base_prices = {
                'BTC-USD': 69000.0, 'SOXL': 45.0, 'NVDA': 480.0, 'RIVN': 15.0, 'TSLA': 250.0, 'SPY': 450.0,
                'META': 300.0, 'TQQQ': 120.0, 'SPXL': 80.0, 'AMD': 110.0,
                'JPM': 150.0, 'BAC': 30.0, 'WFC': 45.0, 'GS': 400.0, 'MS': 90.0,
                'JNJ': 160.0, 'PFE': 30.0, 'UNH': 500.0, 'ABBV': 150.0, 'MRK': 120.0,
                'XOM': 110.0, 'CVX': 150.0, 'COP': 120.0, 'EOG': 130.0, 'SLB': 50.0,
                'PG': 150.0, 'KO': 60.0, 'PEP': 170.0, 'WMT': 160.0, 'HD': 350.0,
                'BA': 200.0, 'CAT': 250.0, 'GE': 100.0, 'HON': 200.0, 'MMM': 100.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate time series data
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            # Generate price data with some trend and volatility
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, periods)  # 0.1% daily return, 2% volatility
            prices = [base_price]
            
            for i in range(1, periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(new_price)
            
            # Generate OHLCV data
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                # Generate OHLC from close price
                volatility = 0.01  # 1% intraday volatility
                high = close * (1 + np.random.uniform(0, volatility))
                low = close * (1 - np.random.uniform(0, volatility))
                open_price = close * (1 + np.random.uniform(-volatility/2, volatility/2))
                
                # Ensure OHLC consistency
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Generate volume
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            return None
    
    async def get_ranking_summary(self) -> TickerRankerSummary:
        """Get summary of ticker ranker performance."""
        try:
            total_ranked = len(self.ranked_opportunities)
            
            top_opportunities = sorted(self.ranked_opportunities, key=lambda x: x.overall_score, reverse=True)[:10]
            
            # Sector distribution
            sector_distribution = {}
            for opportunity in self.ranked_opportunities:
                sector = opportunity.scan_result.metadata.get('sector', 'Unknown')
                sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
            
            # Risk distribution
            risk_distribution = {}
            for opportunity in self.ranked_opportunities:
                risk_level = opportunity.risk_level
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            avg_expected_return = np.mean([o.expected_return for o in self.ranked_opportunities]) if self.ranked_opportunities else 0.0
            avg_sharpe_ratio = np.mean([o.sharpe_ratio for o in self.ranked_opportunities]) if self.ranked_opportunities else 0.0
            avg_confidence = np.mean([o.confidence for o in self.ranked_opportunities]) if self.ranked_opportunities else 0.0
            
            return TickerRankerSummary(
                total_ranked=total_ranked,
                top_opportunities=top_opportunities,
                sector_distribution=sector_distribution,
                risk_distribution=risk_distribution,
                avg_expected_return=avg_expected_return,
                avg_sharpe_ratio=avg_sharpe_ratio,
                avg_confidence=avg_confidence,
                ranking_duration=0.0,  # Will be updated by caller
                last_ranking=datetime.now(),
                criteria_weights={k.value: v for k, v in self.criteria_weights.items()}
            )
            
        except Exception as e:
            logger.error(f"Error getting ranking summary: {e}")
            return TickerRankerSummary(
                total_ranked=0,
                top_opportunities=[],
                sector_distribution={},
                risk_distribution={},
                avg_expected_return=0.0,
                avg_sharpe_ratio=0.0,
                avg_confidence=0.0,
                ranking_duration=0.0,
                last_ranking=datetime.now(),
                criteria_weights={}
            )
    
    def get_top_ranked_opportunities(self, limit: int = 5) -> List[RankedOpportunity]:
        """Get top ranked opportunities."""
        return sorted(self.ranked_opportunities, key=lambda x: x.overall_score, reverse=True)[:limit]
    
    def get_opportunities_by_risk_level(self, risk_level: str) -> List[RankedOpportunity]:
        """Get opportunities filtered by risk level."""
        return [o for o in self.ranked_opportunities if o.risk_level == risk_level]
    
    def get_opportunities_by_recommendation(self, recommendation: str) -> List[RankedOpportunity]:
        """Get opportunities filtered by recommendation."""
        return [o for o in self.ranked_opportunities if o.recommendation == recommendation]
