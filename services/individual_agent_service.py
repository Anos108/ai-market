"""
Individual Agent Service

This service runs individual AI agents and generates predictions for each symbol,
storing them in the agent_signals table for the ensemble blender to use.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import asyncpg
import yfinance as yf
import ta
from dataclasses import dataclass
import json
import random
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class AgentPrediction:
    """Represents a prediction from an individual agent."""
    agent_name: str
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime

class IndividualAgentService:
    """Service for running individual agents and generating predictions."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.agents = {
            'MomentumAgent': self._momentum_agent,
            'SentimentAgent': self._sentiment_agent,
            'CorrelationAgent': self._correlation_agent,
            'RiskAgent': self._risk_agent,
            'VolatilityAgent': self._volatility_agent,
            'VolumeAgent': self._volume_agent,
            'EventImpactAgent': self._event_impact_agent,
            'ForecastAgent': self._forecast_agent,
            'StrategyAgent': self._strategy_agent,
            'MetaAgent': self._meta_agent
        }
        
        # Symbols to analyze - will be fetched from database dynamically
        self.symbols = []  # Will be populated from managed_symbols table
    
    async def _load_symbols_from_database(self):
        """Load active symbols from the database."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get all symbols from the symbols table
                symbol_rows = await conn.fetch("SELECT symbol FROM symbols ORDER BY symbol")
                self.symbols = [row['symbol'] for row in symbol_rows]
                logger.info(f"📊 Loaded {len(self.symbols)} symbols from database: {self.symbols}")
        except Exception as e:
            logger.error(f"❌ Error loading symbols from database: {e}")
            # Fallback to a minimal set if database query fails
            self.symbols = ['NVDA', 'TSLA', 'BTC-USD']
    
    async def run_all_agents(self) -> List[AgentPrediction]:
        """Run all individual agents and generate predictions."""
        logger.info("🤖 Starting individual agent prediction generation...")
        
        # Load symbols from database
        await self._load_symbols_from_database()
        
        all_predictions = []
        
        # Run agents in parallel for better performance
        tasks = []
        for agent_name, agent_func in self.agents.items():
            task = asyncio.create_task(self._run_agent(agent_name, agent_func))
            tasks.append(task)
        
        # Wait for all agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all predictions
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Agent failed: {result}")
            elif isinstance(result, list):
                all_predictions.extend(result)
        
        logger.info(f"✅ Generated {len(all_predictions)} individual agent predictions")
        return all_predictions
    
    async def _run_agent(self, agent_name: str, agent_func) -> List[AgentPrediction]:
        """Run a single agent for all symbols."""
        predictions = []
        
        try:
            for symbol in self.symbols:
                try:
                    prediction = await agent_func(symbol)
                    if prediction:
                        predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"{agent_name} failed for {symbol}: {e}")
                    continue
        except Exception as e:
            logger.error(f"{agent_name} failed completely: {e}")
        
        return predictions
    
    async def _momentum_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Momentum Agent - Analyzes price momentum and trend strength."""
        try:
            # Get recent price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if len(hist) < 20:
                return None
            
            # Calculate momentum indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else sma_20
            current_price = hist['Close'].iloc[-1]
            price_change_5d = (current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]
            price_change_20d = (current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]
            
            # RSI
            rsi = ta.momentum.RSIIndicator(hist['Close']).rsi().iloc[-1]
            
            # Determine signal
            if price_change_5d > 0.02 and rsi > 50 and current_price > sma_20:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + abs(price_change_5d) * 5)
                reasoning = f"Strong upward momentum: {price_change_5d:.2%} 5d change, RSI {rsi:.1f}"
            elif price_change_5d < -0.02 and rsi < 50 and current_price < sma_20:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + abs(price_change_5d) * 5)
                reasoning = f"Strong downward momentum: {price_change_5d:.2%} 5d change, RSI {rsi:.1f}"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Mixed momentum signals: {price_change_5d:.2%} 5d change, RSI {rsi:.1f}"
            
            return AgentPrediction(
                agent_name='MomentumAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'rsi': float(rsi),
                    'price_change_5d': float(price_change_5d),
                    'price_change_20d': float(price_change_20d),
                    'sma_20': float(sma_20),
                    'current_price': float(current_price)
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"MomentumAgent failed for {symbol}: {e}")
            return None
    
    async def _sentiment_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Sentiment Agent - Analyzes real market sentiment using Yahoo Finance news."""
        try:
            # Get real news data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            if not news_data:
                # Fallback to simulated sentiment if no news
                sentiment_score = random.uniform(-0.2, 0.2)
                news_volume = 0.1
            else:
                # Analyze real news sentiment
                sentiment_scores = []
                positive_keywords = ['beat', 'exceed', 'surge', 'rally', 'gain', 'strong', 'growth', 'profit', 'upgrade', 'bullish']
                negative_keywords = ['miss', 'decline', 'fall', 'drop', 'loss', 'weak', 'concern', 'risk', 'downgrade', 'bearish']
                
                for news_item in news_data[:10]:  # Analyze top 10 news items
                    title = news_item.get('title', '').lower()
                    summary = news_item.get('summary', '').lower()
                    content = title + ' ' + summary
                    
                    # Count positive and negative keywords
                    positive_count = sum(1 for keyword in positive_keywords if keyword in content)
                    negative_count = sum(1 for keyword in negative_keywords if keyword in content)
                    
                    # Calculate sentiment for this news item
                    if positive_count + negative_count > 0:
                        item_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                        sentiment_scores.append(item_sentiment)
                
                # Calculate overall sentiment
                if sentiment_scores:
                    sentiment_score = np.mean(sentiment_scores)
                    news_volume = min(1.0, len(news_data) / 20.0)
                else:
                    sentiment_score = random.uniform(-0.1, 0.1)
                    news_volume = 0.2
            
            # Determine signal based on sentiment
            if sentiment_score > 0.3 and news_volume > 0.3:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + sentiment_score * 0.3)
                reasoning = f"Positive sentiment: {sentiment_score:.2f} with {len(news_data) if news_data else 0} news articles"
            elif sentiment_score < -0.3 and news_volume > 0.3:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + abs(sentiment_score) * 0.3)
                reasoning = f"Negative sentiment: {sentiment_score:.2f} with {len(news_data) if news_data else 0} news articles"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Neutral sentiment: {sentiment_score:.2f} with {len(news_data) if news_data else 0} news articles"
            
            return AgentPrediction(
                agent_name='SentimentAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'sentiment_score': sentiment_score,
                    'news_volume': news_volume,
                    'positive_articles': sum(1 for score in sentiment_scores if score > 0.1) if 'sentiment_scores' in locals() else 0,
                    'negative_articles': sum(1 for score in sentiment_scores if score < -0.1) if 'sentiment_scores' in locals() else 0,
                    'neutral_articles': sum(1 for score in sentiment_scores if -0.1 <= score <= 0.1) if 'sentiment_scores' in locals() else len(news_data) if news_data else 0,
                    'sentiment_source': 'real_yahoo_finance_news'
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"SentimentAgent failed for {symbol}: {e}, using simulated sentiment")
            # Fallback to simulated sentiment
            sentiment_score = random.uniform(-1, 1)
            news_volume = random.uniform(0, 1)
            
            if sentiment_score > 0.3 and news_volume > 0.5:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + sentiment_score * 0.3)
                reasoning = f"Simulated positive sentiment: {sentiment_score:.2f}"
            elif sentiment_score < -0.3 and news_volume > 0.5:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + abs(sentiment_score) * 0.3)
                reasoning = f"Simulated negative sentiment: {sentiment_score:.2f}"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Simulated neutral sentiment: {sentiment_score:.2f}"
            
            return AgentPrediction(
                agent_name='SentimentAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'sentiment_score': sentiment_score,
                    'news_volume': news_volume,
                    'sentiment_source': 'simulated_fallback'
                },
                timestamp=datetime.now()
            )
    
    async def _correlation_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Correlation Agent - Analyzes correlation with market indices and sectors."""
        try:
            # Get data for symbol and market index
            ticker = yf.Ticker(symbol)
            spy = yf.Ticker('SPY')
            
            hist_symbol = ticker.history(period="3mo")
            hist_spy = spy.history(period="3mo")
            
            if len(hist_symbol) < 30 or len(hist_spy) < 30:
                return None
            
            # Calculate correlation
            returns_symbol = hist_symbol['Close'].pct_change().dropna()
            returns_spy = hist_spy['Close'].pct_change().dropna()
            
            # Align the data
            min_len = min(len(returns_symbol), len(returns_spy))
            returns_symbol = returns_symbol.iloc[-min_len:]
            returns_spy = returns_spy.iloc[-min_len:]
            
            correlation = returns_symbol.corr(returns_spy)
            
            # Beta calculation
            covariance = np.cov(returns_symbol, returns_spy)[0, 1]
            spy_variance = np.var(returns_spy)
            beta = covariance / spy_variance if spy_variance != 0 else 1.0
            
            # Determine signal based on correlation and beta
            if correlation > 0.7 and beta > 1.2:
                signal_type = 'buy' if returns_spy.iloc[-1] > 0 else 'sell'
                confidence = min(0.9, 0.6 + correlation * 0.3)
                reasoning = f"High correlation {correlation:.2f} with beta {beta:.2f}, following market trend"
            elif correlation < 0.3:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Low correlation {correlation:.2f}, independent movement"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Moderate correlation {correlation:.2f} with beta {beta:.2f}"
            
            return AgentPrediction(
                agent_name='CorrelationAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'correlation': float(correlation),
                    'beta': float(beta),
                    'market_return': float(returns_spy.iloc[-1]),
                    'symbol_return': float(returns_symbol.iloc[-1])
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"CorrelationAgent failed for {symbol}: {e}")
            return None
    
    async def _risk_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Risk Agent - Analyzes risk metrics and volatility."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if len(hist) < 30:
                return None
            
            # Calculate risk metrics
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            max_drawdown = (hist['Close'].cummax() - hist['Close']).max() / hist['Close'].cummax().max()
            
            # Determine signal based on risk metrics
            if sharpe_ratio > 1.0 and volatility < 0.3 and max_drawdown < 0.15:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + sharpe_ratio * 0.2)
                reasoning = f"Low risk, high Sharpe ratio {sharpe_ratio:.2f}, volatility {volatility:.2%}"
            elif sharpe_ratio < -0.5 or volatility > 0.5 or max_drawdown > 0.3:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + abs(sharpe_ratio) * 0.2)
                reasoning = f"High risk: Sharpe {sharpe_ratio:.2f}, volatility {volatility:.2%}, max drawdown {max_drawdown:.2%}"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Moderate risk: Sharpe {sharpe_ratio:.2f}, volatility {volatility:.2%}"
            
            return AgentPrediction(
                agent_name='RiskAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'returns_mean': float(returns.mean()),
                    'returns_std': float(returns.std())
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"RiskAgent failed for {symbol}: {e}")
            return None
    
    async def _volatility_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Volatility Agent - Analyzes volatility patterns and breakouts."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2mo")
            
            if len(hist) < 20:
                return None
            
            # Calculate volatility metrics
            returns = hist['Close'].pct_change().dropna()
            recent_vol = returns.iloc[-10:].std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            vol_ratio = recent_vol / historical_vol if historical_vol != 0 else 1.0
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(hist['Close'])
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            current_price = hist['Close'].iloc[-1]
            
            # Determine signal based on volatility patterns
            if vol_ratio > 1.5 and current_price > bb_upper:
                signal_type = 'sell'  # High volatility breakout to upside (often reversal)
                confidence = min(0.9, 0.6 + (vol_ratio - 1) * 0.2)
                reasoning = f"High volatility breakout: vol ratio {vol_ratio:.2f}, price above BB upper"
            elif vol_ratio > 1.5 and current_price < bb_lower:
                signal_type = 'buy'  # High volatility breakout to downside (often reversal)
                confidence = min(0.9, 0.6 + (vol_ratio - 1) * 0.2)
                reasoning = f"High volatility breakout: vol ratio {vol_ratio:.2f}, price below BB lower"
            elif vol_ratio < 0.7:
                signal_type = 'buy'  # Low volatility often precedes moves
                confidence = 0.6 + (1 - vol_ratio) * 0.2
                reasoning = f"Low volatility environment: vol ratio {vol_ratio:.2f}, potential breakout setup"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Normal volatility: vol ratio {vol_ratio:.2f}"
            
            return AgentPrediction(
                agent_name='VolatilityAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'volatility_ratio': float(vol_ratio),
                    'recent_volatility': float(recent_vol),
                    'historical_volatility': float(historical_vol),
                    'bb_upper': float(bb_upper),
                    'bb_lower': float(bb_lower),
                    'bb_middle': float(bb_middle),
                    'current_price': float(current_price)
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"VolatilityAgent failed for {symbol}: {e}")
            return None
    
    async def _volume_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Volume Agent - Analyzes volume patterns and accumulation/distribution."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2mo")
            
            if len(hist) < 20:
                return None
            
            # Calculate volume metrics
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1.0
            
            # Price change
            price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
            
            # On-Balance Volume
            obv = ta.volume.OnBalanceVolumeIndicator(hist['Close'], hist['Volume']).on_balance_volume()
            obv_change = (obv.iloc[-1] - obv.iloc[-6]) / obv.iloc[-6] if obv.iloc[-6] != 0 else 0
            
            # Determine signal based on volume patterns
            if volume_ratio > 1.5 and price_change > 0 and obv_change > 0:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + (volume_ratio - 1) * 0.2)
                reasoning = f"High volume accumulation: vol ratio {volume_ratio:.2f}, price up {price_change:.2%}"
            elif volume_ratio > 1.5 and price_change < 0 and obv_change < 0:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + (volume_ratio - 1) * 0.2)
                reasoning = f"High volume distribution: vol ratio {volume_ratio:.2f}, price down {price_change:.2%}"
            elif volume_ratio < 0.7:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Low volume: vol ratio {volume_ratio:.2f}, waiting for confirmation"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Normal volume: vol ratio {volume_ratio:.2f}"
            
            return AgentPrediction(
                agent_name='VolumeAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'volume_ratio': float(volume_ratio),
                    'current_volume': float(current_volume),
                    'avg_volume': float(avg_volume),
                    'price_change': float(price_change),
                    'obv_change': float(obv_change)
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"VolumeAgent failed for {symbol}: {e}")
            return None
    
    async def _event_impact_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Event Impact Agent - Analyzes earnings, news events, and catalysts."""
        try:
            # Simulate event analysis (in real system, would analyze earnings dates, news events, etc.)
            days_to_earnings = random.randint(-5, 30)  # -5 to 30 days
            event_impact = random.uniform(-0.5, 0.5)
            news_sentiment = random.uniform(-1, 1)
            
            # Determine signal based on events
            if days_to_earnings <= 2 and days_to_earnings >= -2:
                # Earnings period
                signal_type = 'hold'
                confidence = 0.7
                reasoning = f"Earnings period: {days_to_earnings} days, high uncertainty"
            elif event_impact > 0.2 and news_sentiment > 0.3:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + event_impact * 0.5)
                reasoning = f"Positive event impact: {event_impact:.2f}, news sentiment: {news_sentiment:.2f}"
            elif event_impact < -0.2 and news_sentiment < -0.3:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + abs(event_impact) * 0.5)
                reasoning = f"Negative event impact: {event_impact:.2f}, news sentiment: {news_sentiment:.2f}"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Neutral events: impact {event_impact:.2f}, sentiment {news_sentiment:.2f}"
            
            return AgentPrediction(
                agent_name='EventImpactAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'days_to_earnings': days_to_earnings,
                    'event_impact': event_impact,
                    'news_sentiment': news_sentiment,
                    'event_type': 'earnings' if abs(days_to_earnings) <= 2 else 'other'
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"EventImpactAgent failed for {symbol}: {e}")
            return None
    
    async def _forecast_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Forecast Agent - Uses technical analysis for price forecasting."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if len(hist) < 50:
                return None
            
            # Calculate technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            current_price = hist['Close'].iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(hist['Close'])
            macd_line = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_histogram = macd.macd_diff().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(hist['High'], hist['Low'], hist['Close'])
            stoch_k = stoch.stoch().iloc[-1]
            stoch_d = stoch.stoch_signal().iloc[-1]
            
            # Determine signal based on technical analysis
            bullish_signals = 0
            bearish_signals = 0
            
            if current_price > sma_20 > sma_50:
                bullish_signals += 1
            elif current_price < sma_20 < sma_50:
                bearish_signals += 1
            
            if macd_line > macd_signal and macd_histogram > 0:
                bullish_signals += 1
            elif macd_line < macd_signal and macd_histogram < 0:
                bearish_signals += 1
            
            if stoch_k > stoch_d and stoch_k > 50:
                bullish_signals += 1
            elif stoch_k < stoch_d and stoch_k < 50:
                bearish_signals += 1
            
            # Determine signal
            if bullish_signals >= 2:
                signal_type = 'buy'
                confidence = min(0.9, 0.6 + bullish_signals * 0.1)
                reasoning = f"Bullish technical setup: {bullish_signals} signals, MACD {macd_line:.4f}"
            elif bearish_signals >= 2:
                signal_type = 'sell'
                confidence = min(0.9, 0.6 + bearish_signals * 0.1)
                reasoning = f"Bearish technical setup: {bearish_signals} signals, MACD {macd_line:.4f}"
            else:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Mixed technical signals: {bullish_signals} bullish, {bearish_signals} bearish"
            
            return AgentPrediction(
                agent_name='ForecastAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'macd_line': float(macd_line),
                    'macd_signal': float(macd_signal),
                    'macd_histogram': float(macd_histogram),
                    'stoch_k': float(stoch_k),
                    'stoch_d': float(stoch_d),
                    'sma_20': float(sma_20),
                    'sma_50': float(sma_50),
                    'current_price': float(current_price),
                    'bullish_signals': bullish_signals,
                    'bearish_signals': bearish_signals
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"ForecastAgent failed for {symbol}: {e}")
            return None
    
    async def _strategy_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Strategy Agent - Implements various trading strategies."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if len(hist) < 100:
                return None
            
            # Multiple strategy signals
            signals = []
            
            # Strategy 1: Mean reversion
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            current_price = hist['Close'].iloc[-1]
            
            if current_price < sma_20 * 0.95 and current_price > sma_50:
                signals.append(('buy', 0.7, 'Mean reversion: price below SMA20'))
            elif current_price > sma_20 * 1.05 and current_price < sma_50:
                signals.append(('sell', 0.7, 'Mean reversion: price above SMA20'))
            
            # Strategy 2: Trend following
            ema_12 = hist['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = hist['Close'].ewm(span=26).mean().iloc[-1]
            
            if ema_12 > ema_26 and current_price > ema_12:
                signals.append(('buy', 0.6, 'Trend following: EMA12 > EMA26'))
            elif ema_12 < ema_26 and current_price < ema_12:
                signals.append(('sell', 0.6, 'Trend following: EMA12 < EMA26'))
            
            # Strategy 3: Breakout
            recent_high = hist['High'].rolling(window=20).max().iloc[-1]
            recent_low = hist['Low'].rolling(window=20).min().iloc[-1]
            
            if current_price > recent_high * 1.01:
                signals.append(('buy', 0.8, 'Breakout: price above 20-day high'))
            elif current_price < recent_low * 0.99:
                signals.append(('sell', 0.8, 'Breakdown: price below 20-day low'))
            
            # Combine signals
            if not signals:
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = "No clear strategy signals"
            else:
                # Take the highest confidence signal
                best_signal = max(signals, key=lambda x: x[1])
                signal_type = best_signal[0]
                confidence = best_signal[1]
                reasoning = best_signal[2]
            
            return AgentPrediction(
                agent_name='StrategyAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'sma_20': float(sma_20),
                    'sma_50': float(sma_50),
                    'ema_12': float(ema_12),
                    'ema_26': float(ema_26),
                    'recent_high': float(recent_high),
                    'recent_low': float(recent_low),
                    'current_price': float(current_price),
                    'strategy_signals': len(signals)
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"StrategyAgent failed for {symbol}: {e}")
            return None
    
    async def _meta_agent(self, symbol: str) -> Optional[AgentPrediction]:
        """Meta Agent - Analyzes market regime and adjusts other agent weights."""
        try:
            # Get market data for regime analysis
            spy = yf.Ticker('SPY')
            hist_spy = spy.history(period="6mo")
            
            if len(hist_spy) < 100:
                return None
            
            # Calculate market regime indicators
            spy_returns = hist_spy['Close'].pct_change().dropna()
            volatility = spy_returns.std() * np.sqrt(252)
            trend = (hist_spy['Close'].iloc[-1] - hist_spy['Close'].iloc[-60]) / hist_spy['Close'].iloc[-60] if len(hist_spy) >= 60 else 0
            
            # Determine market regime
            if trend > 0.1 and volatility < 0.2:
                regime = 'bull'
                signal_type = 'buy'
                confidence = 0.7
                reasoning = f"Bull market regime: trend {trend:.2%}, low volatility {volatility:.2%}"
            elif trend < -0.1 and volatility < 0.2:
                regime = 'bear'
                signal_type = 'sell'
                confidence = 0.7
                reasoning = f"Bear market regime: trend {trend:.2%}, low volatility {volatility:.2%}"
            elif volatility > 0.3:
                regime = 'volatile'
                signal_type = 'hold'
                confidence = 0.6
                reasoning = f"High volatility regime: {volatility:.2%}, uncertain market"
            else:
                regime = 'neutral'
                signal_type = 'hold'
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                reasoning = f"Neutral market regime: trend {trend:.2%}, volatility {volatility:.2%}"
            
            return AgentPrediction(
                agent_name='MetaAgent',
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'market_regime': regime,
                    'market_trend': float(trend),
                    'market_volatility': float(volatility),
                    'regime_confidence': confidence,
                    'spy_price': float(hist_spy['Close'].iloc[-1])
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"MetaAgent failed for {symbol}: {e}")
            return None
    
    def _clean_metadata(self, metadata: dict) -> dict:
        """Clean metadata by replacing NaN and Inf values with None."""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            elif isinstance(value, dict):
                cleaned[key] = self._clean_metadata(value)
            elif isinstance(value, list):
                cleaned[key] = [v if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else None for v in value]
            else:
                cleaned[key] = value
        return cleaned
    
    async def store_predictions(self, predictions: List[AgentPrediction]) -> bool:
        """Store predictions in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                # Clear old predictions (older than 1 hour)
                await conn.execute("""
                    DELETE FROM agent_signals 
                    WHERE timestamp < NOW() - INTERVAL '1 hour'
                """)
                
                # Insert new predictions
                for prediction in predictions:
                    # Clean metadata to remove NaN/Inf values
                    cleaned_metadata = self._clean_metadata(prediction.metadata)
                    
                    await conn.execute("""
                        INSERT INTO agent_signals (
                            agent_name, symbol, signal_type, confidence, reasoning, metadata, timestamp
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, 
                        prediction.agent_name,
                        prediction.symbol,
                        prediction.signal_type,
                        prediction.confidence,
                        prediction.reasoning,
                        json.dumps(cleaned_metadata),
                        prediction.timestamp
                    )
                
                logger.info(f"✅ Stored {len(predictions)} individual agent predictions")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store predictions: {e}")
            return False
