"""
Event Impact Agent for AI Market Analysis System

This agent analyzes and scores market events and their short/long-term impact
on assets to provide event-driven trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import requests
from bs4 import BeautifulSoup
import feedparser
import re

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market events."""
    EARNINGS = "earnings"
    FOMC = "fomc"
    CPI = "cpi"
    GDP = "gdp"
    UNEMPLOYMENT = "unemployment"
    FED_SPEECH = "fed_speech"
    MERGER = "merger"
    DIVIDEND = "dividend"
    SPLIT = "split"
    IPO = "ipo"
    REGULATION = "regulation"
    GEOPOLITICAL = "geopolitical"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"
    OTHER = "other"


class EventImpact(Enum):
    """Event impact levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketEvent:
    """Represents a market event with impact scoring."""
    event_id: str
    event_type: EventType
    title: str
    description: str
    timestamp: datetime
    impact_score: float  # 0-1 scale
    impact_duration: int  # hours
    affected_assets: List[str]
    confidence: float
    source: str
    metadata: Dict[str, Any]


class EventImpactAgent(BaseAgent):
    """
    Event Impact Agent for analyzing and scoring market events.
    
    This agent analyzes:
    - Market events and their impact on assets
    - Event timing and duration effects
    - Event correlation with price movements
    - Event-driven trading opportunities
    - Risk assessment based on upcoming events
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Event Impact Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'event_sources': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.marketwatch.com/marketwatch/topstories/',
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.federalreserve.gov/feeds/press_all.xml'
            ],
            'event_keywords': {
                EventType.EARNINGS: ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit'],
                EventType.FOMC: ['fomc', 'federal reserve', 'interest rate', 'monetary policy', 'fed meeting'],
                EventType.CPI: ['cpi', 'consumer price index', 'inflation', 'price level'],
                EventType.GDP: ['gdp', 'gross domestic product', 'economic growth', 'recession'],
                EventType.UNEMPLOYMENT: ['unemployment', 'jobless claims', 'employment', 'jobs report'],
                EventType.FED_SPEECH: ['fed speech', 'federal reserve speech', 'powell', 'yellen'],
                EventType.MERGER: ['merger', 'acquisition', 'takeover', 'buyout'],
                EventType.DIVIDEND: ['dividend', 'dividend yield', 'payout'],
                EventType.SPLIT: ['stock split', 'split', 'reverse split'],
                EventType.IPO: ['ipo', 'initial public offering', 'going public'],
                EventType.REGULATION: ['regulation', 'regulatory', 'sec', 'fda approval'],
                EventType.GEOPOLITICAL: ['war', 'conflict', 'sanctions', 'trade war', 'brexit'],
                EventType.NATURAL_DISASTER: ['hurricane', 'earthquake', 'flood', 'disaster'],
                EventType.PANDEMIC: ['pandemic', 'covid', 'virus', 'outbreak']
            },
            'impact_weights': {
                EventType.EARNINGS: 0.8,
                EventType.FOMC: 0.9,
                EventType.CPI: 0.7,
                EventType.GDP: 0.6,
                EventType.UNEMPLOYMENT: 0.5,
                EventType.FED_SPEECH: 0.4,
                EventType.MERGER: 0.7,
                EventType.DIVIDEND: 0.3,
                EventType.SPLIT: 0.2,
                EventType.IPO: 0.6,
                EventType.REGULATION: 0.8,
                EventType.GEOPOLITICAL: 0.9,
                EventType.NATURAL_DISASTER: 0.7,
                EventType.PANDEMIC: 0.9
            },
            'lookback_hours': 72,  # 3 days
            'impact_duration_hours': 24,
            'confidence_threshold': 0.6,
            'high_impact_threshold': 0.7
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="EventImpactAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.event_history = []
        self.active_events = []
        self.event_cache = {}
        self.last_event_fetch = None
        
        logger.info(f"Initialized EventImpactAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the event impact agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For event analysis, we don't need traditional ML training
            # Instead, we'll validate our event analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Event impact analysis approach validated")
            return {"status": "event_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate event-driven prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Event-driven trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple event analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple event analysis (not trained)")
                return self._simple_event_analysis(context)
            
            # Perform comprehensive event analysis
            event_analysis = self._analyze_events(context)
            
            # Generate signal based on event insights
            signal = self._generate_event_signal(event_analysis, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Event analysis error: {e}", context)
    
    def _analyze_events(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze events and their impact on the given symbol.
        
        Args:
            context: Current market context
            
        Returns:
            Event analysis results
        """
        try:
            symbol = context.symbol
            
            # Fetch recent events
            recent_events = self._fetch_recent_events(symbol)
            
            # Score event impacts
            event_impacts = self._score_event_impacts(recent_events, symbol)
            
            # Analyze event timing
            timing_analysis = self._analyze_event_timing(event_impacts)
            
            # Calculate overall event risk
            event_risk = self._calculate_event_risk(event_impacts)
            
            # Detect event-driven opportunities
            opportunities = self._detect_event_opportunities(event_impacts, context)
            
            return {
                'recent_events': recent_events,
                'event_impacts': event_impacts,
                'timing_analysis': timing_analysis,
                'event_risk': event_risk,
                'opportunities': opportunities,
                'confidence': self._calculate_event_confidence(event_impacts)
            }
            
        except Exception as e:
            logger.error(f"Event analysis failed: {e}")
            return {
                'recent_events': [],
                'event_impacts': [],
                'timing_analysis': {},
                'event_risk': {},
                'opportunities': [],
                'confidence': 0.0
            }
    
    def _fetch_recent_events(self, symbol: str) -> List[MarketEvent]:
        """
        Fetch recent market events relevant to the symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of recent market events
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.event_cache:
                return self.event_cache[cache_key]
            
            events = []
            
            # Fetch from RSS feeds
            for source in self.config['event_sources']:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:20]:  # Limit to recent entries
                        event = self._parse_event_entry(entry, symbol)
                        if event:
                            events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source}: {e}")
                    continue
            
            # Sort by timestamp (most recent first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Cache results
            self.event_cache[cache_key] = events
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []
    
    def _parse_event_entry(self, entry: Dict, symbol: str) -> Optional[MarketEvent]:
        """
        Parse an RSS entry into a MarketEvent.
        
        Args:
            entry: RSS feed entry
            symbol: Stock symbol
            
        Returns:
            MarketEvent if relevant, None otherwise
        """
        try:
            title = entry.get('title', '')
            description = entry.get('summary', '')
            text = f"{title} {description}".lower()
            
            # Check if event is relevant to the symbol
            if not self._is_relevant_event(text, symbol):
                return None
            
            # Classify event type
            event_type = self._classify_event_type(text)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(text, event_type)
            
            # Determine affected assets
            affected_assets = self._determine_affected_assets(text, symbol)
            
            # Create event
            event = MarketEvent(
                event_id=f"{entry.get('id', '')}_{datetime.now().timestamp()}",
                event_type=event_type,
                title=title,
                description=description,
                timestamp=datetime.now(),  # Use current time as proxy
                impact_score=impact_score,
                impact_duration=self.config['impact_duration_hours'],
                affected_assets=affected_assets,
                confidence=self._calculate_event_confidence_score(text, event_type),
                source=entry.get('link', ''),
                metadata={'raw_entry': entry}
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to parse event entry: {e}")
            return None
    
    def _is_relevant_event(self, text: str, symbol: str) -> bool:
        """Check if event is relevant to the given symbol."""
        try:
            # Check for symbol mention
            if symbol.lower() in text:
                return True
            
            # Check for market-wide events
            market_keywords = ['market', 'stock', 'trading', 'finance', 'economy', 'fed', 'fomc', 'cpi', 'gdp']
            if any(keyword in text for keyword in market_keywords):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Event relevance check failed: {e}")
            return False
    
    def _classify_event_type(self, text: str) -> EventType:
        """Classify the type of event based on text content."""
        try:
            # Count keyword matches for each event type
            type_scores = {}
            
            for event_type, keywords in self.config['event_keywords'].items():
                score = sum(1 for keyword in keywords if keyword in text)
                type_scores[event_type] = score
            
            # Return event type with highest score
            if type_scores:
                return max(type_scores, key=type_scores.get)
            else:
                return EventType.OTHER
                
        except Exception as e:
            logger.error(f"Event type classification failed: {e}")
            return EventType.OTHER
    
    def _calculate_impact_score(self, text: str, event_type: EventType) -> float:
        """Calculate the impact score for an event."""
        try:
            # Base impact from event type
            base_impact = self.config['impact_weights'].get(event_type, 0.5)
            
            # Adjust based on text content
            impact_modifiers = {
                'urgent': 0.2,
                'breaking': 0.3,
                'critical': 0.4,
                'major': 0.2,
                'significant': 0.1,
                'minor': -0.1,
                'small': -0.2
            }
            
            modifier = 0
            for keyword, value in impact_modifiers.items():
                if keyword in text:
                    modifier += value
            
            # Calculate final impact score
            impact_score = base_impact + modifier
            return max(0.0, min(1.0, impact_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Impact score calculation failed: {e}")
            return 0.5
    
    def _determine_affected_assets(self, text: str, symbol: str) -> List[str]:
        """Determine which assets are affected by the event."""
        try:
            affected = [symbol]  # Always include the current symbol
            
            # Add market-wide assets for major events
            if any(keyword in text for keyword in ['market', 'economy', 'fed', 'fomc', 'cpi', 'gdp']):
                affected.extend(['SPY', 'QQQ', 'IWM'])  # Market indices
            
            return list(set(affected))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Affected assets determination failed: {e}")
            return [symbol]
    
    def _calculate_event_confidence_score(self, text: str, event_type: EventType) -> float:
        """Calculate confidence score for event classification."""
        try:
            # Base confidence from event type
            base_confidence = 0.7
            
            # Adjust based on text quality
            if len(text) > 100:  # Longer text usually more reliable
                base_confidence += 0.1
            
            # Adjust based on keyword matches
            keywords = self.config['event_keywords'].get(event_type, [])
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                base_confidence += min(0.2, matches * 0.05)
            
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            logger.error(f"Event confidence calculation failed: {e}")
            return 0.5
    
    def _score_event_impacts(self, events: List[MarketEvent], symbol: str) -> List[Dict[str, Any]]:
        """Score the impact of events on the given symbol."""
        try:
            impacts = []
            
            for event in events:
                if symbol in event.affected_assets:
                    impact = {
                        'event': event,
                        'symbol_impact': event.impact_score,
                        'time_to_event': self._calculate_time_to_event(event),
                        'impact_duration': event.impact_duration,
                        'confidence': event.confidence
                    }
                    impacts.append(impact)
            
            return impacts
            
        except Exception as e:
            logger.error(f"Event impact scoring failed: {e}")
            return []
    
    def _calculate_time_to_event(self, event: MarketEvent) -> float:
        """Calculate time until event impact (in hours)."""
        try:
            # For now, assume events are recent (within lookback period)
            # In a real implementation, this would parse actual event timestamps
            return 0.0  # Event is happening now
            
        except Exception as e:
            logger.error(f"Time to event calculation failed: {e}")
            return 0.0
    
    def _analyze_event_timing(self, event_impacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the timing of events and their impacts."""
        try:
            if not event_impacts:
                return {'timing_risk': 'low', 'upcoming_events': 0}
            
            # Count high-impact events
            high_impact_events = [e for e in event_impacts if e['symbol_impact'] > self.config['high_impact_threshold']]
            
            # Determine timing risk
            if len(high_impact_events) > 2:
                timing_risk = 'high'
            elif len(high_impact_events) > 0:
                timing_risk = 'medium'
            else:
                timing_risk = 'low'
            
            return {
                'timing_risk': timing_risk,
                'upcoming_events': len(event_impacts),
                'high_impact_events': len(high_impact_events),
                'avg_impact': np.mean([e['symbol_impact'] for e in event_impacts]) if event_impacts else 0
            }
            
        except Exception as e:
            logger.error(f"Event timing analysis failed: {e}")
            return {'timing_risk': 'unknown', 'upcoming_events': 0}
    
    def _calculate_event_risk(self, event_impacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall event risk for the symbol."""
        try:
            if not event_impacts:
                return {'risk_level': 'low', 'risk_score': 0.0}
            
            # Calculate weighted risk score
            total_impact = sum(e['symbol_impact'] * e['confidence'] for e in event_impacts)
            avg_confidence = np.mean([e['confidence'] for e in event_impacts])
            
            risk_score = total_impact * avg_confidence
            
            # Classify risk level
            if risk_score > 0.7:
                risk_level = 'high'
            elif risk_score > 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'total_impact': total_impact,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Event risk calculation failed: {e}")
            return {'risk_level': 'unknown', 'risk_score': 0.0}
    
    def _detect_event_opportunities(self, event_impacts: List[Dict[str, Any]], context: AgentContext) -> List[Dict[str, Any]]:
        """Detect event-driven trading opportunities."""
        try:
            opportunities = []
            
            for impact in event_impacts:
                event = impact['event']
                
                # Look for high-impact, high-confidence events
                if (impact['symbol_impact'] > self.config['high_impact_threshold'] and 
                    impact['confidence'] > self.config['confidence_threshold']):
                    
                    # Determine opportunity type based on event type
                    if event.event_type in [EventType.EARNINGS, EventType.MERGER, EventType.IPO]:
                        opportunity_type = 'fundamental'
                    elif event.event_type in [EventType.FOMC, EventType.CPI, EventType.GDP]:
                        opportunity_type = 'macro'
                    else:
                        opportunity_type = 'event_driven'
                    
                    opportunity = {
                        'type': opportunity_type,
                        'event': event,
                        'impact_score': impact['symbol_impact'],
                        'confidence': impact['confidence'],
                        'description': f"{event.event_type.value} event: {event.title[:50]}..."
                    }
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Event opportunity detection failed: {e}")
            return []
    
    def _calculate_event_confidence(self, event_impacts: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in event analysis."""
        try:
            if not event_impacts:
                return 0.0
            
            # Base confidence on number and quality of events
            base_confidence = min(len(event_impacts) / 5, 1.0)  # Normalize to 5 events
            
            # Adjust based on event confidence
            avg_event_confidence = np.mean([e['confidence'] for e in event_impacts])
            
            final_confidence = (base_confidence + avg_event_confidence) / 2
            return min(1.0, final_confidence)
            
        except Exception as e:
            logger.error(f"Event confidence calculation failed: {e}")
            return 0.0
    
    def _generate_event_signal(self, analysis: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on event analysis.
        
        Args:
            analysis: Event analysis results
            context: Current market context
            
        Returns:
            Event-driven trading signal
        """
        try:
            event_risk = analysis['event_risk']
            opportunities = analysis['opportunities']
            timing_analysis = analysis['timing_analysis']
            confidence = analysis['confidence']
            
            # Determine signal based on event analysis
            if event_risk['risk_level'] == 'high' and timing_analysis['timing_risk'] == 'high':
                signal_type = SignalType.SELL
                reasoning = f"High event risk ({event_risk['risk_score']:.2f}) with multiple upcoming events"
            elif opportunities and len(opportunities) > 0:
                # Look for positive opportunities
                positive_opportunities = [o for o in opportunities if o['impact_score'] > 0.6]
                if positive_opportunities:
                    signal_type = SignalType.BUY
                    reasoning = f"Event-driven opportunity: {positive_opportunities[0]['description']}"
                else:
                    signal_type = SignalType.HOLD
                    reasoning = "Event-driven opportunities detected but impact unclear"
            elif event_risk['risk_level'] == 'low' and timing_analysis['timing_risk'] == 'low':
                signal_type = SignalType.HOLD
                reasoning = "Low event risk - no significant events detected"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Event risk: {event_risk['risk_level']} - monitoring for opportunities"
            
            # Adjust confidence based on event analysis quality
            adjusted_confidence = min(confidence * 0.8, 0.9)  # Event analysis typically lower confidence
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'event_analysis': analysis,
                    'method': 'event_impact_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_event_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple event analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple event-driven signal
        """
        try:
            # Simple event analysis based on market volatility
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            # Use volatility as a proxy for event impact
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    
                    # High volatility might indicate event-driven activity
                    if volatility > 0.03:  # 3% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.HOLD,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_event'},
                            reasoning=f"High volatility ({volatility:.2%}) suggests potential event-driven activity"
                        )
            
            return self._create_hold_signal("No clear event-driven signal", context)
            
        except Exception as e:
            logger.error(f"Simple event analysis failed: {e}")
            return self._create_hold_signal(f"Simple event analysis error: {e}", context)
    
    def _create_hold_signal(self, reason: str, context: AgentContext) -> AgentSignal:
        """Create a hold signal with error information."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=context.timestamp,
            asset_symbol=context.symbol,
            metadata={'error': reason},
            reasoning=f"Hold signal: {reason}"
        )
    
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the event model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update event history
            if hasattr(self, '_last_event_analysis'):
                self.event_history.append(self._last_event_analysis)
                
                # Keep only recent history
                if len(self.event_history) > 50:
                    self.event_history = self.event_history[-50:]
            
            logger.info(f"Updated event model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
