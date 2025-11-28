import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)

class ExplanationType(str, Enum):
    TRADE_DECISION = "trade_decision"
    AGENT_PERFORMANCE = "agent_performance"
    MARKET_ANALYSIS = "market_analysis"
    FORECAST_ERROR = "forecast_error"
    PORTFOLIO_CHANGE = "portfolio_change"
    RISK_ASSESSMENT = "risk_assessment"

class ExplanationTone(str, Enum):
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"

@dataclass
class ExplanationRequest:
    explanation_type: ExplanationType
    data: Dict[str, Any]
    tone: ExplanationTone = ExplanationTone.PROFESSIONAL
    max_length: int = 500
    include_metrics: bool = True
    include_recommendations: bool = True

@dataclass
class ExplanationResponse:
    explanation_type: ExplanationType
    title: str
    summary: str
    detailed_explanation: str
    key_metrics: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    generated_at: str
    tone: ExplanationTone

@dataclass
class TradeExplanation:
    trade_id: str
    symbol: str
    decision_rationale: str
    risk_factors: List[str]
    opportunity_factors: List[str]
    market_context: str
    agent_contributions: Dict[str, str]
    confidence_analysis: str
    alternative_scenarios: List[str]

@dataclass
class PerformanceExplanation:
    agent_name: str
    performance_summary: str
    strengths: List[str]
    weaknesses: List[str]
    market_conditions: str
    improvement_suggestions: List[str]
    comparison_to_peers: str

@dataclass
class MarketAnalysisExplanation:
    market_regime: str
    regime_explanation: str
    key_drivers: List[str]
    risk_factors: List[str]
    opportunity_areas: List[str]
    historical_context: str
    forward_looking: str

class LLMExplainAgent:
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
        self.metric_formatters = self._load_metric_formatters()
        logger.info("LLMExplainAgent initialized")

    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different types."""
        return {
            "trade_decision": {
                "professional": """
**Trade Decision Analysis: {symbol}**

**Decision Rationale:**
{decision_rationale}

**Risk Assessment:**
{risk_analysis}

**Opportunity Analysis:**
{opportunity_analysis}

**Market Context:**
{market_context}

**Agent Contributions:**
{agent_contributions}

**Confidence Analysis:**
{confidence_analysis}
""",
                "conversational": """
Hey! Let me break down why we made this trade on {symbol}:

**The Big Picture:**
{decision_rationale}

**What could go wrong:**
{risk_analysis}

**What could go right:**
{opportunity_analysis}

**Market vibes:**
{market_context}

**Our AI agents said:**
{agent_contributions}

**How confident are we?**
{confidence_analysis}
""",
                "technical": """
**Technical Trade Analysis: {symbol}**

**Algorithmic Decision Matrix:**
{decision_rationale}

**Risk Metrics:**
{risk_analysis}

**Opportunity Metrics:**
{opportunity_analysis}

**Market Regime Analysis:**
{market_context}

**Multi-Agent Signal Aggregation:**
{agent_contributions}

**Confidence Calibration:**
{confidence_analysis}
""",
                "simplified": """
**Simple Trade Explanation: {symbol}**

**Why we bought/sold:**
{decision_rationale}

**Risks:**
{risk_analysis}

**Opportunities:**
{opportunity_analysis}

**Market situation:**
{market_context}

**AI recommendations:**
{agent_contributions}

**Confidence level:**
{confidence_analysis}
"""
            },
            "agent_performance": {
                "professional": """
**Agent Performance Analysis: {agent_name}**

**Performance Summary:**
{performance_summary}

**Strengths:**
{strengths}

**Areas for Improvement:**
{weaknesses}

**Market Conditions:**
{market_conditions}

**Peer Comparison:**
{comparison_to_peers}

**Recommendations:**
{improvement_suggestions}
""",
                "conversational": """
**How's {agent_name} doing?**

**The good news:**
{performance_summary}

**What they're great at:**
{strengths}

**Where they need work:**
{weaknesses}

**Market conditions they faced:**
{market_conditions}

**Compared to other agents:**
{comparison_to_peers}

**What we should do:**
{improvement_suggestions}
""",
                "technical": """
**Agent Performance Metrics: {agent_name}**

**Quantitative Analysis:**
{performance_summary}

**Performance Strengths:**
{strengths}

**Performance Weaknesses:**
{weaknesses}

**Market Regime Performance:**
{market_conditions}

**Relative Performance Analysis:**
{comparison_to_peers}

**Optimization Recommendations:**
{improvement_suggestions}
""",
                "simplified": """
**Agent Report: {agent_name}**

**How they did:**
{performance_summary}

**Good at:**
{strengths}

**Needs work on:**
{weaknesses}

**Market conditions:**
{market_conditions}

**Vs other agents:**
{comparison_to_peers}

**What to do:**
{improvement_suggestions}
"""
            }
        }

    def _load_metric_formatters(self) -> Dict[str, str]:
        """Load metric formatting templates."""
        return {
            "percentage": "{value:.2f}%",
            "currency": "${value:,.2f}",
            "ratio": "{value:.3f}",
            "days": "{value:.1f} days",
            "score": "{value:.2f}/1.0"
        }

    async def explain_trade_decision(self, trade_data: Dict[str, Any], tone: ExplanationTone = ExplanationTone.PROFESSIONAL) -> ExplanationResponse:
        """Generate explanation for a trade decision."""
        logger.info(f"Generating trade decision explanation for {trade_data.get('symbol', 'unknown')}")
        
        # Extract trade information
        symbol = trade_data.get('symbol', 'Unknown')
        trade_type = trade_data.get('trade_type', 'unknown')
        entry_price = trade_data.get('entry_price', 0)
        quantity = trade_data.get('quantity', 0)
        confidence_score = trade_data.get('confidence_score', 0.5)
        agent_signals = trade_data.get('agent_signals', [])
        market_regime = trade_data.get('market_regime', 'unknown')
        
        # Generate explanation components
        decision_rationale = self._generate_decision_rationale(trade_data)
        risk_analysis = self._generate_risk_analysis(trade_data)
        opportunity_analysis = self._generate_opportunity_analysis(trade_data)
        market_context = self._generate_market_context(trade_data)
        agent_contributions = self._generate_agent_contributions(trade_data)
        confidence_analysis = self._generate_confidence_analysis(trade_data)
        
        # Format the explanation
        template = self.explanation_templates["trade_decision"][tone.value]
        detailed_explanation = template.format(
            symbol=symbol,
            decision_rationale=decision_rationale,
            risk_analysis=risk_analysis,
            opportunity_analysis=opportunity_analysis,
            market_context=market_context,
            agent_contributions=agent_contributions,
            confidence_analysis=confidence_analysis
        )
        
        # Generate key metrics
        key_metrics = {
            "entry_price": entry_price,
            "quantity": quantity,
            "confidence_score": confidence_score,
            "agent_count": len(agent_signals),
            "market_regime": market_regime
        }
        
        # Generate recommendations
        recommendations = self._generate_trade_recommendations(trade_data)
        
        return ExplanationResponse(
            explanation_type=ExplanationType.TRADE_DECISION,
            title=f"Trade Decision Analysis: {symbol}",
            summary=f"Analysis of {trade_type} trade on {symbol} with {confidence_score:.1%} confidence",
            detailed_explanation=detailed_explanation,
            key_metrics=key_metrics,
            recommendations=recommendations,
            confidence_score=confidence_score,
            generated_at=datetime.now().isoformat(),
            tone=tone
        )

    async def explain_agent_performance(self, agent_data: Dict[str, Any], tone: ExplanationTone = ExplanationTone.PROFESSIONAL) -> ExplanationResponse:
        """Generate explanation for agent performance."""
        logger.info(f"Generating agent performance explanation for {agent_data.get('agent_name', 'unknown')}")
        
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        win_rate = agent_data.get('win_rate', 0)
        total_signals = agent_data.get('total_signals', 0)
        avg_confidence = agent_data.get('avg_confidence', 0)
        
        # Generate explanation components
        performance_summary = self._generate_performance_summary(agent_data)
        strengths = self._generate_agent_strengths(agent_data)
        weaknesses = self._generate_agent_weaknesses(agent_data)
        market_conditions = self._generate_market_conditions_analysis(agent_data)
        comparison_to_peers = self._generate_peer_comparison(agent_data)
        improvement_suggestions = self._generate_improvement_suggestions(agent_data)
        
        # Format the explanation
        template = self.explanation_templates["agent_performance"][tone.value]
        detailed_explanation = template.format(
            agent_name=agent_name,
            performance_summary=performance_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            market_conditions=market_conditions,
            comparison_to_peers=comparison_to_peers,
            improvement_suggestions=improvement_suggestions
        )
        
        # Generate key metrics
        key_metrics = {
            "win_rate": win_rate,
            "total_signals": total_signals,
            "avg_confidence": avg_confidence,
            "successful_signals": agent_data.get('successful_signals', 0),
            "failed_signals": agent_data.get('failed_signals', 0)
        }
        
        # Generate recommendations
        recommendations = self._generate_agent_recommendations(agent_data)
        
        return ExplanationResponse(
            explanation_type=ExplanationType.AGENT_PERFORMANCE,
            title=f"Agent Performance Analysis: {agent_name}",
            summary=f"Performance analysis for {agent_name} with {win_rate:.1%} win rate",
            detailed_explanation=detailed_explanation,
            key_metrics=key_metrics,
            recommendations=recommendations,
            confidence_score=avg_confidence,
            generated_at=datetime.now().isoformat(),
            tone=tone
        )

    async def explain_market_analysis(self, market_data: Dict[str, Any], tone: ExplanationTone = ExplanationTone.PROFESSIONAL) -> ExplanationResponse:
        """Generate explanation for market analysis."""
        logger.info("Generating market analysis explanation")
        
        market_regime = market_data.get('market_regime', 'unknown')
        volatility = market_data.get('volatility', 0)
        trend_direction = market_data.get('trend_direction', 'neutral')
        
        # Generate explanation components
        regime_explanation = self._generate_regime_explanation(market_data)
        key_drivers = self._generate_key_drivers(market_data)
        risk_factors = self._generate_risk_factors(market_data)
        opportunity_areas = self._generate_opportunity_areas(market_data)
        historical_context = self._generate_historical_context(market_data)
        forward_looking = self._generate_forward_looking(market_data)
        
        # Format the explanation
        detailed_explanation = f"""
**Market Analysis: {market_regime.title()} Regime**

**Current Market Regime:**
{regime_explanation}

**Key Market Drivers:**
{key_drivers}

**Risk Factors:**
{risk_factors}

**Opportunity Areas:**
{opportunity_areas}

**Historical Context:**
{historical_context}

**Forward Looking:**
{forward_looking}
"""
        
        # Generate key metrics
        key_metrics = {
            "market_regime": market_regime,
            "volatility": volatility,
            "trend_direction": trend_direction,
            "regime_duration": market_data.get('regime_duration', 0),
            "regime_strength": market_data.get('regime_strength', 0)
        }
        
        # Generate recommendations
        recommendations = self._generate_market_recommendations(market_data)
        
        return ExplanationResponse(
            explanation_type=ExplanationType.MARKET_ANALYSIS,
            title=f"Market Analysis: {market_regime.title()} Regime",
            summary=f"Analysis of {market_regime} market regime with {volatility:.1%} volatility",
            detailed_explanation=detailed_explanation,
            key_metrics=key_metrics,
            recommendations=recommendations,
            confidence_score=0.8,  # Market analysis confidence
            generated_at=datetime.now().isoformat(),
            tone=tone
        )

    def _generate_decision_rationale(self, trade_data: Dict[str, Any]) -> str:
        """Generate decision rationale for a trade."""
        symbol = trade_data.get('symbol', 'Unknown')
        trade_type = trade_data.get('trade_type', 'unknown')
        confidence_score = trade_data.get('confidence_score', 0.5)
        
        if trade_type == 'long':
            rationale = f"Long position initiated on {symbol} based on bullish signals with {confidence_score:.1%} confidence. "
        else:
            rationale = f"Short position initiated on {symbol} based on bearish signals with {confidence_score:.1%} confidence. "
        
        rationale += "The decision was supported by multiple technical and fundamental indicators suggesting favorable risk-reward potential."
        
        return rationale

    def _generate_risk_analysis(self, trade_data: Dict[str, Any]) -> str:
        """Generate risk analysis for a trade."""
        symbol = trade_data.get('symbol', 'Unknown')
        market_regime = trade_data.get('market_regime', 'unknown')
        
        risks = [
            f"Market volatility in {market_regime} regime could impact {symbol} performance",
            "Liquidity constraints may affect execution and exit strategies",
            "Sector-specific risks could influence individual stock performance",
            "Macroeconomic factors may override technical signals"
        ]
        
        return "\n".join([f"• {risk}" for risk in risks])

    def _generate_opportunity_analysis(self, trade_data: Dict[str, Any]) -> str:
        """Generate opportunity analysis for a trade."""
        symbol = trade_data.get('symbol', 'Unknown')
        confidence_score = trade_data.get('confidence_score', 0.5)
        
        opportunities = [
            f"Strong technical setup with {confidence_score:.1%} confidence score",
            "Multiple agent signals converging on similar direction",
            "Favorable risk-reward ratio based on historical patterns",
            "Market regime supports the chosen strategy"
        ]
        
        return "\n".join([f"• {opp}" for opp in opportunities])

    def _generate_market_context(self, trade_data: Dict[str, Any]) -> str:
        """Generate market context for a trade."""
        market_regime = trade_data.get('market_regime', 'unknown')
        
        context = f"Current market is in a {market_regime} regime, which historically provides "
        
        if market_regime == 'bull':
            context += "favorable conditions for long positions with strong upward momentum."
        elif market_regime == 'bear':
            context += "challenging conditions requiring careful risk management and potential short opportunities."
        elif market_regime == 'sideways':
            context += "range-bound conditions suitable for mean reversion strategies."
        elif market_regime == 'volatile':
            context += "high volatility conditions requiring increased position sizing caution."
        else:
            context += "trending conditions with clear directional bias."
        
        return context

    def _generate_agent_contributions(self, trade_data: Dict[str, Any]) -> str:
        """Generate agent contributions explanation."""
        agent_signals = trade_data.get('agent_signals', [])
        
        if not agent_signals:
            return "No specific agent signals available for this trade."
        
        contributions = []
        for agent in agent_signals:
            if 'Momentum' in agent:
                contributions.append(f"• {agent}: Identified strong price momentum and trend continuation signals")
            elif 'Sentiment' in agent:
                contributions.append(f"• {agent}: Detected positive market sentiment and news flow")
            elif 'Correlation' in agent:
                contributions.append(f"• {agent}: Found favorable correlation patterns with market leaders")
            elif 'Risk' in agent:
                contributions.append(f"• {agent}: Confirmed acceptable risk levels and position sizing")
            elif 'Volatility' in agent:
                contributions.append(f"• {agent}: Identified optimal volatility conditions for entry")
            else:
                contributions.append(f"• {agent}: Provided supporting technical and fundamental analysis")
        
        return "\n".join(contributions)

    def _generate_confidence_analysis(self, trade_data: Dict[str, Any]) -> str:
        """Generate confidence analysis for a trade."""
        confidence_score = trade_data.get('confidence_score', 0.5)
        
        if confidence_score >= 0.8:
            confidence_level = "Very High"
            explanation = "Multiple strong signals align with high historical accuracy patterns."
        elif confidence_score >= 0.7:
            confidence_level = "High"
            explanation = "Strong signal convergence with good historical performance."
        elif confidence_score >= 0.6:
            confidence_level = "Moderate"
            explanation = "Mixed signals with reasonable confidence based on current market conditions."
        else:
            confidence_level = "Low"
            explanation = "Limited signal strength requiring careful monitoring and risk management."
        
        return f"Confidence Level: {confidence_level} ({confidence_score:.1%})\n\n{explanation}"

    def _generate_performance_summary(self, agent_data: Dict[str, Any]) -> str:
        """Generate performance summary for an agent."""
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        win_rate = agent_data.get('win_rate', 0)
        total_signals = agent_data.get('total_signals', 0)
        
        if win_rate >= 0.7:
            performance = "excellent"
        elif win_rate >= 0.6:
            performance = "good"
        elif win_rate >= 0.5:
            performance = "average"
        else:
            performance = "below average"
        
        return f"{agent_name} has shown {performance} performance with a {win_rate:.1%} win rate across {total_signals} signals. The agent demonstrates consistent signal generation with varying degrees of accuracy depending on market conditions."

    def _generate_agent_strengths(self, agent_data: Dict[str, Any]) -> str:
        """Generate agent strengths analysis."""
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        win_rate = agent_data.get('win_rate', 0)
        avg_confidence = agent_data.get('avg_confidence', 0)
        
        strengths = []
        
        if win_rate >= 0.6:
            strengths.append("High win rate indicating strong signal accuracy")
        
        if avg_confidence >= 0.7:
            strengths.append("High confidence in signal generation")
        
        if 'Momentum' in agent_name:
            strengths.append("Excellent at identifying trend continuations")
            strengths.append("Strong performance in trending markets")
        elif 'Sentiment' in agent_name:
            strengths.append("Effective at capturing market sentiment shifts")
            strengths.append("Good performance during news-driven events")
        elif 'Risk' in agent_name:
            strengths.append("Consistent risk management and position sizing")
            strengths.append("Effective at avoiding high-risk scenarios")
        
        strengths.append("Reliable signal generation across different market conditions")
        
        return "\n".join([f"• {strength}" for strength in strengths])

    def _generate_agent_weaknesses(self, agent_data: Dict[str, Any]) -> str:
        """Generate agent weaknesses analysis."""
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        win_rate = agent_data.get('win_rate', 0)
        
        weaknesses = []
        
        if win_rate < 0.5:
            weaknesses.append("Below-average win rate requiring performance improvement")
        
        if 'Momentum' in agent_name:
            weaknesses.append("May struggle in sideways or choppy markets")
            weaknesses.append("Potential for false breakouts in low-volume conditions")
        elif 'Sentiment' in agent_name:
            weaknesses.append("Can be affected by noise in social media sentiment")
            weaknesses.append("May lag behind rapid sentiment changes")
        elif 'Risk' in agent_name:
            weaknesses.append("Conservative approach may miss high-reward opportunities")
            weaknesses.append("May be overly cautious in favorable market conditions")
        
        weaknesses.append("Performance varies significantly across different market regimes")
        
        return "\n".join([f"• {weakness}" for weakness in weaknesses])

    def _generate_market_conditions_analysis(self, agent_data: Dict[str, Any]) -> str:
        """Generate market conditions analysis for an agent."""
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        best_regime = agent_data.get('best_performing_regime', 'unknown')
        worst_regime = agent_data.get('worst_performing_regime', 'unknown')
        
        return f"{agent_name} performs best in {best_regime} market conditions, where its specialized algorithms can effectively identify and capitalize on market patterns. However, the agent struggles more in {worst_regime} conditions, where its traditional approaches may be less effective due to increased market noise and unpredictable price movements."

    def _generate_peer_comparison(self, agent_data: Dict[str, Any]) -> str:
        """Generate peer comparison analysis."""
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        win_rate = agent_data.get('win_rate', 0)
        
        if win_rate >= 0.7:
            comparison = f"{agent_name} ranks among the top-performing agents in the system, consistently outperforming the average win rate and demonstrating superior signal accuracy."
        elif win_rate >= 0.6:
            comparison = f"{agent_name} performs above average compared to other agents, showing solid signal generation capabilities with room for further optimization."
        elif win_rate >= 0.5:
            comparison = f"{agent_name} performs at the system average, providing reliable signals but with opportunities for improvement through enhanced algorithms or market regime adaptation."
        else:
            comparison = f"{agent_name} currently underperforms compared to other agents in the system, requiring focused optimization efforts to improve signal accuracy and market adaptation."
        
        return comparison

    def _generate_improvement_suggestions(self, agent_data: Dict[str, Any]) -> str:
        """Generate improvement suggestions for an agent."""
        agent_name = agent_data.get('agent_name', 'Unknown Agent')
        win_rate = agent_data.get('win_rate', 0)
        
        suggestions = []
        
        if win_rate < 0.6:
            suggestions.append("Implement additional signal validation filters to improve accuracy")
            suggestions.append("Enhance market regime detection for better adaptation")
        
        if 'Momentum' in agent_name:
            suggestions.append("Add volume confirmation to reduce false breakout signals")
            suggestions.append("Implement multi-timeframe analysis for better trend confirmation")
        elif 'Sentiment' in agent_name:
            suggestions.append("Improve noise filtering in sentiment data sources")
            suggestions.append("Add sentiment momentum indicators for better timing")
        elif 'Risk' in agent_name:
            suggestions.append("Develop dynamic position sizing based on market volatility")
            suggestions.append("Enhance correlation analysis for better risk assessment")
        
        suggestions.append("Implement machine learning models for continuous improvement")
        suggestions.append("Add ensemble methods to combine with other agent signals")
        
        return "\n".join([f"• {suggestion}" for suggestion in suggestions])

    def _generate_regime_explanation(self, market_data: Dict[str, Any]) -> str:
        """Generate regime explanation for market analysis."""
        market_regime = market_data.get('market_regime', 'unknown')
        volatility = market_data.get('volatility', 0)
        
        explanations = {
            'bull': f"The market is currently in a bullish regime with {volatility:.1%} volatility, characterized by upward price trends, positive investor sentiment, and strong buying pressure across major indices.",
            'bear': f"The market is in a bearish regime with {volatility:.1%} volatility, marked by downward price trends, negative investor sentiment, and selling pressure that may continue until key support levels are reached.",
            'sideways': f"The market is in a sideways regime with {volatility:.1%} volatility, showing range-bound price action without clear directional bias, creating opportunities for mean reversion strategies.",
            'volatile': f"The market is experiencing high volatility ({volatility:.1%}), with rapid price swings and increased uncertainty, requiring careful risk management and position sizing.",
            'trending': f"The market is in a trending regime with {volatility:.1%} volatility, showing clear directional movement with consistent momentum that favors trend-following strategies."
        }
        
        return explanations.get(market_regime, f"The market is in a {market_regime} regime with {volatility:.1%} volatility, showing unique characteristics that require specialized analysis and strategy adaptation.")

    def _generate_key_drivers(self, market_data: Dict[str, Any]) -> str:
        """Generate key market drivers."""
        drivers = [
            "• Economic indicators and central bank policy decisions",
            "• Corporate earnings reports and guidance updates",
            "• Geopolitical events and trade policy developments",
            "• Market sentiment and investor risk appetite",
            "• Technical support and resistance levels",
            "• Sector rotation and institutional flow patterns"
        ]
        
        return "\n".join(drivers)

    def _generate_risk_factors(self, market_data: Dict[str, Any]) -> str:
        """Generate risk factors for market analysis."""
        risks = [
            "• Increased market volatility and uncertainty",
            "• Potential for sudden trend reversals",
            "• Liquidity constraints in certain market segments",
            "• Regulatory changes affecting market structure",
            "• Economic data surprises and policy shifts",
            "• Geopolitical tensions and global trade disruptions"
        ]
        
        return "\n".join(risks)

    def _generate_opportunity_areas(self, market_data: Dict[str, Any]) -> str:
        """Generate opportunity areas for market analysis."""
        opportunities = [
            "• Sector-specific opportunities based on current regime",
            "• Mean reversion opportunities in oversold/overbought conditions",
            "• Momentum continuation in trending markets",
            "• Volatility trading opportunities in range-bound markets",
            "• Cross-asset correlation opportunities",
            "• Event-driven trading around earnings and announcements"
        ]
        
        return "\n".join(opportunities)

    def _generate_historical_context(self, market_data: Dict[str, Any]) -> str:
        """Generate historical context for market analysis."""
        return "Historical analysis shows that similar market regimes have typically lasted 2-6 weeks, with performance patterns that can inform current strategy selection. Past transitions between regimes have often been preceded by specific technical and fundamental indicators that we continue to monitor."

    def _generate_forward_looking(self, market_data: Dict[str, Any]) -> str:
        """Generate forward-looking analysis."""
        return "Looking ahead, we expect the current market regime to continue for the near term, with potential transition signals to monitor including key economic data releases, central bank communications, and technical breakouts from current ranges. Risk management remains paramount as regime transitions can occur rapidly."

    def _generate_trade_recommendations(self, trade_data: Dict[str, Any]) -> List[str]:
        """Generate trade recommendations."""
        return [
            "Monitor position closely for any changes in market conditions",
            "Set appropriate stop-loss levels based on risk tolerance",
            "Consider scaling out of position if confidence decreases",
            "Review and adjust position size based on portfolio allocation"
        ]

    def _generate_agent_recommendations(self, agent_data: Dict[str, Any]) -> List[str]:
        """Generate agent recommendations."""
        return [
            "Continue monitoring agent performance across different market regimes",
            "Consider adjusting agent weights based on current market conditions",
            "Implement suggested improvements to enhance signal accuracy",
            "Regular review of agent parameters and optimization opportunities"
        ]

    def _generate_market_recommendations(self, market_data: Dict[str, Any]) -> List[str]:
        """Generate market recommendations."""
        return [
            "Adjust portfolio allocation based on current market regime",
            "Implement regime-specific trading strategies",
            "Monitor key indicators for regime transition signals",
            "Maintain appropriate risk management for current volatility levels"
        ]

    async def generate_explanation(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation based on request."""
        logger.info(f"Generating {request.explanation_type.value} explanation with {request.tone.value} tone")
        
        if request.explanation_type == ExplanationType.TRADE_DECISION:
            return await self.explain_trade_decision(request.data, request.tone)
        elif request.explanation_type == ExplanationType.AGENT_PERFORMANCE:
            return await self.explain_agent_performance(request.data, request.tone)
        elif request.explanation_type == ExplanationType.MARKET_ANALYSIS:
            return await self.explain_market_analysis(request.data, request.tone)
        else:
            raise ValueError(f"Unsupported explanation type: {request.explanation_type}")

    async def get_explanation_summary(self) -> Dict[str, Any]:
        """Get explanation agent summary."""
        return {
            "explanation_types_supported": [e.value for e in ExplanationType],
            "tones_available": [t.value for t in ExplanationTone],
            "templates_loaded": len(self.explanation_templates),
            "last_explanation_generated": datetime.now().isoformat()
        }
