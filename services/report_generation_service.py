import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
from pathlib import Path

from agents.report_agent import ReportSummary, ReportType, ReportFormat
from agents.llm_explain_agent import LLMExplainAgent, ExplanationType, ExplanationTone

logger = logging.getLogger(__name__)

@dataclass
class ReportDeliveryConfig:
    email_enabled: bool = False
    slack_enabled: bool = False
    telegram_enabled: bool = False
    webhook_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    webhook_url: str = ""

@dataclass
class GeneratedReport:
    report_id: str
    report_type: ReportType
    format: ReportFormat
    content: str
    file_path: Optional[str] = None
    generated_at: str = ""
    file_size: int = 0
    delivery_status: Dict[str, bool] = field(default_factory=dict)

@dataclass
class ReportTemplate:
    template_id: str
    name: str
    description: str
    report_type: ReportType
    format: ReportFormat
    template_content: str
    variables: List[str] = field(default_factory=list)

class ReportGenerationService:
    def __init__(self, output_directory: str = "reports"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.llm_explain_agent = LLMExplainAgent()
        self.delivery_config = ReportDeliveryConfig()
        self.templates = self._load_default_templates()
        
        logger.info(f"ReportGenerationService initialized with output directory: {self.output_directory}")

    def _load_default_templates(self) -> Dict[str, ReportTemplate]:
        """Load default report templates."""
        templates = {}
        
        # Daily Report HTML Template
        daily_html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Trading Report - {report_period}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #007bff; margin: 0; }
        .header p { color: #666; margin: 5px 0; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background-color: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }
        .summary-card h3 { margin: 0 0 10px 0; color: #333; }
        .summary-card .value { font-size: 24px; font-weight: bold; color: #007bff; }
        .summary-card .label { color: #666; font-size: 14px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        .table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .table th { background-color: #f8f9fa; font-weight: bold; color: #333; }
        .table tr:hover { background-color: #f5f5f5; }
        .positive { color: #28a745; font-weight: bold; }
        .negative { color: #dc3545; font-weight: bold; }
        .neutral { color: #6c757d; }
        .explanation { background-color: #e9ecef; padding: 15px; border-radius: 6px; margin-top: 15px; }
        .explanation h4 { margin-top: 0; color: #495057; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Daily Trading Report</h1>
            <p>{report_period}</p>
            <p>Generated on {generated_at}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Trades</h3>
                <div class="value">{total_trades}</div>
                <div class="label">Trades executed</div>
            </div>
            <div class="summary-card">
                <h3>Win Rate</h3>
                <div class="value {win_rate_class}">{win_rate:.1f}%</div>
                <div class="label">Successful trades</div>
            </div>
            <div class="summary-card">
                <h3>Total P&L</h3>
                <div class="value {pnl_class}">${total_pnl:,.2f}</div>
                <div class="label">Profit/Loss</div>
            </div>
            <div class="summary-card">
                <h3>Best Agent</h3>
                <div class="value">{best_performing_agent}</div>
                <div class="label">Top performer</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Trade Summary</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Successful Trades</td>
                        <td class="positive">{successful_trades}</td>
                        <td class="neutral">-</td>
                    </tr>
                    <tr>
                        <td>Failed Trades</td>
                        <td class="negative">{failed_trades}</td>
                        <td class="neutral">-</td>
                    </tr>
                    <tr>
                        <td>Open Trades</td>
                        <td class="neutral">{open_trades}</td>
                        <td class="neutral">-</td>
                    </tr>
                    <tr>
                        <td>Average Trade Duration</td>
                        <td class="neutral">{avg_trade_duration:.1f} days</td>
                        <td class="neutral">-</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Agent Performance</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Win Rate</th>
                        <th>Total Signals</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
                    {agent_performance_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Market Regime Analysis</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Regime</th>
                        <th>Duration</th>
                        <th>Trades</th>
                        <th>Success Rate</th>
                        <th>Best Agent</th>
                    </tr>
                </thead>
                <tbody>
                    {market_regime_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>AI Explanation</h2>
            <div class="explanation">
                <h4>Daily Performance Analysis</h4>
                <p>{daily_explanation}</p>
            </div>
        </div>
        
        <div class="footer">
            <p>This report was generated automatically by the AI Market Analysis System</p>
            <p>For questions or support, please contact the system administrator</p>
        </div>
    </div>
</body>
</html>
"""
        
        templates["daily_html"] = ReportTemplate(
            template_id="daily_html",
            name="Daily Report HTML",
            description="HTML format daily trading report with comprehensive analysis",
            report_type=ReportType.DAILY,
            format=ReportFormat.HTML,
            template_content=daily_html_template,
            variables=["report_period", "generated_at", "total_trades", "win_rate", "win_rate_class", 
                      "total_pnl", "pnl_class", "best_performing_agent", "successful_trades", 
                      "failed_trades", "open_trades", "avg_trade_duration", "agent_performance_rows",
                      "market_regime_rows", "daily_explanation"]
        )
        
        # Weekly Report Markdown Template
        weekly_markdown_template = """
# Weekly Trading Report

**Period:** {report_period}  
**Generated:** {generated_at}

## Executive Summary

- **Total Trades:** {total_trades}
- **Win Rate:** {win_rate:.1f}%
- **Total P&L:** ${total_pnl:,.2f}
- **Best Performing Agent:** {best_performing_agent}

## Performance Metrics

| Metric | Value | Change |
|--------|-------|--------|
| Successful Trades | {successful_trades} | - |
| Failed Trades | {failed_trades} | - |
| Open Trades | {open_trades} | - |
| Average Duration | {avg_trade_duration:.1f} days | - |

## Agent Performance

| Agent | Win Rate | Signals | Performance |
|-------|----------|---------|-------------|
{agent_performance_rows}

## Market Regime Analysis

| Regime | Duration | Trades | Success Rate | Best Agent |
|--------|----------|--------|--------------|------------|
{market_regime_rows}

## AI Analysis

{daily_explanation}

## Recommendations

{recommendations}

---
*Generated by AI Market Analysis System*
"""
        
        templates["weekly_markdown"] = ReportTemplate(
            template_id="weekly_markdown",
            name="Weekly Report Markdown",
            description="Markdown format weekly trading report",
            report_type=ReportType.WEEKLY,
            format=ReportFormat.MARKDOWN,
            template_content=weekly_markdown_template,
            variables=["report_period", "generated_at", "total_trades", "win_rate", "total_pnl",
                      "best_performing_agent", "successful_trades", "failed_trades", "open_trades",
                      "avg_trade_duration", "agent_performance_rows", "market_regime_rows",
                      "daily_explanation", "recommendations"]
        )
        
        return templates

    async def generate_report(
        self, 
        report_summary: ReportSummary, 
        format: ReportFormat = ReportFormat.HTML,
        include_explanations: bool = True
    ) -> GeneratedReport:
        """Generate a report in the specified format."""
        logger.info(f"Generating {report_summary.report_type.value} report in {format.value} format")
        
        report_id = f"{report_summary.report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate AI explanations if requested
        explanations = {}
        if include_explanations:
            explanations = await self._generate_explanations(report_summary)
        
        # Generate content based on format
        if format == ReportFormat.HTML:
            content = await self._generate_html_report(report_summary, explanations)
            file_extension = "html"
        elif format == ReportFormat.MARKDOWN:
            content = await self._generate_markdown_report(report_summary, explanations)
            file_extension = "md"
        elif format == ReportFormat.JSON:
            content = await self._generate_json_report(report_summary, explanations)
            file_extension = "json"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file
        filename = f"{report_id}.{file_extension}"
        file_path = self.output_directory / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = file_path.stat().st_size
        
        generated_report = GeneratedReport(
            report_id=report_id,
            report_type=report_summary.report_type,
            format=format,
            content=content,
            file_path=str(file_path),
            generated_at=datetime.now().isoformat(),
            file_size=file_size
        )
        
        logger.info(f"Report generated successfully: {file_path} ({file_size} bytes)")
        return generated_report

    async def _generate_explanations(self, report_summary: ReportSummary) -> Dict[str, str]:
        """Generate AI explanations for the report."""
        explanations = {}
        
        # Generate daily performance explanation
        if report_summary.report_type in [ReportType.DAILY, ReportType.WEEKLY]:
            market_data = {
                "market_regime": "mixed",  # Could be derived from report data
                "volatility": 0.15,
                "trend_direction": "neutral"
            }
            
            explanation_response = await self.llm_explain_agent.explain_market_analysis(
                market_data, 
                ExplanationTone.PROFESSIONAL
            )
            explanations["daily_explanation"] = explanation_response.detailed_explanation
            explanations["recommendations"] = "\n".join([f"- {rec}" for rec in explanation_response.recommendations])
        
        return explanations

    async def _generate_html_report(self, report_summary: ReportSummary, explanations: Dict[str, str]) -> str:
        """Generate HTML report content."""
        template = self.templates["daily_html"]
        
        # Prepare template variables
        variables = {
            "report_period": report_summary.report_period,
            "generated_at": report_summary.generated_at,
            "total_trades": report_summary.total_trades,
            "win_rate": report_summary.win_rate,
            "win_rate_class": "positive" if report_summary.win_rate >= 60 else "negative" if report_summary.win_rate < 50 else "neutral",
            "total_pnl": report_summary.total_pnl,
            "pnl_class": "positive" if report_summary.total_pnl >= 0 else "negative",
            "best_performing_agent": report_summary.best_performing_agent,
            "successful_trades": report_summary.successful_trades,
            "failed_trades": report_summary.failed_trades,
            "open_trades": report_summary.open_trades,
            "avg_trade_duration": report_summary.avg_trade_duration,
            "agent_performance_rows": self._generate_agent_performance_rows(report_summary),
            "market_regime_rows": self._generate_market_regime_rows(report_summary),
            "daily_explanation": explanations.get("daily_explanation", "No explanation available."),
            "recommendations": explanations.get("recommendations", "No recommendations available.")
        }
        
        return template.template_content.format(**variables)

    async def _generate_markdown_report(self, report_summary: ReportSummary, explanations: Dict[str, str]) -> str:
        """Generate Markdown report content."""
        template = self.templates["weekly_markdown"]
        
        # Prepare template variables
        variables = {
            "report_period": report_summary.report_period,
            "generated_at": report_summary.generated_at,
            "total_trades": report_summary.total_trades,
            "win_rate": report_summary.win_rate,
            "total_pnl": report_summary.total_pnl,
            "best_performing_agent": report_summary.best_performing_agent,
            "successful_trades": report_summary.successful_trades,
            "failed_trades": report_summary.failed_trades,
            "open_trades": report_summary.open_trades,
            "avg_trade_duration": report_summary.avg_trade_duration,
            "agent_performance_rows": self._generate_agent_performance_rows_markdown(report_summary),
            "market_regime_rows": self._generate_market_regime_rows_markdown(report_summary),
            "daily_explanation": explanations.get("daily_explanation", "No explanation available."),
            "recommendations": explanations.get("recommendations", "No recommendations available.")
        }
        
        return template.template_content.format(**variables)

    async def _generate_json_report(self, report_summary: ReportSummary, explanations: Dict[str, str]) -> str:
        """Generate JSON report content."""
        report_data = {
            "report_id": f"{report_summary.report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_type": report_summary.report_type.value,
            "report_period": report_summary.report_period,
            "generated_at": report_summary.generated_at,
            "summary": {
                "total_trades": report_summary.total_trades,
                "successful_trades": report_summary.successful_trades,
                "failed_trades": report_summary.failed_trades,
                "open_trades": report_summary.open_trades,
                "total_pnl": report_summary.total_pnl,
                "total_pnl_percent": report_summary.total_pnl_percent,
                "win_rate": report_summary.win_rate,
                "avg_trade_duration": report_summary.avg_trade_duration,
                "best_performing_agent": report_summary.best_performing_agent,
                "worst_performing_agent": report_summary.worst_performing_agent
            },
            "market_regimes": [asdict(regime) for regime in report_summary.market_regimes],
            "forecast_errors": [asdict(error) for error in report_summary.forecast_errors],
            "explanations": explanations
        }
        
        return json.dumps(report_data, indent=2, default=str)

    def _generate_agent_performance_rows(self, report_summary: ReportSummary) -> str:
        """Generate agent performance table rows for HTML."""
        # Mock agent performance data
        agents = [
            {"name": "MomentumAgent", "win_rate": 65.2, "signals": 45, "performance": "Good"},
            {"name": "SentimentAgent", "win_rate": 58.7, "signals": 38, "performance": "Average"},
            {"name": "RiskAgent", "win_rate": 72.1, "signals": 52, "performance": "Excellent"},
            {"name": "VolatilityAgent", "win_rate": 61.3, "signals": 41, "performance": "Good"}
        ]
        
        rows = []
        for agent in agents:
            performance_class = "positive" if agent["win_rate"] >= 65 else "negative" if agent["win_rate"] < 55 else "neutral"
            rows.append(f"""
                    <tr>
                        <td>{agent['name']}</td>
                        <td class="{performance_class}">{agent['win_rate']:.1f}%</td>
                        <td>{agent['signals']}</td>
                        <td class="{performance_class}">{agent['performance']}</td>
                    </tr>
            """)
        
        return "\n".join(rows)

    def _generate_market_regime_rows(self, report_summary: ReportSummary) -> str:
        """Generate market regime table rows for HTML."""
        # Mock market regime data
        regimes = [
            {"regime": "Bull", "duration": 5, "trades": 12, "success_rate": 75.0, "best_agent": "MomentumAgent"},
            {"regime": "Sideways", "duration": 3, "trades": 8, "success_rate": 62.5, "best_agent": "RiskAgent"},
            {"regime": "Volatile", "duration": 2, "trades": 6, "success_rate": 50.0, "best_agent": "VolatilityAgent"}
        ]
        
        rows = []
        for regime in regimes:
            success_class = "positive" if regime["success_rate"] >= 65 else "negative" if regime["success_rate"] < 55 else "neutral"
            rows.append(f"""
                    <tr>
                        <td>{regime['regime']}</td>
                        <td>{regime['duration']} days</td>
                        <td>{regime['trades']}</td>
                        <td class="{success_class}">{regime['success_rate']:.1f}%</td>
                        <td>{regime['best_agent']}</td>
                    </tr>
            """)
        
        return "\n".join(rows)

    def _generate_agent_performance_rows_markdown(self, report_summary: ReportSummary) -> str:
        """Generate agent performance table rows for Markdown."""
        agents = [
            {"name": "MomentumAgent", "win_rate": 65.2, "signals": 45, "performance": "Good"},
            {"name": "SentimentAgent", "win_rate": 58.7, "signals": 38, "performance": "Average"},
            {"name": "RiskAgent", "win_rate": 72.1, "signals": 52, "performance": "Excellent"},
            {"name": "VolatilityAgent", "win_rate": 61.3, "signals": 41, "performance": "Good"}
        ]
        
        rows = []
        for agent in agents:
            rows.append(f"| {agent['name']} | {agent['win_rate']:.1f}% | {agent['signals']} | {agent['performance']} |")
        
        return "\n".join(rows)

    def _generate_market_regime_rows_markdown(self, report_summary: ReportSummary) -> str:
        """Generate market regime table rows for Markdown."""
        regimes = [
            {"regime": "Bull", "duration": 5, "trades": 12, "success_rate": 75.0, "best_agent": "MomentumAgent"},
            {"regime": "Sideways", "duration": 3, "trades": 8, "success_rate": 62.5, "best_agent": "RiskAgent"},
            {"regime": "Volatile", "duration": 2, "trades": 6, "success_rate": 50.0, "best_agent": "VolatilityAgent"}
        ]
        
        rows = []
        for regime in regimes:
            rows.append(f"| {regime['regime']} | {regime['duration']} days | {regime['trades']} | {regime['success_rate']:.1f}% | {regime['best_agent']} |")
        
        return "\n".join(rows)

    async def deliver_report(self, report: GeneratedReport, delivery_config: Optional[ReportDeliveryConfig] = None) -> Dict[str, bool]:
        """Deliver report via configured channels."""
        if delivery_config is None:
            delivery_config = self.delivery_config
        
        delivery_status = {}
        
        # Email delivery
        if delivery_config.email_enabled and delivery_config.email_recipients:
            try:
                await self._deliver_via_email(report, delivery_config.email_recipients)
                delivery_status["email"] = True
                logger.info(f"Report delivered via email to {len(delivery_config.email_recipients)} recipients")
            except Exception as e:
                delivery_status["email"] = False
                logger.error(f"Failed to deliver report via email: {e}")
        
        # Slack delivery
        if delivery_config.slack_enabled and delivery_config.slack_webhook_url:
            try:
                await self._deliver_via_slack(report, delivery_config.slack_webhook_url)
                delivery_status["slack"] = True
                logger.info("Report delivered via Slack")
            except Exception as e:
                delivery_status["slack"] = False
                logger.error(f"Failed to deliver report via Slack: {e}")
        
        # Telegram delivery
        if delivery_config.telegram_enabled and delivery_config.telegram_bot_token and delivery_config.telegram_chat_id:
            try:
                await self._deliver_via_telegram(report, delivery_config.telegram_bot_token, delivery_config.telegram_chat_id)
                delivery_status["telegram"] = True
                logger.info("Report delivered via Telegram")
            except Exception as e:
                delivery_status["telegram"] = False
                logger.error(f"Failed to deliver report via Telegram: {e}")
        
        # Webhook delivery
        if delivery_config.webhook_enabled and delivery_config.webhook_url:
            try:
                await self._deliver_via_webhook(report, delivery_config.webhook_url)
                delivery_status["webhook"] = True
                logger.info("Report delivered via webhook")
            except Exception as e:
                delivery_status["webhook"] = False
                logger.error(f"Failed to deliver report via webhook: {e}")
        
        return delivery_status

    async def _deliver_via_email(self, report: GeneratedReport, recipients: List[str]):
        """Deliver report via email (mock implementation)."""
        # In a real implementation, this would use an email service like SendGrid, AWS SES, etc.
        logger.info(f"Mock email delivery to {recipients} for report {report.report_id}")
        await asyncio.sleep(0.1)  # Simulate network delay

    async def _deliver_via_slack(self, report: GeneratedReport, webhook_url: str):
        """Deliver report via Slack (mock implementation)."""
        # In a real implementation, this would send a POST request to the Slack webhook
        logger.info(f"Mock Slack delivery to {webhook_url} for report {report.report_id}")
        await asyncio.sleep(0.1)  # Simulate network delay

    async def _deliver_via_telegram(self, report: GeneratedReport, bot_token: str, chat_id: str):
        """Deliver report via Telegram (mock implementation)."""
        # In a real implementation, this would use the Telegram Bot API
        logger.info(f"Mock Telegram delivery to chat {chat_id} for report {report.report_id}")
        await asyncio.sleep(0.1)  # Simulate network delay

    async def _deliver_via_webhook(self, report: GeneratedReport, webhook_url: str):
        """Deliver report via webhook (mock implementation)."""
        # In a real implementation, this would send a POST request to the webhook URL
        logger.info(f"Mock webhook delivery to {webhook_url} for report {report.report_id}")
        await asyncio.sleep(0.1)  # Simulate network delay

    async def get_report_summary(self) -> Dict[str, Any]:
        """Get report generation service summary."""
        return {
            "output_directory": str(self.output_directory),
            "templates_available": len(self.templates),
            "delivery_channels": {
                "email": self.delivery_config.email_enabled,
                "slack": self.delivery_config.slack_enabled,
                "telegram": self.delivery_config.telegram_enabled,
                "webhook": self.delivery_config.webhook_enabled
            },
            "supported_formats": [f.value for f in ReportFormat],
            "supported_report_types": [t.value for t in ReportType]
        }
