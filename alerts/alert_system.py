"""
Real-time Alert and Notification System

This module provides comprehensive alerting capabilities for
the AI Market Analysis System.
"""

import asyncio
import smtplib
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import websockets
import threading
import time
from pathlib import Path

from ..agents.base_agent import AgentSignal, SignalType

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts."""
    SIGNAL_ALERT = "signal_alert"
    PERFORMANCE_ALERT = "performance_alert"
    SYSTEM_ALERT = "system_alert"
    RISK_ALERT = "risk_alert"
    REGIME_ALERT = "regime_alert"
    ERROR_ALERT = "error_alert"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    LOG = "log"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Python expression to evaluate
    threshold: float
    cooldown_minutes: int = 60
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.LOG])
    description: str = ""


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class NotificationConfig:
    """Notification configuration."""
    email_config: Optional[Dict[str, Any]] = None
    webhook_config: Optional[Dict[str, Any]] = None
    websocket_config: Optional[Dict[str, Any]] = None


class AlertSystem:
    """
    Real-time alert and notification system.
    
    This class provides:
    - Real-time alert monitoring
    - Multiple notification channels
    - Alert rule management
    - Alert history and tracking
    - Cooldown and throttling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Alert System.
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            'check_interval': 30,  # seconds
            'max_alerts_per_hour': 100,
            'alert_history_size': 1000,
            'notification_config': {
                'email_config': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_email': '',
                    'to_emails': []
                },
                'webhook_config': {
                    'url': '',
                    'headers': {'Content-Type': 'application/json'},
                    'timeout': 10
                },
                'websocket_config': {
                    'host': 'localhost',
                    'port': 8765,
                    'path': '/alerts'
                }
            },
            'default_rules': [
                {
                    'name': 'high_confidence_signal',
                    'alert_type': AlertType.SIGNAL_ALERT,
                    'severity': AlertSeverity.MEDIUM,
                    'condition': 'signal.confidence > 0.8',
                    'threshold': 0.8,
                    'cooldown_minutes': 30,
                    'channels': [NotificationChannel.LOG, NotificationChannel.DASHBOARD],
                    'description': 'High confidence trading signal detected'
                },
                {
                    'name': 'system_error',
                    'alert_type': AlertType.ERROR_ALERT,
                    'severity': AlertSeverity.HIGH,
                    'condition': 'error_count > 5',
                    'threshold': 5,
                    'cooldown_minutes': 15,
                    'channels': [NotificationChannel.LOG, NotificationChannel.EMAIL],
                    'description': 'Multiple system errors detected'
                },
                {
                    'name': 'low_performance',
                    'alert_type': AlertType.PERFORMANCE_ALERT,
                    'severity': AlertSeverity.MEDIUM,
                    'condition': 'win_rate < 0.4',
                    'threshold': 0.4,
                    'cooldown_minutes': 60,
                    'channels': [NotificationChannel.LOG, NotificationChannel.DASHBOARD],
                    'description': 'Agent performance below threshold'
                },
                {
                    'name': 'high_risk',
                    'alert_type': AlertType.RISK_ALERT,
                    'severity': AlertSeverity.HIGH,
                    'condition': 'var_95 > 0.05',
                    'threshold': 0.05,
                    'cooldown_minutes': 30,
                    'channels': [NotificationChannel.LOG, NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                    'description': 'High risk detected (VaR > 5%)'
                }
            ]
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.notification_config = NotificationConfig(**self.config['notification_config'])
        self.alert_counters = {}
        self.last_alert_times = {}
        self.running = False
        self.websocket_clients = set()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info(f"Initialized AlertSystem with {len(self.alert_rules)} rules")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        try:
            for rule_config in self.config['default_rules']:
                rule = AlertRule(
                    name=rule_config['name'],
                    alert_type=AlertType(rule_config['alert_type']),
                    severity=AlertSeverity(rule_config['severity']),
                    condition=rule_config['condition'],
                    threshold=rule_config['threshold'],
                    cooldown_minutes=rule_config.get('cooldown_minutes', 60),
                    enabled=rule_config.get('enabled', True),
                    channels=[NotificationChannel(ch) for ch in rule_config.get('channels', ['log'])],
                    description=rule_config.get('description', '')
                )
                self.alert_rules[rule.name] = rule
                
        except Exception as e:
            logger.error(f"Failed to initialize default rules: {e}")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        try:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
        except Exception as e:
            logger.error(f"Failed to add alert rule {rule.name}: {e}")
    
    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        try:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
        except Exception as e:
            logger.error(f"Failed to remove alert rule {rule_name}: {e}")
    
    def update_alert_rule(self, rule_name: str, updates: Dict[str, Any]) -> None:
        """Update an alert rule."""
        try:
            if rule_name in self.alert_rules:
                rule = self.alert_rules[rule_name]
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logger.info(f"Updated alert rule: {rule_name}")
        except Exception as e:
            logger.error(f"Failed to update alert rule {rule_name}: {e}")
    
    def check_signal_alerts(self, signal: AgentSignal) -> None:
        """Check for signal-based alerts."""
        try:
            for rule_name, rule in self.alert_rules.items():
                if (rule.alert_type == AlertType.SIGNAL_ALERT and 
                    rule.enabled and 
                    self._is_rule_triggered(rule, signal)):
                    
                    self._trigger_alert(rule, signal)
                    
        except Exception as e:
            logger.error(f"Signal alert check failed: {e}")
    
    def check_performance_alerts(self, performance_data: Dict[str, Any]) -> None:
        """Check for performance-based alerts."""
        try:
            for rule_name, rule in self.alert_rules.items():
                if (rule.alert_type == AlertType.PERFORMANCE_ALERT and 
                    rule.enabled and 
                    self._is_rule_triggered(rule, performance_data)):
                    
                    self._trigger_alert(rule, performance_data)
                    
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
    
    def check_system_alerts(self, system_data: Dict[str, Any]) -> None:
        """Check for system-based alerts."""
        try:
            for rule_name, rule in self.alert_rules.items():
                if (rule.alert_type == AlertType.SYSTEM_ALERT and 
                    rule.enabled and 
                    self._is_rule_triggered(rule, system_data)):
                    
                    self._trigger_alert(rule, system_data)
                    
        except Exception as e:
            logger.error(f"System alert check failed: {e}")
    
    def check_risk_alerts(self, risk_data: Dict[str, Any]) -> None:
        """Check for risk-based alerts."""
        try:
            for rule_name, rule in self.alert_rules.items():
                if (rule.alert_type == AlertType.RISK_ALERT and 
                    rule.enabled and 
                    self._is_rule_triggered(rule, risk_data)):
                    
                    self._trigger_alert(rule, risk_data)
                    
        except Exception as e:
            logger.error(f"Risk alert check failed: {e}")
    
    def _is_rule_triggered(self, rule: AlertRule, data: Any) -> bool:
        """Check if a rule is triggered."""
        try:
            # Check cooldown
            if self._is_in_cooldown(rule):
                return False
            
            # Check rate limiting
            if self._is_rate_limited():
                return False
            
            # Evaluate condition
            # Create a safe evaluation context
            context = {
                'signal': data if hasattr(data, 'confidence') else None,
                'performance': data if isinstance(data, dict) and 'win_rate' in data else None,
                'system': data if isinstance(data, dict) and 'error_count' in data else None,
                'risk': data if isinstance(data, dict) and 'var_95' in data else None,
                'data': data
            }
            
            # Simple condition evaluation (in production, use a safer method)
            try:
                result = eval(rule.condition, {"__builtins__": {}}, context)
                return bool(result)
            except:
                # Fallback to threshold comparison
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)) and value > rule.threshold:
                            return True
                return False
                
        except Exception as e:
            logger.error(f"Rule evaluation failed for {rule.name}: {e}")
            return False
    
    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period."""
        try:
            if rule.name not in self.last_alert_times:
                return False
            
            last_alert_time = self.last_alert_times[rule.name]
            cooldown_duration = timedelta(minutes=rule.cooldown_minutes)
            
            return datetime.now() - last_alert_time < cooldown_duration
            
        except Exception as e:
            logger.error(f"Cooldown check failed for {rule.name}: {e}")
            return False
    
    def _is_rate_limited(self) -> bool:
        """Check if we're hitting rate limits."""
        try:
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            if current_hour not in self.alert_counters:
                self.alert_counters[current_hour] = 0
            
            return self.alert_counters[current_hour] >= self.config['max_alerts_per_hour']
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, data: Any) -> None:
        """Trigger an alert."""
        try:
            # Create alert
            alert_id = f"{rule.name}_{datetime.now().isoformat()}"
            alert = Alert(
                id=alert_id,
                rule_name=rule.name,
                alert_type=rule.alert_type,
                severity=rule.severity,
                message=self._generate_alert_message(rule, data),
                timestamp=datetime.now(),
                data=self._extract_alert_data(data)
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update counters
            self.last_alert_times[rule.name] = datetime.now()
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            self.alert_counters[current_hour] = self.alert_counters.get(current_hour, 0) + 1
            
            # Send notifications
            self._send_notifications(alert, rule)
            
            # Cleanup old alerts
            self._cleanup_old_alerts()
            
            logger.info(f"Alert triggered: {rule.name} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Alert triggering failed for {rule.name}: {e}")
    
    def _generate_alert_message(self, rule: AlertRule, data: Any) -> str:
        """Generate alert message."""
        try:
            if rule.alert_type == AlertType.SIGNAL_ALERT and hasattr(data, 'confidence'):
                return f"High confidence signal from {data.agent_name}: {data.signal_type.value} (confidence: {data.confidence:.2f})"
            elif rule.alert_type == AlertType.PERFORMANCE_ALERT:
                return f"Performance alert: {rule.description} (current: {data.get('win_rate', 0):.2f})"
            elif rule.alert_type == AlertType.SYSTEM_ALERT:
                return f"System alert: {rule.description} (errors: {data.get('error_count', 0)})"
            elif rule.alert_type == AlertType.RISK_ALERT:
                return f"Risk alert: {rule.description} (VaR: {data.get('var_95', 0):.2%})"
            else:
                return f"Alert: {rule.description}"
                
        except Exception as e:
            logger.error(f"Alert message generation failed: {e}")
            return f"Alert: {rule.description}"
    
    def _extract_alert_data(self, data: Any) -> Dict[str, Any]:
        """Extract relevant data for alert."""
        try:
            if hasattr(data, '__dict__'):
                return {k: v for k, v in data.__dict__.items() if not k.startswith('_')}
            elif isinstance(data, dict):
                return data
            else:
                return {'data': str(data)}
                
        except Exception as e:
            logger.error(f"Alert data extraction failed: {e}")
            return {}
    
    def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications through configured channels."""
        try:
            for channel in rule.channels:
                if channel == NotificationChannel.LOG:
                    self._send_log_notification(alert)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(alert)
                elif channel == NotificationChannel.WEBSOCKET:
                    self._send_websocket_notification(alert)
                elif channel == NotificationChannel.DASHBOARD:
                    self._send_dashboard_notification(alert)
                    
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
    
    def _send_log_notification(self, alert: Alert) -> None:
        """Send log notification."""
        try:
            severity_map = {
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }
            
            log_level = severity_map.get(alert.severity, logging.INFO)
            logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
            
        except Exception as e:
            logger.error(f"Log notification failed: {e}")
    
    def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification."""
        try:
            if not self.notification_config.email_config:
                return
            
            config = self.notification_config.email_config
            
            if not config.get('username') or not config.get('password'):
                logger.warning("Email configuration incomplete")
                return
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"AI Market Analysis Alert: {alert.severity.value.upper()}"
            
            # Create body
            body = f"""
            Alert Details:
            - Type: {alert.alert_type.value}
            - Severity: {alert.severity.value}
            - Time: {alert.timestamp.isoformat()}
            - Message: {alert.message}
            
            Alert Data:
            {json.dumps(alert.data, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
    
    def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification."""
        try:
            if not self.notification_config.webhook_config:
                return
            
            config = self.notification_config.webhook_config
            
            if not config.get('url'):
                logger.warning("Webhook URL not configured")
                return
            
            # Prepare payload
            payload = {
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            # Send webhook
            response = requests.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=config.get('timeout', 10)
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent for alert {alert.id}")
            else:
                logger.warning(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    def _send_websocket_notification(self, alert: Alert) -> None:
        """Send websocket notification."""
        try:
            if not self.websocket_clients:
                return
            
            # Prepare message
            message = {
                'type': 'alert',
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    asyncio.create_task(client.send(json.dumps(message)))
                except:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
            if self.websocket_clients:
                logger.info(f"Websocket notification sent to {len(self.websocket_clients)} clients")
                
        except Exception as e:
            logger.error(f"Websocket notification failed: {e}")
    
    def _send_dashboard_notification(self, alert: Alert) -> None:
        """Send dashboard notification (stored for dashboard to retrieve)."""
        try:
            # This is handled by the dashboard polling the alert system
            logger.info(f"Dashboard notification queued for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Dashboard notification failed: {e}")
    
    def _cleanup_old_alerts(self) -> None:
        """Cleanup old alerts from history."""
        try:
            max_history = self.config['alert_history_size']
            
            if len(self.alert_history) > max_history:
                self.alert_history = self.alert_history[-max_history:]
            
            # Cleanup old active alerts (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            old_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.timestamp < cutoff_time
            ]
            
            for alert_id in old_alerts:
                del self.active_alerts[alert_id]
                
        except Exception as e:
            logger.error(f"Alert cleanup failed: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            total_alerts = len(self.alert_history)
            active_alerts = len(self.active_alerts)
            
            # Count by severity
            severity_counts = {}
            for alert in self.alert_history:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by type
            type_counts = {}
            for alert in self.alert_history:
                alert_type = alert.alert_type.value
                type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'severity_distribution': severity_counts,
                'type_distribution': type_counts,
                'rules_count': len(self.alert_rules),
                'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled])
            }
            
        except Exception as e:
            logger.error(f"Alert statistics calculation failed: {e}")
            return {}
    
    def start_monitoring(self) -> None:
        """Start the alert monitoring system."""
        try:
            self.running = True
            logger.info("Alert monitoring system started")
            
        except Exception as e:
            logger.error(f"Alert monitoring start failed: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop the alert monitoring system."""
        try:
            self.running = False
            logger.info("Alert monitoring system stopped")
            
        except Exception as e:
            logger.error(f"Alert monitoring stop failed: {e}")
