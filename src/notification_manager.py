import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import threading
import time

class NotificationManager:
    def __init__(self, max_notifications: int = 50):
        self.max_notifications = max_notifications
        self.notifications = deque(maxlen=max_notifications)
        self.notification_lock = threading.Lock()
        self.subscribers = []  
        
        logging.info("Notification Manager initialized")
    
    def add_notification(self, notification: Dict) -> bool:
        """Add a new notification"""
        try:
            with self.notification_lock:
                if "timestamp" not in notification:
                    notification["timestamp"] = datetime.now().isoformat()

                notification["id"] = f"notif_{int(time.time() * 1000)}"

                self.notifications.append(notification)

                logging.info(f"Notification added: {notification.get('message', 'Unknown')}")

                self._notify_subscribers(notification)
                
                return True
                
        except Exception as e:
            logging.error(f"Error adding notification: {str(e)}")
            return False
    
    def _notify_subscribers(self, notification: Dict):
        """Notify all subscribers about new notification"""

        logging.info(f"Broadcasting notification: {notification.get('type', 'UNKNOWN')}")
    
    def get_recent_notifications(self, limit: int = 10) -> List[Dict]:
        """Get recent notifications"""
        try:
            with self.notification_lock:
                notifications_list = list(self.notifications)
                return notifications_list[-limit:] if notifications_list else []
        except Exception as e:
            logging.error(f"Error getting recent notifications: {str(e)}")
            return []
    
    def get_unread_notifications(self) -> List[Dict]:
        """Get unread notifications"""
        try:
            with self.notification_lock:
                return [notif for notif in self.notifications if not notif.get("read", False)]
        except Exception as e:
            logging.error(f"Error getting unread notifications: {str(e)}")
            return []
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        try:
            with self.notification_lock:
                for notif in self.notifications:
                    if notif.get("id") == notification_id:
                        notif["read"] = True
                        return True
                return False
        except Exception as e:
            logging.error(f"Error marking notification as read: {str(e)}")
            return False
    
    def mark_all_as_read(self) -> bool:
        """Mark all notifications as read"""
        try:
            with self.notification_lock:
                for notif in self.notifications:
                    notif["read"] = True
                return True
        except Exception as e:
            logging.error(f"Error marking all notifications as read: {str(e)}")
            return False
    
    def clear_notifications(self) -> bool:
        """Clear all notifications"""
        try:
            with self.notification_lock:
                self.notifications.clear()
                logging.info("All notifications cleared")
                return True
        except Exception as e:
            logging.error(f"Error clearing notifications: {str(e)}")
            return False
    
    def create_opportunity_notification(self, opportunity: Dict) -> Dict:
        """Create a formatted opportunity notification"""
        return {
            "type": "OPPORTUNITY",
            "priority": "HIGH",
            "title": f"🚀 Trading Opportunity: {opportunity.get('symbol', 'Unknown')}",
            "message": opportunity.get("message", "Opportunity detected"),
            "symbol": opportunity.get("symbol"),
            "alert_type": opportunity.get("alert_type"),
            "quote_data": opportunity.get("quote_data", {}),
            "action_required": True,
            "read": False
        }
    
    def create_order_notification(self, order: Dict) -> Dict:
        """Create a formatted order notification"""
        symbol = order.get("symbol", "Unknown")
        action = order.get("action", "Unknown")
        quantity = order.get("quantity", 0)
        status = order.get("status", "Unknown")
        
        return {
            "type": "ORDER",
            "priority": "MEDIUM",
            "title": f"📋 Order {status}: {symbol}",
            "message": f"{action} {quantity} shares of {symbol} - Status: {status}",
            "symbol": symbol,
            "order_id": order.get("order_id"),
            "order_data": order,
            "action_required": status in ["REJECTED", "CANCELLED"],
            "read": False
        }
    
    def create_portfolio_notification(self, portfolio: Dict) -> Dict:
        """Create a formatted portfolio notification"""
        total_pnl = portfolio.get("total_pnl", 0)
        pnl_percent = portfolio.get("total_pnl_percent", 0)
        
        emoji = "📈" if total_pnl >= 0 else "📉"
        
        return {
            "type": "PORTFOLIO",
            "priority": "LOW",
            "title": f"{emoji} Portfolio Update",
            "message": f"Total P&L: ₹{total_pnl:,.2f} ({pnl_percent:+.1f}%)",
            "portfolio_data": portfolio,
            "action_required": False,
            "read": False
        }
    
    def create_system_notification(self, message: str, priority: str = "LOW") -> Dict:
        """Create a system notification"""
        return {
            "type": "SYSTEM",
            "priority": priority.upper(),
            "title": "🔧 System Update",
            "message": message,
            "action_required": False,
            "read": False
        }
    
    def create_error_notification(self, error_message: str, context: str = "") -> Dict:
        """Create an error notification"""
        return {
            "type": "ERROR",
            "priority": "HIGH",
            "title": "⚠️ Error Alert",
            "message": f"{error_message} {context}".strip(),
            "action_required": True,
            "read": False
        }
    
    def create_signal_notification(self, signal: Dict) -> Dict:
        """Create a formatted trading signal notification.

        Used by the SmartOpportunityScanner for BUY/SELL/HOLD alerts.
        action_required is always False -- signals are alerts only, never auto-executed.
        """
        signal_type = signal.get("signal_type", "HOLD")
        confidence = signal.get("confidence", 0)
        symbol = signal.get("symbol", "Unknown")

        if signal_type == "BUY":
            emoji = "BUY SIGNAL"
            priority = "HIGH"
        elif signal_type == "SELL":
            emoji = "SELL SIGNAL"
            priority = "HIGH"
        else:
            emoji = "HOLD"
            priority = "MEDIUM"

        reasons = signal.get("reasons", [])
        reasons_text = "; ".join(reasons[:3]) if reasons else "Multiple technical indicators"

        title = f"{emoji}: {symbol} | Confidence: {confidence:.0f}%"

        message = signal.get("message", "")
        if not message:
            snapshot = signal.get("snapshot", {})
            price = snapshot.get("current_price", 0)
            message = (
                f"{signal_type} signal for {symbol} @ Rs.{price:.2f}\n"
                f"Confidence: {confidence:.0f}%\n"
                f"Reasons: {reasons_text}"
            )

        news_context = signal.get("news_context", "")
        if news_context:
            message += f"\n\n{news_context}"

        return {
            "type": "SIGNAL",
            "priority": priority,
            "title": title,
            "message": message,
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "reasons": reasons,
            "snapshot": signal.get("snapshot", {}),
            "news_context": news_context,
            "action_required": False,
            "read": False
        }

    def create_portfolio_guard_notification(self, alert: Dict) -> Dict:
        """Create a notification from the portfolio guard service."""
        severity = alert.get("severity", "INFO")
        symbol = alert.get("symbol", "Unknown")
        check_type = alert.get("check_type", "")
        message = alert.get("message", "")

        priority_map = {"URGENT": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"}
        priority = priority_map.get(severity, "LOW")
        tag = f"[{severity}]"

        return {
            "type": "PORTFOLIO_GUARD",
            "priority": priority,
            "title": f"{tag} Portfolio Alert: {symbol}",
            "message": message,
            "symbol": symbol,
            "check_type": check_type,
            "severity": severity,
            "action_required": severity == "URGENT",
            "read": False,
        }

    def create_radar_notification(self, results: List[Dict]) -> Dict:
        """Create a summary notification from market radar scan."""
        count = len(results)
        symbols = [r.get("symbol", "?") for r in results[:5]]
        summary = ", ".join(symbols)
        if count > 5:
            summary += f" +{count - 5} more"

        return {
            "type": "RADAR",
            "priority": "MEDIUM",
            "title": f"Market Radar: {count} stocks flagged",
            "message": f"Scan found {count} interesting stocks: {summary}",
            "results": results,
            "action_required": False,
            "read": False,
        }

    def create_news_notification(self, alert: Dict) -> Dict:
        """Create a notification for news events."""
        symbol = alert.get("symbol", "Market")
        urgent = alert.get("urgent", False)
        headline = alert.get("headline", "")
        source = alert.get("source", "")

        priority = "HIGH" if urgent else "LOW"
        tag = "[URGENT]" if urgent else "[NEWS]"

        return {
            "type": "NEWS",
            "priority": priority,
            "title": f"{tag} {symbol}: {headline[:60]}",
            "message": f"{headline}\nSource: {source}" if source else headline,
            "symbol": symbol,
            "urgent": urgent,
            "action_required": urgent,
            "read": False,
        }

    def get_notification_summary(self) -> Dict:
        """Get notification summary"""
        try:
            with self.notification_lock:
                total = len(self.notifications)
                unread = len(self.get_unread_notifications())

                type_counts = {}
                priority_counts = {}
                
                for notif in self.notifications:
                    notif_type = notif.get("type", "UNKNOWN")
                    priority = notif.get("priority", "LOW")
                    
                    type_counts[notif_type] = type_counts.get(notif_type, 0) + 1
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                return {
                    "total_notifications": total,
                    "unread_notifications": unread,
                    "type_breakdown": type_counts,
                    "priority_breakdown": priority_counts,
                    "last_notification": self.notifications[-1] if self.notifications else None
                }
                
        except Exception as e:
            logging.error(f"Error getting notification summary: {str(e)}")
            return {
                "total_notifications": 0,
                "unread_notifications": 0,
                "type_breakdown": {},
                "priority_breakdown": {},
                "last_notification": None
            }
    
    def format_notification_for_display(self, notification: Dict) -> str:
        """Format notification for display in Claude"""
        try:
            timestamp = notification.get("timestamp", "")
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M")
            else:
                time_str = "Unknown"
            
            title = notification.get("title", "Notification")
            message = notification.get("message", "")
            priority = notification.get("priority", "LOW")
            read_status = "✓" if notification.get("read", False) else "●"

            priority_emojis = {
                "HIGH": "🔴",
                "MEDIUM": "🟡",
                "LOW": "🔵"
            }
            
            priority_emoji = priority_emojis.get(priority, "⚪")
            
            return f"{read_status} {priority_emoji} [{time_str}] {title}\n   {message}"
            
        except Exception as e:
            logging.error(f"Error formatting notification: {str(e)}")
            return f"Error formatting notification: {str(e)}"
    
    def get_formatted_notifications(self, limit: int = 10) -> str:
        """Get formatted notifications for display"""
        try:
            notifications = self.get_recent_notifications(limit)
            
            if not notifications:
                return "No recent notifications"
            
            formatted_notifications = []
            for notif in reversed(notifications): 
                formatted_notifications.append(self.format_notification_for_display(notif))
            
            return "\n\n".join(formatted_notifications)
            
        except Exception as e:
            logging.error(f"Error getting formatted notifications: {str(e)}")
            return f"Error retrieving notifications: {str(e)}"

notification_manager = NotificationManager()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Notification Manager...")

    opportunity_data = {
        "symbol": "RELIANCE",
        "alert_type": "volume_spike",
        "message": "🚀 RELIANCE: Volume spike detected (4.2x average)",
        "quote_data": {
            "current_price": 2456.75,
            "change_percent": 3.2,
            "volume": 1500000
        }
    }
    
    notification_manager.add_notification(
        notification_manager.create_opportunity_notification(opportunity_data)
    )
    
    order_data = {
        "symbol": "TCS",
        "action": "BUY",
        "quantity": 10,
        "status": "PLACED",
        "order_id": "UP123456"
    }
    
    notification_manager.add_notification(
        notification_manager.create_order_notification(order_data)
    )
    
    notification_manager.add_notification(
        notification_manager.create_system_notification("Scanner started successfully", "MEDIUM")
    )

    print("\nRecent Notifications:")
    print(notification_manager.get_formatted_notifications())

    print("\nNotification Summary:")
    summary = notification_manager.get_notification_summary()
    print(json.dumps(summary, indent=2))