# analytics_engine.py
# Advanced Analytics System for WhatsApp Groups

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import statistics
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
import logging
from dataclasses import dataclass
from enum import Enum
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsTimeframe(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class MetricType(Enum):
    MESSAGE_VOLUME = "message_volume"
    SENTIMENT_DISTRIBUTION = "sentiment_distribution"
    RESPONSE_TIME = "response_time"
    USER_ENGAGEMENT = "user_engagement"
    CONTENT_FLAGGING = "content_flagging"
    LANGUAGE_USAGE = "language_usage"
    ACTIVITY_PATTERNS = "activity_patterns"

@dataclass
class AnalyticsMetric:
    metric_name: str
    value: float
    timestamp: datetime
    group_id: Optional[str] = None
    metadata: Dict = None

class AdvancedAnalyticsEngine:
    """Advanced analytics engine for comprehensive insights"""
    
    def __init__(self, group_db_path='whatsapp_groups.db', messages_db_path='whatsapp_messages.db'):
        self.group_db_path = group_db_path
        self.messages_db_path = messages_db_path
        self.init_analytics_tables()
    
    def init_analytics_tables(self):
        """Initialize analytics-specific tables"""
        try:
            conn = sqlite3.connect(self.group_db_path)
            cursor = conn.cursor()
            
            # Advanced metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT,
                    metric_type TEXT,
                    metric_name TEXT,
                    value REAL,
                    timestamp TIMESTAMP,
                    timeframe TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX(group_id, metric_type, timeframe)
                )
            ''')
            
            # Engagement tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_engagement (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT,
                    user_phone TEXT,
                    date DATE,
                    messages_sent INTEGER DEFAULT 0,
                    messages_received INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0,
                    engagement_score REAL DEFAULT 0,
                    influence_score REAL DEFAULT 0,
                    sentiment_score REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_id, user_phone, date)
                )
            ''')
            
            # Trend analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trend_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT,
                    trend_type TEXT,
                    trend_direction TEXT,
                    confidence_score REAL,
                    start_date DATE,
                    end_date DATE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Analytics tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analytics tables: {str(e)}")
    
    def calculate_group_health_score(self, group_id: str, timeframe_days: int = 30) -> Dict:
        """Calculate comprehensive group health score"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=timeframe_days)
            
            # Get message data
            messages_df = self._get_group_messages(group_id, start_date, end_date)
            
            if messages_df.empty:
                return {
                    'health_score': 0,
                    'components': {},
                    'recommendations': ['No message data available for analysis']
                }
            
            # Calculate component scores
            activity_score = self._calculate_activity_score(messages_df)
            sentiment_score = self._calculate_sentiment_health(messages_df)
            engagement_score = self._calculate_engagement_score(messages_df)
            safety_score = self._calculate_safety_score(messages_df)
            response_score = self._calculate_response_time_score(messages_df)
            
            # Weighted health score
            health_score = (
                activity_score * 0.25 +
                sentiment_score * 0.20 +
                engagement_score * 0.20 +
                safety_score * 0.25 +
                response_score * 0.10
            )
            
            components = {
                'activity_score': round(activity_score, 2),
                'sentiment_score': round(sentiment_score, 2),
                'engagement_score': round(engagement_score, 2),
                'safety_score': round(safety_score, 2),
                'response_score': round(response_score, 2)
            }
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(components)
            
            # Store metric
            self._store_metric(group_id, 'health_score', 'overall_health', health_score, 
                             metadata={'components': components})
            
            return {
                'health_score': round(health_score, 2),
                'components': components,
                'recommendations': recommendations,
                'analysis_period': f"{start_date.date()} to {end_date.date()}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_activity_score(self, messages_df: pd.DataFrame) -> float:
        """Calculate activity score based on message volume and distribution"""
        if messages_df.empty:
            return 0.0
        
        # Daily message counts
        daily_counts = messages_df.groupby(messages_df['timestamp'].dt.date).size()
        
        # Base score from message volume
        avg_daily_messages = daily_counts.mean()
        volume_score = min(avg_daily_messages / 50, 1.0) * 40  # Max 40 points for volume
        
        # Consistency score
        if len(daily_counts) > 1:
            cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else 0
            consistency_score = max(0, (1 - cv)) * 30  # Max 30 points for consistency
        else:
            consistency_score = 30
        
        # Peak usage patterns
        hourly_dist = messages_df['timestamp'].dt.hour.value_counts()
        peak_usage_score = min(len(hourly_dist[hourly_dist > 0]), 24) / 24 * 30  # Max 30 points
        
        return volume_score + consistency_score + peak_usage_score
    
    def _calculate_sentiment_health(self, messages_df: pd.DataFrame) -> float:
        """Calculate sentiment health score"""
        if messages_df.empty or 'sentiment' not in messages_df.columns:
            return 50.0  # Neutral baseline
        
        sentiment_counts = messages_df['sentiment'].value_counts()
        total_messages = len(messages_df)
        
        positive_ratio = sentiment_counts.get('positive', 0) / total_messages
        negative_ratio = sentiment_counts.get('negative', 0) / total_messages
        neutral_ratio = sentiment_counts.get('neutral', 0) / total_messages
        
        # Score calculation
        sentiment_score = (
            positive_ratio * 100 +
            neutral_ratio * 60 +
            negative_ratio * 20
        )
        
        return min(sentiment_score, 100)
    
    def _calculate_engagement_score(self, messages_df: pd.DataFrame) -> float:
        """Calculate user engagement score"""
        if messages_df.empty:
            return 0.0
        
        # Unique participants
        unique_senders = messages_df['sender_phone'].nunique()
        
        # Message distribution among users
        user_messages = messages_df['sender_phone'].value_counts()
        
        # Gini coefficient for message distribution (lower = more equal participation)
        gini = self._calculate_gini_coefficient(user_messages.values)
        equality_score = (1 - gini) * 50  # Max 50 points for equal participation
        
        # Participation breadth
        participation_score = min(unique_senders / 20, 1.0) * 50  # Max 50 points
        
        return equality_score + participation_score
    
    def _calculate_safety_score(self, messages_df: pd.DataFrame) -> float:
        """Calculate safety score based on flagged content"""
        if messages_df.empty:
            return 100.0  # Perfect safety if no messages
        
        total_messages = len(messages_df)
        
        if 'is_flagged' in messages_df.columns:
            flagged_messages = messages_df['is_flagged'].sum()
            flagged_ratio = flagged_messages / total_messages
            
            # Priority-weighted flagging
            if 'priority' in messages_df.columns:
                critical_flags = len(messages_df[messages_df['priority'] == 'CRITICAL'])
                high_flags = len(messages_df[messages_df['priority'] == 'HIGH'])
                
                weighted_flags = critical_flags * 3 + high_flags * 2 + flagged_messages
                safety_score = max(0, 100 - (weighted_flags / total_messages * 100))
            else:
                safety_score = max(0, 100 - (flagged_ratio * 100))
        else:
            safety_score = 100.0
        
        return safety_score
    
    def _calculate_response_time_score(self, messages_df: pd.DataFrame) -> float:
        """Calculate response time score"""
        if messages_df.empty or 'response_time_minutes' not in messages_df.columns:
            return 50.0  # Neutral baseline
        
        response_times = messages_df['response_time_minutes'].dropna()
        
        if response_times.empty:
            return 50.0
        
        avg_response_time = response_times.mean()
        
        # Score based on response time (lower is better)
        if avg_response_time <= 5:
            return 100.0
        elif avg_response_time <= 30:
            return 80.0
        elif avg_response_time <= 120:
            return 60.0
        elif avg_response_time <= 360:
            return 40.0
        else:
            return 20.0
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    def _generate_health_recommendations(self, components: Dict) -> List[str]:
        """Generate actionable recommendations based on health components"""
        recommendations = []
        
        if components['activity_score'] < 50:
            recommendations.append("📈 Increase group activity by posting engaging content and questions")
        
        if components['sentiment_score'] < 60:
            recommendations.append("😊 Monitor sentiment and address negative discussions proactively")
        
        if components['engagement_score'] < 50:
            recommendations.append("👥 Encourage broader participation from group members")
        
        if components['safety_score'] < 80:
            recommendations.append("🛡️ Review and address flagged content to improve group safety")
        
        if components['response_score'] < 60:
            recommendations.append("⚡ Improve response times to member queries and concerns")
        
        if all(score > 80 for score in components.values()):
            recommendations.append("🎉 Excellent group health! Continue current practices")
        
        return recommendations
    
    def generate_trend_analysis(self, group_id: str = None, timeframe_days: int = 90) -> Dict:
        """Generate comprehensive trend analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=timeframe_days)
            
            if group_id:
                messages_df = self._get_group_messages(group_id, start_date, end_date)
                groups_data = [group_id]
            else:
                messages_df = self._get_all_messages(start_date, end_date)
                groups_data = messages_df['group_id'].unique() if not messages_df.empty else []
            
            trends = {
                'message_volume_trend': self._analyze_volume_trend(messages_df),
                'sentiment_trend': self._analyze_sentiment_trend(messages_df),
                'user_engagement_trend': self._analyze_engagement_trend(messages_df),
                'content_safety_trend': self._analyze_safety_trend(messages_df),
                'language_usage_trend': self._analyze_language_trend(messages_df),
                'peak_activity_patterns': self._analyze_activity_patterns(messages_df),
                'emerging_topics': self._extract_trending_topics(messages_df)
            }
            
            # Generate insights
            insights = self._generate_trend_insights(trends)
            
            return {
                'trends': trends,
                'insights': insights,
                'analysis_period': f"{start_date.date()} to {end_date.date()}",
                'groups_analyzed': len(groups_data) if groups_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_volume_trend(self, messages_df: pd.DataFrame) -> Dict:
        """Analyze message volume trends"""
        if messages_df.empty:
            return {'trend': 'no_data', 'confidence': 0}
        
        # Daily message counts
        daily_counts = messages_df.groupby(messages_df['timestamp'].dt.date).size()
        
        if len(daily_counts) < 7:
            return {'trend': 'insufficient_data', 'confidence': 0}
        
        # Calculate trend using linear regression
        x = np.arange(len(daily_counts))
        y = daily_counts.values
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine trend direction and strength
        relative_slope = slope / np.mean(y) if np.mean(y) > 0 else 0
        
        if abs(relative_slope) < 0.01:
            trend_direction = 'stable'
        elif relative_slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate confidence based on R-squared
        y_pred = slope * x + intercept
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'confidence': max(0, min(r_squared, 1)),
            'daily_average': np.mean(y),
            'growth_rate': relative_slope * 100
        }
    
    def _analyze_sentiment_trend(self, messages_df: pd.DataFrame) -> Dict:
        """Analyze sentiment trends over time"""
        if messages_df.empty or 'sentiment' not in messages_df.columns:
            return {'trend': 'no_data'}
        
        # Daily sentiment ratios
        daily_sentiment = messages_df.groupby([
            messages_df['timestamp'].dt.date, 'sentiment'
        ]).size().unstack(fill_value=0)
        
        if daily_sentiment.empty:
            return {'trend': 'no_data'}
        
        # Calculate positive sentiment ratio over time
        daily_sentiment['total'] = daily_sentiment.sum(axis=1)
        daily_sentiment['positive_ratio'] = (
            daily_sentiment.get('positive', 0) / daily_sentiment['total']
        )
        
        # Trend analysis
        x = np.arange(len(daily_sentiment))
        y = daily_sentiment['positive_ratio'].values
        
        if len(y) > 3:
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 0.001:
                trend = 'improving'
            elif slope < -0.001:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'slope': slope if len(y) > 3 else 0,
            'current_positive_ratio': y[-1] if len(y) > 0 else 0,
            'average_positive_ratio': np.mean(y) if len(y) > 0 else 0
        }
    
    def _analyze_engagement_trend(self, messages_df: pd.DataFrame) -> Dict:
        """Analyze user engagement trends"""
        if messages_df.empty:
            return {'trend': 'no_data'}
        
        # Daily unique users
        daily_users = messages_df.groupby(
            messages_df['timestamp'].dt.date
        )['sender_phone'].nunique()
        
        # Messages per user per day
        daily_msg_per_user = messages_df.groupby([
            messages_df['timestamp'].dt.date, 'sender_phone'
        ]).size().groupby(level=0).mean()
        
        # Trend analysis for active users
        if len(daily_users) > 3:
            x = np.arange(len(daily_users))
            y = daily_users.values
            slope_users, _ = np.polyfit(x, y, 1)
            
            # Trend analysis for messages per user
            y_msg = daily_msg_per_user.values
            slope_intensity, _ = np.polyfit(x, y_msg, 1)
            
            return {
                'user_count_trend': 'increasing' if slope_users > 0.1 else 'decreasing' if slope_users < -0.1 else 'stable',
                'engagement_intensity_trend': 'increasing' if slope_intensity > 0.01 else 'decreasing' if slope_intensity < -0.01 else 'stable',
                'avg_daily_active_users': np.mean(y),
                'avg_messages_per_user': np.mean(y_msg)
            }
        
        return {'trend': 'insufficient_data'}
    
    def _analyze_safety_trend(self, messages_df: pd.DataFrame) -> Dict:
        """Analyze content safety trends"""
        if messages_df.empty or 'is_flagged' not in messages_df.columns:
            return {'trend': 'no_data'}
        
        # Daily flagged message ratios
        daily_stats = messages_df.groupby(messages_df['timestamp'].dt.date).agg({
            'is_flagged': ['sum', 'count']
        }).round(4)
        
        daily_stats.columns = ['flagged', 'total']
        daily_stats['flagged_ratio'] = daily_stats['flagged'] / daily_stats['total']
        
        if len(daily_stats) > 3:
            x = np.arange(len(daily_stats))
            y = daily_stats['flagged_ratio'].values
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 0.001:
                trend = 'deteriorating'
            elif slope < -0.001:
                trend = 'improving'
            else:
                trend = 'stable'
                
            return {
                'trend': trend,
                'slope': slope,
                'current_flagged_ratio': y[-1] if len(y) > 0 else 0,
                'average_flagged_ratio': np.mean(y)
            }
        
        return {'trend': 'insufficient_data'}
    
    def _analyze_language_trend(self, messages_df: pd.DataFrame) -> Dict:
        """Analyze language usage trends"""
        if messages_df.empty or 'language' not in messages_df.columns:
            return {'trend': 'no_data'}
        
        # Language distribution over time
        daily_languages = messages_df.groupby([
            messages_df['timestamp'].dt.date, 'language'
        ]).size().unstack(fill_value=0)
        
        # Calculate percentages
        daily_languages_pct = daily_languages.div(daily_languages.sum(axis=1), axis=0)
        
        # Identify trending languages
        trending = {}
        for lang in daily_languages_pct.columns:
            if len(daily_languages_pct) > 3:
                x = np.arange(len(daily_languages_pct))
                y = daily_languages_pct[lang].values
                slope, _ = np.polyfit(x, y, 1)
                
                if abs(slope) > 0.01:  # Significant trend
                    trending[lang] = {
                        'trend': 'increasing' if slope > 0 else 'decreasing',
                        'slope': slope,
                        'current_usage': y[-1],
                        'average_usage': np.mean(y)
                    }
        
        return {
            'trending_languages': trending,
            'language_diversity': len(daily_languages_pct.columns),
            'dominant_language': daily_languages_pct.mean().idxmax() if not daily_languages_pct.empty else None
        }
    
    def _analyze_activity_patterns(self, messages_df: pd.DataFrame) -> Dict:
        """Analyze peak activity patterns"""
        if messages_df.empty:
            return {'pattern': 'no_data'}
        
        # Hourly patterns
        hourly_activity = messages_df['timestamp'].dt.hour.value_counts().sort_index()
        peak_hours = hourly_activity.nlargest(3).index.tolist()
        
        # Daily patterns
        daily_activity = messages_df['timestamp'].dt.day_name().value_counts()
        peak_days = daily_activity.nlargest(2).index.tolist()
        
        # Activity concentration (Gini coefficient)
        activity_concentration = self._calculate_gini_coefficient(hourly_activity.values)
        
        return {
            'peak_hours': peak_hours,
            'peak_days': peak_days,
            'activity_concentration': activity_concentration,
            'most_active_hour': hourly_activity.idxmax(),
            'least_active_hour': hourly_activity.idxmin(),
            'total_active_hours': len(hourly_activity[hourly_activity > 0])
        }
    
    def _extract_trending_topics(self, messages_df: pd.DataFrame) -> Dict:
        """Extract trending topics from message content"""
        if messages_df.empty or 'message_content' not in messages_df.columns:
            return {'topics': []}
        
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        from collections import Counter
        
        # Clean and tokenize messages
        all_text = ' '.join(messages_df['message_content'].dropna())
        words = re.findall(r'\b\w{4,}\b', all_text.lower())
        
        # Filter common words (basic stop words)
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
            'come', 'here', 'your', 'what', 'said', 'each', 'which', 'their'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get top trending words
        word_counts = Counter(filtered_words)
        trending_topics = [
            {'topic': word, 'frequency': count}
            for word, count in word_counts.most_common(10)
        ]
        
        return {'topics': trending_topics}
    
    def _generate_trend_insights(self, trends: Dict) -> List[str]:
        """Generate actionable insights from trends"""
        insights = []
        
        # Volume insights
        if trends['message_volume_trend']['trend'] == 'decreasing':
            insights.append("📉 Message volume is declining. Consider engaging content strategies.")
        elif trends['message_volume_trend']['trend'] == 'increasing':
            insights.append("📈 Message volume is growing. Monitor group capacity and moderation needs.")
        
        # Sentiment insights
        if trends['sentiment_trend'].get('trend') == 'declining':
            insights.append("😟 Sentiment is declining. Address negative discussions proactively.")
        elif trends['sentiment_trend'].get('trend') == 'improving':
            insights.append("😊 Sentiment is improving. Current strategies are effective.")
        
        # Safety insights
        if trends['content_safety_trend'].get('trend') == 'deteriorating':
            insights.append("⚠️ Content safety issues are increasing. Review moderation policies.")
        
        # Activity patterns
        peak_hours = trends['peak_activity_patterns'].get('peak_hours', [])
        if peak_hours:
            insights.append(f"⏰ Peak activity hours: {', '.join(map(str, peak_hours))}. Schedule important messages accordingly.")
        
        return insights
    
    def create_comprehensive_dashboard_data(self, group_id: str = None, timeframe_days: int = 30) -> Dict:
        """Create comprehensive dashboard data"""
        try:
            dashboard_data = {
                'summary_metrics': {},
                'charts_data': {},
                'insights': [],
                'recommendations': []
            }
            
            if group_id:
                # Single group analysis
                health_score = self.calculate_group_health_score(group_id, timeframe_days)
                dashboard_data['health_score'] = health_score
                
                # Group-specific metrics
                group_metrics = self._calculate_group_metrics(group_id, timeframe_days)
                dashboard_data['summary_metrics'] = group_metrics
                
            else:
                # Platform-wide analysis
                platform_metrics = self._calculate_platform_metrics(timeframe_days)
                dashboard_data['summary_metrics'] = platform_metrics
            
            # Generate trend analysis
            trends = self.generate_trend_analysis(group_id, timeframe_days)
            dashboard_data['trends'] = trends
            
            # Create chart data
            dashboard_data['charts_data'] = self._generate_chart_data(group_id, timeframe_days)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating dashboard data: {str(e)}")
            return {'error': str(e)}
    
    def _get_group_messages(self, group_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get messages for a specific group within date range"""
        try:
            conn = sqlite3.connect(self.group_db_path)
            
            query = '''
                SELECT * FROM group_messages 
                WHERE group_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=(group_id, start_date, end_date))
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting group messages: {str(e)}")
            return pd.DataFrame()
    
    def _get_all_messages(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get all messages within date range"""
        try:
            conn = sqlite3.connect(self.group_db_path)
            
            query = '''
                SELECT * FROM group_messages 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting all messages: {str(e)}")
            return pd.DataFrame()
    
    def _store_metric(self, group_id: str, metric_type: str, metric_name: str, 
                     value: float, metadata: Dict = None, timeframe: str = 'daily'):
        """Store analytics metric in database"""
        try:
            conn = sqlite3.connect(self.group_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analytics_metrics 
                (group_id, metric_type, metric_name, value, timestamp, timeframe, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                group_id, metric_type, metric_name, value, datetime.now(),
                timeframe, json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metric: {str(e)}")
    
    def _calculate_group_metrics(self, group_id: str, timeframe_days: int) -> Dict:
        """Calculate comprehensive metrics for a specific group"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        
        messages_df = self._get_group_messages(group_id, start_date, end_date)
        
        if messages_df.empty:
            return {
                'total_messages': 0,
                'unique_senders': 0,
                'avg_messages_per_day': 0,
                'flagged_messages': 0,
                'response_time_avg': 0
            }
        
        metrics = {
            'total_messages': len(messages_df),
            'unique_senders': messages_df['sender_phone'].nunique(),
            'avg_messages_per_day': len(messages_df) / timeframe_days,
            'flagged_messages': messages_df['is_flagged'].sum() if 'is_flagged' in messages_df.columns else 0,
            'response_time_avg': messages_df['response_time_minutes'].mean() if 'response_time_minutes' in messages_df.columns else 0
        }
        
        # Sentiment distribution
        if 'sentiment' in messages_df.columns:
            sentiment_counts = messages_df['sentiment'].value_counts()
            metrics.update({
                'sentiment_positive': sentiment_counts.get('positive', 0),
                'sentiment_negative': sentiment_counts.get('negative', 0),
                'sentiment_neutral': sentiment_counts.get('neutral', 0)
            })
        
        return metrics
    
    def _calculate_platform_metrics(self, timeframe_days: int) -> Dict:
        """Calculate platform-wide metrics"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        
        messages_df = self._get_all_messages(start_date, end_date)
        
        if messages_df.empty:
            return {
                'total_messages': 0,
                'total_groups': 0,
                'active_groups': 0,
                'total_users': 0,
                'avg_group_activity': 0
            }
        
        metrics = {
            'total_messages': len(messages_df),
            'total_groups': messages_df['group_id'].nunique(),
            'active_groups': messages_df['group_id'].nunique(),
            'total_users': messages_df['sender_phone'].nunique(),
            'avg_group_activity': len(messages_df) / messages_df['group_id'].nunique() if messages_df['group_id'].nunique() > 0 else 0,
            'flagged_messages': messages_df['is_flagged'].sum() if 'is_flagged' in messages_df.columns else 0
        }
        
        return metrics
    
    def _generate_chart_data(self, group_id: str = None, timeframe_days: int = 30) -> Dict:
        """Generate data for various charts"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        
        if group_id:
            messages_df = self._get_group_messages(group_id, start_date, end_date)
        else:
            messages_df = self._get_all_messages(start_date, end_date)
        
        if messages_df.empty:
            return {}
        
        charts_data = {}
        
        # Daily message volume chart
        daily_counts = messages_df.groupby(messages_df['timestamp'].dt.date).size()
        charts_data['daily_volume'] = {
            'dates': [str(date) for date in daily_counts.index],
            'counts': daily_counts.tolist()
        }
        
        # Hourly activity pattern
        hourly_counts = messages_df['timestamp'].dt.hour.value_counts().sort_index()
        charts_data['hourly_pattern'] = {
            'hours': hourly_counts.index.tolist(),
            'counts': hourly_counts.tolist()
        }
        
        # Sentiment distribution over time
        if 'sentiment' in messages_df.columns:
            sentiment_daily = messages_df.groupby([
                messages_df['timestamp'].dt.date, 'sentiment'
            ]).size().unstack(fill_value=0)
            
            charts_data['sentiment_trend'] = {
                'dates': [str(date) for date in sentiment_daily.index],
                'positive': sentiment_daily.get('positive', pd.Series(0)).tolist(),
                'negative': sentiment_daily.get('negative', pd.Series(0)).tolist(),
                'neutral': sentiment_daily.get('neutral', pd.Series(0)).tolist()
            }
        
        # Language usage
        if 'language' in messages_df.columns:
            language_counts = messages_df['language'].value_counts()
            charts_data['language_distribution'] = {
                'languages': language_counts.index.tolist(),
                'counts': language_counts.tolist()
            }
        
        # User engagement (top contributors)
        user_messages = messages_df['sender_phone'].value_counts().head(10)
        charts_data['top_contributors'] = {
            'users': [f"User_{i+1}" for i in range(len(user_messages))],  # Anonymized
            'message_counts': user_messages.tolist()
        }
        
        return charts_data

    def generate_plotly_charts(self, group_id: str = None, timeframe_days: int = 30) -> Dict:
        """Generate Plotly charts for advanced visualization"""
        try:
            chart_data = self._generate_chart_data(group_id, timeframe_days)
            
            if not chart_data:
                return {'error': 'No data available for charts'}
            
            charts = {}
            
            # 1. Daily Message Volume Chart
            if 'daily_volume' in chart_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data['daily_volume']['dates'],
                    y=chart_data['daily_volume']['counts'],
                    mode='lines+markers',
                    name='Daily Messages',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='Daily Message Volume Trend',
                    xaxis_title='Date',
                    yaxis_title='Number of Messages',
                    template='plotly_white',
                    height=400
                )
                
                charts['daily_volume'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
            # 2. Hourly Activity Heatmap
            if 'hourly_pattern' in chart_data:
                fig = go.Figure(data=go.Bar(
                    x=chart_data['hourly_pattern']['hours'],
                    y=chart_data['hourly_pattern']['counts'],
                    marker_color='#764ba2',
                    name='Messages by Hour'
                ))
                
                fig.update_layout(
                    title='Activity Pattern by Hour',
                    xaxis_title='Hour of Day',
                    yaxis_title='Number of Messages',
                    template='plotly_white',
                    height=400
                )
                
                charts['hourly_pattern'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
            # 3. Sentiment Trend Chart
            if 'sentiment_trend' in chart_data:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=chart_data['sentiment_trend']['dates'],
                    y=chart_data['sentiment_trend']['positive'],
                    mode='lines+markers',
                    name='Positive',
                    line=dict(color='#27ae60', width=2),
                    stackgroup='one'
                ))
                
                fig.add_trace(go.Scatter(
                    x=chart_data['sentiment_trend']['dates'],
                    y=chart_data['sentiment_trend']['neutral'],
                    mode='lines+markers',
                    name='Neutral',
                    line=dict(color='#f39c12', width=2),
                    stackgroup='one'
                ))
                
                fig.add_trace(go.Scatter(
                    x=chart_data['sentiment_trend']['dates'],
                    y=chart_data['sentiment_trend']['negative'],
                    mode='lines+markers',
                    name='Negative',
                    line=dict(color='#e74c3c', width=2),
                    stackgroup='one'
                ))
                
                fig.update_layout(
                    title='Sentiment Trend Over Time',
                    xaxis_title='Date',
                    yaxis_title='Number of Messages',
                    template='plotly_white',
                    height=400
                )
                
                charts['sentiment_trend'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
            # 4. Language Distribution Pie Chart
            if 'language_distribution' in chart_data:
                fig = go.Figure(data=[go.Pie(
                    labels=chart_data['language_distribution']['languages'],
                    values=chart_data['language_distribution']['counts'],
                    hole=0.3,
                    marker_colors=['#667eea', '#764ba2', '#27ae60', '#f39c12', '#e74c3c']
                )])
                
                fig.update_layout(
                    title='Language Distribution',
                    template='plotly_white',
                    height=400
                )
                
                charts['language_distribution'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
            # 5. User Engagement Chart
            if 'top_contributors' in chart_data:
                fig = go.Figure(data=[go.Bar(
                    x=chart_data['top_contributors']['users'],
                    y=chart_data['top_contributors']['message_counts'],
                    marker_color='#667eea',
                    name='Messages Sent'
                )])
                
                fig.update_layout(
                    title='Top Contributors',
                    xaxis_title='Users (Anonymized)',
                    yaxis_title='Messages Sent',
                    template='plotly_white',
                    height=400
                )
                
                charts['top_contributors'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating Plotly charts: {str(e)}")
            return {'error': str(e)}
    
    def generate_advanced_report(self, group_id: str = None, timeframe_days: int = 30) -> Dict:
        """Generate comprehensive advanced analytics report"""
        try:
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'timeframe_days': timeframe_days,
                    'group_id': group_id,
                    'report_type': 'group_analysis' if group_id else 'platform_analysis'
                },
                'executive_summary': {},
                'detailed_analysis': {},
                'recommendations': [],
                'charts': {},
                'raw_data': {}
            }
            
            # Generate health score and trends
            if group_id:
                health_data = self.calculate_group_health_score(group_id, timeframe_days)
                trend_data = self.generate_trend_analysis(group_id, timeframe_days)
                
                report['executive_summary'] = {
                    'health_score': health_data.get('health_score', 0),
                    'health_components': health_data.get('components', {}),
                    'primary_concerns': self._identify_primary_concerns(health_data),
                    'key_trends': self._summarize_key_trends(trend_data)
                }
                
                report['detailed_analysis'] = {
                    'health_analysis': health_data,
                    'trend_analysis': trend_data,
                    'group_metrics': self._calculate_group_metrics(group_id, timeframe_days)
                }
                
            else:
                platform_metrics = self._calculate_platform_metrics(timeframe_days)
                trend_data = self.generate_trend_analysis(None, timeframe_days)
                
                report['executive_summary'] = {
                    'platform_metrics': platform_metrics,
                    'key_trends': self._summarize_key_trends(trend_data),
                    'platform_health': self._assess_platform_health(platform_metrics)
                }
                
                report['detailed_analysis'] = {
                    'platform_metrics': platform_metrics,
                    'trend_analysis': trend_data
                }
            
            # Generate charts
            report['charts'] = self.generate_plotly_charts(group_id, timeframe_days)
            
            # Generate recommendations
            report['recommendations'] = self._generate_comprehensive_recommendations(
                report['detailed_analysis']
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating advanced report: {str(e)}")
            return {'error': str(e)}
    
    def _identify_primary_concerns(self, health_data: Dict) -> List[str]:
        """Identify primary concerns from health analysis"""
        concerns = []
        components = health_data.get('components', {})
        
        for component, score in components.items():
            if score < 50:
                concerns.append(f"Low {component.replace('_', ' ')}: {score}")
        
        return concerns[:3]  # Top 3 concerns
    
    def _summarize_key_trends(self, trend_data: Dict) -> List[str]:
        """Summarize key trends"""
        if 'trends' not in trend_data:
            return []
        
        trends = trend_data['trends']
        key_trends = []
        
        # Message volume trend
        vol_trend = trends.get('message_volume_trend', {})
        if vol_trend.get('trend') != 'stable':
            key_trends.append(f"Message volume is {vol_trend.get('trend', 'unknown')}")
        
        # Sentiment trend
        sent_trend = trends.get('sentiment_trend', {})
        if sent_trend.get('trend') and sent_trend['trend'] != 'stable':
            key_trends.append(f"Sentiment is {sent_trend['trend']}")
        
        # Safety trend
        safety_trend = trends.get('content_safety_trend', {})
        if safety_trend.get('trend') == 'deteriorating':
            key_trends.append("Content safety issues increasing")
        
        return key_trends[:5]
    
    def _assess_platform_health(self, metrics: Dict) -> str:
        """Assess overall platform health"""
        if metrics.get('total_messages', 0) == 0:
            return 'No Activity'
        
        flagged_ratio = metrics.get('flagged_messages', 0) / metrics.get('total_messages', 1)
        
        if flagged_ratio < 0.05:
            return 'Healthy'
        elif flagged_ratio < 0.1:
            return 'Moderate'
        else:
            return 'Needs Attention'
    
    def _generate_comprehensive_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Health-based recommendations
        if 'health_analysis' in analysis:
            health_components = analysis['health_analysis'].get('components', {})
            
            for component, score in health_components.items():
                if score < 60:
                    recommendations.append({
                        'type': 'improvement',
                        'priority': 'high' if score < 40 else 'medium',
                        'category': component,
                        'recommendation': self._get_component_recommendation(component, score),
                        'expected_impact': 'high' if score < 40 else 'medium'
                    })
        
        # Trend-based recommendations
        if 'trend_analysis' in analysis:
            trends = analysis['trend_analysis'].get('trends', {})
            
            # Volume trend recommendations
            vol_trend = trends.get('message_volume_trend', {})
            if vol_trend.get('trend') == 'decreasing':
                recommendations.append({
                    'type': 'action',
                    'priority': 'medium',
                    'category': 'engagement',
                    'recommendation': 'Implement engagement strategies to increase message volume',
                    'expected_impact': 'medium'
                })
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _get_component_recommendation(self, component: str, score: float) -> str:
        """Get specific recommendation for health component"""
        recommendations_map = {
            'activity_score': 'Increase group activity through engaging content and regular updates',
            'sentiment_score': 'Monitor and address negative sentiment through proactive moderation',
            'engagement_score': 'Encourage broader participation from group members',
            'safety_score': 'Strengthen content moderation and review flagged messages',
            'response_score': 'Improve response times to member queries and concerns'
        }
        
        return recommendations_map.get(component, 'Review and improve this aspect of group management')

# Initialize global analytics engine
analytics_engine = AdvancedAnalyticsEngine()