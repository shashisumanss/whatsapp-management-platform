# sensitivity_manager.py
# Complete AI Model Sensitivity Management System

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class SensitivityLevel(Enum):
    """Predefined sensitivity levels"""
    VERY_LOW = "very_low"      # Only extreme cases (5% false positive rate)
    LOW = "low"                # Conservative flagging (10% false positive rate)
    MEDIUM = "medium"          # Balanced approach (15% false positive rate)
    HIGH = "high"              # Liberal flagging (25% false positive rate)
    VERY_HIGH = "very_high"    # Flag almost everything (40% false positive rate)

class ModelType(Enum):
    """Types of models that can be tuned"""
    KEYWORD_DETECTION = "keyword_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ML_CLASSIFICATION = "ml_classification"
    BERT_TOXICITY = "bert_toxicity"
    HEALTH_SCORING = "health_scoring"

@dataclass
class SensitivityConfig:
    """Configuration for model sensitivity"""
    model_type: str
    sensitivity_level: str
    threshold_values: Dict[str, float]
    keyword_sets: Dict[str, List[str]]
    enabled_models: List[str]
    custom_rules: Dict[str, Any]
    last_updated: str
    updated_by: str

class SensitivityManager:
    """Manages sensitivity configurations for all AI models"""
    
    def __init__(self, db_path='whatsapp_groups.db'):
        self.db_path = db_path
        self.configs = {}
        self.init_database()
        self.load_configurations()
    
    def init_database(self):
        """Initialize sensitivity management tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main sensitivity configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensitivity_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT UNIQUE,
                sensitivity_level TEXT,
                threshold_values TEXT,
                keyword_sets TEXT,
                enabled_models TEXT,
                custom_rules TEXT,
                last_updated TIMESTAMP,
                updated_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                sensitivity_level TEXT,
                true_positives INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                true_negatives INTEGER DEFAULT 0,
                false_negatives INTEGER DEFAULT 0,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                date_recorded DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sensitivity adjustment logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensitivity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                old_level TEXT,
                new_level TEXT,
                old_thresholds TEXT,
                new_thresholds TEXT,
                reason TEXT,
                adjusted_by TEXT,
                adjustment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_impact TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Insert default configurations if not exists
        self._insert_default_configs()
    
    def _insert_default_configs(self):
        """Insert default configurations for all model types"""
        default_configs = self._get_default_configs()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for model_type, config in default_configs.items():
            cursor.execute('''
                INSERT OR IGNORE INTO sensitivity_configs 
                (model_type, sensitivity_level, threshold_values, keyword_sets, 
                 enabled_models, custom_rules, last_updated, updated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_type.value,
                config.sensitivity_level,
                json.dumps(config.threshold_values),
                json.dumps(config.keyword_sets),
                json.dumps(config.enabled_models),
                json.dumps(config.custom_rules),
                datetime.now().isoformat(),
                'system'
            ))
        
        conn.commit()
        conn.close()
    
    def _get_default_configs(self) -> Dict[ModelType, SensitivityConfig]:
        """Define default sensitivity configurations"""
        return {
            ModelType.KEYWORD_DETECTION: SensitivityConfig(
                model_type=ModelType.KEYWORD_DETECTION.value,
                sensitivity_level=SensitivityLevel.MEDIUM.value,
                threshold_values={
                    "exact_match_confidence": 0.95,
                    "partial_match_confidence": 0.75,
                    "context_weight": 0.8,
                    "language_confidence_threshold": 0.6,
                    "minimum_keyword_length": 3
                },
                keyword_sets={
                    "violence_critical": ["kill", "murder", "attack", "bomb", "weapon", "shoot"],
                    "violence_high": ["fight", "beat", "hurt", "destroy", "threat", "revenge"],
                    "emergency_critical": ["emergency", "urgent", "sos", "help", "ambulance", "911"],
                    "emergency_high": ["accident", "injured", "fire", "police", "rescue"],
                    "political_sensitive": ["election", "vote", "fraud", "rigging", "booth", "ballot"],
                    "kannada_violence": ["ಕೊಲ್ಲು", "ಸಾವು", "ಹೊಡೆ", "ಗಾಯ", "ಬೆದರಿಕೆ"],
                    "kannada_emergency": ["ತುರ್ತು", "ಸಹಾಯ", "ಅಪಘಾತ", "ಆಸ್ಪತ್ರೆ", "ಪೊಲೀಸ್"]
                },
                enabled_models=["exact_match", "fuzzy_match", "context_analysis"],
                custom_rules={
                    "require_multiple_indicators": False,
                    "context_window_size": 5,
                    "enable_negation_detection": True,
                    "escalation_keywords": ["immediately", "now", "asap"]
                },
                last_updated=datetime.now().isoformat(),
                updated_by="system"
            ),
            
            ModelType.SENTIMENT_ANALYSIS: SensitivityConfig(
                model_type=ModelType.SENTIMENT_ANALYSIS.value,
                sensitivity_level=SensitivityLevel.MEDIUM.value,
                threshold_values={
                    "positive_threshold": 0.1,      # Above this = positive
                    "negative_threshold": -0.1,     # Below this = negative
                    "confidence_threshold": 0.5,    # Minimum confidence to classify
                    "neutral_range": 0.2,           # Range around 0 considered neutral
                    "extreme_sentiment_threshold": 0.7  # Very positive/negative
                },
                keyword_sets={
                    "positive_boosters": ["excellent", "amazing", "wonderful", "fantastic"],
                    "negative_boosters": ["terrible", "awful", "horrible", "disgusting"],
                    "kannada_positive": ["ಚೆನ್ನಾಗಿದೆ", "ಸಂತೋಷ", "ಖುಷಿ", "ಸುಂದರ"],
                    "kannada_negative": ["ಕೆಟ್ಟ", "ದುಃಖ", "ಕಷ್ಟ", "ಸಮಸ್ಯೆ"]
                },
                enabled_models=["textblob", "vader", "custom_kannada"],
                custom_rules={
                    "weight_textblob": 0.4,
                    "weight_vader": 0.4,
                    "weight_kannada": 0.2,
                    "enable_sarcasm_detection": False,
                    "boost_caps_text": True
                },
                last_updated=datetime.now().isoformat(),
                updated_by="system"
            ),
            
            ModelType.ML_CLASSIFICATION: SensitivityConfig(
                model_type=ModelType.ML_CLASSIFICATION.value,
                sensitivity_level=SensitivityLevel.MEDIUM.value,
                threshold_values={
                    "critical_threshold": 0.7,      # Probability for CRITICAL
                    "high_threshold": 0.5,          # Probability for HIGH
                    "medium_threshold": 0.3,        # Probability for MEDIUM
                    "ensemble_agreement": 0.6,      # 60% models must agree
                    "confidence_boost": 0.1         # Boost for keyword matches
                },
                keyword_sets={
                    "training_violence": ["kill", "death", "murder", "attack", "bomb"],
                    "training_emergency": ["emergency", "urgent", "help", "sos"],
                    "training_normal": ["hello", "good", "thanks", "morning"]
                },
                enabled_models=["random_forest", "naive_bayes", "logistic_regression"],
                custom_rules={
                    "use_ensemble_voting": True,
                    "retrain_frequency_days": 30,
                    "min_training_samples": 100,
                    "feature_selection_method": "tfidf"
                },
                last_updated=datetime.now().isoformat(),
                updated_by="system"
            ),
            
            ModelType.BERT_TOXICITY: SensitivityConfig(
                model_type=ModelType.BERT_TOXICITY.value,
                sensitivity_level=SensitivityLevel.MEDIUM.value,
                threshold_values={
                    "toxicity_threshold": 0.7,      # Probability for toxic
                    "severe_toxicity_threshold": 0.9,  # Very toxic content
                    "confidence_threshold": 0.5,    # Minimum confidence
                    "text_length_min": 5,           # Minimum chars to analyze
                    "text_length_max": 512          # Maximum chars (BERT limit)
                },
                keyword_sets={
                    "toxic_indicators": ["hate", "racist", "discriminate", "harass"],
                    "severe_indicators": ["kill", "die", "murder", "rape"]
                },
                enabled_models=["bert_toxicity"],
                custom_rules={
                    "enable_gpu": False,
                    "batch_processing": True,
                    "cache_results": True,
                    "fallback_on_error": True
                },
                last_updated=datetime.now().isoformat(),
                updated_by="system"
            ),
            
            ModelType.HEALTH_SCORING: SensitivityConfig(
                model_type=ModelType.HEALTH_SCORING.value,
                sensitivity_level=SensitivityLevel.MEDIUM.value,
                threshold_values={
                    "excellent_health_threshold": 80,   # Above 80 = excellent
                    "good_health_threshold": 60,        # 60-80 = good
                    "poor_health_threshold": 40,        # Below 40 = poor
                    "activity_weight": 0.25,            # Activity score weight
                    "sentiment_weight": 0.20,           # Sentiment score weight
                    "engagement_weight": 0.20,          # Engagement score weight
                    "safety_weight": 0.25,              # Safety score weight
                    "response_weight": 0.10             # Response time weight
                },
                keyword_sets={
                    "health_indicators": ["active", "engaging", "positive", "safe"],
                    "risk_indicators": ["inactive", "negative", "toxic", "unresponsive"]
                },
                enabled_models=["activity_analyzer", "sentiment_analyzer", "safety_analyzer"],
                custom_rules={
                    "timeframe_days": 30,
                    "minimum_messages_for_analysis": 10,
                    "update_frequency_hours": 24,
                    "alert_on_decline": True
                },
                last_updated=datetime.now().isoformat(),
                updated_by="system"
            )
        }
    
    def load_configurations(self):
        """Load all configurations from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sensitivity_configs')
        rows = cursor.fetchall()
        
        for row in rows:
            config = SensitivityConfig(
                model_type=row[1],
                sensitivity_level=row[2],
                threshold_values=json.loads(row[3]),
                keyword_sets=json.loads(row[4]),
                enabled_models=json.loads(row[5]),
                custom_rules=json.loads(row[6]),
                last_updated=row[7],
                updated_by=row[8]
            )
            self.configs[row[1]] = config
        
        conn.close()
    
    def get_config(self, model_type: str) -> Optional[SensitivityConfig]:
        """Get configuration for a specific model type"""
        return self.configs.get(model_type)
    
    def update_sensitivity_level(self, model_type: str, new_level: str, 
                               updated_by: str, reason: str = "") -> bool:
        """Update sensitivity level for a model"""
        try:
            if model_type not in self.configs:
                return False
            
            old_config = self.configs[model_type]
            old_level = old_config.sensitivity_level
            old_thresholds = old_config.threshold_values.copy()
            
            # Update thresholds based on sensitivity level
            new_thresholds = self._calculate_thresholds_for_level(model_type, new_level)
            
            # Update configuration
            self.configs[model_type].sensitivity_level = new_level
            self.configs[model_type].threshold_values.update(new_thresholds)
            self.configs[model_type].last_updated = datetime.now().isoformat()
            self.configs[model_type].updated_by = updated_by
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sensitivity_configs 
                SET sensitivity_level = ?, threshold_values = ?, 
                    last_updated = ?, updated_by = ?
                WHERE model_type = ?
            ''', (
                new_level,
                json.dumps(self.configs[model_type].threshold_values),
                self.configs[model_type].last_updated,
                updated_by,
                model_type
            ))
            
            # Log the change
            cursor.execute('''
                INSERT INTO sensitivity_logs 
                (model_type, old_level, new_level, old_thresholds, new_thresholds, 
                 reason, adjusted_by, performance_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_type, old_level, new_level,
                json.dumps(old_thresholds), json.dumps(new_thresholds),
                reason, updated_by, "To be measured"
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated {model_type} sensitivity from {old_level} to {new_level}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating sensitivity: {str(e)}")
            return False
    
    def _calculate_thresholds_for_level(self, model_type: str, level: str) -> Dict[str, float]:
        """Calculate threshold values based on sensitivity level"""
        level_multipliers = {
            SensitivityLevel.VERY_LOW.value: {
                "threshold_multiplier": 1.5,    # Stricter thresholds
                "confidence_boost": 0.2
            },
            SensitivityLevel.LOW.value: {
                "threshold_multiplier": 1.2,
                "confidence_boost": 0.1
            },
            SensitivityLevel.MEDIUM.value: {
                "threshold_multiplier": 1.0,    # Default
                "confidence_boost": 0.0
            },
            SensitivityLevel.HIGH.value: {
                "threshold_multiplier": 0.8,    # More lenient
                "confidence_boost": -0.1
            },
            SensitivityLevel.VERY_HIGH.value: {
                "threshold_multiplier": 0.6,
                "confidence_boost": -0.2
            }
        }
        
        multiplier_config = level_multipliers.get(level, level_multipliers[SensitivityLevel.MEDIUM.value])
        current_config = self.configs[model_type]
        new_thresholds = {}
        
        # Apply multipliers based on model type
        if model_type == ModelType.KEYWORD_DETECTION.value:
            new_thresholds = {
                "exact_match_confidence": max(0.1, min(1.0, 
                    current_config.threshold_values["exact_match_confidence"] * multiplier_config["threshold_multiplier"])),
                "partial_match_confidence": max(0.1, min(1.0,
                    current_config.threshold_values["partial_match_confidence"] * multiplier_config["threshold_multiplier"]))
            }
        
        elif model_type == ModelType.ML_CLASSIFICATION.value:
            new_thresholds = {
                "critical_threshold": max(0.1, min(1.0,
                    current_config.threshold_values["critical_threshold"] * multiplier_config["threshold_multiplier"])),
                "high_threshold": max(0.1, min(1.0,
                    current_config.threshold_values["high_threshold"] * multiplier_config["threshold_multiplier"]))
            }
        
        elif model_type == ModelType.BERT_TOXICITY.value:
            new_thresholds = {
                "toxicity_threshold": max(0.1, min(1.0,
                    current_config.threshold_values["toxicity_threshold"] * multiplier_config["threshold_multiplier"]))
            }
        
        return new_thresholds
    
    def add_custom_keywords(self, model_type: str, category: str, 
                           keywords: List[str], updated_by: str) -> bool:
        """Add custom keywords to a model's keyword sets"""
        try:
            if model_type not in self.configs:
                return False
            
            if category not in self.configs[model_type].keyword_sets:
                self.configs[model_type].keyword_sets[category] = []
            
            # Add new keywords (avoid duplicates)
            existing_keywords = set(self.configs[model_type].keyword_sets[category])
            new_keywords = [kw for kw in keywords if kw.lower() not in existing_keywords]
            
            self.configs[model_type].keyword_sets[category].extend(new_keywords)
            self.configs[model_type].last_updated = datetime.now().isoformat()
            self.configs[model_type].updated_by = updated_by
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sensitivity_configs 
                SET keyword_sets = ?, last_updated = ?, updated_by = ?
                WHERE model_type = ?
            ''', (
                json.dumps(self.configs[model_type].keyword_sets),
                self.configs[model_type].last_updated,
                updated_by,
                model_type
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added {len(new_keywords)} keywords to {model_type}.{category}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding keywords: {str(e)}")
            return False
    
    def get_performance_metrics(self, model_type: str, days: int = 30) -> Dict:
        """Get performance metrics for a model over specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            cursor.execute('''
                SELECT AVG(precision_score), AVG(recall_score), AVG(f1_score),
                       SUM(true_positives), SUM(false_positives), 
                       SUM(true_negatives), SUM(false_negatives)
                FROM model_performance 
                WHERE model_type = ? AND date_recorded >= ?
            ''', (model_type, start_date))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and any(x is not None for x in result):
                return {
                    "avg_precision": round(result[0] or 0, 3),
                    "avg_recall": round(result[1] or 0, 3),
                    "avg_f1_score": round(result[2] or 0, 3),
                    "total_true_positives": result[3] or 0,
                    "total_false_positives": result[4] or 0,
                    "total_true_negatives": result[5] or 0,
                    "total_false_negatives": result[6] or 0,
                    "accuracy": self._calculate_accuracy(result[3:7])
                }
            else:
                return {
                    "avg_precision": 0,
                    "avg_recall": 0,
                    "avg_f1_score": 0,
                    "total_true_positives": 0,
                    "total_false_positives": 0,
                    "total_true_negatives": 0,
                    "total_false_negatives": 0,
                    "accuracy": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    def _calculate_accuracy(self, confusion_matrix_values: Tuple) -> float:
        """Calculate accuracy from confusion matrix values"""
        tp, fp, tn, fn = [x or 0 for x in confusion_matrix_values]
        total = tp + fp + tn + fn
        if total == 0:
            return 0
        return round((tp + tn) / total, 3)
    
    def get_sensitivity_recommendations(self, model_type: str) -> List[Dict]:
        """Get recommendations for sensitivity adjustments based on performance"""
        try:
            metrics = self.get_performance_metrics(model_type)
            recommendations = []
            
            if metrics.get("avg_precision", 0) < 0.7:
                recommendations.append({
                    "type": "decrease_sensitivity",
                    "reason": f"Low precision ({metrics['avg_precision']:.2f}). Too many false positives.",
                    "suggested_action": "Increase thresholds to reduce false positives",
                    "impact": "Will reduce false alarms but might miss some real issues"
                })
            
            if metrics.get("avg_recall", 0) < 0.7:
                recommendations.append({
                    "type": "increase_sensitivity", 
                    "reason": f"Low recall ({metrics['avg_recall']:.2f}). Missing too many real issues.",
                    "suggested_action": "Decrease thresholds to catch more issues",
                    "impact": "Will catch more real issues but may increase false alarms"
                })
            
            if metrics.get("avg_f1_score", 0) > 0.85:
                recommendations.append({
                    "type": "maintain_current",
                    "reason": f"Excellent F1 score ({metrics['avg_f1_score']:.2f}). Model performing well.",
                    "suggested_action": "Maintain current sensitivity settings",
                    "impact": "Continue current performance level"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def export_configurations(self) -> Dict:
        """Export all configurations for backup/transfer"""
        return {
            "configs": {k: asdict(v) for k, v in self.configs.items()},
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def import_configurations(self, config_data: Dict, imported_by: str) -> bool:
        """Import configurations from backup/transfer"""
        try:
            configs = config_data.get("configs", {})
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for model_type, config_dict in configs.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO sensitivity_configs 
                    (model_type, sensitivity_level, threshold_values, keyword_sets, 
                     enabled_models, custom_rules, last_updated, updated_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_type,
                    config_dict["sensitivity_level"],
                    json.dumps(config_dict["threshold_values"]),
                    json.dumps(config_dict["keyword_sets"]),
                    json.dumps(config_dict["enabled_models"]),
                    json.dumps(config_dict["custom_rules"]),
                    datetime.now().isoformat(),
                    imported_by
                ))
            
            conn.commit()
            conn.close()
            
            # Reload configurations
            self.load_configurations()
            
            logger.info(f"Imported {len(configs)} model configurations")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configurations: {str(e)}")
            return False

# Initialize global sensitivity manager
sensitivity_manager = SensitivityManager()