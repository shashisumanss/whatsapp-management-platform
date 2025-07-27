# WhatsApp Sentiment Analysis with ML-based Content Flagging and Management Platform
# Production version - CRITICAL-only ticket creation and working bulk messaging

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import json
import uuid
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
import numpy as np
import zipfile
import threading
import time
from datetime import datetime, timedelta
import uuid
import io
scheduled_messages_storage = []
bulk_message_history = []

from collections import defaultdict


# Imports for multilingual support
from googletrans import Translator
import langdetect
from langdetect import detect_langs
import warnings
warnings.filterwarnings("ignore")

# ML imports for content flagging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

from whatsapp_service import get_whatsapp_service, init_whatsapp_service
import os

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize WhatsApp service
WHATSAPP_CONFIG = {
    'ACCESS_TOKEN': os.getenv('WHATSAPP_ACCESS_TOKEN', 'YOUR_WHATSAPP_ACCESS_TOKEN'),
    'PHONE_NUMBER_ID': os.getenv('WHATSAPP_PHONE_NUMBER_ID', 'YOUR_PHONE_NUMBER_ID'),
    'WEBHOOK_VERIFY_TOKEN': os.getenv('WHATSAPP_WEBHOOK_TOKEN', 'YOUR_WEBHOOK_VERIFY_TOKEN'),
    'BUSINESS_ACCOUNT_ID': os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID', 'YOUR_BUSINESS_ACCOUNT_ID'),
    'APP_SECRET': os.getenv('WHATSAPP_APP_SECRET', 'YOUR_APP_SECRET')
}

# Initialize the WhatsApp service
whatsapp_service = init_whatsapp_service(WHATSAPP_CONFIG)
# Handle transformers for advanced models
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    print("⚠️  Transformers not available. Using basic ML methods.")
    HAS_TRANSFORMERS = False

matplotlib.use('Agg')

# Global storage for management features
tickets_storage = []
bulk_message_history = []

# JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


class ContentFlagger:
    """ML-based system to flag sensitive content"""
    
    def __init__(self):
        # Priority levels for flagged content
        self.priority_levels = {
            'CRITICAL': {'level': 5, 'color': '#FF0000', 'response_time': '5 minutes'},
            'HIGH': {'level': 4, 'color': '#FF6B6B', 'response_time': '30 minutes'},
            'MEDIUM': {'level': 3, 'color': '#FFA500', 'response_time': '2 hours'},
            'LOW': {'level': 2, 'color': '#FFD93D', 'response_time': '24 hours'},
            'NORMAL': {'level': 1, 'color': '#4CAF50', 'response_time': 'No action'}
        }
        
        # Keywords for different threat categories
        self.threat_keywords = {
            'violence': ['kill', 'death', 'murder', 'attack', 'bomb', 'shoot', 'hurt', 'harm', 
                        'destroy', 'violence', 'threat', 'revenge', 'weapon', 'fight', 'beat'],
            'emergency': ['emergency', 'urgent', 'help', 'sos', 'critical', 'immediately', 
                         '911', 'ambulance', 'police', 'accident', 'injured', 'fire', 'rescue'],
            'election': ['election', 'vote', 'ballot', 'polling', 'fraud', 'rigging', 
                        'booth capturing', 'voter', 'campaign', 'candidate'],
            'harassment': ['hate', 'racist', 'discriminate', 'bully', 'harass', 'abuse', 'threaten'],
            'grievance': ['complaint', 'issue', 'problem', 'unfair', 'corrupt', 'bribe', 'scam'],
            'infrastructure': ['water', 'electricity', 'power', 'road', 'garbage', 'street', 'drain', 
                             'sewage', 'light', 'broken', 'damaged', 'not working', 'problem'],
            'healthcare': ['doctor', 'hospital', 'medicine', 'health', 'medical', 'treatment', 
                          'sick', 'patient', 'clinic', 'pharmacy']
        }
        
        # Kannada keywords
        self.kannada_keywords = {
            'violence': ['ಕೊಲ್ಲು', 'ಸಾವು', 'ಹೊಡೆ', 'ಗಾಯ', 'ಬೆದರಿಕೆ', 'ಹಿಂಸೆ'],
            'emergency': ['ತುರ್ತು', 'ಸಹಾಯ', 'ಅಪಘಾತ', 'ಆಸ್ಪತ್ರೆ', 'ಪೊಲೀಸ್', 'ಬೆಂಕಿ'],
            'election': ['ಚುನಾವಣೆ', 'ಮತ', 'ಅಭ್ಯರ್ಥಿ', 'ಮತದಾನ', 'ಕೇಂದ್ರ'],
            'grievance': ['ದೂರು', 'ಸಮಸ್ಯೆ', 'ಭ್ರಷ್ಟಾಚಾರ', 'ಲಂಚ', 'ತೊಂದರೆ'],
            'infrastructure': ['ನೀರು', 'ವಿದ್ಯುತ್', 'ರಸ್ತೆ', 'ಕಸ', 'ದೀಪ', 'ಹಾನಿ'],
            'healthcare': ['ಆಸ್ಪತ್ರೆ', 'ವೈದ್ಯ', 'ಔಷಧ', 'ಚಿಕಿತ್ಸೆ', 'ಆರೋಗ್ಯ', 'ರೋಗಿ']
        }
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.models = {}
        self.is_trained = False
        
        # Try to load BERT for toxicity detection
        if HAS_TRANSFORMERS:
            try:
                self.toxicity_pipeline = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=-1
                )
                self.has_bert = True
                print("✅ BERT toxicity model loaded")
            except:
                self.has_bert = False
        else:
            self.has_bert = False
    
    def train_models(self):
        """Train ML models on synthetic data"""
        # Generate training data
        training_samples = []
        
        # Violence samples (CRITICAL)
        violence_texts = [
            "I will kill you", "death threat", "attack tomorrow",
            "bring weapons", "hurt someone badly", "beat him up", "murder plan"
        ]
        for text in violence_texts:
            training_samples.append({'text': text, 'category': 'violence', 'priority': 'CRITICAL'})
        
        # Emergency samples (HIGH)
        emergency_texts = [
            "Help emergency", "urgent need ambulance", "accident happened",
            "call police now", "fire help urgent", "someone injured badly"
        ]
        for text in emergency_texts:
            training_samples.append({'text': text, 'category': 'emergency', 'priority': 'HIGH'})
        
        # Infrastructure samples (HIGH)
        infrastructure_texts = [
            "no water supply", "power cut since morning", "road is damaged",
            "garbage not collected", "street light not working", "drain blocked"
        ]
        for text in infrastructure_texts:
            training_samples.append({'text': text, 'category': 'infrastructure', 'priority': 'HIGH'})
        
        # Healthcare samples (HIGH)
        healthcare_texts = [
            "no doctor available", "medicine shortage", "hospital closed",
            "urgent medical help needed", "patient very sick"
        ]
        for text in healthcare_texts:
            training_samples.append({'text': text, 'category': 'healthcare', 'priority': 'HIGH'})
        
        # Normal samples
        normal_texts = [
            "meet for coffee", "weather nice today", "happy birthday",
            "send documents", "thanks for help", "good morning"
        ]
        for text in normal_texts:
            training_samples.append({'text': text, 'category': 'normal', 'priority': 'NORMAL'})
        
        # Create DataFrame and train
        df = pd.DataFrame(training_samples)
        X = self.vectorizer.fit_transform(df['text'])
        y = df['priority']
        
        # Train multiple models
        self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['nb'] = MultinomialNB()
        self.models['lr'] = LogisticRegression(max_iter=1000)
        
        for name, model in self.models.items():
            model.fit(X, y)
        
        self.is_trained = True
        print("✅ Content flagging models trained successfully")
    
    def check_keywords(self, text):
        """Check for threat keywords"""
        text_lower = text.lower()
        threats = []
        
        # Check English keywords
        for category, keywords in self.threat_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category == 'violence':
                        severity = 'CRITICAL'
                    elif category in ['emergency', 'healthcare', 'infrastructure']:
                        severity = 'HIGH'
                    else:
                        severity = 'MEDIUM'
                    
                    threats.append({
                        'category': category,
                        'keyword': keyword,
                        'severity': severity
                    })
        
        # Check Kannada keywords
        for category, keywords in self.kannada_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if category == 'violence':
                        severity = 'CRITICAL'
                    elif category in ['emergency', 'healthcare', 'infrastructure']:
                        severity = 'HIGH'
                    else:
                        severity = 'MEDIUM'
                    
                    threats.append({
                        'category': category,
                        'keyword': keyword,
                        'severity': severity,
                        'language': 'kannada'
                    })
        
        return threats
    
    def predict_with_ml(self, text):
        """Use ML models to predict priority"""
        if not self.is_trained:
            return None
        
        try:
            X = self.vectorizer.transform([text.lower()])
            predictions = {}
            
            # Get predictions from all models
            for name, model in self.models.items():
                pred = model.predict(X)[0]
                predictions[name] = pred
                
            # Majority voting
            from collections import Counter
            priority = Counter(predictions.values()).most_common(1)[0][0]
            
            return priority
        except Exception as e:
            return None
    
    def check_with_bert(self, text):
        """Check toxicity with BERT"""
        if not self.has_bert:
            return None
        
        try:
            results = self.toxicity_pipeline(text[:512])
            for result in results:
                if result['label'] == 'TOXIC' and result['score'] > 0.7:
                    return True
            return False
        except Exception as e:
            return None
    
    def flag_content(self, text, sender=None, timestamp=None):
        """Main method to flag content"""
        if not text or pd.isna(text):
            return {
                'is_flagged': False,
                'priority': 'NORMAL',
                'category': 'normal',
                'reasons': []
            }
        
        flagged = False
        max_priority = 'NORMAL'
        reasons = []
        detected_category = 'normal'
        
        # 1. Check keywords
        keyword_threats = self.check_keywords(text)
        if keyword_threats:
            flagged = True
            for threat in keyword_threats:
                reasons.append({
                    'type': 'keyword',
                    'category': threat['category'],
                    'detail': threat['keyword']
                })
                # Update priority and category
                if self.priority_levels[threat['severity']]['level'] > self.priority_levels[max_priority]['level']:
                    max_priority = threat['severity']
                    detected_category = threat['category']
        
        # 2. ML prediction
        ml_priority = self.predict_with_ml(text)
        if ml_priority and ml_priority != 'NORMAL':
            flagged = True
            reasons.append({
                'type': 'ml_model',
                'category': 'ml_prediction',
                'detail': ml_priority
            })
            if self.priority_levels[ml_priority]['level'] > self.priority_levels[max_priority]['level']:
                max_priority = ml_priority
        
        # 3. BERT toxicity check
        is_toxic = self.check_with_bert(text)
        if is_toxic:
            flagged = True
            reasons.append({
                'type': 'bert',
                'category': 'toxicity',
                'detail': 'toxic content detected'
            })
            if self.priority_levels['HIGH']['level'] > self.priority_levels[max_priority]['level']:
                max_priority = 'HIGH'
        
        return {
            'is_flagged': flagged,
            'priority': max_priority,
            'category': detected_category,
            'priority_info': self.priority_levels[max_priority],
            'reasons': reasons,
            'message': text,
            'sender': sender,
            'timestamp': timestamp
        }


class MultilingualSentimentAnalyzer:
    """Sentiment analyzer with multilingual support"""
    
    def __init__(self):
        self.translator = Translator()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Kannada sentiment words
        self.kannada_positive_words = {
            'ಚೆನ್ನಾಗಿದೆ', 'ಸಂತೋಷ', 'ಖುಷಿ', 'ಸುಂದರ', 'ಉತ್ತಮ', 'ಒಳ್ಳೆಯ', 'ಧನ್ಯವಾದ'
        }
        self.kannada_negative_words = {
            'ಕೆಟ್ಟ', 'ದುಃಖ', 'ಕಷ್ಟ', 'ಸಮಸ್ಯೆ', 'ತೊಂದರೆ', 'ಬೇಸರ', 'ಕೋಪ'
        }
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            # Check for Kannada characters
            if re.search(r'[\u0C80-\u0CFF]', text):
                return 'kannada'
            
            # Try language detection
            detected = langdetect.detect(text)
            return detected
        except:
            return 'unknown'
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text:
            return {'sentiment': 'neutral', 'confidence': 0, 'language': 'unknown'}
        
        language = self.detect_language(text)
        
        # For Kannada text
        if language == 'kannada' or language == 'kn':
            positive_score = sum(1 for word in self.kannada_positive_words if word in text)
            negative_score = sum(1 for word in self.kannada_negative_words if word in text)
            
            if positive_score > negative_score:
                return {'sentiment': 'positive', 'confidence': 0.7, 'language': 'kannada'}
            elif negative_score > positive_score:
                return {'sentiment': 'negative', 'confidence': 0.7, 'language': 'kannada'}
            else:
                return {'sentiment': 'neutral', 'confidence': 0.5, 'language': 'kannada'}
        
        # For English and other languages
        try:
            # TextBlob analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # VADER analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            compound = vader_scores['compound']
            
            # Combine scores
            final_score = (polarity + compound) / 2
            
            if final_score > 0.1:
                sentiment = 'positive'
            elif final_score < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(final_score),
                'language': language
            }
        except:
            return {'sentiment': 'neutral', 'confidence': 0, 'language': language}


class EnhancedWhatsAppAnalyzer:
    """Main analyzer with content flagging"""
    
    def __init__(self):
        self.sentiment_analyzer = MultilingualSentimentAnalyzer()
        self.content_flagger = ContentFlagger()
        self.content_flagger.train_models()
        self.df = None
        self.flagged_messages = []
    
    def parse_chat(self, file_path):
        """Parse WhatsApp chat file"""
        try:
            # Read file with different encodings
            encodings = ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    break
                except:
                    continue
            
            if not content:
                return False
            
            # WhatsApp message patterns
            patterns = [
                r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',
                r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)',
            ]
            
            messages = []
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        date_str, time_str, sender, message = match
                        
                        # Parse datetime
                        try:
                            dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M")
                        except:
                            try:
                                dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
                            except:
                                continue
                        
                        messages.append({
                            'datetime': dt,
                            'date': dt.date(),
                            'hour': dt.hour,
                            'sender': sender.strip(),
                            'message': message.strip()
                        })
                    break
            
            if messages:
                self.df = pd.DataFrame(messages)
                
                # Filter system messages
                system_keywords = ['image omitted', 'video omitted', 'joined using', 'left']
                initial_count = len(self.df)
                for keyword in system_keywords:
                    self.df = self.df[~self.df['message'].str.contains(keyword, case=False, na=False)]
                
                return len(self.df) > 0
            
            return False
            
        except Exception as e:
            print(f"Error parsing chat: {e}")
            return False
    
    def analyze_messages(self):
        """Analyze all messages for sentiment and flag sensitive content"""
        if self.df is None:
            return False
        
        print(f"Analyzing {len(self.df)} messages...")
        
        sentiments = []
        self.flagged_messages = []
        
        for idx, row in self.df.iterrows():
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(row['message'])
            sentiments.append(sentiment_result)
            
            # Content flagging
            flag_result = self.content_flagger.flag_content(
                row['message'],
                sender=row['sender'],
                timestamp=row['datetime']
            )
            
            if flag_result['is_flagged']:
                flag_result['index'] = idx
                self.flagged_messages.append(flag_result)
        
        # Add results to dataframe
        self.df['sentiment'] = [s['sentiment'] for s in sentiments]
        self.df['confidence'] = [s['confidence'] for s in sentiments]
        self.df['language'] = [s['language'] for s in sentiments]
        
        # Add flagging info
        self.df['is_flagged'] = False
        self.df['flag_priority'] = 'NORMAL'
        
        for flag in self.flagged_messages:
            self.df.loc[flag['index'], 'is_flagged'] = True
            self.df.loc[flag['index'], 'flag_priority'] = flag['priority']
        
        print(f"Analysis complete. Found {len(self.flagged_messages)} flagged messages.")
        
        return True
    
    def get_insights(self):
        """Generate insights including flagged content statistics"""
        if self.df is None:
            return {}
        
        total = len(self.df)
        
        # Basic stats
        insights = {
            'total_messages': total,
            'participants': self.df['sender'].nunique(),
            'date_range': {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max())
            }
        }
        
        # Sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        insights['sentiment_distribution'] = {
            'positive': int(sentiment_counts.get('positive', 0)),
            'negative': int(sentiment_counts.get('negative', 0)),
            'neutral': int(sentiment_counts.get('neutral', 0))
        }
        
        # Language distribution
        lang_counts = self.df['language'].value_counts()
        insights['languages'] = {
            lang: int(count) for lang, count in lang_counts.items()
        }
        
        # Flagged content statistics
        if self.flagged_messages:
            priority_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for flag in self.flagged_messages:
                priority_counts[flag['priority']] += 1
                category = flag.get('category', 'unknown')
                category_counts[category] += 1
            
            insights['flagged_content'] = {
                'total_flagged': len(self.flagged_messages),
                'by_priority': dict(priority_counts),
                'by_category': dict(category_counts),
                'critical_messages': [
                    {
                        'message': f['message'][:100] + '...' if len(f['message']) > 100 else f['message'],
                        'sender': f['sender'],
                        'priority': f['priority'],
                        'timestamp': f['timestamp'].isoformat() if f['timestamp'] else None
                    }
                    for f in sorted(self.flagged_messages, 
                                  key=lambda x: self.content_flagger.priority_levels[x['priority']]['level'],
                                  reverse=True)[:10]
                ]
            }
        else:
            insights['flagged_content'] = {
                'total_flagged': 0,
                'by_priority': {},
                'by_category': {}
            }
        
        return insights
    
    def create_visualizations(self, analysis_id):
        """Create analysis charts"""
        if self.df is None:
            return None
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                     autopct='%1.1f%%', colors=colors)
        axes[0,0].set_title('Sentiment Distribution')
        
        # 2. Language distribution
        lang_counts = self.df['language'].value_counts().head(5)
        axes[0,1].bar(lang_counts.index, lang_counts.values)
        axes[0,1].set_title('Language Distribution')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Flagged content by priority
        if self.flagged_messages:
            priority_counts = defaultdict(int)
            for flag in self.flagged_messages:
                priority_counts[flag['priority']] += 1
            
            priorities = list(self.content_flagger.priority_levels.keys())
            counts = [priority_counts.get(p, 0) for p in priorities]
            colors = [self.content_flagger.priority_levels[p]['color'] for p in priorities]
            
            axes[0,2].bar(priorities, counts, color=colors)
            axes[0,2].set_title('Flagged Messages by Priority')
        else:
            axes[0,2].text(0.5, 0.5, 'No Flagged Content', ha='center', va='center')
            axes[0,2].set_title('Flagged Messages')
        
        # 4. Messages by hour
        hourly = self.df['hour'].value_counts().sort_index()
        axes[1,0].bar(hourly.index, hourly.values)
        axes[1,0].set_title('Messages by Hour')
        axes[1,0].set_xlabel('Hour')
        
        # 5. Top contributors
        top_senders = self.df['sender'].value_counts().head(10)
        axes[1,1].barh(range(len(top_senders)), top_senders.values)
        axes[1,1].set_yticks(range(len(top_senders)))
        axes[1,1].set_yticklabels([s[:15] + '...' if len(s) > 15 else s 
                                   for s in top_senders.index])
        axes[1,1].set_title('Top Contributors')
        
        # 6. Messages over time
        daily = self.df.groupby('date').size()
        axes[1,2].plot(daily.index, daily.values)
        axes[1,2].set_title('Messages Over Time')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save
        chart_path = f'static/charts/{analysis_id}_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path


# Ticket Management Functions
def create_tickets_from_analysis():
    """Create tickets from CRITICAL flagged messages only"""
    global tickets_storage
    
    tickets_created = 0
    
    if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
        print(f"Found {len(analyzer.flagged_messages)} flagged messages")
        
        for flag_data in analyzer.flagged_messages:
            priority = flag_data.get('priority', 'NORMAL')
            
            # ONLY create tickets for CRITICAL priority
            if priority == 'CRITICAL':
                ticket = create_ticket_from_flagged_data(flag_data)
                if ticket:
                    tickets_created += 1
    
    print(f"Created {tickets_created} CRITICAL tickets")
    return tickets_created

def create_ticket_from_flagged_data(flag_data):
    """Create a single ticket from flagged message data"""
    global tickets_storage
    
    try:
        # Generate ticket ID
        ticket_id = f"T{len(tickets_storage) + 1:03d}"
        
        # Get message content
        message_content = flag_data.get('message', 'No message content')
        sender = flag_data.get('sender', 'Unknown User')
        priority = flag_data.get('priority', 'MEDIUM')
        category = flag_data.get('category', 'general')
        
        # Enhanced category detection for violence
        if category == 'general' or category == 'normal' or not category:
            message_lower = message_content.lower()
            
            if any(word in message_lower for word in ['kill', 'death', 'murder', 'attack', 'bomb', 'weapon']):
                category = 'violence'
            elif any(word in message_lower for word in ['water', 'electricity', 'power', 'road']):
                category = 'infrastructure'
            elif any(word in message_lower for word in ['doctor', 'hospital', 'medicine']):
                category = 'healthcare'
            elif any(word in message_lower for word in ['emergency', 'urgent', 'help']):
                category = 'emergency'
            else:
                category = 'general'
        
        # Assignment rules
        assignment_map = {
            'violence': 'Police Department',
            'infrastructure': 'Municipal Engineer',
            'healthcare': 'Health Officer',
            'emergency': 'Emergency Response Team',
            'corruption': 'Anti-Corruption Officer',
            'general': 'General Administrator'
        }
        
        assigned_to = assignment_map.get(category, 'General Administrator')
        
        # Create ticket
        ticket = {
            'id': ticket_id,
            'title': f"{category.title()} Issue - {sender}",
            'description': message_content,
            'category': category,
            'priority': priority,
            'status': 'OPEN',
            'assigned_to': assigned_to,
            'sender': sender,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Auto-created from CRITICAL flagged message'
        }
        
        # Add to storage
        tickets_storage.append(ticket)
        print(f"Created ticket {ticket_id}: {ticket['title']}")
        
        return ticket
        
    except Exception as e:
        print(f"Error creating ticket: {e}")
        return None


# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/reports', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize analyzer
analyzer = EnhancedWhatsAppAnalyzer()

# ===== FLASK ROUTES =====

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.txt'):
            analysis_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
            file.save(file_path)
            
            # Parse and analyze
            if analyzer.parse_chat(file_path):
                if analyzer.analyze_messages():
                    # Auto-create tickets from CRITICAL flagged messages only
                    tickets_created = create_tickets_from_analysis()
                    
                    # Get insights
                    insights = analyzer.get_insights()
                    insights['tickets_created'] = tickets_created
                    
                    # Create visualizations
                    chart_path = analyzer.create_visualizations(analysis_id)
                    
                    # Save data
                    insights_path = f'static/reports/{analysis_id}_insights.json'
                    with open(insights_path, 'w') as f:
                        json.dump(insights, f, indent=2, cls=NumpyEncoder)
                    
                    # Save flagged messages
                    if analyzer.flagged_messages:
                        flagged_path = f'static/reports/{analysis_id}_flagged.json'
                        with open(flagged_path, 'w') as f:
                            json.dump(analyzer.flagged_messages, f, indent=2, cls=NumpyEncoder)
                    
                    return jsonify({
                        'success': True,
                        'analysis_id': analysis_id,
                        'insights': insights,
                        'tickets_created': tickets_created
                    })
        
        return jsonify({'error': 'Invalid file format or processing failed'}), 400
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    try:
        # Load insights
        insights_path = f'static/reports/{analysis_id}_insights.json'
        with open(insights_path, 'r') as f:
            insights = json.load(f)
        
        chart_url = f'/static/charts/{analysis_id}_analysis.png'
        
        return render_template('results.html', 
                             insights=insights, 
                             chart_url=chart_url,
                             analysis_id=analysis_id)
    except Exception as e:
        print(f"Error showing results: {e}")
        return redirect(url_for('index'))

# ===== MANAGEMENT DASHBOARD ROUTES =====

@app.route('/management')
def management_dashboard():
    """Management dashboard page"""
    return render_template('management.html')

@app.route('/api/dashboard_stats')
def dashboard_stats():
    """API endpoint for dashboard statistics"""
    total_messages = len(analyzer.df) if analyzer.df is not None else 0
    flagged_messages = len(analyzer.flagged_messages) if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages else 0
    
    # Count critical messages specifically
    critical_messages = 0
    if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
        critical_messages = len([f for f in analyzer.flagged_messages if f.get('priority') == 'CRITICAL'])
    
    open_tickets = len([t for t in tickets_storage if t['status'] == 'OPEN'])
    total_tickets = len(tickets_storage)
    
    return jsonify({
        'total_messages': total_messages,
        'flagged_messages': flagged_messages,
        'critical_messages': critical_messages,
        'open_tickets': open_tickets,
        'total_tickets': total_tickets,
        'active_groups': 3
    })

@app.route('/api/recent_flagged')
def recent_flagged():
    """Get recent flagged messages"""
    if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
        return jsonify([{
            'message': flag.get('message', 'No message')[:100],
            'sender': flag.get('sender', 'Unknown'),
            'priority': flag.get('priority', 'MEDIUM'),
            'category': flag.get('category', 'general'),
            'timestamp': '30 minutes ago'
        } for flag in analyzer.flagged_messages[:10]])
    
    return jsonify([])

@app.route('/api/create_ticket', methods=['POST'])
def create_ticket():
    """Create a ticket manually"""
    global tickets_storage
    
    try:
        data = request.get_json()
        
        # Generate ticket ID
        ticket_id = f"T{len(tickets_storage) + 1:03d}"
        
        ticket = {
            'id': ticket_id,
            'title': data.get('title', f"Manual Ticket from {data.get('sender', 'Unknown')}"),
            'description': data.get('message', ''),
            'category': 'manual',
            'priority': data.get('priority', 'MEDIUM'),
            'status': 'OPEN',
            'assigned_to': 'General Administrator',
            'sender': data.get('sender', 'Unknown'),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Manually created'
        }
        
        tickets_storage.append(ticket)
        
        return jsonify({
            'success': True,
            'ticket_id': ticket_id,
            'message': f'Ticket {ticket_id} created successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tickets')
def get_tickets():
    """Get all tickets"""
    return jsonify(tickets_storage)

@app.route('/api/tickets/<ticket_id>/update', methods=['POST'])
def update_ticket(ticket_id):
    """Update ticket status"""
    try:
        data = request.get_json()
        new_status = data.get('status', 'OPEN')
        
        for ticket in tickets_storage:
            if ticket['id'] == ticket_id:
                ticket['status'] = new_status
                ticket['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if new_status == 'RESOLVED':
                    ticket['resolved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({
                    'success': True,
                    'message': f'Ticket {ticket_id} updated to {new_status}'
                })
        
        return jsonify({
            'success': False,
            'error': 'Ticket not found'
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analytics/summary')
def analytics_summary():
    """Get analytics summary"""
    if analyzer.df is not None and not analyzer.df.empty:
        total_messages = len(analyzer.df)
        
        # Sentiment distribution
        sentiment_counts = analyzer.df['sentiment'].value_counts() if 'sentiment' in analyzer.df.columns else {}
        
        # Language distribution
        language_counts = analyzer.df['language'].value_counts() if 'language' in analyzer.df.columns else {}
        
        # Flagged messages analysis
        flagged_count = len(analyzer.flagged_messages) if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages else 0
        
        # Priority distribution
        priority_dist = {}
        if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
            for flag in analyzer.flagged_messages:
                priority = flag.get('priority', 'NORMAL')
                priority_dist[priority] = priority_dist.get(priority, 0) + 1
        
        return jsonify({
            'total_messages': total_messages,
            'flagged_messages': flagged_count,
            'flagged_percentage': round((flagged_count / total_messages * 100), 2) if total_messages > 0 else 0,
            'sentiment_distribution': dict(sentiment_counts),
            'language_distribution': dict(language_counts),
            'priority_distribution': priority_dist,
            'tickets_created': len(tickets_storage),
            'open_tickets': len([t for t in tickets_storage if t['status'] == 'OPEN']),
            'resolved_tickets': len([t for t in tickets_storage if t['status'] == 'RESOLVED'])
        })
    
    return jsonify({
        'total_messages': 0,
        'flagged_messages': 0,
        'flagged_percentage': 0,
        'sentiment_distribution': {},
        'language_distribution': {},
        'priority_distribution': {},
        'tickets_created': len(tickets_storage),
        'open_tickets': len([t for t in tickets_storage if t['status'] == 'OPEN']),
        'resolved_tickets': len([t for t in tickets_storage if t['status'] == 'RESOLVED'])
    })

@app.route('/api/flag_message', methods=['POST'])
def flag_message_api():
    """API endpoint to flag a single message"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        result = analyzer.content_flagger.flag_content(message)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk_messages', methods=['GET'])
def get_bulk_messages():
    """Get bulk message history"""
    global bulk_message_history
    return jsonify(bulk_message_history)

@app.route('/api/simulate_message', methods=['POST'])  
def simulate_incoming_message():
    """Simulate an incoming WhatsApp message for testing"""
    data = request.get_json()
    message_content = data.get('message', '')
    sender = data.get('sender', 'Test User')
    
    if not message_content:
        return jsonify({'error': 'Message content required'}), 400
    
    # Analyze the message using existing system
    sentiment_result = analyzer.sentiment_analyzer.analyze_sentiment(message_content)
    flag_result = analyzer.content_flagger.flag_content(message_content, sender=sender)
    
    # Create ticket ONLY if flagged as CRITICAL
    ticket_created = None
    if flag_result['is_flagged'] and flag_result['priority'] == 'CRITICAL':
        ticket_created = create_ticket_from_flagged_data({
            'message': message_content,
            'sender': sender,
            'priority': flag_result['priority'],
            'category': flag_result.get('category', 'general')
        })
    
    return jsonify({
        'message_analyzed': True,
        'sentiment': sentiment_result,
        'flagging': flag_result,
        'ticket_created': ticket_created['id'] if ticket_created else None,
        'auto_response': get_auto_response(flag_result) if flag_result['is_flagged'] else None
    })

def get_auto_response(flag_result):
    """Generate auto-response based on flag result"""
    responses = {
        'violence': "⚠️ CRITICAL ALERT: This message has been flagged for immediate attention. Emergency response team notified.",
        'infrastructure': "Thank you for reporting this infrastructure issue. We have noted your concern.",
        'healthcare': "Your healthcare concern has been noted. For medical emergencies, please call 108.",
        'emergency': "Emergency alert received. Appropriate authorities have been notified.",
        'corruption': "Thank you for this report. It has been forwarded to the appropriate authorities.",
        'election': "Your election-related concern has been noted."
    }
    
    category = flag_result.get('category', 'general')
    priority = flag_result.get('priority', 'NORMAL')
    
    if priority == 'CRITICAL':
        return responses.get('violence')  # All critical gets emergency response
    else:
        return responses.get(category, "Thank you for your message. We have received it and will respond appropriately.")

# ===== ADD THESE ROUTES TO YOUR APP =====

# WhatsApp Webhook (replace or add to your existing webhook route)
@app.route('/webhook', methods=['GET', 'POST'])
def whatsapp_webhook():
    """WhatsApp webhook endpoint"""
    if request.method == 'GET':
        # Webhook verification
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        
        if mode == 'subscribe' and token == WHATSAPP_CONFIG['WEBHOOK_VERIFY_TOKEN']:
            logger.info("WhatsApp webhook verified successfully")
            return challenge
        else:
            logger.warning("WhatsApp webhook verification failed")
            return 'Forbidden', 403
    
    elif request.method == 'POST':
        try:
            # Verify signature if app secret is configured
            if WHATSAPP_CONFIG.get('APP_SECRET'):
                signature = request.headers.get('X-Hub-Signature-256', '')
                payload = request.get_data(as_text=True)
                
                if not whatsapp_service.verify_webhook_signature(payload, signature):
                    logger.warning("Webhook signature verification failed")
                    return 'Unauthorized', 401
            
            # Process webhook data
            data = request.get_json()
            result = whatsapp_service.process_webhook(data)
            
            # If there's an incoming message, analyze it with your existing system
            if result.get('incoming_message'):
                incoming_msg = result['incoming_message']
                
                # Use your existing analyzer
                sentiment_result = analyzer.sentiment_analyzer.analyze_sentiment(incoming_msg['content'])
                flag_result = analyzer.content_flagger.flag_content(
                    incoming_msg['content'],
                    sender=incoming_msg['from'],
                    timestamp=incoming_msg['timestamp']
                )
                
                # Auto-create ticket if CRITICAL
                if flag_result['is_flagged'] and flag_result['priority'] == 'CRITICAL':
                    ticket_created = create_ticket_from_flagged_data({
                        'message': incoming_msg['content'],
                        'sender': incoming_msg['from'],
                        'priority': flag_result['priority'],
                        'category': flag_result.get('category', 'general')
                    })
                    
                    # Send auto-response
                    auto_response = get_auto_response(flag_result)
                    if auto_response:
                        whatsapp_service.send_text_message(
                            to=incoming_msg['from'],
                            message=auto_response
                        )
            
            return jsonify({'status': 'success'})
            
        except Exception as e:
            logger.error(f"Webhook processing error: {str(e)}")
            return jsonify({'error': str(e)}), 500

# WhatsApp Configuration Check
@app.route('/api/whatsapp/config_check')
def check_whatsapp_config():
    """Check WhatsApp API configuration"""
    try:
        is_configured, config_status = whatsapp_service.is_configured()
        
        return jsonify({
            'configured': is_configured,
            'status': config_status,
            'message': 'WhatsApp API ready' if is_configured else 'WhatsApp API configuration incomplete'
        })
        
    except Exception as e:
        return jsonify({
            'configured': False,
            'error': str(e)
        }), 500

# Send Single WhatsApp Message
@app.route('/api/whatsapp/send_message', methods=['POST'])
def send_whatsapp_message():
    """Send a single WhatsApp message"""
    try:
        data = request.get_json()
        to = data.get('to', '').strip()
        message = data.get('message', '').strip()
        
        if not to or not message:
            return jsonify({
                'success': False,
                'error': 'Both "to" and "message" are required'
            }), 400
        
        # Validate phone number format
        import re
        phone_regex = r'^\+[1-9]\d{1,14}$'
        if not re.match(phone_regex, to):
            return jsonify({
                'success': False,
                'error': 'Invalid phone number format. Use international format: +1234567890'
            }), 400
        
        # Send message via WhatsApp service
        result = whatsapp_service.send_text_message(to, message)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Send Media Message
@app.route('/api/whatsapp/send_media', methods=['POST'])
def send_whatsapp_media():
    """Send media message via WhatsApp"""
    try:
        data = request.get_json()
        to = data.get('to', '').strip()
        media_type = data.get('media_type', 'image')
        media_url = data.get('media_url', '').strip()
        caption = data.get('caption', '')
        filename = data.get('filename', '')
        
        if not to or not media_url:
            return jsonify({
                'success': False,
                'error': 'Recipient and media URL are required'
            }), 400
        
        if media_type not in ['image', 'document', 'audio', 'video']:
            return jsonify({
                'success': False,
                'error': 'Invalid media type. Use: image, document, audio, video'
            }), 400
        
        result = whatsapp_service.send_media_message(
            to=to,
            media_type=media_type,
            media_url=media_url,
            caption=caption,
            filename=filename
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error sending media message: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Enhanced Bulk Messaging with WhatsApp
@app.route('/api/whatsapp/bulk_send', methods=['POST'])
def send_bulk_whatsapp():
    """Send bulk WhatsApp messages"""
    try:
        data = request.get_json()
        recipients = data.get('recipients', [])
        message = data.get('message', '').strip()
        template_name = data.get('template_name')
        delay = data.get('delay_seconds', 0.1)
        campaign_name = data.get('campaign_name', f'Bulk Campaign {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        
        if not recipients:
            return jsonify({
                'success': False,
                'error': 'Recipients list is required'
            }), 400
        
        if not message and not template_name:
            return jsonify({
                'success': False,
                'error': 'Either message or template_name is required'
            }), 400
        
        # Validate phone numbers
        import re
        phone_regex = r'^\+[1-9]\d{1,14}$'
        invalid_numbers = [num for num in recipients if not re.match(phone_regex, num)]
        
        if invalid_numbers:
            return jsonify({
                'success': False,
                'error': f'Invalid phone numbers found: {", ".join(invalid_numbers[:5])}'
            }), 400
        
        # Send bulk messages
        result = whatsapp_service.send_bulk_messages(
            recipients=recipients,
            message=message,
            template_name=template_name,
            delay_seconds=delay,
            campaign_name=campaign_name
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in bulk WhatsApp sending: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/bulk_message', methods=['POST'])
def send_bulk_message():
    """Enhanced bulk message endpoint with scheduling support"""
    global bulk_message_history, scheduled_messages_storage
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        groups = data.get('groups', [])
        recipients = data.get('recipients', [])
        use_whatsapp = data.get('use_whatsapp', False)
        message_type = data.get('message_type', 'text')
        media_url = data.get('media_url', '')
        
        # Scheduling parameters
        is_scheduled = data.get('is_scheduled', False)
        schedule_date = data.get('schedule_date', '')
        schedule_time = data.get('schedule_time', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message content is required'
            }), 400
        
        if not groups and not recipients:
            return jsonify({
                'success': False,
                'error': 'At least one group or recipient must be selected'
            }), 400
        
        # Handle scheduling
        if is_scheduled and schedule_date and schedule_time:
            return handle_scheduled_message(data)
        
        # Immediate sending (existing logic)
        return send_immediate_message(data)
        
    except Exception as e:
        logger.error(f"Error in bulk messaging: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 3. ADD these new routes (make sure they don't exist already):

@app.route('/api/bulk_message_enhanced', methods=['POST'])  
def send_bulk_message_enhanced():
    """Alternative endpoint name for enhanced bulk messaging"""
    return send_bulk_message()  # Just calls the main function

@app.route('/api/scheduled_messages', methods=['GET'])
def get_scheduled_messages():
    """Get all scheduled messages"""
    global scheduled_messages_storage
    
    try:
        # Add time remaining for pending messages
        for msg in scheduled_messages_storage:
            if msg['status'] == 'pending':
                scheduled_time = datetime.strptime(msg['scheduled_for'], '%Y-%m-%d %H:%M:%S')
                time_remaining = scheduled_time - datetime.now()
                
                if time_remaining.total_seconds() > 0:
                    days = time_remaining.days
                    hours, remainder = divmod(time_remaining.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    
                    if days > 0:
                        msg['time_remaining'] = f"{days}d {hours}h {minutes}m"
                    elif hours > 0:
                        msg['time_remaining'] = f"{hours}h {minutes}m"
                    else:
                        msg['time_remaining'] = f"{minutes}m"
                else:
                    msg['time_remaining'] = "Overdue"
        
        return jsonify(scheduled_messages_storage)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# 4. ADD the helper functions (if they don't exist):

def handle_scheduled_message(data):
    """Handle scheduling of messages"""
    global scheduled_messages_storage
    
    try:
        # Parse scheduled datetime
        schedule_datetime_str = f"{data['schedule_date']} {data['schedule_time']}"
        scheduled_datetime = datetime.strptime(schedule_datetime_str, "%Y-%m-%d %H:%M")
        
        # Validate scheduling time
        if scheduled_datetime <= datetime.now():
            return jsonify({
                'success': False,
                'error': 'Cannot schedule messages in the past'
            }), 400
        
        # Create scheduled message record
        scheduled_id = f"SCH{len(scheduled_messages_storage) + 1:03d}"
        
        scheduled_message = {
            'id': scheduled_id,
            'message': data['message'][:100] + '...' if len(data['message']) > 100 else data['message'],
            'full_message': data['message'],
            'message_type': data.get('message_type', 'text'),
            'media_url': data.get('media_url', ''),
            'groups': data.get('groups', []),
            'recipients': data.get('recipients', []),
            'total_targets': len(data.get('groups', [])) + len(data.get('recipients', [])),
            'scheduled_for': scheduled_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'pending',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'use_whatsapp': data.get('use_whatsapp', False)
        }
        
        scheduled_messages_storage.append(scheduled_message)
        
        # Calculate delay and schedule
        delay_seconds = (scheduled_datetime - datetime.now()).total_seconds()
        
        # Start background thread for sending
        def delayed_send():
            time.sleep(delay_seconds)
            send_scheduled_message_now(scheduled_id)
        
        thread = threading.Thread(target=delayed_send, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'scheduled_id': scheduled_id,
            'scheduled_for': scheduled_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'delay_minutes': int(delay_seconds / 60),
            'message': f'Message scheduled successfully for {scheduled_datetime.strftime("%B %d, %Y at %I:%M %p")}'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid date/time format'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def send_immediate_message(data):
    """Send message immediately (existing logic enhanced)"""
    global bulk_message_history
    
    try:
        message = data['message']
        groups = data.get('groups', [])
        recipients = data.get('recipients', [])
        use_whatsapp = data.get('use_whatsapp', False)
        
        if use_whatsapp and recipients:
            # Send via WhatsApp Business API
            campaign_name = f"Immediate Message {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            result = whatsapp_service.send_bulk_messages(
                recipients=recipients,
                message=message,
                delay_seconds=0.1,
                campaign_name=campaign_name
            )
            
            if result.get('success'):
                bulk_message = {
                    'id': result['campaign_id'],
                    'message': message[:100] + '...' if len(message) > 100 else message,
                    'full_message': message,
                    'target_recipients': recipients,
                    'recipient_count': len(recipients),
                    'status': 'Started',
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'sent_count': 0,
                    'failed_count': 0,
                    'platform': 'WhatsApp Business API'
                }
                
                bulk_message_history.append(bulk_message)
                
                return jsonify({
                    'success': True,
                    'message_id': result['campaign_id'],
                    'sent_to': len(recipients),
                    'message': f'Message sent immediately to {len(recipients)} recipients!',
                    'platform': 'WhatsApp Business API'
                })
            else:
                return jsonify(result), 500
        
        else:
            # Simulated sending for groups
            bulk_message = {
                'id': f"BM{len(bulk_message_history) + 1:03d}",
                'message': message[:100] + '...' if len(message) > 100 else message,
                'full_message': message,
                'target_groups': groups,
                'group_count': len(groups),
                'status': 'Sent',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sent_count': len(groups),
                'failed_count': 0,
                'platform': 'Simulated'
            }
            
            bulk_message_history.append(bulk_message)
            
            return jsonify({
                'success': True,
                'message_id': bulk_message['id'],
                'sent_to': len(groups),
                'message': f'Message sent immediately to {len(groups)} groups!',
                'platform': 'Simulated'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def send_scheduled_message_now(scheduled_id):
    """Execute scheduled message sending"""
    global scheduled_messages_storage
    
    try:
        # Find scheduled message
        scheduled_msg = None
        for msg in scheduled_messages_storage:
            if msg['id'] == scheduled_id:
                scheduled_msg = msg
                break
        
        if not scheduled_msg:
            logger.error(f"Scheduled message {scheduled_id} not found")
            return
        
        # Update status to sending
        scheduled_msg['status'] = 'sending'
        scheduled_msg['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare data for sending
        send_data = {
            'message': scheduled_msg['full_message'],
            'message_type': scheduled_msg['message_type'],
            'media_url': scheduled_msg['media_url'],
            'groups': scheduled_msg['groups'],
            'recipients': scheduled_msg['recipients'],
            'use_whatsapp': scheduled_msg['use_whatsapp']
        }
        
        # Send the message
        result = send_immediate_message(send_data)
        
        if result.get_json().get('success'):
            scheduled_msg['status'] = 'completed'
            scheduled_msg['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            scheduled_msg['sent_count'] = scheduled_msg['total_targets']
            logger.info(f"Scheduled message {scheduled_id} sent successfully")
        else:
            scheduled_msg['status'] = 'failed'
            scheduled_msg['error'] = result.get_json().get('error', 'Unknown error')
            logger.error(f"Scheduled message {scheduled_id} failed to send")
            
    except Exception as e:
        # Update status to failed
        for msg in scheduled_messages_storage:
            if msg['id'] == scheduled_id:
                msg['status'] = 'failed'
                msg['error'] = str(e)
                break
        logger.error(f"Error sending scheduled message {scheduled_id}: {str(e)}")
# Get WhatsApp Templates
@app.route('/api/whatsapp/templates')
def get_whatsapp_templates():
    """Get approved WhatsApp message templates"""
    try:
        templates = whatsapp_service.get_message_templates()
        return jsonify({
            'success': True,
            'templates': templates,
            'count': len(templates)
        })
    except Exception as e:
        logger.error(f"Error getting WhatsApp templates: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Get WhatsApp Delivery Statistics
@app.route('/api/whatsapp/delivery_stats')
def get_whatsapp_delivery_stats():
    """Get WhatsApp message delivery statistics"""
    try:
        stats = whatsapp_service.get_delivery_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting delivery stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Get Campaign Statistics
@app.route('/api/whatsapp/campaigns')
def get_whatsapp_campaigns():
    """Get WhatsApp campaign statistics"""
    try:
        campaigns = whatsapp_service.get_campaign_stats()
        return jsonify({
            'success': True,
            'campaigns': campaigns
        })
    except Exception as e:
        logger.error(f"Error getting campaigns: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/whatsapp/campaigns/<campaign_id>')
def get_whatsapp_campaign_details(campaign_id):
    """Get specific campaign details"""
    try:
        stats = whatsapp_service.get_campaign_stats(campaign_id)
        return jsonify({
            'success': True,
            'campaign': stats
        })
    except Exception as e:
        logger.error(f"Error getting campaign details: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WhatsApp Health Check
@app.route('/api/whatsapp/health')
def whatsapp_health_check():
    """Get WhatsApp service health status"""
    try:
        health = whatsapp_service.get_health_status()
        return jsonify(health)
    except Exception as e:
        return jsonify({
            'healthy': False,
            'error': str(e)
        }), 500

# Enhanced Dashboard Stats (modify your existing endpoint)
@app.route('/api/dashboard_stats_enhanced')
def dashboard_stats_enhanced():
    """Enhanced dashboard statistics with WhatsApp data"""
    total_messages = len(analyzer.df) if analyzer.df is not None else 0
    flagged_messages = len(analyzer.flagged_messages) if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages else 0
    
    # Count critical messages specifically
    critical_messages = 0
    if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
        critical_messages = len([f for f in analyzer.flagged_messages if f.get('priority') == 'CRITICAL'])
    
    open_tickets = len([t for t in tickets_storage if t['status'] == 'OPEN'])
    total_tickets = len(tickets_storage)
    
    # Get WhatsApp stats
    whatsapp_stats = whatsapp_service.get_delivery_stats()
    whatsapp_messages = whatsapp_stats.get('total_tracked', 0)
    
    return jsonify({
        'total_messages': total_messages,
        'flagged_messages': flagged_messages,
        'critical_messages': critical_messages,
        'open_tickets': open_tickets,
        'total_tickets': total_tickets,
        'whatsapp_messages': whatsapp_messages,
        'active_groups': 3,
        'whatsapp_health': whatsapp_service.get_health_status().get('healthy', False)
    })


# Test WhatsApp Integration
@app.route('/api/whatsapp/test', methods=['POST'])
def test_whatsapp_integration():
    """Test WhatsApp integration with a single message"""
    try:
        data = request.get_json()
        test_number = data.get('test_number', '').strip()
        
        if not test_number:
            return jsonify({
                'success': False,
                'error': 'Test phone number is required'
            }), 400
        
        # Send test message
        test_message = f"🧪 WhatsApp API Test from your platform - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        result = whatsapp_service.send_text_message(
            to=test_number,
            message=test_message,
            message_id="test_message"
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Test message sent successfully!',
                'whatsapp_message_id': result.get('whatsapp_message_id'),
                'test_number': test_number
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'test_number': test_number
            }), 400
        
    except Exception as e:
        logger.error(f"Error testing WhatsApp integration: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Maintenance endpoint
@app.route('/api/whatsapp/maintenance/cleanup', methods=['POST'])
def cleanup_whatsapp_data():
    """Clean up old WhatsApp message data"""
    try:
        data = request.get_json() or {}
        days_to_keep = data.get('days_to_keep', 30)
        
        result = whatsapp_service.cleanup_old_data(days_to_keep)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error cleaning up WhatsApp data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/scheduled_messages/<message_id>/cancel', methods=['POST'])
def cancel_scheduled_message(message_id):
    """Cancel a scheduled message"""
    global scheduled_messages_storage
    
    try:
        for msg in scheduled_messages_storage:
            if msg['id'] == message_id and msg['status'] == 'pending':
                msg['status'] = 'cancelled'
                msg['cancelled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({
                    'success': True,
                    'message': f'Scheduled message {message_id} cancelled successfully'
                })
        
        return jsonify({
            'success': False,
            'error': 'Message not found or cannot be cancelled'
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scheduled_messages/<message_id>/reschedule', methods=['POST'])
def reschedule_message(message_id):
    """Reschedule a pending message"""
    global scheduled_messages_storage
    
    try:
        data = request.get_json()
        new_schedule_date = data.get('schedule_date')
        new_schedule_time = data.get('schedule_time')
        
        if not new_schedule_date or not new_schedule_time:
            return jsonify({
                'success': False,
                'error': 'New schedule date and time are required'
            }), 400
        
        # Parse new datetime
        new_datetime_str = f"{new_schedule_date} {new_schedule_time}"
        new_datetime = datetime.strptime(new_datetime_str, "%Y-%m-%d %H:%M")
        
        if new_datetime <= datetime.now():
            return jsonify({
                'success': False,
                'error': 'Cannot reschedule to a past time'
            }), 400
        
        # Find and update message
        for msg in scheduled_messages_storage:
            if msg['id'] == message_id and msg['status'] == 'pending':
                msg['scheduled_for'] = new_datetime.strftime('%Y-%m-%d %H:%M:%S')
                msg['rescheduled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Restart the scheduling thread
                delay_seconds = (new_datetime - datetime.now()).total_seconds()
                
                def delayed_send():
                    time.sleep(delay_seconds)
                    send_scheduled_message_now(message_id)
                
                thread = threading.Thread(target=delayed_send, daemon=True)
                thread.start()
                
                return jsonify({
                    'success': True,
                    'message': f'Message rescheduled for {new_datetime.strftime("%B %d, %Y at %I:%M %p")}'
                })
        
        return jsonify({
            'success': False,
            'error': 'Message not found or cannot be rescheduled'
        }), 404
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid date/time format'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scheduling_stats', methods=['GET'])
def get_scheduling_stats():
    """Get scheduling statistics for dashboard"""
    global scheduled_messages_storage
    
    try:
        total_scheduled = len(scheduled_messages_storage)
        pending = len([msg for msg in scheduled_messages_storage if msg['status'] == 'pending'])
        completed = len([msg for msg in scheduled_messages_storage if msg['status'] == 'completed'])
        failed = len([msg for msg in scheduled_messages_storage if msg['status'] == 'failed'])
        cancelled = len([msg for msg in scheduled_messages_storage if msg['status'] == 'cancelled'])
        
        # Next scheduled message
        next_message = None
        pending_messages = [msg for msg in scheduled_messages_storage if msg['status'] == 'pending']
        if pending_messages:
            # Sort by scheduled time
            pending_messages.sort(key=lambda x: datetime.strptime(x['scheduled_for'], '%Y-%m-%d %H:%M:%S'))
            next_msg = pending_messages[0]
            next_time = datetime.strptime(next_msg['scheduled_for'], '%Y-%m-%d %H:%M:%S')
            time_until = next_time - datetime.now()
            
            if time_until.total_seconds() > 0:
                days = time_until.days
                hours, remainder = divmod(time_until.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                if days > 0:
                    time_str = f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    time_str = f"{hours}h {minutes}m"
                else:
                    time_str = f"{minutes}m"
                
                next_message = {
                    'id': next_msg['id'],
                    'message_preview': next_msg['message'],
                    'scheduled_for': next_msg['scheduled_for'],
                    'time_until': time_str
                }
        
        return jsonify({
            'total_scheduled': total_scheduled,
            'pending': pending,
            'completed': completed,
            'failed': failed,
            'cancelled': cancelled,
            'next_message': next_message,
            'success_rate': round((completed / total_scheduled * 100), 1) if total_scheduled > 0 else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Add this endpoint for quick scheduling stats in dashboard
@app.route('/api/dashboard_stats_with_scheduling')
def dashboard_stats_with_scheduling():
    """Enhanced dashboard stats including scheduling data"""
    try:
        # Get existing stats
        total_messages = len(analyzer.df) if analyzer.df is not None else 0
        flagged_messages = len(analyzer.flagged_messages) if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages else 0
        
        critical_messages = 0
        if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
            critical_messages = len([f for f in analyzer.flagged_messages if f.get('priority') == 'CRITICAL'])
        
        open_tickets = len([t for t in tickets_storage if t['status'] == 'OPEN'])
        total_tickets = len(tickets_storage)
        
        # Add scheduling stats
        pending_scheduled = len([msg for msg in scheduled_messages_storage if msg['status'] == 'pending'])
        total_scheduled = len(scheduled_messages_storage)
        
        # WhatsApp stats
        whatsapp_stats = whatsapp_service.get_delivery_stats()
        whatsapp_messages = whatsapp_stats.get('total_tracked', 0)
        
        return jsonify({
            'total_messages': total_messages,
            'flagged_messages': flagged_messages,
            'critical_messages': critical_messages,
            'open_tickets': open_tickets,
            'total_tickets': total_tickets,
            'pending_scheduled': pending_scheduled,
            'total_scheduled': total_scheduled,
            'whatsapp_messages': whatsapp_messages,
            'active_groups': 3,
            'whatsapp_health': whatsapp_service.get_health_status().get('healthy', False)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility function for scheduling validation
def validate_schedule_time(schedule_date, schedule_time):
    """Validate scheduling parameters"""
    try:
        if not schedule_date or not schedule_time:
            return False, "Date and time are required"
        
        # Parse datetime
        datetime_str = f"{schedule_date} {schedule_time}"
        scheduled_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        
        # Check if in future
        if scheduled_datetime <= datetime.now():
            return False, "Cannot schedule in the past"
        
        # Check if not too far in future (optional - e.g., max 30 days)
        max_future = datetime.now() + timedelta(days=30)
        if scheduled_datetime > max_future:
            return False, "Cannot schedule more than 30 days in advance"
        
        return True, "Valid schedule time"
        
    except ValueError:
        return False, "Invalid date/time format"
    except Exception as e:
        return False, str(e)

@app.route('/check_tickets')
def check_tickets():
    """Quick check of ticket storage"""
    return f"""
    <h2>Ticket Storage Check</h2>
    <p><strong>Total tickets in storage:</strong> {len(tickets_storage)}</p>
    <p><strong>Tickets:</strong></p>
    <pre>{json.dumps(tickets_storage, indent=2, default=str)}</pre>
    <a href="/management">Back to Management</a>
    """
if __name__ == '__main__':
    print("🚀 Starting Enhanced WhatsApp Management Platform...")
    print("🛡️ Features:")
    print("   ✓ Sentiment analysis (English & Kannada)")
    print("   ✓ ML-based threat detection") 
    print("   ✓ CRITICAL-only automatic ticket creation")
    print("   ✓ WhatsApp Business API integration")
    print("   ✓ Real-time message delivery tracking")
    print("   ✓ Bulk messaging campaigns")
    print("   ✓ Template message support")
    print("   ✓ Media message capabilities")
    print("   ✓ Management dashboard with real-time stats")
    print("   ✓ Team collaboration and ticket management")
    print()
    print("📱 Main Analysis: http://localhost:5000")
    print("🎛️ Management Dashboard: http://localhost:5000/management")
    print("🏥 WhatsApp Health: http://localhost:5000/api/whatsapp/health")
    print()
    print("⚠️  TICKET POLICY: Only CRITICAL priority messages create tickets")
    print("📊 CRITICAL: violence, death, threats, weapons")
    print("📝 HIGH: emergencies, infrastructure, healthcare (flagged but no tickets)")
    print()
    print("⚙️  WhatsApp Configuration:")
    is_configured, config_status = whatsapp_service.is_configured()
    for key, value in config_status.items():
        status = "✅ SET" if value else "❌ NOT SET"
        print(f"   {key.upper()}: {status}")
    print()
    if is_configured:
        print("✅ WhatsApp Business API is ready!")
    else:
        print("❌ WhatsApp Business API needs configuration")
        print("📖 See setup guide for configuration steps")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    