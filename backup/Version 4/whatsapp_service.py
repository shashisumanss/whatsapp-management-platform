# whatsapp_service.py
# WhatsApp Business API Integration Service

import os
import requests
import json
import sqlite3
import hashlib
import hmac
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhatsAppService:
    """Complete WhatsApp Business API service"""
    
    def __init__(self, config: Dict = None):
        """Initialize WhatsApp service with configuration"""
        self.config = config or self._get_config_from_env()
        
        # API configuration
        self.access_token = self.config.get('ACCESS_TOKEN')
        self.phone_number_id = self.config.get('PHONE_NUMBER_ID')
        self.webhook_verify_token = self.config.get('WEBHOOK_VERIFY_TOKEN')
        self.business_account_id = self.config.get('BUSINESS_ACCOUNT_ID')
        self.app_secret = self.config.get('APP_SECRET')
        
        # API endpoints
        self.base_url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Initialize database and tracking
        self.init_database()
        
        # Rate limiting
        self.daily_message_count = 0
        self.daily_limit = 10000
        self.last_reset_date = datetime.now().date()
        self.rate_limit_per_second = 80
        
        # Message tracking
        self.message_status = {}
        self.delivery_callbacks = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Templates cache
        self.templates_cache = {}
        self.templates_last_updated = None
        
        logger.info("WhatsApp Service initialized")
    
    def _get_config_from_env(self) -> Dict:
        """Get configuration from environment variables"""
        return {
            'ACCESS_TOKEN': os.getenv('WHATSAPP_ACCESS_TOKEN', ''),
            'PHONE_NUMBER_ID': os.getenv('WHATSAPP_PHONE_NUMBER_ID', ''),
            'WEBHOOK_VERIFY_TOKEN': os.getenv('WHATSAPP_WEBHOOK_TOKEN', ''),
            'BUSINESS_ACCOUNT_ID': os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID', ''),
            'APP_SECRET': os.getenv('WHATSAPP_APP_SECRET', '')
        }
    
    def init_database(self):
        """Initialize SQLite database for message tracking"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db', check_same_thread=False)
            cursor = conn.cursor()
            
            # Messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whatsapp_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE,
                    whatsapp_id TEXT,
                    to_number TEXT,
                    message_content TEXT,
                    message_type TEXT DEFAULT 'text',
                    status TEXT DEFAULT 'sent',
                    sent_at TIMESTAMP,
                    delivered_at TIMESTAMP,
                    read_at TIMESTAMP,
                    failed_at TIMESTAMP,
                    error_message TEXT,
                    template_name TEXT,
                    campaign_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Campaigns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whatsapp_campaigns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT UNIQUE,
                    name TEXT,
                    message_content TEXT,
                    template_name TEXT,
                    total_recipients INTEGER,
                    sent_count INTEGER DEFAULT 0,
                    delivered_count INTEGER DEFAULT 0,
                    read_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whatsapp_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_name TEXT UNIQUE,
                    category TEXT,
                    language TEXT,
                    status TEXT,
                    components TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("WhatsApp database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def is_configured(self) -> Tuple[bool, Dict]:
        """Check if WhatsApp service is properly configured"""
        required_fields = ['ACCESS_TOKEN', 'PHONE_NUMBER_ID', 'WEBHOOK_VERIFY_TOKEN']
        status = {}
        
        for field in required_fields:
            value = self.config.get(field, '')
            is_set = bool(value and value != f'YOUR_{field}')
            status[field.lower() + '_set'] = is_set
        
        # Optional fields
        status['business_account_id_set'] = bool(
            self.config.get('BUSINESS_ACCOUNT_ID', '') and 
            self.config.get('BUSINESS_ACCOUNT_ID') != 'YOUR_BUSINESS_ACCOUNT_ID'
        )
        
        all_configured = all(status[key] for key in [
            'access_token_set', 'phone_number_id_set', 'webhook_verify_token_set'
        ])
        
        return all_configured, status
    
    def verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature for security"""
        if not self.app_secret or not signature:
            return False
        
        try:
            signature = signature.replace('sha256=', '')
            expected_signature = hmac.new(
                self.app_secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {str(e)}")
            return False
    
    def update_daily_limits(self):
        """Update daily message count and reset if needed"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_message_count = 0
            self.last_reset_date = current_date
    
    def can_send_message(self) -> Tuple[bool, str]:
        """Check if message can be sent based on rate limits"""
        self.update_daily_limits()
        
        if self.daily_message_count >= self.daily_limit:
            return False, f"Daily limit of {self.daily_limit} messages exceeded"
        
        return True, "OK"
    
    def send_text_message(self, to: str, message: str, message_id: str = None) -> Dict:
        """Send a text message via WhatsApp Business API"""
        try:
            # Check configuration and rate limits
            is_configured, _ = self.is_configured()
            if not is_configured:
                return {
                    'success': False,
                    'error': 'WhatsApp service not properly configured',
                    'message_id': message_id
                }
            
            can_send, reason = self.can_send_message()
            if not can_send:
                return {
                    'success': False,
                    'error': reason,
                    'message_id': message_id,
                    'rate_limited': True
                }
            
            # Prepare payload
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": message}
            }
            
            # Send request
            response = requests.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            
            if response.status_code == 200:
                wa_message_id = result.get('messages', [{}])[0].get('id')
                
                # Track message in database
                if message_id:
                    self.track_message_in_db(
                        message_id=message_id,
                        whatsapp_id=wa_message_id,
                        to_number=to,
                        message_content=message,
                        message_type='text',
                        status='sent'
                    )
                
                # Update daily count
                self.daily_message_count += 1
                
                logger.info(f"Text message sent successfully to {to}: {wa_message_id}")
                return {
                    'success': True,
                    'whatsapp_message_id': wa_message_id,
                    'message_id': message_id
                }
            else:
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                logger.error(f"Failed to send message to {to}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'message_id': message_id
                }
                
        except Exception as e:
            logger.error(f"Exception sending message to {to}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message_id': message_id
            }
    
    def send_template_message(self, to: str, template_name: str, language: str = "en", 
                            components: List[Dict] = None) -> Dict:
        """Send a template message"""
        try:
            can_send, reason = self.can_send_message()
            if not can_send:
                return {'success': False, 'error': reason, 'rate_limited': True}
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {"code": language}
                }
            }
            
            if components:
                payload["template"]["components"] = components
            
            response = requests.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            
            if response.status_code == 200:
                wa_message_id = result.get('messages', [{}])[0].get('id')
                
                # Track message
                self.track_message_in_db(
                    message_id=f"template_{int(datetime.now().timestamp())}",
                    whatsapp_id=wa_message_id,
                    to_number=to,
                    message_content=f"Template: {template_name}",
                    message_type='template',
                    status='sent',
                    template_name=template_name
                )
                
                self.daily_message_count += 1
                
                logger.info(f"Template message sent to {to}: {wa_message_id}")
                return {
                    'success': True,
                    'whatsapp_message_id': wa_message_id,
                    'template_name': template_name
                }
            else:
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                logger.error(f"Failed to send template to {to}: {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"Exception sending template to {to}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def send_media_message(self, to: str, media_type: str, media_url: str, 
                          caption: str = None, filename: str = None) -> Dict:
        """Send media message (image, document, audio, video)"""
        try:
            can_send, reason = self.can_send_message()
            if not can_send:
                return {'success': False, 'error': reason, 'rate_limited': True}
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": media_type
            }
            
            media_obj = {"link": media_url}
            if caption:
                media_obj["caption"] = caption
            if filename:
                media_obj["filename"] = filename
            
            payload[media_type] = media_obj
            
            response = requests.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            
            if response.status_code == 200:
                wa_message_id = result.get('messages', [{}])[0].get('id')
                
                # Track message
                self.track_message_in_db(
                    message_id=f"media_{int(datetime.now().timestamp())}",
                    whatsapp_id=wa_message_id,
                    to_number=to,
                    message_content=f"[{media_type.upper()}] {caption or filename or media_url}",
                    message_type=media_type,
                    status='sent'
                )
                
                self.daily_message_count += 1
                
                logger.info(f"Media message sent to {to}: {wa_message_id}")
                return {
                    'success': True,
                    'whatsapp_message_id': wa_message_id,
                    'media_type': media_type
                }
            else:
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"Exception sending media to {to}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def send_bulk_messages(self, recipients: List[str], message: str = None, 
                          template_name: str = None, template_components: List[Dict] = None,
                          delay_seconds: float = 0.1, campaign_name: str = None) -> Dict:
        """Send bulk messages with rate limiting"""
        try:
            # Create campaign record
            campaign_id = f"campaign_{int(datetime.now().timestamp())}"
            
            if campaign_name:
                self.create_campaign_record(
                    campaign_id=campaign_id,
                    name=campaign_name,
                    message_content=message,
                    template_name=template_name,
                    total_recipients=len(recipients)
                )
            
            results = {
                'campaign_id': campaign_id,
                'total': len(recipients),
                'sent': 0,
                'failed': 0,
                'errors': []
            }
            
            # Execute bulk sending in background
            def bulk_send_worker():
                sent_count = 0
                failed_count = 0
                
                for i, recipient in enumerate(recipients):
                    try:
                        # Rate limiting delay
                        if delay_seconds > 0:
                            time.sleep(delay_seconds)
                        
                        # Send message
                        if template_name:
                            result = self.send_template_message(
                                to=recipient,
                                template_name=template_name,
                                components=template_components
                            )
                        else:
                            result = self.send_text_message(
                                to=recipient,
                                message=message,
                                message_id=f"{campaign_id}_{i}"
                            )
                        
                        if result.get('success'):
                            sent_count += 1
                        else:
                            failed_count += 1
                            results['errors'].append({
                                'recipient': recipient,
                                'error': result.get('error', 'Unknown error')
                            })
                        
                        # Update progress every 10 messages
                        if (i + 1) % 10 == 0 and campaign_name:
                            self.update_campaign_progress(campaign_id, sent_count, failed_count)
                            logger.info(f"Bulk sending progress: {i + 1}/{len(recipients)}")
                    
                    except Exception as e:
                        failed_count += 1
                        results['errors'].append({
                            'recipient': recipient,
                            'error': str(e)
                        })
                
                # Final update
                results['sent'] = sent_count
                results['failed'] = failed_count
                
                if campaign_name:
                    self.update_campaign_progress(campaign_id, sent_count, failed_count, completed=True)
                
                logger.info(f"Bulk sending complete: {sent_count}/{len(recipients)} sent")
            
            # Start bulk sending in background thread
            self.executor.submit(bulk_send_worker)
            
            return {
                'success': True,
                'message': f'Bulk sending started for {len(recipients)} recipients',
                'campaign_id': campaign_id,
                'estimated_duration': f'{len(recipients) * delay_seconds:.1f} seconds'
            }
            
        except Exception as e:
            logger.error(f"Error in bulk messaging: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_message_templates(self) -> List[Dict]:
        """Get approved message templates"""
        try:
            # Check cache
            if (self.templates_cache and self.templates_last_updated and 
                datetime.now() - self.templates_last_updated < timedelta(minutes=30)):
                return self.templates_cache.get('templates', [])
            
            if not self.business_account_id:
                logger.warning("Business Account ID not configured for templates")
                return []
            
            url = f"https://graph.facebook.com/v18.0/{self.business_account_id}/message_templates"
            params = {'limit': 100}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                templates = result.get('data', [])
                
                # Cache templates
                self.templates_cache = {'templates': templates}
                self.templates_last_updated = datetime.now()
                
                logger.info(f"Retrieved {len(templates)} message templates")
                return templates
            else:
                logger.error(f"Failed to get templates: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Exception getting templates: {str(e)}")
            return []
    
    def process_webhook(self, data: Dict) -> Dict:
        """Process incoming webhook data"""
        try:
            entry = data.get('entry', [{}])[0]
            changes = entry.get('changes', [{}])[0]
            value = changes.get('value', {})
            
            # Process status updates
            if 'statuses' in value:
                for status in value['statuses']:
                    message_id = status.get('id')
                    status_type = status.get('status')
                    timestamp = status.get('timestamp')
                    
                    # Update message status in database
                    if timestamp:
                        timestamp_dt = datetime.fromtimestamp(int(timestamp))
                        self.update_message_status_in_db(message_id, status_type, timestamp_dt)
                    
                    logger.info(f"Message {message_id} status: {status_type}")
            
            # Process incoming messages
            if 'messages' in value:
                for message in value['messages']:
                    msg_id = message.get('id')
                    from_number = message.get('from')
                    msg_type = message.get('type')
                    timestamp = message.get('timestamp')
                    
                    # Extract message content
                    content = self.extract_message_content(message, msg_type)
                    
                    incoming_message = {
                        'id': msg_id,
                        'from': from_number,
                        'type': msg_type,
                        'content': content,
                        'timestamp': datetime.fromtimestamp(int(timestamp)) if timestamp else datetime.now(),
                        'raw_data': message
                    }
                    
                    return {
                        'message_processed': True,
                        'incoming_message': incoming_message
                    }
            
            return {'status': 'processed'}
            
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return {'error': str(e)}
    
    def extract_message_content(self, message: Dict, msg_type: str) -> str:
        """Extract content from different message types"""
        if msg_type == 'text':
            return message.get('text', {}).get('body', '')
        elif msg_type == 'image':
            return message.get('image', {}).get('caption', '[Image]')
        elif msg_type == 'document':
            filename = message.get('document', {}).get('filename', 'Unknown')
            return f"[Document: {filename}]"
        elif msg_type == 'audio':
            return '[Audio]'
        elif msg_type == 'video':
            return '[Video]'
        elif msg_type == 'location':
            return '[Location]'
        elif msg_type == 'interactive':
            return '[Interactive Response]'
        else:
            return f'[{msg_type.upper()}]'
    
    def track_message_in_db(self, message_id: str, whatsapp_id: str, to_number: str,
                          message_content: str, message_type: str, status: str,
                          template_name: str = None, campaign_id: str = None):
        """Track message in database"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO whatsapp_messages 
                (message_id, whatsapp_id, to_number, message_content, message_type,
                 status, sent_at, template_name, campaign_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message_id, whatsapp_id, to_number, message_content[:500],
                message_type, status, datetime.now(), template_name, campaign_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error tracking message in database: {str(e)}")
    
    def update_message_status_in_db(self, whatsapp_id: str, status: str, timestamp: datetime = None):
        """Update message status in database"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            timestamp = timestamp or datetime.now()
            
            update_fields = {'status': status}
            if status == 'delivered':
                update_fields['delivered_at'] = timestamp
            elif status == 'read':
                update_fields['read_at'] = timestamp
            elif status == 'failed':
                update_fields['failed_at'] = timestamp
            
            # Build dynamic update query
            set_clause = ', '.join([f"{key} = ?" for key in update_fields.keys()])
            values = list(update_fields.values()) + [whatsapp_id]
            
            cursor.execute(f'''
                UPDATE whatsapp_messages 
                SET {set_clause}
                WHERE whatsapp_id = ?
            ''', values)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating message status: {str(e)}")
    
    def create_campaign_record(self, campaign_id: str, name: str, message_content: str,
                             template_name: str, total_recipients: int):
        """Create campaign record in database"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO whatsapp_campaigns 
                (campaign_id, name, message_content, template_name, total_recipients, 
                 status, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                campaign_id, name, message_content, template_name, 
                total_recipients, 'started', datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating campaign record: {str(e)}")
    
    def update_campaign_progress(self, campaign_id: str, sent_count: int, 
                               failed_count: int, completed: bool = False):
        """Update campaign progress"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            if completed:
                cursor.execute('''
                    UPDATE whatsapp_campaigns 
                    SET sent_count = ?, failed_count = ?, status = ?, completed_at = ?
                    WHERE campaign_id = ?
                ''', (sent_count, failed_count, 'completed', datetime.now(), campaign_id))
            else:
                cursor.execute('''
                    UPDATE whatsapp_campaigns 
                    SET sent_count = ?, failed_count = ?
                    WHERE campaign_id = ?
                ''', (sent_count, failed_count, campaign_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating campaign progress: {str(e)}")
    
    def get_delivery_stats(self) -> Dict:
        """Get delivery statistics"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'sent' THEN 1 END) as sent,
                    COUNT(CASE WHEN status = 'delivered' THEN 1 END) as delivered,
                    COUNT(CASE WHEN status = 'read' THEN 1 END) as read,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                FROM whatsapp_messages
                WHERE sent_at >= date('now', '-7 days')
            ''')
            
            stats = cursor.fetchone()
            conn.close()
            
            return {
                'total_tracked': stats[0] or 0,
                'sent': stats[1] or 0,
                'delivered': stats[2] or 0,
                'read': stats[3] or 0,
                'failed': stats[4] or 0,
                'daily_limit': self.daily_limit,
                'daily_used': self.daily_message_count,
                'daily_remaining': self.daily_limit - self.daily_message_count
            }
            
        except Exception as e:
            logger.error(f"Error getting delivery stats: {str(e)}")
            return {
                'total_tracked': 0, 'sent': 0, 'delivered': 0, 
                'read': 0, 'failed': 0, 'error': str(e)
            }
    
    def get_campaign_stats(self, campaign_id: str = None) -> Dict:
        """Get campaign statistics"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            if campaign_id:
                cursor.execute('''
                    SELECT * FROM whatsapp_campaigns WHERE campaign_id = ?
                ''', (campaign_id,))
                campaign = cursor.fetchone()
                
                if campaign:
                    conn.close()
                    return {
                        'campaign_id': campaign[1],
                        'name': campaign[2],
                        'total_recipients': campaign[5],
                        'sent_count': campaign[6],
                        'delivered_count': campaign[7],
                        'read_count': campaign[8],
                        'failed_count': campaign[9],
                        'status': campaign[10],
                        'started_at': campaign[11],
                        'completed_at': campaign[12]
                    }
            else:
                cursor.execute('''
                    SELECT * FROM whatsapp_campaigns 
                    ORDER BY created_at DESC LIMIT 20
                ''')
                campaigns = cursor.fetchall()
                conn.close()
                
                return {
                    'campaigns': [
                        {
                            'campaign_id': c[1],
                            'name': c[2],
                            'total_recipients': c[5],
                            'sent_count': c[6],
                            'status': c[10],
                            'started_at': c[11],
                            'completed_at': c[12]
                        } for c in campaigns
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting campaign stats: {str(e)}")
            return {'error': str(e)}
    
    def get_health_status(self) -> Dict:
        """Get health status of WhatsApp service"""
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'configured': False,
                'database': 'unknown',
                'rate_limits': 'unknown',
                'api_connection': 'unknown'
            }
            
            # Check configuration
            is_configured, config_status = self.is_configured()
            health['configured'] = is_configured
            health['configuration_details'] = config_status
            
            # Check database
            try:
                conn = sqlite3.connect('whatsapp_messages.db')
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM whatsapp_messages LIMIT 1')
                conn.close()
                health['database'] = 'healthy'
            except Exception as e:
                health['database'] = f'error: {str(e)}'
            
            # Check rate limits
            can_send, reason = self.can_send_message()
            health['rate_limits'] = {
                'can_send': can_send,
                'reason': reason,
                'daily_used': self.daily_message_count,
                'daily_limit': self.daily_limit
            }
            
            # Overall health
            health['healthy'] = (
                health['configured'] and
                health['database'] == 'healthy' and
                can_send
            )
            
            return health
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict:
        """Clean up old message and campaign data"""
        try:
            conn = sqlite3.connect('whatsapp_messages.db')
            cursor = conn.cursor()
            
            # Delete old messages
            cursor.execute('''
                DELETE FROM whatsapp_messages 
                WHERE sent_at < datetime('now', '-{} days')
            '''.format(days_to_keep))
            deleted_messages = cursor.rowcount
            
            # Delete old campaigns
            cursor.execute('''
                DELETE FROM whatsapp_campaigns 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days_to_keep))
            deleted_campaigns = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleanup complete: {deleted_messages} messages, {deleted_campaigns} campaigns deleted")
            
            return {
                'success': True,
                'deleted_messages': deleted_messages,
                'deleted_campaigns': deleted_campaigns,
                'days_kept': days_to_keep
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Singleton instance
whatsapp_service = None

def get_whatsapp_service(config: Dict = None) -> WhatsAppService:
    """Get or create WhatsApp service instance"""
    global whatsapp_service
    if whatsapp_service is None:
        whatsapp_service = WhatsAppService(config)
    return whatsapp_service

def init_whatsapp_service(config: Dict = None) -> WhatsAppService:
    """Initialize WhatsApp service with configuration"""
    global whatsapp_service
    whatsapp_service = WhatsAppService(config)
    return whatsapp_service

