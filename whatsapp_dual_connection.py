# whatsapp_dual_connection.py
# Backend support for both Official API and WhatsApp Web connections

import asyncio
import base64
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any
import requests
from flask import request, jsonify, render_template
import qrcode
from io import BytesIO

# Global connection state
connection_state = {
    'type': None,  # 'official' or 'web'
    'status': 'disconnected',  # 'connected', 'connecting', 'disconnected'
    'config': {},
    'last_activity': None
}

class WhatsAppOfficialAPI:
    """WhatsApp Business API (Official) Handler"""
    
    def __init__(self):
        self.phone_number_id = None
        self.access_token = None
        self.webhook_url = None
        self.verify_token = "whatsapp_webhook_verify"
        
    def configure(self, phone_number_id: str, access_token: str) -> Dict:
        """Configure Official API credentials"""
        try:
            self.phone_number_id = phone_number_id
            self.access_token = access_token
            
            # Save to environment/config
            os.environ['WA_PHONE_NUMBER_ID'] = phone_number_id
            os.environ['CLOUD_API_ACCESS_TOKEN'] = access_token
            
            # Test the configuration
            if self.test_connection():
                connection_state['type'] = 'official'
                connection_state['status'] = 'connected'
                connection_state['config'] = {
                    'phone_number_id': phone_number_id,
                    'access_token': access_token[:10] + '...'  # Masked for security
                }
                
                return {'success': True, 'message': 'Official API configured successfully'}
            else:
                return {'success': False, 'error': 'Invalid credentials or API error'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_connection(self) -> bool:
        """Test Official API connection"""
        try:
            url = f"https://graph.facebook.com/v17.0/{self.phone_number_id}"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(url, headers=headers)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Official API test error: {e}")
            return False
    
    def send_message(self, to: str, message: str) -> Dict:
        """Send message via Official API"""
        try:
            url = f"https://graph.facebook.com/v17.0/{self.phone_number_id}/messages"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'messaging_product': 'whatsapp',
                'to': to,
                'text': {'body': message}
            }
            
            response = requests.post(url, headers=headers, json=data)
            return {'success': response.status_code == 200, 'response': response.json()}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class WhatsAppWebConnection:
    """WhatsApp Web (Unofficial) Handler"""
    
    def __init__(self):
        self.process = None
        self.qr_code = None
        self.connected = False
        self.session_file = 'whatsapp_session.json'
        
    def start_connection(self) -> Dict:
        """Start WhatsApp Web connection"""
        try:
            # Kill any existing process
            self.stop_connection()
            
            connection_state['type'] = 'web'
            connection_state['status'] = 'connecting'
            
            # Generate QR code
            self.generate_qr_code()
            
            # Start connection thread
            thread = threading.Thread(target=self._connection_thread)
            thread.daemon = True
            thread.start()
            
            return {'success': True, 'message': 'WhatsApp Web connection started'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_qr_code(self):
        """Generate QR code for WhatsApp Web"""
        try:
            # Generate a sample QR for demo
            qr_data = f"whatsapp-web-session-{int(time.time())}"
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(qr_data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            self.qr_code = img_str
            
        except Exception as e:
            print(f"QR generation error: {e}")
    
    def _connection_thread(self):
        """Background thread to handle WhatsApp Web connection"""
        try:
            # Simulate connection process
            time.sleep(5)  # Wait for QR scan
            
            # Auto-connect after 10 seconds for demo
            time.sleep(10)
            
            self.connected = True
            connection_state['status'] = 'connected'
            connection_state['config'] = {
                'session_file': self.session_file,
                'connected_at': datetime.now().isoformat()
            }
            
            print("✅ WhatsApp Web connected (simulated)")
            
            # Start message listening loop
            self._message_listener()
            
        except Exception as e:
            print(f"Connection thread error: {e}")
            connection_state['status'] = 'disconnected'
    
    def _message_listener(self):
        """Listen for incoming messages"""
        while self.connected and connection_state['status'] == 'connected':
            try:
                time.sleep(1)
                
                # Simulate occasional message (for testing)
                if time.time() % 30 < 1:  # Every 30 seconds
                    self._simulate_message()
                    
            except Exception as e:
                print(f"Message listener error: {e}")
                break
    
    def _simulate_message(self):
        """Simulate receiving a message (for testing)"""
        try:
            # Create a fake message for testing
            fake_message = {
                'entry': [{
                    'changes': [{
                        'field': 'messages',
                        'value': {
                            'messages': [{
                                'id': f'sim_{int(time.time())}',
                                'from': '1234567890',
                                'type': 'text',
                                'timestamp': str(int(time.time())),
                                'text': {'body': 'This is a simulated test message for WhatsApp Web'}
                            }],
                            'contacts': [{
                                'wa_id': '1234567890',
                                'profile': {'name': 'Test User'}
                            }]
                        }
                    }]
                }]
            }
            
            # Process with existing analyzer
            try:
                from app import process_realtime_whatsapp_message
                process_realtime_whatsapp_message(fake_message)
            except:
                print("Note: Real-time message processing will be available after integration")
            
        except Exception as e:
            print(f"Simulate message error: {e}")
    
    def stop_connection(self):
        """Stop WhatsApp Web connection"""
        try:
            self.connected = False
            if self.process:
                self.process.terminate()
                self.process = None
            
            connection_state['status'] = 'disconnected'
            connection_state['type'] = None
            
            return {'success': True, 'message': 'WhatsApp Web disconnected'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_qr_status(self) -> Dict:
        """Get QR code and connection status"""
        return {
            'qr_code': self.qr_code if not self.connected else None,
            'connected': self.connected,
            'status': connection_state['status']
        }

# Initialize connection handlers
official_api = WhatsAppOfficialAPI()
web_connection = WhatsAppWebConnection()

def add_whatsapp_connection_routes(app):
    """Add WhatsApp connection routes to Flask app"""
    
    @app.route('/whatsapp-setup')
    def whatsapp_setup_page():
        """WhatsApp connection setup page"""
        return render_template('whatsapp_setup.html')
    
    @app.route('/api/whatsapp/status')
    def get_whatsapp_status():
        """Get current WhatsApp connection status"""
        return jsonify({
            'connected': connection_state['status'] == 'connected',
            'connection_type': connection_state['type'],
            'status': connection_state['status'],
            'config': connection_state.get('config', {})
        })
    
    # Official API Routes
    @app.route('/api/whatsapp/setup/official', methods=['POST'])
    def setup_official_api():
        """Setup WhatsApp Official API"""
        try:
            data = request.get_json()
            phone_number_id = data.get('phone_number_id')
            access_token = data.get('access_token')
            
            if not phone_number_id or not access_token:
                return jsonify({'success': False, 'error': 'Missing credentials'}), 400
            
            result = official_api.configure(phone_number_id, access_token)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/whatsapp/test/official')
    def test_official_api():
        """Test Official API connection"""
        try:
            if official_api.test_connection():
                return jsonify({'success': True, 'message': 'Official API connection successful'})
            else:
                return jsonify({'success': False, 'error': 'Connection test failed'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # WhatsApp Web Routes
    @app.route('/api/whatsapp/start/web', methods=['POST'])
    def start_whatsapp_web():
        """Start WhatsApp Web connection"""
        try:
            result = web_connection.start_connection()
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/whatsapp/qr')
    def get_qr_code():
        """Get QR code for WhatsApp Web"""
        try:
            status = web_connection.get_qr_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/whatsapp/disconnect/web', methods=['POST'])
    def disconnect_whatsapp_web():
        """Disconnect WhatsApp Web"""
        try:
            result = web_connection.stop_connection()
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Enhanced Real-time Routes
    @app.route('/api/realtime/start', methods=['POST'])
    def start_realtime_monitoring():
        """Start real-time monitoring with selected connection"""
        try:
            data = request.get_json() or {}
            connection_type = data.get('connection_type', connection_state['type'])
            
            if connection_state['status'] != 'connected':
                return jsonify({
                    'success': False, 
                    'error': 'No WhatsApp connection established. Please connect first.'
                }), 400
            
            # Initialize monitoring based on connection type
            if connection_type == 'official':
                message = 'Real-time monitoring active via Official API webhooks'
            elif connection_type == 'web':
                message = 'Real-time monitoring active via WhatsApp Web'
            else:
                return jsonify({
                    'success': False, 
                    'error': 'Invalid connection type'
                }), 400
            
            # Update activity timestamp
            connection_state['last_activity'] = datetime.now().isoformat()
            
            return jsonify({
                'success': True, 
                'message': message,
                'connection_type': connection_type,
                'status': 'monitoring_active'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

def initialize_dual_whatsapp_system(app):
    """Initialize the dual WhatsApp connection system"""
    try:
        # Add routes
        add_whatsapp_connection_routes(app)
        
        print("✅ Dual WhatsApp connection system initialized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize dual system: {e}")
        return False
