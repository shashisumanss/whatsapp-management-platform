# app.py Integration
# Add these imports and routes to your existing app.py

# ===== ADD THESE IMPORTS AT THE TOP =====
from whatsapp_service import get_whatsapp_service, init_whatsapp_service
import os

# ===== ADD AFTER YOUR EXISTING IMPORTS =====

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
@app.route('/api/dashboard_stats')
def dashboard_stats():
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

# Enhanced Bulk Message Endpoint (modify your existing one)
@app.route('/api/bulk_message', methods=['POST'])
def send_bulk_message():
    """Enhanced bulk message with WhatsApp API integration"""
    global bulk_message_history
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        groups = data.get('groups', [])
        recipients = data.get('recipients', [])  # WhatsApp phone numbers
        use_whatsapp = data.get('use_whatsapp', False)
        
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
        
        if use_whatsapp and recipients:
            # Send via WhatsApp Business API
            campaign_name = f"Bulk Message {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            result = whatsapp_service.send_bulk_messages(
                recipients=recipients,
                message=message,
                delay_seconds=0.1,
                campaign_name=campaign_name
            )
            
            if result.get('success'):
                # Add to bulk message history
                bulk_message = {
                    'id': result['campaign_id'],
                    'message': message[:100] + '...' if len(message) > 100 else message,
                    'full_message': message,
                    'target_recipients': recipients,
                    'recipient_count': len(recipients),
                    'status': 'Started',
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'sent_count': 0,  # Will be updated by background process
                    'failed_count': 0,
                    'platform': 'WhatsApp Business API'
                }
                
                bulk_message_history.append(bulk_message)
                
                return jsonify({
                    'success': True,
                    'message_id': result['campaign_id'],
                    'sent_to': len(recipients),
                    'message': f'WhatsApp bulk campaign started for {len(recipients)} recipients!',
                    'platform': 'WhatsApp Business API',
                    'estimated_duration': result.get('estimated_duration')
                })
            else:
                return jsonify(result), 500
        
        else:
            # Original simulated sending for groups
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
                'message': f'Message sent to {len(groups)} groups successfully!',
                'platform': 'Simulated'
            })
        
    except Exception as e:
        logger.error(f"Error in bulk messaging: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

# ===== ADD TO YOUR MAIN SECTION =====
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