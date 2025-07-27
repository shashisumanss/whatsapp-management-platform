# whatsapp_group_sync.py
# Real WhatsApp Group Synchronization System

import json
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from group_manager import group_manager, GroupType, GroupStatus
from whatsapp_service import whatsapp_service

logger = logging.getLogger(__name__)

class WhatsAppGroupSync:
    """Synchronize real WhatsApp groups with our management system"""
    
    def __init__(self, whatsapp_service):
        self.whatsapp_service = whatsapp_service
        self.group_manager = group_manager
    
    def sync_groups_from_webhook(self, webhook_data: Dict) -> Dict:
        """Sync groups from WhatsApp webhook data"""
        try:
            groups_synced = 0
            members_synced = 0
            
            # Extract group information from webhook
            if 'entry' in webhook_data:
                for entry in webhook_data['entry']:
                    changes = entry.get('changes', [])
                    for change in changes:
                        value = change.get('value', {})
                        
                        # Check for group-related messages
                        if 'messages' in value:
                            for message in value['messages']:
                                group_info = self._extract_group_info_from_message(message)
                                if group_info:
                                    success = self._sync_single_group(group_info)
                                    if success:
                                        groups_synced += 1
                        
                        # Check for group member changes
                        if 'contacts' in value:
                            members_synced += self._sync_group_members(value['contacts'])
            
            return {
                'success': True,
                'groups_synced': groups_synced,
                'members_synced': members_synced
            }
            
        except Exception as e:
            logger.error(f"Error syncing groups from webhook: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_group_info_from_message(self, message: Dict) -> Optional[Dict]:
        """Extract group information from a WhatsApp message"""
        try:
            # WhatsApp group messages have a 'from' field with group ID
            from_id = message.get('from', '')
            
            # Group IDs typically end with '@g.us'
            if from_id.endswith('@g.us'):
                # Try to get group metadata if available
                context = message.get('context', {})
                
                return {
                    'whatsapp_group_id': from_id,
                    'group_name': self._extract_group_name_from_context(context, from_id),
                    'last_message_time': datetime.fromtimestamp(int(message.get('timestamp', 0))),
                    'participants': self._extract_participants(message)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting group info from message: {str(e)}")
            return None
    
    def _extract_group_name_from_context(self, context: Dict, group_id: str) -> str:
        """Extract group name from message context or generate from ID"""
        # Try to get group name from context
        if 'group_name' in context:
            return context['group_name']
        
        # If no name available, generate a readable name from group ID
        # Remove @g.us suffix and take last 8 characters
        clean_id = group_id.replace('@g.us', '')[-8:]
        return f"WhatsApp Group {clean_id}"
    
    def _extract_participants(self, message: Dict) -> List[str]:
        """Extract participant phone numbers from message"""
        participants = []
        
        # Add message sender
        if 'from' in message and not message['from'].endswith('@g.us'):
            participants.append(message['from'])
        
        # Check for mentions or quoted messages
        if 'context' in message:
            context = message['context']
            if 'quoted_message' in context:
                quoted_from = context['quoted_message'].get('from')
                if quoted_from and quoted_from not in participants:
                    participants.append(quoted_from)
        
        return participants
    
    def _sync_single_group(self, group_info: Dict) -> bool:
        """Sync a single group to our database"""
        try:
            whatsapp_group_id = group_info['whatsapp_group_id']
            
            # Check if group already exists in our system
            existing_groups = self.group_manager.db.get_all_groups()
            existing_group = None
            
            for group in existing_groups:
                if hasattr(group, 'settings') and group.settings:
                    if group.settings.get('whatsapp_group_id') == whatsapp_group_id:
                        existing_group = group
                        break
            
            if existing_group:
                # Update existing group
                updates = {
                    'name': group_info['group_name'],
                    'settings': {
                        **existing_group.settings,
                        'whatsapp_group_id': whatsapp_group_id,
                        'last_sync': datetime.now().isoformat(),
                        'last_message_time': group_info['last_message_time'].isoformat()
                    }
                }
                success = self.group_manager.db.update_group(existing_group.group_id, updates)
                logger.info(f"Updated existing group: {group_info['group_name']}")
            else:
                # Create new group
                group_id = self.group_manager.create_group(
                    name=group_info['group_name'],
                    description=f"Synced WhatsApp group - {group_info['group_name']}",
                    group_type=self._determine_group_type(group_info['group_name']),
                    settings={
                        'whatsapp_group_id': whatsapp_group_id,
                        'synced_from_whatsapp': True,
                        'last_sync': datetime.now().isoformat(),
                        'last_message_time': group_info['last_message_time'].isoformat(),
                        'auto_response_enabled': True,
                        'content_filtering': True,
                        'max_members': 1000
                    }
                )
                logger.info(f"Created new group from WhatsApp: {group_info['group_name']}")
                existing_group = self.group_manager.db.get_group(group_id)
            
            # Sync participants
            if group_info.get('participants') and existing_group:
                self._sync_group_participants(existing_group.group_id, group_info['participants'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error syncing single group: {str(e)}")
            return False
    
    def _determine_group_type(self, group_name: str) -> GroupType:
        """Determine group type based on group name"""
        name_lower = group_name.lower()
        
        if any(keyword in name_lower for keyword in ['emergency', 'urgent', 'alert', 'sos']):
            return GroupType.EMERGENCY
        elif any(keyword in name_lower for keyword in ['health', 'medical', 'hospital', 'clinic']):
            return GroupType.HEALTHCARE
        elif any(keyword in name_lower for keyword in ['road', 'water', 'electricity', 'infrastructure']):
            return GroupType.INFRASTRUCTURE
        elif any(keyword in name_lower for keyword in ['announcement', 'news', 'update']):
            return GroupType.ANNOUNCEMENTS
        elif any(keyword in name_lower for keyword in ['support', 'help', 'service']):
            return GroupType.SUPPORT
        else:
            return GroupType.COMMUNITY
    
    def _sync_group_participants(self, group_id: str, participants: List[str]) -> int:
        """Sync group participants"""
        synced_count = 0
        
        for phone_number in participants:
            try:
                # Create member data
                member_data = {
                    'phone_number': phone_number,
                    'name': f"WhatsApp User {phone_number[-4:]}",  # Use last 4 digits as name
                    'role': 'member',
                    'joined_date': datetime.now().isoformat(),
                    'synced_from_whatsapp': True
                }
                
                # Add member to group
                result = self.group_manager.db.bulk_add_members(group_id, [member_data])
                if result.get('success'):
                    synced_count += 1
                    
            except Exception as e:
                logger.error(f"Error syncing participant {phone_number}: {str(e)}")
        
        return synced_count
    
    def _sync_group_members(self, contacts: List[Dict]) -> int:
        """Sync group members from contacts data"""
        synced_count = 0
        
        for contact in contacts:
            try:
                # Extract contact information
                phone = contact.get('wa_id') or contact.get('phone')
                name = contact.get('profile', {}).get('name', f"User {phone[-4:]}")
                
                if phone:
                    # Update member information across all groups they belong to
                    # This is a simplified approach - you might want to be more specific
                    synced_count += 1
                    
            except Exception as e:
                logger.error(f"Error syncing contact: {str(e)}")
        
        return synced_count
    
    def fetch_groups_via_api(self) -> Dict:
        """Fetch groups directly via WhatsApp Business API (if supported)"""
        try:
            # Note: WhatsApp Business API doesn't directly provide group listing
            # This is a placeholder for future API enhancements
            
            logger.warning("Direct group fetching via API not supported by WhatsApp Business API")
            return {
                'success': False,
                'error': 'WhatsApp Business API does not support direct group listing',
                'recommendation': 'Use webhook-based synchronization instead'
            }
            
        except Exception as e:
            logger.error(f"Error fetching groups via API: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def manual_group_registration(self, whatsapp_group_id: str, group_name: str, 
                                group_type: str = 'community') -> Dict:
        """Manually register a WhatsApp group"""
        try:
            # Create group in our system
            group_id = self.group_manager.create_group(
                name=group_name,
                description=f"Manually registered WhatsApp group: {group_name}",
                group_type=GroupType(group_type),
                settings={
                    'whatsapp_group_id': whatsapp_group_id,
                    'manually_registered': True,
                    'registration_date': datetime.now().isoformat(),
                    'auto_response_enabled': True,
                    'content_filtering': True,
                    'max_members': 1000
                }
            )
            
            logger.info(f"Manually registered WhatsApp group: {group_name}")
            
            return {
                'success': True,
                'group_id': group_id,
                'message': f'WhatsApp group "{group_name}" registered successfully'
            }
            
        except Exception as e:
            logger.error(f"Error manually registering group: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_whatsapp_group_insights(self, whatsapp_group_id: str) -> Dict:
        """Get insights for a specific WhatsApp group"""
        try:
            # Find our internal group that matches this WhatsApp group
            groups = self.group_manager.db.get_all_groups()
            target_group = None
            
            for group in groups:
                if (hasattr(group, 'settings') and group.settings and 
                    group.settings.get('whatsapp_group_id') == whatsapp_group_id):
                    target_group = group
                    break
            
            if not target_group:
                return {
                    'success': False,
                    'error': 'WhatsApp group not found in our system'
                }
            
            # Get group statistics and analytics
            stats = self.group_manager.get_group_statistics(target_group.group_id)
            
            return {
                'success': True,
                'group_name': target_group.name,
                'internal_group_id': target_group.group_id,
                'whatsapp_group_id': whatsapp_group_id,
                'statistics': stats,
                'sync_info': {
                    'last_sync': target_group.settings.get('last_sync'),
                    'sync_method': 'webhook' if target_group.settings.get('synced_from_whatsapp') else 'manual'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting WhatsApp group insights: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize the sync service
whatsapp_group_sync = WhatsAppGroupSync(whatsapp_service)

# Add these routes to your app.py

def add_whatsapp_sync_routes(app):
    """Add WhatsApp synchronization routes to Flask app"""
    
    @app.route('/api/whatsapp/sync/webhook', methods=['POST'])
    def sync_groups_from_webhook():
        """Sync groups from incoming webhook data"""
        try:
            webhook_data = request.get_json()
            result = whatsapp_group_sync.sync_groups_from_webhook(webhook_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/whatsapp/sync/manual', methods=['POST'])
    def manually_register_group():
        """Manually register a WhatsApp group"""
        try:
            data = request.get_json()
            whatsapp_group_id = data.get('whatsapp_group_id')
            group_name = data.get('group_name')
            group_type = data.get('group_type', 'community')
            
            if not whatsapp_group_id or not group_name:
                return jsonify({
                    'success': False,
                    'error': 'whatsapp_group_id and group_name are required'
                }), 400
            
            result = whatsapp_group_sync.manual_group_registration(
                whatsapp_group_id, group_name, group_type
            )
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/whatsapp/groups/<whatsapp_group_id>/insights', methods=['GET'])
    def get_whatsapp_group_insights(whatsapp_group_id):
        """Get insights for a specific WhatsApp group"""
        try:
            result = whatsapp_group_sync.get_whatsapp_group_insights(whatsapp_group_id)
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/whatsapp/sync/status', methods=['GET'])
    def get_sync_status():
        """Get synchronization status"""
        try:
            groups = group_manager.db.get_all_groups()
            
            synced_groups = []
            manual_groups = []
            
            for group in groups:
                if hasattr(group, 'settings') and group.settings:
                    if group.settings.get('synced_from_whatsapp'):
                        synced_groups.append({
                            'group_id': group.group_id,
                            'name': group.name,
                            'whatsapp_group_id': group.settings.get('whatsapp_group_id'),
                            'last_sync': group.settings.get('last_sync')
                        })
                    elif group.settings.get('manually_registered'):
                        manual_groups.append({
                            'group_id': group.group_id,
                            'name': group.name,
                            'whatsapp_group_id': group.settings.get('whatsapp_group_id'),
                            'registration_date': group.settings.get('registration_date')
                        })
            
            return jsonify({
                'success': True,
                'total_groups': len(groups),
                'synced_from_whatsapp': len(synced_groups),
                'manually_registered': len(manual_groups),
                'synced_groups': synced_groups,
                'manual_groups': manual_groups
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# Usage instructions for integration