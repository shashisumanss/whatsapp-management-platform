# group_manager.py
# Comprehensive Group Management System

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import uuid
from collections import defaultdict, Counter
import statistics
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroupType(Enum):
    COMMUNITY = "community"
    EMERGENCY = "emergency"
    ANNOUNCEMENTS = "announcements"
    SUPPORT = "support"
    FEEDBACK = "feedback"
    OFFICIAL = "official"
    HEALTHCARE = "healthcare"
    INFRASTRUCTURE = "infrastructure"

class MemberRole(Enum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"
    RESTRICTED = "restricted"

class GroupStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    SUSPENDED = "suspended"

@dataclass
class GroupMember:
    phone_number: str
    name: str
    role: MemberRole
    joined_date: datetime
    last_active: Optional[datetime] = None
    message_count: int = 0
    is_active: bool = True
    tags: List[str] = None
    constituency: Optional[str] = None
    booth_number: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class WhatsAppGroup:
    group_id: str
    name: str
    description: str
    group_type: GroupType
    created_date: datetime
    status: GroupStatus = GroupStatus.ACTIVE
    member_count: int = 0
    invite_link: Optional[str] = None
    tags: List[str] = None
    settings: Dict = None
    constituency: Optional[str] = None
    region: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.settings is None:
            self.settings = {
                'auto_response_enabled': True,
                'content_filtering': True,
                'max_members': 1000,
                'member_approval_required': False,
                'message_throttling': True,
                'emergency_escalation': True
            }

class GroupDatabase:
    """Enhanced database manager for groups and members"""
    
    def __init__(self, db_path='whatsapp_groups.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Groups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                group_type TEXT NOT NULL,
                created_date TIMESTAMP,
                status TEXT DEFAULT 'active',
                member_count INTEGER DEFAULT 0,
                invite_link TEXT,
                tags TEXT,
                settings TEXT,
                constituency TEXT,
                region TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Members table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT,
                phone_number TEXT,
                name TEXT,
                role TEXT DEFAULT 'member',
                joined_date TIMESTAMP,
                last_active TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                tags TEXT,
                constituency TEXT,
                booth_number TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES groups (group_id),
                UNIQUE(group_id, phone_number)
            )
        ''')
        
        # Group messages table for analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT,
                sender_phone TEXT,
                message_content TEXT,
                message_type TEXT DEFAULT 'text',
                timestamp TIMESTAMP,
                sentiment TEXT,
                is_flagged BOOLEAN DEFAULT 0,
                priority TEXT DEFAULT 'normal',
                language TEXT,
                response_time_minutes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES groups (group_id)
            )
        ''')
        
        # Group analytics summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT,
                date DATE,
                total_messages INTEGER DEFAULT 0,
                unique_senders INTEGER DEFAULT 0,
                flagged_messages INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0,
                sentiment_positive INTEGER DEFAULT 0,
                sentiment_negative INTEGER DEFAULT 0,
                sentiment_neutral INTEGER DEFAULT 0,
                member_growth INTEGER DEFAULT 0,
                activity_score REAL DEFAULT 0,
                health_score REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES groups (group_id),
                UNIQUE(group_id, date)
            )
        ''')
        
        # Member activity tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS member_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT,
                phone_number TEXT,
                activity_date DATE,
                messages_sent INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0,
                sentiment_score REAL DEFAULT 0,
                engagement_score REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES groups (group_id),
                UNIQUE(group_id, phone_number, activity_date)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Group database initialized successfully")
    
    def add_group(self, group: WhatsAppGroup) -> bool:
        """Add a new group"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO groups 
                (group_id, name, description, group_type, created_date, status, 
                 member_count, invite_link, tags, settings, constituency, region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                group.group_id, group.name, group.description, group.group_type.value,
                group.created_date, group.status.value, group.member_count,
                group.invite_link, json.dumps(group.tags), json.dumps(group.settings),
                group.constituency, group.region
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Group {group.name} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding group: {str(e)}")
            return False
    
    def get_group(self, group_id: str) -> Optional[WhatsAppGroup]:
        """Get group by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM groups WHERE group_id = ?', (group_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return WhatsAppGroup(
                    group_id=row[0],
                    name=row[1],
                    description=row[2],
                    group_type=GroupType(row[3]),
                    created_date=datetime.fromisoformat(row[4]),
                    status=GroupStatus(row[5]),
                    member_count=row[6],
                    invite_link=row[7],
                    tags=json.loads(row[8]) if row[8] else [],
                    settings=json.loads(row[9]) if row[9] else {},
                    constituency=row[10],
                    region=row[11]
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting group: {str(e)}")
            return None
    
    def get_all_groups(self) -> List[WhatsAppGroup]:
        """Get all groups"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM groups ORDER BY name')
            rows = cursor.fetchall()
            conn.close()
            
            groups = []
            for row in rows:
                group = WhatsAppGroup(
                    group_id=row[0],
                    name=row[1],
                    description=row[2],
                    group_type=GroupType(row[3]),
                    created_date=datetime.fromisoformat(row[4]),
                    status=GroupStatus(row[5]),
                    member_count=row[6],
                    invite_link=row[7],
                    tags=json.loads(row[8]) if row[8] else [],
                    settings=json.loads(row[9]) if row[9] else {},
                    constituency=row[10],
                    region=row[11]
                )
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error getting all groups: {str(e)}")
            return []
    
    def add_member_to_group(self, group_id: str, member: GroupMember) -> bool:
        """Add member to group"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Add member
            cursor.execute('''
                INSERT OR REPLACE INTO group_members 
                (group_id, phone_number, name, role, joined_date, last_active,
                 message_count, is_active, tags, constituency, booth_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                group_id, member.phone_number, member.name, member.role.value,
                member.joined_date, member.last_active, member.message_count,
                member.is_active, json.dumps(member.tags), member.constituency,
                member.booth_number
            ))
            
            # Update group member count
            cursor.execute('''
                UPDATE groups 
                SET member_count = (
                    SELECT COUNT(*) FROM group_members 
                    WHERE group_id = ? AND is_active = 1
                )
                WHERE group_id = ?
            ''', (group_id, group_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Member {member.name} added to group {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding member to group: {str(e)}")
            return False
    
    def get_group_members(self, group_id: str) -> List[GroupMember]:
        """Get all members of a group"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT phone_number, name, role, joined_date, last_active,
                       message_count, is_active, tags, constituency, booth_number
                FROM group_members 
                WHERE group_id = ? AND is_active = 1
                ORDER BY name
            ''', (group_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            members = []
            for row in rows:
                member = GroupMember(
                    phone_number=row[0],
                    name=row[1],
                    role=MemberRole(row[2]),
                    joined_date=datetime.fromisoformat(row[3]),
                    last_active=datetime.fromisoformat(row[4]) if row[4] else None,
                    message_count=row[5],
                    is_active=bool(row[6]),
                    tags=json.loads(row[7]) if row[7] else [],
                    constituency=row[8],
                    booth_number=row[9]
                )
                members.append(member)
            
            return members
            
        except Exception as e:
            logger.error(f"Error getting group members: {str(e)}")
            return []
    
    def bulk_add_members(self, group_id: str, members_data: List[Dict]) -> Dict:
        """Bulk add members from CSV data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            added_count = 0
            failed_count = 0
            errors = []
            
            for member_data in members_data:
                try:
                    # Validate required fields
                    if not all(key in member_data for key in ['phone_number', 'name']):
                        errors.append(f"Missing required fields for {member_data.get('name', 'Unknown')}")
                        failed_count += 1
                        continue
                    
                    # Create member object
                    member = GroupMember(
                        phone_number=member_data['phone_number'],
                        name=member_data['name'],
                        role=MemberRole(member_data.get('role', 'member')),
                        joined_date=datetime.now(),
                        constituency=member_data.get('constituency'),
                        booth_number=member_data.get('booth_number'),
                        tags=member_data.get('tags', '').split(',') if member_data.get('tags') else []
                    )
                    
                    # Insert member
                    cursor.execute('''
                        INSERT OR REPLACE INTO group_members 
                        (group_id, phone_number, name, role, joined_date,
                         constituency, booth_number, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        group_id, member.phone_number, member.name, member.role.value,
                        member.joined_date, member.constituency, member.booth_number,
                        json.dumps(member.tags)
                    ))
                    
                    added_count += 1
                    
                except Exception as e:
                    errors.append(f"Error adding {member_data.get('name', 'Unknown')}: {str(e)}")
                    failed_count += 1
            
            # Update group member count
            cursor.execute('''
                UPDATE groups 
                SET member_count = (
                    SELECT COUNT(*) FROM group_members 
                    WHERE group_id = ? AND is_active = 1
                )
                WHERE group_id = ?
            ''', (group_id, group_id))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'added_count': added_count,
                'failed_count': failed_count,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Error in bulk add members: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'added_count': 0,
                'failed_count': len(members_data)
            }
    
    def remove_member_from_group(self, group_id: str, phone_number: str) -> bool:
        """Remove member from group"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE group_members 
                SET is_active = 0 
                WHERE group_id = ? AND phone_number = ?
            ''', (group_id, phone_number))
            
            # Update group member count
            cursor.execute('''
                UPDATE groups 
                SET member_count = (
                    SELECT COUNT(*) FROM group_members 
                    WHERE group_id = ? AND is_active = 1
                )
                WHERE group_id = ?
            ''', (group_id, group_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error removing member: {str(e)}")
            return False
    
    def update_group(self, group_id: str, updates: Dict) -> bool:
        """Update group information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for field, value in updates.items():
                if field in ['name', 'description', 'status', 'invite_link', 'constituency', 'region']:
                    set_clauses.append(f"{field} = ?")
                    values.append(value)
                elif field in ['tags', 'settings']:
                    set_clauses.append(f"{field} = ?")
                    values.append(json.dumps(value))
            
            if set_clauses:
                set_clauses.append("updated_at = ?")
                values.append(datetime.now())
                values.append(group_id)
                
                query = f"UPDATE groups SET {', '.join(set_clauses)} WHERE group_id = ?"
                cursor.execute(query, values)
                
                conn.commit()
                conn.close()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating group: {str(e)}")
            return False
    
    def get_groups_by_type(self, group_type: GroupType) -> List[WhatsAppGroup]:
        """Get groups by type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM groups WHERE group_type = ? ORDER BY name', (group_type.value,))
            rows = cursor.fetchall()
            conn.close()
            
            groups = []
            for row in rows:
                group = WhatsAppGroup(
                    group_id=row[0],
                    name=row[1],
                    description=row[2],
                    group_type=GroupType(row[3]),
                    created_date=datetime.fromisoformat(row[4]),
                    status=GroupStatus(row[5]),
                    member_count=row[6],
                    invite_link=row[7],
                    tags=json.loads(row[8]) if row[8] else [],
                    settings=json.loads(row[9]) if row[9] else {},
                    constituency=row[10],
                    region=row[11]
                )
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error getting groups by type: {str(e)}")
            return []
    
    def search_groups(self, query: str, filters: Dict = None) -> List[WhatsAppGroup]:
        """Search groups with filters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            where_clauses = ["(name LIKE ? OR description LIKE ?)"]
            params = [f"%{query}%", f"%{query}%"]
            
            if filters:
                if filters.get('group_type'):
                    where_clauses.append("group_type = ?")
                    params.append(filters['group_type'])
                
                if filters.get('status'):
                    where_clauses.append("status = ?")
                    params.append(filters['status'])
                
                if filters.get('constituency'):
                    where_clauses.append("constituency = ?")
                    params.append(filters['constituency'])
                
                if filters.get('region'):
                    where_clauses.append("region = ?")
                    params.append(filters['region'])
            
            query_sql = f"SELECT * FROM groups WHERE {' AND '.join(where_clauses)} ORDER BY name"
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            groups = []
            for row in rows:
                group = WhatsAppGroup(
                    group_id=row[0],
                    name=row[1],
                    description=row[2],
                    group_type=GroupType(row[3]),
                    created_date=datetime.fromisoformat(row[4]),
                    status=GroupStatus(row[5]),
                    member_count=row[6],
                    invite_link=row[7],
                    tags=json.loads(row[8]) if row[8] else [],
                    settings=json.loads(row[9]) if row[9] else {},
                    constituency=row[10],
                    region=row[11]
                )
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error searching groups: {str(e)}")
            return []

class GroupManager:
    """Main class for group management operations"""
    
    def __init__(self, db_path='whatsapp_groups.db'):
        self.db = GroupDatabase(db_path)
    
    def create_group(self, name: str, description: str, group_type: GroupType, 
                    constituency: str = None, region: str = None, 
                    settings: Dict = None) -> str:
        """Create a new group"""
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        
        group = WhatsAppGroup(
            group_id=group_id,
            name=name,
            description=description,
            group_type=group_type,
            created_date=datetime.now(),
            constituency=constituency,
            region=region,
            settings=settings
        )
        
        if self.db.add_group(group):
            logger.info(f"Group created successfully: {group_id}")
            return group_id
        else:
            raise Exception("Failed to create group")
    
    def add_members_from_csv(self, group_id: str, csv_content: str) -> Dict:
        """Add members from CSV content"""
        try:
            # Parse CSV
            csv_file = io.StringIO(csv_content)
            reader = csv.DictReader(csv_file)
            
            members_data = []
            for row in reader:
                # Clean and validate data
                member_data = {
                    'phone_number': row.get('phone_number', '').strip(),
                    'name': row.get('name', '').strip(),
                    'role': row.get('role', 'member').strip().lower(),
                    'constituency': row.get('constituency', '').strip(),
                    'booth_number': row.get('booth_number', '').strip(),
                    'tags': row.get('tags', '').strip()
                }
                
                # Validate phone number format
                if not member_data['phone_number'].startswith('+'):
                    member_data['phone_number'] = '+91' + member_data['phone_number'].lstrip('+0')
                
                members_data.append(member_data)
            
            return self.db.bulk_add_members(group_id, members_data)
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'added_count': 0,
                'failed_count': 0
            }
    
    def add_members_from_file_upload(self, group_id: str, file_data) -> Dict:
        """Add members from uploaded file"""
        try:
            # Read file content
            if hasattr(file_data, 'read'):
                content = file_data.read().decode('utf-8')
            else:
                content = file_data
            
            return self.add_members_from_csv(group_id, content)
            
        except Exception as e:
            logger.error(f"Error processing file upload: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_group_statistics(self, group_id: str) -> Dict:
        """Get comprehensive group statistics"""
        try:
            group = self.db.get_group(group_id)
            if not group:
                return {'error': 'Group not found'}
            
            members = self.db.get_group_members(group_id)
            
            # Basic stats
            stats = {
                'group_info': {
                    'name': group.name,
                    'type': group.group_type.value,
                    'status': group.status.value,
                    'created_date': group.created_date.isoformat(),
                    'constituency': group.constituency,
                    'region': group.region
                },
                'member_stats': {
                    'total_members': len(members),
                    'active_members': len([m for m in members if m.is_active]),
                    'admins': len([m for m in members if m.role == MemberRole.ADMIN]),
                    'moderators': len([m for m in members if m.role == MemberRole.MODERATOR]),
                    'members': len([m for m in members if m.role == MemberRole.MEMBER])
                },
                'activity_stats': {
                    'total_messages': sum(m.message_count for m in members),
                    'avg_messages_per_member': statistics.mean([m.message_count for m in members]) if members else 0,
                    'most_active_member': max(members, key=lambda m: m.message_count).name if members else None
                }
            }
            
            # Role distribution
            role_counts = Counter(m.role.value for m in members)
            stats['role_distribution'] = dict(role_counts)
            
            # Constituency distribution
            if any(m.constituency for m in members):
                constituency_counts = Counter(m.constituency for m in members if m.constituency)
                stats['constituency_distribution'] = dict(constituency_counts)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting group statistics: {str(e)}")
            return {'error': str(e)}
    
    def export_group_members(self, group_id: str, format_type: str = 'csv') -> Optional[str]:
        """Export group members in various formats"""
        try:
            members = self.db.get_group_members(group_id)
            
            if format_type.lower() == 'csv':
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow([
                    'phone_number', 'name', 'role', 'joined_date', 
                    'last_active', 'message_count', 'constituency', 
                    'booth_number', 'tags'
                ])
                
                # Data
                for member in members:
                    writer.writerow([
                        member.phone_number,
                        member.name,
                        member.role.value,
                        member.joined_date.isoformat(),
                        member.last_active.isoformat() if member.last_active else '',
                        member.message_count,
                        member.constituency or '',
                        member.booth_number or '',
                        ','.join(member.tags)
                    ])
                
                return output.getvalue()
            
            elif format_type.lower() == 'json':
                members_data = []
                for member in members:
                    member_dict = asdict(member)
                    member_dict['role'] = member.role.value
                    member_dict['joined_date'] = member.joined_date.isoformat()
                    if member.last_active:
                        member_dict['last_active'] = member.last_active.isoformat()
                    members_data.append(member_dict)
                
                return json.dumps(members_data, indent=2)
            
            return None
            
        except Exception as e:
            logger.error(f"Error exporting members: {str(e)}")
            return None
    
    def validate_group_settings(self, settings: Dict) -> Tuple[bool, List[str]]:
        """Validate group settings"""
        errors = []
        
        # Required settings
        required_settings = ['max_members', 'auto_response_enabled', 'content_filtering']
        for setting in required_settings:
            if setting not in settings:
                errors.append(f"Missing required setting: {setting}")
        
        # Validate max_members
        if 'max_members' in settings:
            try:
                max_members = int(settings['max_members'])
                if max_members < 1 or max_members > 5000:
                    errors.append("max_members must be between 1 and 5000")
            except (ValueError, TypeError):
                errors.append("max_members must be a valid integer")
        
        # Validate boolean settings
        boolean_settings = ['auto_response_enabled', 'content_filtering', 'member_approval_required']
        for setting in boolean_settings:
            if setting in settings and not isinstance(settings[setting], bool):
                errors.append(f"{setting} must be a boolean value")
        
        return len(errors) == 0, errors
    
    def get_groups_summary(self) -> Dict:
        """Get summary of all groups"""
        try:
            groups = self.db.get_all_groups()
            
            summary = {
                'total_groups': len(groups),
                'active_groups': len([g for g in groups if g.status == GroupStatus.ACTIVE]),
                'total_members': sum(g.member_count for g in groups),
                'groups_by_type': {},
                'groups_by_status': {},
                'groups_by_constituency': {},
                'average_group_size': statistics.mean([g.member_count for g in groups]) if groups else 0
            }
            
            # Group by type
            type_counts = Counter(g.group_type.value for g in groups)
            summary['groups_by_type'] = dict(type_counts)
            
            # Group by status
            status_counts = Counter(g.status.value for g in groups)
            summary['groups_by_status'] = dict(status_counts)
            
            # Group by constituency
            constituency_counts = Counter(g.constituency for g in groups if g.constituency)
            summary['groups_by_constituency'] = dict(constituency_counts)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting groups summary: {str(e)}")
            return {'error': str(e)}

# Initialize global group manager
group_manager = GroupManager()