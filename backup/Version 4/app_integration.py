# app_integration.py
# Integration routes for group management and analytics
# Add these routes to your existing app.py

from flask import Flask, request, jsonify, render_template, send_file
from group_manager import group_manager, GroupType, MemberRole, GroupStatus
from analytics_engine import analytics_engine
import json
import io
import csv
from datetime import datetime, timedelta

# ===== GROUP MANAGEMENT ROUTES =====

@app.route('/api/groups', methods=['GET'])
def get_groups():
    """Get all groups with optional filtering"""
    try:
        group_type = request.args.get('type')
        status = request.args.get('status')
        constituency = request.args.get('constituency')
        
        if group_type or status or constituency:
            # Filtered search
            filters = {}
            if group_type:
                filters['group_type'] = group_type
            if status:
                filters['status'] = status
            if constituency:
                filters['constituency'] = constituency
            
            groups = group_manager.db.search_groups('', filters)
        else:
            # Get all groups
            groups = group_manager.db.get_all_groups()
        
        groups_data = []
        for group in groups:
            groups_data.append({
                'group_id': group.group_id,
                'name': group.name,
                'description': group.description,
                'type': group.group_type.value,
                'status': group.status.value,
                'member_count': group.member_count,
                'created_date': group.created_date.isoformat(),
                'constituency': group.constituency,
                'region': group.region,
                'tags': group.tags,
                'invite_link': group.invite_link
            })
        
        return jsonify({
            'success': True,
            'groups': groups_data,
            'total_count': len(groups_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups', methods=['POST'])
def create_group():
    """Create a new WhatsApp group"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description', 'group_type']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'{field} is required'}), 400
        
        # Validate group type
        try:
            group_type = GroupType(data['group_type'])
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid group type'}), 400
        
        # Create group
        group_id = group_manager.create_group(
            name=data['name'],
            description=data['description'],
            group_type=group_type,
            constituency=data.get('constituency'),
            region=data.get('region'),
            settings=data.get('settings')
        )
        
        return jsonify({
            'success': True,
            'group_id': group_id,
            'message': f'Group "{data["name"]}" created successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/<group_id>', methods=['GET'])
def get_group_details(group_id):
    """Get detailed information about a specific group"""
    try:
        group = group_manager.db.get_group(group_id)
        
        if not group:
            return jsonify({'success': False, 'error': 'Group not found'}), 404
        
        # Get group statistics
        stats = group_manager.get_group_statistics(group_id)
        
        group_data = {
            'group_id': group.group_id,
            'name': group.name,
            'description': group.description,
            'type': group.group_type.value,
            'status': group.status.value,
            'member_count': group.member_count,
            'created_date': group.created_date.isoformat(),
            'constituency': group.constituency,
            'region': group.region,
            'tags': group.tags,
            'settings': group.settings,
            'invite_link': group.invite_link,
            'statistics': stats
        }
        
        return jsonify({
            'success': True,
            'group': group_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/<group_id>', methods=['PUT'])
def update_group(group_id):
    """Update group information"""
    try:
        data = request.get_json()
        
        # Validate settings if provided
        if 'settings' in data:
            is_valid, errors = group_manager.validate_group_settings(data['settings'])
            if not is_valid:
                return jsonify({'success': False, 'errors': errors}), 400
        
        # Update group
        success = group_manager.db.update_group(group_id, data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Group updated successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to update group'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/<group_id>/members', methods=['GET'])
def get_group_members(group_id):
    """Get all members of a group"""
    try:
        members = group_manager.db.get_group_members(group_id)
        
        members_data = []
        for member in members:
            members_data.append({
                'phone_number': member.phone_number,
                'name': member.name,
                'role': member.role.value,
                'joined_date': member.joined_date.isoformat(),
                'last_active': member.last_active.isoformat() if member.last_active else None,
                'message_count': member.message_count,
                'is_active': member.is_active,
                'constituency': member.constituency,
                'booth_number': member.booth_number,
                'tags': member.tags
            })
        
        return jsonify({
            'success': True,
            'members': members_data,
            'total_count': len(members_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/<group_id>/members/bulk', methods=['POST'])
def bulk_add_members(group_id):
    """Bulk add members to a group from CSV upload"""
    try:
        # Check if group exists
        group = group_manager.db.get_group(group_id)
        if not group:
            return jsonify({'success': False, 'error': 'Group not found'}), 404
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                result = group_manager.add_members_from_file_upload(group_id, file)
            else:
                return jsonify({'success': False, 'error': 'Please upload a CSV file'}), 400
        elif request.is_json:
            # Handle JSON data
            data = request.get_json()
            if 'members' in data:
                result = group_manager.db.bulk_add_members(group_id, data['members'])
            else:
                return jsonify({'success': False, 'error': 'Members data required'}), 400
        else:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/<group_id>/members/<phone_number>', methods=['DELETE'])
def remove_member(group_id, phone_number):
    """Remove a member from a group"""
    try:
        success = group_manager.db.remove_member_from_group(group_id, phone_number)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Member removed successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to remove member'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/<group_id>/export', methods=['GET'])
def export_group_members(group_id):
    """Export group members in CSV or JSON format"""
    try:
        format_type = request.args.get('format', 'csv').lower()
        
        if format_type not in ['csv', 'json']:
            return jsonify({'success': False, 'error': 'Invalid format. Use csv or json'}), 400
        
        export_data = group_manager.export_group_members(group_id, format_type)
        
        if export_data:
            # Get group info for filename
            group = group_manager.db.get_group(group_id)
            filename = f"{group.name}_members_{datetime.now().strftime('%Y%m%d')}.{format_type}"
            
            # Create response
            output = io.StringIO(export_data)
            output.seek(0)
            
            return send_file(
                io.BytesIO(export_data.encode()),
                mimetype='text/csv' if format_type == 'csv' else 'application/json',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({'success': False, 'error': 'Export failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/search', methods=['GET'])
def search_groups():
    """Search groups with query and filters"""
    try:
        query = request.args.get('q', '')
        group_type = request.args.get('type')
        status = request.args.get('status')
        constituency = request.args.get('constituency')
        region = request.args.get('region')
        
        filters = {}
        if group_type:
            filters['group_type'] = group_type
        if status:
            filters['status'] = status
        if constituency:
            filters['constituency'] = constituency
        if region:
            filters['region'] = region
        
        groups = group_manager.db.search_groups(query, filters)
        
        groups_data = []
        for group in groups:
            groups_data.append({
                'group_id': group.group_id,
                'name': group.name,
                'description': group.description,
                'type': group.group_type.value,
                'status': group.status.value,
                'member_count': group.member_count,
                'constituency': group.constituency,
                'region': group.region
            })
        
        return jsonify({
            'success': True,
            'groups': groups_data,
            'query': query,
            'filters': filters,
            'total_count': len(groups_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/summary', methods=['GET'])
def get_groups_summary():
    """Get summary statistics for all groups"""
    try:
        summary = group_manager.get_groups_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== ANALYTICS ROUTES =====

@app.route('/api/analytics/health/<group_id>', methods=['GET'])
def get_group_health_score(group_id):
    """Get health score for a specific group"""
    try:
        timeframe_days = int(request.args.get('timeframe', 30))
        health_data = analytics_engine.calculate_group_health_score(group_id, timeframe_days)
        
        return jsonify({
            'success': True,
            'health_data': health_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/trends', methods=['GET'])
def get_trends_analysis():
    """Get trends analysis for group or platform"""
    try:
        group_id = request.args.get('group_id')
        timeframe_days = int(request.args.get('timeframe', 90))
        
        trends = analytics_engine.generate_trend_analysis(group_id, timeframe_days)
        
        return jsonify({
            'success': True,
            'trends': trends
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/dashboard', methods=['GET'])
def get_analytics_dashboard():
    """Get comprehensive dashboard data"""
    try:
        group_id = request.args.get('group_id')
        timeframe_days = int(request.args.get('timeframe', 30))
        
        dashboard_data = analytics_engine.create_comprehensive_dashboard_data(group_id, timeframe_days)
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/charts', methods=['GET'])
def get_analytics_charts():
    """Get Plotly charts data"""
    try:
        group_id = request.args.get('group_id')
        timeframe_days = int(request.args.get('timeframe', 30))
        
        charts = analytics_engine.generate_plotly_charts(group_id, timeframe_days)
        
        return jsonify({
            'success': True,
            'charts': charts
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/report', methods=['GET'])
def generate_analytics_report():
    """Generate comprehensive analytics report"""
    try:
        group_id = request.args.get('group_id')
        timeframe_days = int(request.args.get('timeframe', 30))
        
        report = analytics_engine.generate_advanced_report(group_id, timeframe_days)
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/metrics/store', methods=['POST'])
def store_custom_metric():
    """Store custom analytics metric"""
    try:
        data = request.get_json()
        
        required_fields = ['group_id', 'metric_type', 'metric_name', 'value']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'{field} is required'}), 400
        
        analytics_engine._store_metric(
            group_id=data['group_id'],
            metric_type=data['metric_type'],
            metric_name=data['metric_name'],
            value=data['value'],
            metadata=data.get('metadata'),
            timeframe=data.get('timeframe', 'daily')
        )
        
        return jsonify({
            'success': True,
            'message': 'Metric stored successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== ENHANCED MANAGEMENT DASHBOARD ROUTES =====

@app.route('/groups')
def groups_management_page():
    """Groups management dashboard page"""
    return render_template('groups_management.html')

@app.route('/analytics')
def analytics_page():
    """Analytics dashboard page"""
    return render_template('analytics_dashboard.html')

@app.route('/api/dashboard/enhanced_stats')
def enhanced_dashboard_stats():
    """Enhanced dashboard statistics including groups and analytics"""
    try:
        # Get existing stats
        total_messages = len(analyzer.df) if analyzer.df is not None else 0
        flagged_messages = len(analyzer.flagged_messages) if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages else 0
        critical_messages = 0
        if hasattr(analyzer, 'flagged_messages') and analyzer.flagged_messages:
            critical_messages = len([f for f in analyzer.flagged_messages if f.get('priority') == 'CRITICAL'])
        
        open_tickets = len([t for t in tickets_storage if t['status'] == 'OPEN'])
        total_tickets = len(tickets_storage)
        
        # Add group statistics
        groups_summary = group_manager.get_groups_summary()
        
        # Add WhatsApp stats
        whatsapp_stats = whatsapp_service.get_delivery_stats()
        
        return jsonify({
            'total_messages': total_messages,
            'flagged_messages': flagged_messages,
            'critical_messages': critical_messages,
            'open_tickets': open_tickets,
            'total_tickets': total_tickets,
            'whatsapp_messages': whatsapp_stats.get('total_tracked', 0),
            'pending_scheduled': len([msg for msg in scheduled_messages_storage if msg['status'] == 'pending']),
            
            # Group statistics
            'total_groups': groups_summary.get('total_groups', 0),
            'active_groups': groups_summary.get('active_groups', 0),
            'total_group_members': groups_summary.get('total_members', 0),
            'average_group_size': round(groups_summary.get('average_group_size', 0), 1),
            
            # Health indicators
            'whatsapp_health': whatsapp_service.get_health_status().get('healthy', False),
            'platform_health': 'healthy' if flagged_messages / max(total_messages, 1) < 0.05 else 'needs_attention'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== INTEGRATION WITH EXISTING MESSAGE ANALYSIS =====

def integrate_message_with_groups(message_data, analysis_result):
    """Integrate analyzed message with group system"""
    try:
        # Determine group_id from message context (you'll need to implement this logic)
        group_id = determine_group_from_message(message_data)
        
        if group_id:
            # Store message in group analytics
            conn = sqlite3.connect('whatsapp_groups.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO group_messages 
                (group_id, sender_phone, message_content, message_type, timestamp,
                 sentiment, is_flagged, priority, language, response_time_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                group_id,
                message_data.get('sender', 'unknown'),
                message_data.get('message', ''),
                message_data.get('type', 'text'),
                datetime.now(),
                analysis_result.get('sentiment', {}).get('sentiment', 'neutral'),
                analysis_result.get('flagging', {}).get('is_flagged', False),
                analysis_result.get('flagging', {}).get('priority', 'normal'),
                analysis_result.get('sentiment', {}).get('language', 'unknown'),
                None  # Response time to be calculated later
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Message integrated with group {group_id}")
            
    except Exception as e:
        logger.error(f"Error integrating message with groups: {str(e)}")

def determine_group_from_message(message_data):
    """Determine which group a message belongs to"""
    # Implement your logic here based on:
    # - WhatsApp group ID from webhook
    # - Phone number mapping
    # - Message context
    # For now, return a default group for testing
    return "group_default"

# ===== BULK OPERATIONS FOR GROUPS =====

@app.route('/api/groups/bulk/create', methods=['POST'])
def bulk_create_groups():
    """Bulk create groups from CSV data"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a CSV file'}), 400
        
        # Read CSV
        csv_content = file.read().decode('utf-8')
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        
        created_groups = []
        failed_groups = []
        
        for row in reader:
            try:
                # Validate required fields
                if not all(key in row for key in ['name', 'description', 'group_type']):
                    failed_groups.append({
                        'row': row,
                        'error': 'Missing required fields'
                    })
                    continue
                
                # Create group
                group_id = group_manager.create_group(
                    name=row['name'].strip(),
                    description=row['description'].strip(),
                    group_type=GroupType(row['group_type'].strip().lower()),
                    constituency=row.get('constituency', '').strip() or None,
                    region=row.get('region', '').strip() or None
                )
                
                created_groups.append({
                    'group_id': group_id,
                    'name': row['name']
                })
                
            except Exception as e:
                failed_groups.append({
                    'row': row,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'created_count': len(created_groups),
            'failed_count': len(failed_groups),
            'created_groups': created_groups,
            'failed_groups': failed_groups
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/groups/bulk/update_status', methods=['POST'])
def bulk_update_group_status():
    """Bulk update status of multiple groups"""
    try:
        data = request.get_json()
        group_ids = data.get('group_ids', [])
        new_status = data.get('status')
        
        if not group_ids:
            return jsonify({'success': False, 'error': 'No groups specified'}), 400
        
        if not new_status or new_status not in [s.value for s in GroupStatus]:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        
        updated_count = 0
        failed_count = 0
        
        for group_id in group_ids:
            try:
                success = group_manager.db.update_group(group_id, {'status': new_status})
                if success:
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'failed_count': failed_count,
            'message': f'Updated {updated_count} groups, {failed_count} failed'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== ADVANCED ANALYTICS ENDPOINTS =====

@app.route('/api/analytics/comparative', methods=['GET'])
def comparative_analytics():
    """Compare analytics between multiple groups"""
    try:
        group_ids = request.args.getlist('group_ids')
        timeframe_days = int(request.args.get('timeframe', 30))
        
        if not group_ids:
            return jsonify({'success': False, 'error': 'No groups specified'}), 400
        
        comparison_data = {}
        
        for group_id in group_ids:
            try:
                health_data = analytics_engine.calculate_group_health_score(group_id, timeframe_days)
                group = group_manager.db.get_group(group_id)
                
                comparison_data[group_id] = {
                    'group_name': group.name if group else f'Group {group_id}',
                    'health_score': health_data.get('health_score', 0),
                    'components': health_data.get('components', {}),
                    'member_count': group.member_count if group else 0
                }
                
            except Exception as e:
                comparison_data[group_id] = {
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'comparison': comparison_data,
            'timeframe_days': timeframe_days
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/leaderboard', methods=['GET'])
def analytics_leaderboard():
    """Get leaderboard of top performing groups"""
    try:
        metric = request.args.get('metric', 'health_score')
        limit = int(request.args.get('limit', 10))
        timeframe_days = int(request.args.get('timeframe', 30))
        
        # Get all active groups
        groups = group_manager.db.get_all_groups()
        active_groups = [g for g in groups if g.status == GroupStatus.ACTIVE]
        
        leaderboard = []
        
        for group in active_groups:
            try:
                if metric == 'health_score':
                    health_data = analytics_engine.calculate_group_health_score(group.group_id, timeframe_days)
                    score = health_data.get('health_score', 0)
                elif metric == 'member_count':
                    score = group.member_count
                elif metric == 'activity_score':
                    health_data = analytics_engine.calculate_group_health_score(group.group_id, timeframe_days)
                    score = health_data.get('components', {}).get('activity_score', 0)
                else:
                    score = 0
                
                leaderboard.append({
                    'group_id': group.group_id,
                    'group_name': group.name,
                    'group_type': group.group_type.value,
                    'constituency': group.constituency,
                    'score': round(score, 2),
                    'member_count': group.member_count
                })
                
            except Exception as e:
                logger.error(f"Error calculating metric for group {group.group_id}: {str(e)}")
        
        # Sort by score (descending) and limit
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        leaderboard = leaderboard[:limit]
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard,
            'metric': metric,
            'timeframe_days': timeframe_days
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/alerts', methods=['GET'])
def get_analytics_alerts():
    """Get alerts based on analytics thresholds"""
    try:
        timeframe_days = int(request.args.get('timeframe', 7))
        
        alerts = []
        groups = group_manager.db.get_all_groups()
        active_groups = [g for g in groups if g.status == GroupStatus.ACTIVE]
        
        for group in active_groups:
            try:
                health_data = analytics_engine.calculate_group_health_score(group.group_id, timeframe_days)
                health_score = health_data.get('health_score', 0)
                components = health_data.get('components', {})
                
                # Critical health score
                if health_score < 40:
                    alerts.append({
                        'type': 'critical',
                        'group_id': group.group_id,
                        'group_name': group.name,
                        'message': f'Critical health score: {health_score}',
                        'priority': 'high',
                        'category': 'health'
                    })
                
                # Low safety score
                if components.get('safety_score', 100) < 70:
                    alerts.append({
                        'type': 'warning',
                        'group_id': group.group_id,
                        'group_name': group.name,
                        'message': f'Low safety score: {components["safety_score"]}',
                        'priority': 'medium',
                        'category': 'safety'
                    })
                
                # Low activity
                if components.get('activity_score', 100) < 30:
                    alerts.append({
                        'type': 'info',
                        'group_id': group.group_id,
                        'group_name': group.name,
                        'message': f'Low activity detected',
                        'priority': 'low',
                        'category': 'activity'
                    })
                
            except Exception as e:
                logger.error(f"Error checking alerts for group {group.group_id}: {str(e)}")
        
        # Sort alerts by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        alerts.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'total_alerts': len(alerts),
            'timeframe_days': timeframe_days
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== DATA EXPORT AND REPORTING =====

@app.route('/api/export/comprehensive_report', methods=['GET'])
def export_comprehensive_report():
    """Export comprehensive platform report"""
    try:
        format_type = request.args.get('format', 'json').lower()
        timeframe_days = int(request.args.get('timeframe', 30))
        
        # Generate comprehensive report
        platform_report = analytics_engine.generate_advanced_report(None, timeframe_days)
        groups_summary = group_manager.get_groups_summary()
        
        # Combine data
        comprehensive_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'timeframe_days': timeframe_days,
                'report_type': 'comprehensive_platform_report'
            },
            'platform_analytics': platform_report,
            'groups_summary': groups_summary,
            'system_health': {
                'whatsapp_health': whatsapp_service.get_health_status(),
                'total_tickets': len(tickets_storage),
                'open_tickets': len([t for t in tickets_storage if t['status'] == 'OPEN']),
                'scheduled_messages': len(scheduled_messages_storage)
            }
        }
        
        if format_type == 'json':
            filename = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            return send_file(
                io.BytesIO(json.dumps(comprehensive_report, indent=2).encode()),
                mimetype='application/json',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({'success': False, 'error': 'Only JSON format supported'}), 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== HELPER FUNCTIONS FOR INTEGRATION =====

def initialize_sample_groups():
    """Initialize sample groups for testing"""
    try:
        sample_groups = [
            {
                'name': 'Community Group 1',
                'description': 'Main community discussion group',
                'group_type': GroupType.COMMUNITY,
                'constituency': 'North Bangalore',
                'region': 'Karnataka'
            },
            {
                'name': 'Emergency Alerts',
                'description': 'Emergency notifications and alerts',
                'group_type': GroupType.EMERGENCY,
                'constituency': 'North Bangalore',
                'region': 'Karnataka'
            },
            {
                'name': 'Health Services',
                'description': 'Healthcare related discussions',
                'group_type': GroupType.HEALTHCARE,
                'constituency': 'South Bangalore',
                'region': 'Karnataka'
            },
            {
                'name': 'Infrastructure Updates',
                'description': 'Infrastructure and civic issues',
                'group_type': GroupType.INFRASTRUCTURE,
                'constituency': 'East Bangalore',
                'region': 'Karnataka'
            }
        ]
        
        for group_data in sample_groups:
            try:
                # Check if group already exists
                existing_groups = group_manager.db.get_all_groups()
                if not any(g.name == group_data['name'] for g in existing_groups):
                    group_manager.create_group(**group_data)
                    logger.info(f"Created sample group: {group_data['name']}")
            except Exception as e:
                logger.error(f"Error creating sample group {group_data['name']}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error initializing sample groups: {str(e)}")

# Initialize sample groups on startup
initialize_sample_groups()

# ===== ENHANCED ERROR HANDLING =====

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def handle_500(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('500.html'), 500