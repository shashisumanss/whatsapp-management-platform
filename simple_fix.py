#!/usr/bin/env python3
import os
import shutil
from datetime import datetime

def backup_and_fix():
    template_path = "templates/realtime.html"
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{template_path}.backup_{timestamp}"
    if os.path.exists(template_path):
        shutil.copy2(template_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    # Simple, working template with proper escaping
    fixed_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Real-time Monitoring</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-6">⚡ Real-time Monitoring</h1>
        
        <!-- Connection Status -->
        <div class="bg-white rounded-lg shadow p-4 mb-6">
            <div class="flex items-center">
                <div class="w-3 h-3 rounded-full mr-2 bg-red-500" id="connection-indicator"></div>
                <span id="connection-status">Connecting...</span>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow p-6 mb-6">
            <p class="text-gray-600 mb-4">Real-time WhatsApp message monitoring and analytics</p>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-blue-50 p-4 rounded">
                    <h3 class="font-semibold text-blue-800">Live Messages</h3>
                    <p class="text-2xl font-bold text-blue-600" id="live-count">0</p>
                </div>
                <div class="bg-red-50 p-4 rounded">
                    <h3 class="font-semibold text-red-800">Threats Detected</h3>
                    <p class="text-2xl font-bold text-red-600" id="threat-count">0</p>
                </div>
                <div class="bg-green-50 p-4 rounded">
                    <h3 class="font-semibold text-green-800">Active Groups</h3>
                    <p class="text-2xl font-bold text-green-600" id="group-count">0</p>
                </div>
            </div>
        </div>

        <!-- Messages Feed -->
        <div class="bg-white rounded-lg shadow p-6 mb-6">
            <h2 class="text-xl font-bold mb-4">📱 Live Messages</h2>
            <div id="messages-feed" class="h-64 overflow-y-auto border border-gray-200 rounded p-4 bg-gray-50">
                <div class="text-gray-500 text-center" id="waiting-message">Waiting for messages...</div>
            </div>
        </div>

        <div class="mt-6">
            <a href="/management" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                Back to Management
            </a>
        </div>
    </div>

    <script>
        console.log('Real-time monitoring - SIMPLE FIXED VERSION');
        
        var messageCount = 0;
        var threatCount = 0;
        var activeGroups = [];
        
        console.log('Connecting to WebSocket...');
        var socket = io();
        
        socket.on('connect', function() {
            console.log('WebSocket Connected');
            document.getElementById('connection-indicator').className = 'w-3 h-3 rounded-full mr-2 bg-green-500';
            document.getElementById('connection-status').textContent = 'Connected to Real-time Monitoring';
        });
        
        socket.on('disconnect', function() {
            console.log('WebSocket Disconnected');
            document.getElementById('connection-indicator').className = 'w-3 h-3 rounded-full mr-2 bg-red-500';
            document.getElementById('connection-status').textContent = 'Disconnected';
        });

        socket.on('new_message', function(data) {
            console.log('New message received:', data);
            handleMessage(data);
        });

        function handleMessage(data) {
            try {
                var message = data;
                if (typeof data === 'string') {
                    message = { text: data, sender: 'Unknown', timestamp: new Date().toISOString() };
                }

                var messageText = message.text || 'No content';
                var sender = message.sender || 'Unknown';

                messageCount++;
                document.getElementById('live-count').textContent = messageCount;

                if (messageText.toLowerCase().indexOf('urgent') !== -1 || 
                    messageText.toLowerCase().indexOf('help') !== -1) {
                    threatCount++;
                    document.getElementById('threat-count').textContent = threatCount;
                }

                displayMessage(messageText, sender);

            } catch (error) {
                console.error('Error handling message:', error);
            }
        }

        function displayMessage(text, sender) {
            var feed = document.getElementById('messages-feed');
            var waitingMessage = document.getElementById('waiting-message');
            
            if (waitingMessage) {
                waitingMessage.remove();
            }

            var messageDiv = document.createElement('div');
            messageDiv.className = 'mb-3 p-3 border rounded-lg bg-white border-gray-200';
            messageDiv.innerHTML = '<div class="font-semibold text-sm text-gray-700">💬 ' + sender + '</div><div class="text-gray-800">' + text + '</div>';
            
            feed.insertBefore(messageDiv, feed.firstChild);
        }

        window.testMessage = function() {
            handleMessage({
                text: 'Test message from console!',
                sender: 'Console Tester'
            });
        };

        console.log('Dashboard ready. Test with: testMessage()');
    </script>
</body>
</html>"""
    
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(fixed_template)
    
    print(f"✅ Successfully updated {template_path}")
    print("Refresh the page now!")

if __name__ == "__main__":
    backup_and_fix()
