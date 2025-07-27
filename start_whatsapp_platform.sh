#!/bin/bash
# start_whatsapp_platform.sh - Startup script for WhatsApp Management Platform

echo "🚀 Starting WhatsApp Management Platform..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  .env file not found. Using default configuration."
fi

# Check if WhatsApp service is configured
python3 -c "
from whatsapp_service import get_whatsapp_service
service = get_whatsapp_service()
is_configured, status = service.is_configured()
if is_configured:
    print('✅ WhatsApp Business API is configured')
else:
    print('❌ WhatsApp Business API needs configuration')
    print('📖 Please edit .env file with your API credentials')
"

# Start the Flask application
echo "🌟 Starting Flask server..."
python3 app.py
