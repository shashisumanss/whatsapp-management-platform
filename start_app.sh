#!/bin/bash

echo "🛑 Stopping existing processes..."
sudo pkill -f gunicorn
sudo pkill -f "python.*app"
sleep 2

echo "🔧 Installing dependencies..."
pip3 install eventlet flask-socketio

echo "📂 Moving to app directory..."
cd ~/whatsapp-management-platform

echo "🚀 Starting application with eventlet..."
gunicorn \
  --worker-class eventlet \
  --workers 1 \
  --bind 127.0.0.1:5000 \
  --timeout 120 \
  --keepalive 2 \
  --max-requests 1000 \
  --access-logfile /tmp/gunicorn_access.log \
  --error-logfile /tmp/gunicorn_error.log \
  --log-level info \
  --preload \
  --daemon \
  app:app

sleep 3

echo "✅ Startup complete. Checking status..."
ps aux | grep gunicorn | grep -v grep
netstat -tlnp | grep :5000

echo "📊 Testing Socket.IO status..."
curl -s http://localhost:5000/test/socket-status | jq .

echo ""
echo "📝 Recent logs:"
tail -5 /tmp/gunicorn_error.log
