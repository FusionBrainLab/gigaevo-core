#!/bin/bash

# GigaEvo DAG Builder - Quick Startup Script
# Simple version without health checks

set -e

echo "🚀 Starting GigaEvo DAG Builder (Quick Mode)"
echo ""

# Cleanup
echo "🧹 Cleaning up..."
pkill -f "uvicorn.*api:app" 2>/dev/null || true
pkill -f "react-scripts start" 2>/dev/null || true
pkill -f "python.*run.py" 2>/dev/null || true
lsof -ti:8081 | xargs kill -9 2>/dev/null || true
lsof -ti:8082 | xargs kill -9 2>/dev/null || true
sleep 2

# Start backend
echo "🔧 Starting backend on port 8081..."
cd /workspace-SR008.fs2/mathemage/gigaevo/tools/dag_builder
python run.py &
BACKEND_PID=$!

# Start frontend
echo "🎨 Starting frontend on port 8082..."
cd frontend
export HOST=localhost PORT=8082 BROWSER=none
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers starting..."
echo "📱 Frontend: http://localhost:8082"
echo "🔗 Backend: http://localhost:8081"
echo "📚 API Docs: http://localhost:8081/docs"
echo ""
echo "⏳ Please wait 30-60 seconds for full initialization"
echo "Press Ctrl+C to stop both servers"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Servers stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
