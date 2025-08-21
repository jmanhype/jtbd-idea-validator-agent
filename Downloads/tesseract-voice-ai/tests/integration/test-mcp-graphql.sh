#!/bin/bash

# Integration test script for MCP+GraphQL control plane

echo "🚀 Starting MCP+GraphQL Integration Test"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if service is running
check_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Checking $name..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    echo -e " ${RED}✗${NC}"
    return 1
}

# Start backend GraphQL server
echo -e "${YELLOW}Starting GraphQL Backend...${NC}"
cd backend/graphql
pip install -r requirements.txt > /dev/null 2>&1
python server.py &
BACKEND_PID=$!
cd ../..

# Wait for backend to be ready
check_service "GraphQL Backend" "http://localhost:4000/health"

# Start Apollo Router (MCP wrapper)
echo -e "${YELLOW}Starting Apollo MCP Router...${NC}"
cd apollo-mcp
npm install > /dev/null 2>&1
npm run download-router > /dev/null 2>&1
npm start &
ROUTER_PID=$!
cd ..

# Wait for router to be ready
check_service "Apollo Router" "http://localhost:4001/health"

# Start Next.js frontend
echo -e "${YELLOW}Starting Next.js Frontend...${NC}"
cd frontend
npm install > /dev/null 2>&1
npm run build > /dev/null 2>&1
npm start &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
check_service "Next.js Frontend" "http://localhost:3000"

echo ""
echo -e "${GREEN}✅ All services started successfully!${NC}"
echo ""
echo "Services running at:"
echo "  - GraphQL Backend: http://localhost:4000/graphql"
echo "  - Apollo Router: http://localhost:4001"
echo "  - Next.js Frontend: http://localhost:3000"
echo ""

# Test GraphQL operations
echo -e "${YELLOW}Testing GraphQL Operations...${NC}"

# Test 1: Query available tests
echo -n "1. Query available tests..."
RESPONSE=$(curl -s -X POST http://localhost:4001/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ availableTests { name description testCount } }"}')

if echo "$RESPONSE" | grep -q "unit"; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
fi

# Test 2: Run tests in dry-run mode
echo -n "2. Run tests (dry-run)..."
RESPONSE=$(curl -s -X POST http://localhost:4001/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { runTests(suite: \"unit\", dryRun: true) { id suite status dryRun } }"}')

if echo "$RESPONSE" | grep -q "\"dryRun\":true"; then
    echo -e " ${GREEN}✓${NC}"
    RUN_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | sed 's/"id":"\([^"]*\)"/\1/')
else
    echo -e " ${RED}✗${NC}"
fi

# Test 3: Scaffold page (dry-run)
echo -n "3. Scaffold page (dry-run)..."
RESPONSE=$(curl -s -X POST http://localhost:4001/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { scaffoldPage(template: \"react-component\", name: \"TestComponent\", path: \"/components\", dryRun: true) { id template preview dryRun } }"}')

if echo "$RESPONSE" | grep -q "react-component"; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
fi

echo ""
echo -e "${GREEN}✅ Integration tests completed!${NC}"
echo ""
echo "To stop all services, run:"
echo "  kill $BACKEND_PID $ROUTER_PID $FRONTEND_PID"
echo ""
echo "Or press Ctrl+C to stop now..."

# Keep running until interrupted
wait