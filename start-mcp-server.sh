#!/bin/bash
# Start JTBD Validator MCP Server for Claude Desktop

cd /Users/speed/straughter/jtbd-idea-validator-agent
source .venv/bin/activate

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Verify GROQ_API_KEY is set
export GROQ_API_KEY="${GROQ_API_KEY:?Error: GROQ_API_KEY not set. Add it to .env file}"
export JTBD_DSPY_MODEL="groq/llama-3.3-70b-versatile"
export JTBD_API_KEY="dev-key-jtbd-0123456789abcdef"
export DEBUG="false"

echo "Starting JTBD Validator MCP Server on http://127.0.0.1:8001..."
python mcp-server/server.py --http
