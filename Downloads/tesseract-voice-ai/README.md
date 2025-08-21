# 🎯 Tesseract Voice AI - Multi-Modal Platform

A sophisticated voice AI system featuring **MCP+GraphQL Control Plane**, 6 black-box modules orchestrated for <300ms latency, with natural language navigation across Task/Scope/Time/Mode axes.

## 🚀 **Featured: MCP+GraphQL Control Plane**

This repository includes a **groundbreaking MCP+GraphQL integration** - the first complete bridge between Model Context Protocol and GraphQL with:
- **Real-time test orchestration** with dry-run workflows
- **Interactive confirmation dialogs** before execution
- **Apollo Router federation** exposing GraphQL as MCP tools
- **Full-stack implementation** from backend to frontend

👉 **See [README_MCP_GRAPHQL.md](README_MCP_GRAPHQL.md) for the complete MCP+GraphQL guide**

## Architecture Overview

### 6 Black-Box Modules

1. **Session Core** (WebRTC/Pipecat) - Real-time communication management
2. **ASR Engine** (MLX Whisper) - Speech-to-text conversion
3. **NLU/Orchestrator** (NLTK+LLM) - Intent understanding & routing
4. **Tool Plugins** (JSON-RPC) - External tool integration
5. **TTS Engine** (Kokoro) - Text-to-speech synthesis
6. **Memory/Logs** (SQLite+Redis) - State persistence & analytics

### 4D Navigation Axes

- **Task**: What action to perform
- **Scope**: Context boundaries
- **Time**: Temporal awareness
- **Mode**: Operational state

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis
- Docker (optional)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
docker-compose up
```

## Performance Specifications

- **End-to-End Latency**: <300ms
- **ASR Processing**: <50ms
- **NLU Processing**: <50ms
- **Tool Execution**: <100ms
- **TTS Synthesis**: <50ms
- **Memory Operations**: <50ms

## Module Details

### Session Core
- WebRTC connection management
- Audio streaming with Pipecat
- Session lifecycle management
- Thread-safe operations

### ASR Engine
- MLX Whisper integration
- Real-time transcription
- Voice activity detection
- <50ms processing latency

### NLU/Orchestrator
- Intent classification
- NLTK linguistic analysis
- LLM integration
- 4D navigation support

### Tool Plugins
- JSON-RPC 2.0 protocol
- Dynamic plugin loading
- Sandboxed execution
- Built-in tools (calculator, weather, search)

### TTS Engine
- Kokoro TTS integration
- Multiple voices/languages
- SSML support
- Voice cloning capabilities

### Memory/Logs
- SQLite persistence
- Redis caching
- Full-text search
- Analytics & insights

## API Endpoints

- `GET /` - System status
- `WS /ws/voice` - Voice session WebSocket
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

## Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_session_core.py
pytest tests/test_asr_engine.py
pytest tests/test_nlu_orchestrator.py
```

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    4D Navigation Layer                   │
│     Task ◆ Scope ◆ Time ◆ Mode                          │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
│     WebRTC ◆ UI ◆ Visualization ◆ Controls              │
└─────────────────────────────────────────────────────────┘
                            │
                      WebSocket/REST
                            │
┌─────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                      │
│     Orchestration ◆ Pipeline ◆ Routing                  │
└─────────────────────────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───▼────┐ ┌────▼────┐ ┌───▼────┐ ┌────▼────┐ ┌───▼────┐
│Session │ │  ASR    │ │  NLU   │ │  Tools  │ │  TTS   │
│ Core   │ │ Engine  │ │Orchest.│ │ Plugins │ │ Engine │
└────────┘ └─────────┘ └────────┘ └─────────┘ └────────┘
                            │
                    ┌───────▼────────┐
                    │  Memory/Logs   │
                    │ SQLite + Redis │
                    └────────────────┘
```

## Support

For issues and questions, please open an issue on GitHub.