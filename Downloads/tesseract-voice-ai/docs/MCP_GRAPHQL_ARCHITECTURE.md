# MCP + GraphQL Control Plane Architecture

## Overview

This implementation provides a tight end-to-end integration between MCP (Model Context Protocol) and GraphQL, creating a powerful control plane for test orchestration and code generation with built-in dry-run support.

## Architecture Components

### 1. Backend GraphQL Server (Strawberry + FastAPI)
**Location:** `/backend/graphql/`

- **server.py**: Main GraphQL server implementation
- **schema.graphql**: GraphQL schema definition
- **Features:**
  - Test execution with dry-run mode
  - Component scaffolding with preview
  - Real-time progress subscriptions
  - WebSocket support for live updates

### 2. Apollo MCP Wrapper
**Location:** `/apollo-mcp/`

- **supergraph.yaml**: Federation configuration
- **router.yaml**: MCP tool exposure configuration
- **Features:**
  - Exposes GraphQL operations as MCP tools
  - Automatic parameter validation
  - Built-in telemetry and monitoring
  - Rate limiting and caching

### 3. Next.js Frontend
**Location:** `/frontend/`

- **RunTestsPanel.tsx**: Main control panel component
- **apollo-client.ts**: Apollo Client configuration
- **Features:**
  - Real-time progress updates via subscriptions
  - Dry-run preview with confirmation dialog
  - Interactive test suite selection
  - Live test result display

## Key Operations

### 1. runTests
Execute test suites with optional dry-run mode:
```graphql
mutation RunTests($suite: String!, $dryRun: Boolean!) {
  runTests(suite: $suite, dryRun: $dryRun) {
    id
    status
    results { name status }
  }
}
```

### 2. scaffoldPage
Generate components from templates:
```graphql
mutation ScaffoldPage($template: String!, $name: String!, $path: String!) {
  scaffoldPage(template: $template, name: $name, path: $path, dryRun: true) {
    id
    preview
    files { path action content }
  }
}
```

### 3. runProgress
Subscribe to real-time progress updates:
```graphql
subscription RunProgress($runId: String!) {
  runProgress(runId: $runId) {
    progress
    message
    type
  }
}
```

## Dry-Run Workflow

1. **Preview Phase**: All operations default to `dryRun: true`
2. **Review**: User reviews what will be executed/created
3. **Confirmation**: Interactive dialog for approval
4. **Apply**: Execute actual changes with `applyChanges` mutation

## MCP Tool Exposure

The Apollo Router exposes GraphQL operations as MCP tools:

```yaml
tools:
  - name: "RunTests"
    operation: |
      mutation RunTests($suite: String!, $dryRun: Boolean!) {
        runTests(suite: $suite, dryRun: $dryRun) { ... }
      }
    parameters:
      suite: { type: string, required: true }
      dryRun: { type: boolean, default: true }
```

## Running the Stack

### Development Mode

1. **Start GraphQL Backend:**
```bash
cd backend/graphql
pip install -r requirements.txt
python server.py
```

2. **Start Apollo Router:**
```bash
cd apollo-mcp
npm install
npm run setup  # Downloads router binary
npm start
```

3. **Start Next.js Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Production Mode

Use the provided Docker Compose:
```bash
docker-compose -f docker-compose-integrated.yml up
```

## Integration Testing

Run the complete integration test:
```bash
./tests/integration/test-mcp-graphql.sh
```

## API Endpoints

- **GraphQL Backend:** http://localhost:4000/graphql
- **Apollo Router:** http://localhost:4001/graphql
- **WebSocket:** ws://localhost:4001/graphql/ws
- **Next.js Frontend:** http://localhost:3000
- **Health Checks:** http://localhost:{port}/health

## Security Features

- JWT authentication ready (configure JWKS_URL)
- CORS configuration for cross-origin requests
- Rate limiting per subgraph
- Request deduplication
- Input validation at schema level

## Monitoring & Observability

- Prometheus metrics at `/metrics`
- OpenTelemetry tracing support
- Apollo Studio integration ready
- Request/response logging with x-mcp-debug header

## Extension Points

1. **Add New Operations**: Update schema.graphql and server.py
2. **New MCP Tools**: Add to router.yaml plugins.mcp.expose.tools
3. **Custom Templates**: Extend ScaffoldTemplate in server
4. **Authentication**: Configure JWT in router.yaml
5. **Caching**: Redis configuration in supergraph.yaml

## Best Practices

1. Always start with dry-run mode
2. Use subscriptions for long-running operations
3. Batch mutations when possible
4. Implement proper error boundaries in UI
5. Use Apollo Client cache for optimistic updates

## Troubleshooting

- **Connection refused**: Check all services are running
- **Subscription not working**: Verify WebSocket URL
- **CORS errors**: Check CORS configuration in server.py
- **Rate limiting**: Adjust capacity in router.yaml

## Next Steps

1. Add authentication/authorization
2. Implement more scaffold templates
3. Add test coverage reporting
4. Integrate with CI/CD pipelines
5. Add monitoring dashboards