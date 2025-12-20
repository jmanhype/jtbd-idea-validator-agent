# JTBD MCP Server Authentication

Simple bearer token authentication for the JTBD MCP server.

## Configuration

Set the API key via environment variable:

```bash
export JTBD_API_KEY="your-secret-key-here"
```

Default value: `dev-key-jtbd`

## Usage

### Starting the Server

```bash
# With default dev key
python server.py --http

# With custom key
JTBD_API_KEY="production-key" python server.py --http
```

### Making Authenticated Requests

Include the Bearer token in the Authorization header:

```bash
# Using curl
curl -H "Authorization: Bearer dev-key-jtbd" \
  http://localhost:8001/sse

# Using Python requests
import requests

headers = {
    "Authorization": "Bearer dev-key-jtbd"
}
response = requests.get("http://localhost:8001/sse", headers=headers)
```

### MCP Client Configuration

For Claude Desktop or other MCP clients, configure the API key in your MCP settings:

```json
{
  "mcpServers": {
    "jtbd-validator": {
      "url": "http://localhost:8001/sse",
      "headers": {
        "Authorization": "Bearer dev-key-jtbd"
      }
    }
  }
}
```

## Security Features

- **Timing-safe comparison**: Uses `secrets.compare_digest()` to prevent timing attacks
- **Standard Bearer scheme**: Follows RFC 6750 Bearer Token specification
- **Environment-based configuration**: API key stored in environment variables, not code
- **Clear error messages**: Helpful messages for missing or invalid tokens

## Error Responses

### Missing Authorization Header

```json
{
  "error": "Unauthorized",
  "message": "Missing or invalid Authorization header. Expected: Bearer <token>"
}
```

### Invalid API Key

```json
{
  "error": "Unauthorized",
  "message": "Invalid API key"
}
```

## Implementation Details

The authentication middleware is based on the hextropian-regulatory reference implementation:

- Simple bearer token validation
- No rate limiting (yet)
- Middleware-based approach for clean separation of concerns
- Easy to extend with additional security features

## Future Enhancements

Potential additions:

- Rate limiting per client/token
- Multiple API keys with different permissions
- Token expiration and rotation
- Audit logging for authentication attempts
- IP allowlisting/denylisting
