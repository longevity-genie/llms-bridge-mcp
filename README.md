# llms-bridge-mcp

[![Tests](https://github.com/winternewt/llm-bridge-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/winternewt/llm-bridge-mcp/actions/workflows/test.yml)

An MCP (Model Context Protocol) server to bridge LLM calls via the Model Context Protocol.

## Installation

Using `uv`:

```bash
uv sync
```

## Usage

### Run in stdio mode (for MCP clients like Claude Desktop)

```bash
uv run stdio
```

### Run in HTTP mode

```bash
uv run http --host 0.0.0.0 --port 3011
```

### Run in SSE mode

```bash
uv run sse --host 0.0.0.0 --port 3011
```

### Development mode with Smithery

```bash
uv run dev
```

## Available Tools

The server dynamically registers tools based on available API keys. Each tool allows you to call different LLM providers through the MCP interface.

### llms_bridge_call_chatgpt

Call OpenAI ChatGPT models with a prompt.

**Parameters:**
- `prompt` (str, required): The user message/prompt
- `model` (str, optional): OpenAI model (default: gpt-4o)
- `temperature` (float, optional): Sampling temperature 0-2 (default: 0.7)
- `max_tokens` (int, optional): Maximum response tokens (default: 4096)
- `system_message` (str, optional): System message for context

**Requires:** `OPENAI_API_KEY` environment variable

### llms_bridge_call_gemini

Call Google Gemini models with a prompt.

**Parameters:**
- `prompt` (str, required): The user message/prompt
- `model` (str, optional): Gemini model (default: gemini/gemini-2.0-flash-exp)
- `temperature` (float, optional): Sampling temperature 0-2 (default: 0.7)
- `max_tokens` (int, optional): Maximum response tokens (default: 4096)
- `system_message` (str, optional): System message for context

**Requires:** `GEMINI_API_KEY` environment variable

### llms_bridge_call_perplexity

Call Perplexity Sonar API for web-search enhanced responses.

**Parameters:**
- `prompt` (str, required): The user message/prompt
- `model` (str, optional): Perplexity model (default: perplexity/sonar)
- `temperature` (float, optional): Sampling temperature 0-2 (default: 0.7)
- `max_tokens` (int, optional): Maximum response tokens (default: 4096)
- `system_message` (str, optional): System message for context

**Requires:** `PERPLEXITY_API_KEY` environment variable

### llms_bridge_server_misconfiguration

This tool appears when no API keys are configured. It provides setup instructions.

## Long-Running Task Support

The server implements FastMCP's long-running task pattern, allowing LLM calls to be handled asynchronously when they may take extended time to complete.

### Task Mode

The server is configured with `task_mode="optional"` by default, which provides:

- **Fallback behavior**: Supports async task patterns when the client supports them, but falls back to synchronous operation if not
- **Polling support**: Tasks are polled every 10 seconds (configurable via `poll_interval` parameter)
- **Better UX**: Prevents timeouts on long-running LLM operations while maintaining compatibility with all MCP clients

### How It Works

When a tool is called:
1. If the client supports MCP task protocol, the operation runs asynchronously
2. The server polls the task status at the configured interval
3. Results are returned when the LLM completes its response
4. If the client doesn't support tasks, operations run synchronously as normal

This is particularly useful for:
- Complex prompts that require extended reasoning
- Large context processing
- Multi-step LLM operations
- Clients that may have shorter timeout windows

### Configuration

To customize task behavior, you can instantiate the server with different parameters:

```python
from llms_bridge_mcp.server import LLMBridgeMCP
from datetime import timedelta

# Optional mode (default) - fallback to sync if client doesn't support tasks
server = LLMBridgeMCP(task_mode="optional", poll_interval=timedelta(seconds=10))

# Required mode - force task-based operation
server = LLMBridgeMCP(task_mode="required", poll_interval=timedelta(seconds=5))

# Disabled mode - always synchronous
server = LLMBridgeMCP(task_mode="disabled")
```

## Testing

Run tests with pytest:

```bash
uv run pytest
```

## Configuration

### Environment Variables

Set the following environment variables:

- `OPENAI_API_KEY`: OpenAI API key for ChatGPT models
- `GEMINI_API_KEY`: Google Gemini API key
- `PERPLEXITY_API_KEY`: Perplexity API key
- `MCP_HOST`: Host to bind to (default: 0.0.0.0)
- `MCP_PORT`: Port to bind to (default: 3011)
- `MCP_TRANSPORT`: Transport type (default: streamable-http)

### MCP Configuration Files

The project includes several MCP configuration files for different deployment modes:

- `mcp-config-stdio.json` - Stdio mode for Claude Desktop and other MCP clients (uses uvx)
- `mcp-config-http.json` - HTTP mode for REST API integration
- `mcp-config-sse.json` - SSE (Server-Sent Events) mode for streaming
- `mcp-config-local.json` - Local development using uv

#### Using with Claude Desktop

For published package via uvx (recommended):
```bash
# Linux/Mac
cat mcp-config-stdio.json >> ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Or manually merge the JSON content into your Claude config
```

For local development:
```bash
# Update the path in mcp-config-local.json to match your installation
# Then merge into Claude config
cat mcp-config-local.json >> ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

#### Running via uvx (without installation)

```bash
# Stdio mode
uvx --prerelease=allow llms-bridge-mcp@latest stdio

# HTTP mode
uvx --prerelease=allow llms-bridge-mcp@latest http --host 0.0.0.0 --port 3011

# SSE mode
uvx --prerelease=allow llms-bridge-mcp@latest sse --host 0.0.0.0 --port 3011
```

## Features

- **Dynamic Tool Registration**: Tools are automatically registered based on available API keys
- **Multiple LLM Providers**: Support for OpenAI, Google Gemini, and Perplexity
- **Flexible Deployment**: Run via stdio, HTTP, or SSE transports
- **MCP Protocol**: Full Model Context Protocol support via FastMCP
- **LiteLLM Integration**: Unified interface to multiple LLM providers
- **Graceful Shutdown**: Proper signal handling for clean exits
- **Long-Running Task Support**: Optional async task handling with polling for extended LLM operations

## Project Structure

```
llms-bridge-mcp/
├── src/
│   └── llms_bridge_mcp/
│       ├── __init__.py
│       └── server.py           # Main server implementation
├── tests/
│   └── test_server.py          # Test suite including real LLM calls
├── mcp-config-stdio.json       # Stdio mode for Claude Desktop
├── mcp-config-http.json        # HTTP mode config
├── mcp-config-sse.json         # SSE mode config
├── mcp-config-local.json       # Local development config
├── .env.template               # Environment variables template
├── pyproject.toml              # Project dependencies and metadata
└── README.md
```
