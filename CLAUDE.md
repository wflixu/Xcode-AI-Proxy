# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Xcode AI Proxy is a FastAPI-based proxy server that enables Xcode to use Chinese AI models (Zhipu GLM, Kimi, DeepSeek, QWen) that are not natively supported. It presents an OpenAI-compatible API interface that Xcode can connect to, while internally routing requests to the appropriate Chinese AI provider.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment (requires uv)
uv venv
source .venv/bin/activate
uv sync
```

### Running the Server
```bash
# Start the proxy server (default port: 8899)
python3 server.py

# Custom port/host
python3 server.py --port 3000 --host 0.0.0.0
```

### Configuration
Copy `.env.example` to `.env` and configure API keys:
- `ZHIPU_API_KEY` - For GLM-4.6 model (from https://open.bigmodel.cn/)
- `KIMI_API_KEY` - For Kimi K2 model (from https://platform.moonshot.cn/)
- `DEEPSEEK_API_KEY` - For DeepSeek models (from https://platform.deepseek.com/)

## Architecture

### Single-File Design
The entire proxy logic is contained in [server.py](server.py). This is intentional for simplicity.

### Model Configuration ([server.py:56-105](server.py#L56-L105))
Models are dynamically loaded based on available API keys. The `API_CONFIGS` dictionary is built at startup by checking which environment variables are set. Each model config includes:
- `api_url` - The provider's API endpoint
- `api_key` - From environment variable
- `type` - Provider identifier (zhipu/kimi/deepseek)
- `name` - Display name

### Request Flow
1. Xcode sends request to `/v1/chat/completions` (OpenAI-compatible)
2. [chat_completions()](server.py#L517-L563) validates request
3. [handle_proxy()](server.py#L445-L514) routes to provider-specific handler
4. Provider handlers ([handle_zhipu_request](server.py#L220-L261), [handle_kimi_request](server.py#L264-L305), [handle_deepseek_request](server.py#L356-L443)) forward to upstream APIs
5. Streaming responses are proxied directly without modification

### Key Components

**Retry Logic** ([with_retry](server.py#L135-L154))
- Configurable max retries and delay via env vars
- Exponential backoff (delay × attempt number)

**Message Sanitization** ([sanitize_messages](server.py#L308-L353))
- DeepSeek requires `message['content']` to be a string
- Converts complex content (arrays, objects) to JSON strings
- Called in handle_deepseek_request before proxying

**Provider-Specific Handling**
- Zhipu/Kimi: Direct proxy with minimal transformation
- DeepSeek: Filters unsupported parameters, sanitizes messages

## Adding New Models

To add a new AI provider:

1. Add environment variable check in `REQUIRED_ENV_VARS` ([server.py:45-49](server.py#L45-L49))
2. Add config to `API_CONFIGS` ([server.py:56-105](server.py#L56-L105))
3. Create handler function like `handle_<provider>_request()`
4. Route in `handle_proxy()` ([server.py:465-471](server.py#L465-L471))

## Testing in Xcode

Configure Xcode with:
- **Base URL**: `http://localhost:8899`
- **API Key**: `any-string-works` (not validated)
- **Models**: Will be auto-discovered via `/v1/models` endpoint
