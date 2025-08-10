# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an STT (Speech-to-Text) proxy service that acts as a middleware layer for MLX Omni Server. The primary function is to add default parameters to Whisper STT requests to improve transcription accuracy, particularly for Chinese language content.

## Architecture

The service is built with:
- **Python 3.x** with FastAPI for the web framework
- **httpx** for HTTP client operations with connection pooling
- **JSON configuration system** for parameter override management

### Core Architecture Components

1. **Request Interceptor** (`src/stt_proxy.py:183-192`): Routes STT requests to parameter enhancement handler
2. **Parameter Override Engine** (`src/stt_proxy.py:194-246`): Forcefully applies config-defined parameters regardless of client values
3. **Transparent Proxy** (`src/stt_proxy.py:92-174`): Forwards non-STT requests unchanged to target service
4. **Connection Pool Management** (`src/stt_proxy.py:55-82`): HTTP/2 enabled client with keep-alive optimization

### Critical Parameter Handling Logic

The service implements a **forced override** strategy where configuration file parameters take absolute precedence:
- Config-defined fields: **Always override** client values (logged as "强制设置参数")
- Non-config fields: **Passthrough** client values unchanged
- File uploads: Preserved and forwarded correctly

## Development Commands

### Service Management
```bash
# Quick start (uses convenience scripts)
./start_service.sh

# Manual start with default config
python src/stt_proxy.py

# Start with custom parameters
python src/stt_proxy.py --host 0.0.0.0 --port 10241 --target-host localhost --target-port 10240 --config config/stt_config.json
```

### Testing
```bash
# Run all tests (uses convenience script)
./run_tests.sh

# Run specific test categories
PYTHONPATH=src python tests/test_config_override.py    # Parameter override logic
PYTHONPATH=src python tests/test_full_functionality.py # End-to-end functionality
PYTHONPATH=src python tests/test_curl_simulation.py    # curl command simulation
```

### Development Workflow
```bash
# Test configuration changes
python tests/test_config_override.py

# Test with real audio files
python scripts/send_test_request.py

# Monitor service health
curl http://localhost:10241/health
curl http://localhost:10241/stats
```

## Configuration Management

### Core Configuration (config/stt_config.json)
The current configuration is **heavily optimized** for Chinese dual-speaker business conversations:

- `temperature: 0.1` - Critical for stability (even 0.02 difference causes dramatic output changes)
- `suppress_tokens: [50364]` - Suppresses known problematic tokens
- `condition_on_previous_text: false` - Prevents cumulative errors
- `initial_prompt` - Multi-turn conversation template with speaker identification

### MLX Omni Server Compatibility
**Important**: MLX Omni Server strictly follows official Whisper API spec. These parameters are **NOT supported**:
- Temperature arrays (causes 422 errors)
- Sampling parameters (`top_p`, `num_beams`, `early_stopping`)
- Custom repetition control parameters
- VAD-related parameters
- Third-party extensions

### Environment Variables
- `STT_CONFIG_PATH`: Configuration file path (default: config/stt_config.json)
- `TARGET_HOST`: Target MLX Omni Server host (default: localhost)
- `TARGET_PORT`: Target MLX Omni Server port (default: 10240)

## Key Implementation Details

### Request Flow Architecture
1. **Client Request** → **Proxy:10241** → **Parameter Enhancement** → **MLX Omni Server:10240**
2. STT requests are intercepted and processed through `handle_stt_request()`
3. Non-STT requests bypass parameter processing via `forward_request()`

### File Upload Handling
The service correctly handles multipart/form-data with files:
- Extracts file from form data
- Preserves file metadata (filename, content-type)
- Forwards file content as binary data
- Maintains form field separation

### Deployment Integration
- **macOS LaunchDaemon**: `deployment/com.sttproxy.plist` configured for system service
- **Logging**: Structured logging to `logs/` directory with separate error streams
- **Process Management**: Proper startup/shutdown event handling with resource cleanup

## Testing Strategy

The test suite covers:
- **Configuration Override Logic**: Validates forced parameter replacement behavior
- **HTTP Request Simulation**: Tests actual multipart/form-data handling
- **End-to-End Functionality**: Validates complete request processing pipeline
- **Edge Cases**: Empty configs, partial field matches, client-only parameters

When modifying parameter handling logic, always run the full test suite to ensure override behavior remains consistent.