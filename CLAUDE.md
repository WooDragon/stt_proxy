# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

本地中文语音转写服务，基于 mlx-audio + Qwen3-ASR，运行在 Apple Silicon Mac 上。零自定义代码——mlx-audio 原生提供 OpenAI 兼容的 FastAPI server，项目只包含启动脚本、部署配置和测试。

## Architecture

单进程架构，无代理层：

```
Client → mlx-audio server:10241 (Qwen3-ASR, 直接推理)
```

核心依赖：
- **mlx-audio** >=0.3.1：提供 FastAPI server + STT 推理
- **Qwen3-ASR-1.7B-8bit**：中文 ASR 模型（HuggingFace，mlx-community 量化版）

### 历史背景

本项目前身是 STT Proxy (FastAPI:10241) → MLX Omni Server (Whisper:10240) 的双进程代理模式。因 Qwen3-ASR 中文 WER 远优于 Whisper（4.97% vs 15.30%），且 mlx-audio 自带 OpenAI 兼容 server，已简化为单进程直连架构。旧的代理逻辑、参数覆盖引擎、Whisper 配置已全部删除。

## File Structure

```
├── .env                    # 环境配置（MODEL, PORT, HOST）
├── .gitignore
├── requirements.txt        # Python 依赖
├── start_service.sh        # 手动启动（后台 nohup + 健康检查 + 预热）
├── stop_service.sh         # 手动停止（SIGTERM + wait + SIGKILL）
├── assets/
│   └── silence.wav         # 0.5s 静音 WAV，模型预热用
├── scripts/
│   ├── warmup.sh           # 模型预热（发送 silence.wav 触发首次加载）
│   ├── run_server.sh       # launchd wrapper（source .env + 绝对路径 python + exec 前台）
│   └── send_test_request.py # 手动转写测试
├── deployment/
│   └── com.sttproxy.plist  # macOS LaunchAgent 配置
├── tests/
│   └── test_service.py     # 黑盒端到端测试（需服务运行）
├── docs/
│   └── operations.md       # 运维手册
└── logs/                   # 运行时日志（.gitignore 排除）
```

## Development Commands

### Service Management

```bash
# 启动服务（后台 + 健康检查 + 模型预热）
./start_service.sh

# 停止服务
./stop_service.sh

# 手动前台启动（调试用）
python -m mlx_audio.server --host 0.0.0.0 --port 10241
```

### Testing

```bash
# 端到端测试（需要服务运行中）
pytest tests/ -v

# 手动测试转写
python scripts/send_test_request.py audio.wav
```

### Health Check

```bash
curl http://localhost:10241/v1/models
```

## Configuration

所有配置在 `.env` 文件中：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `mlx-community/Qwen3-ASR-1.7B-8bit` | HuggingFace 模型名 |
| `PORT` | `10241` | 服务端口 |
| `HOST` | `0.0.0.0` | 监听地址 |

## Deployment

两种部署方式互斥：

1. **手动启动**：`./start_service.sh`（nohup 后台模式）
2. **macOS LaunchAgent**：`deployment/com.sttproxy.plist`（前台 exec，launchd 管理生命周期）

详细运维指南见 `docs/operations.md`。

## Testing Strategy

`tests/test_service.py` 是纯黑盒端到端测试：
- 不 mock 第三方库内部
- 直接向运行中的服务发 HTTP 请求
- 服务未运行时自动跳过（`pytest.mark.skipif`）
- 标记为 `@pytest.mark.integration`

测试覆盖：
- `GET /v1/models` 返回正确 JSON
- `POST /v1/audio/transcriptions` 接受 multipart/form-data 返回 `{"text": "..."}`
- 参数校验（缺少 model/file 返回 422）

## Key Notes

- **model 参数是必填的**：mlx-audio server 要求每个请求传 `model` 字段（HuggingFace 模型名）
- **模型首次加载慢**：首次请求触发模型下载和 GPU 加载（10-30s），`start_service.sh` 包含预热步骤
- **HuggingFace 缓存**：模型缓存在 `~/.cache/huggingface/hub/`
