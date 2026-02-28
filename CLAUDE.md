# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

本地中文语音转写服务，基于 mlx-audio + Qwen3-ASR，运行在 Apple Silicon Mac 上。mlx-audio 原生提供 OpenAI 兼容的 FastAPI server，项目包含启动脚本、部署配置、测试和本地 monkey-patch（修复 mlx-audio 已知 bug）。

## Architecture

单进程架构，无代理层：

```
请求到达（audio file）
│
├─ _spool_to_tempfile（async 64KB 分块写磁盘 + 实时大小熔断 → 413）
│
├─ diarize=false（默认）:
│   └─ model.generate(tempfile) → [单次 decode] → [VAD chunking] → [Qwen3-ASR MLX GPU] → NDJSON
│
└─ diarize=true:
    ├─ ASR: model.generate(tempfile) → [单次 decode] → STTOutput
    ├─ Diarization: tempfile → torchaudio → VAD → sliding window
    │   → fbank+CMN → [CoreML WeSpeaker GPU/ANE] → AHC clustering
    └─ 对齐: 时间加权投票 → JSONResponse (含 speaker 标签)
```

核心依赖：
- **mlx-audio** >=0.3.1：提供 FastAPI server + STT 推理
- **Qwen3-ASR-1.7B-8bit**：中文 ASR 模型（HuggingFace，mlx-community 量化版，缓存在项目本地 `models/`）
- **WeSpeaker ResNet34-LM**：speaker embedding 模型（CoreML FP16，离线转换，缓存在 `models/wespeaker/`）
- **scikit-learn**：AHC 聚类（speaker diarization）

硬件分工：
| 组件 | 硬件 | 框架 |
|------|------|------|
| ASR | GPU (Metal) | MLX |
| Speaker Embedding | GPU/ANE | CoreML |
| VAD / fbank / 聚类 | CPU | webrtcvad / torchaudio / sklearn |

### 历史背景

本项目前身是 STT Proxy (FastAPI:10241) → MLX Omni Server (Whisper:10240) 的双进程代理模式。因 Qwen3-ASR 中文 WER 远优于 Whisper（4.97% vs 15.30%），且 mlx-audio 自带 OpenAI 兼容 server，已简化为单进程直连架构。旧的代理逻辑、参数覆盖引擎、Whisper 配置已全部删除。

## File Structure

```
├── .env                    # 环境配置（MODEL, PORT, HOST, HF_HUB_CACHE, WESPEAKER_CACHE, DIARIZE_THRESHOLD, MAX_UPLOAD_BYTES）
├── .gitignore
├── requirements.txt        # Python 依赖（mlx-audio, scikit-learn）
├── run_server.py           # 启动入口（patches → diarize setup → uvicorn.run）
├── start_service.sh        # 手动启动（后台 nohup + 健康检查 + 预热）
├── stop_service.sh         # 手动停止（SIGTERM + wait + SIGKILL）
├── patches/
│   ├── __init__.py         # apply_all() 统一加载所有 monkey-patches
│   ├── audio_io_fix.py     # M4A/AAC pipe seek bug 修复（mlx-audio <=0.3.1）
│   ├── generate_enhance.py # generate() 默认参数修正 + 重复抑制 + language 回填
│   ├── vad_chunking.py     # VAD 驱动音频分段（替换能量分段）
│   ├── speaker_diarization.py # SpeakerDiarizer（CoreML embedding + AHC 聚类）
│   └── server_diarize.py   # 端点包装：direct-to-model + tempfile spool + diarize + 签名校验
├── assets/
│   └── silence.wav         # 0.5s 静音 WAV，模型预热用
├── scripts/
│   ├── warmup.sh           # 模型预热（发送 silence.wav 触发首次加载）
│   ├── run_server.sh       # launchd wrapper（source .env + 绝对路径 python + exec 前台）
│   ├── send_test_request.py # 手动转写测试
│   └── convert_wespeaker.py # 离线模型转换（ONNX → CoreML FP16，一次性）
├── deployment/
│   └── com.sttproxy.plist  # macOS LaunchAgent 配置
├── tests/
│   ├── test_service.py     # 黑盒端到端测试（需服务运行）
│   └── test_patches.py     # patches 单元测试（无需服务运行）
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

# 手动前台启动（调试用，必须通过 run_server.py 启动以加载 patches）
python run_server.py --host 0.0.0.0 --port 10241 --workers 1
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
| `HF_HUB_CACHE` | `./models` | 模型缓存目录（脚本自动展开为绝对路径） |
| `REPETITION_PENALTY` | `1.2` | 重复惩罚系数（1.0=禁用，中文建议 1.1-1.3） |
| `WESPEAKER_CACHE` | `./models/wespeaker` | CoreML speaker embedding 模型目录 |
| `DIARIZE_THRESHOLD` | `0.75` | AHC cosine distance 聚类阈值 |
| `MAX_UPLOAD_BYTES` | `524288000`（500MB） | 单次上传大小上限，超限返回 413 |
| `PYTHON_BIN` | （空，使用 PATH） | Python 解释器绝对路径（launchd/conda 环境下建议设置） |
| `EXTRA_PATH` | `/opt/homebrew/bin:/usr/local/bin` | 额外 PATH 前缀，start_service.sh 和 run_server.sh 共用 |

## Deployment

两种部署方式互斥：

1. **手动启动**：`./start_service.sh`（nohup 后台模式）
2. **macOS LaunchAgent**：`deployment/com.sttproxy.plist`（前台 exec，launchd 管理生命周期）

详细运维指南见 `docs/operations.md`。

## Testing Strategy

两层测试：

### 单元测试（`tests/test_patches.py`）
- 无需运行中的服务
- 测试 patch 逻辑隔离正确性：VAD 切分边界、尾帧安全、默认参数注入、language 回填、speaker 对齐、滑动窗口、AHC 聚类、tempfile spool（含 413 熔断）、gen_kwargs 动态过滤、文件路径输入
- `pytest tests/test_patches.py -v`

### 端到端测试（`tests/test_service.py`）
- 纯黑盒：不 mock 第三方库内部，直接向运行中的服务发 HTTP 请求
- 服务未运行时自动跳过（`pytest.mark.skipif`）
- 标记为 `@pytest.mark.integration`
- 覆盖：`GET /v1/models`、`POST /v1/audio/transcriptions`、参数校验（422）、diarize=true/false、stream+diarize 400、413 上传超限

## Key Notes

- **model 参数是必填的**：mlx-audio server 要求每个请求传 `model` 字段（HuggingFace 模型名）
- **模型首次加载慢**：首次请求触发模型下载和 GPU 加载（10-30s），`start_service.sh` 包含预热步骤
- **模型缓存**：通过 `HF_HUB_CACHE` 配置在项目本地 `models/` 目录（已 gitignore）
- **必须通过 run_server.py 启动**：直接 `python -m mlx_audio.server` 不会加载 patches 和 diarization 端点
- **--workers 1 是必须的**：macOS Python multiprocessing 使用 spawn 模式，多 worker 进程不继承 monkey-patch；且单 GPU 推理多 worker 无收益。`run_server.py` 有硬防护。
- **run_server.py 已弃用 runpy**：改为 `import mlx_audio.server` + `setup_diarize(app)` + `uvicorn.run(app)`。这提供了 import 和 serve 之间的控制窗口来注入 diarization 路由。
- **max_tokens 默认值**：`server.py` 原始 endpoint 默认 `max_tokens=128`，但 `server_diarize.py` wrapper 已覆盖为 `Form(8192)`（与 `generate()` 本身一致），长音频不截断。客户端显式传值时以客户端为准。
- **Diarization 需要预转换模型**：首次使用前必须运行 `python scripts/convert_wespeaker.py` 转换 CoreML 模型。模型缺失时 diarize=true 请求会降级返回纯 ASR 结果 + `diarization_error` 字段。
- **stream + diarize 不兼容**：`stream=true` + `diarize=true` 返回 HTTP 400。Diarization 需要完整 ASR 结果后才能做时间对齐。
- **422 vs 500 错误区分**：wrapper 层捕获 `generate()` 阶段的异常，坏音频文件（格式不支持、解码失败）返回 HTTP 422 + `"Audio decode failed: ..."` detail，客户端可安全跳过；服务端配置错误（ffmpeg 缺失等）和未知异常保持 500，客户端应重试。分类器使用精确消息匹配 + 保守默认 500，避免 false positive。
- **Direct-to-model 架构**：`server_diarize.py` 不再委托原始 handler（`stt_transcriptions`），而是直接调用 `model_provider.load_model()` + `stt_model.generate(tempfile_path)`。消除了原始 handler 的 decode→reencode→decode 路径（double-decode），大文件内存占用从 ~13GB 降至 ~6.6GB 峰值。
- **Upload 流式 spool**：`_spool_to_tempfile` 用 async 64KB 分块读写 + 实时字节计数，超过 `MAX_UPLOAD_BYTES` 立即中断 + 清理 + HTTP 413。不将整个文件拉进内存。
