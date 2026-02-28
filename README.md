# 本地中文语音转写服务

基于 [mlx-audio](https://github.com/Blaizzy/mlx-audio) + [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR) 的本地语音转写服务，专为 Apple Silicon Mac 优化。

## 系统要求

- macOS（Apple Silicon：M1/M2/M3/M4 系列）
- Python 3.11+
- 约 4GB 可用内存（统一内存）

## 安装

```bash
pip install -r requirements.txt
```

首次使用时，模型会自动从 HuggingFace 下载到项目本地 `models/` 目录（约 2.3GB）。

## 快速启动

```bash
# 启动服务（后台运行 + 自动预热模型）
./start_service.sh

# 停止服务
./stop_service.sh
```

## API 使用

服务提供 OpenAI 兼容的 API 接口。

### 转写音频

```bash
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit
```

### 多说话人分离（Speaker Diarization）

```bash
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F diarize=true
```

首次使用前需转换 speaker embedding 模型（一次性）：

```bash
pip install onnx onnx2torch
python scripts/convert_wespeaker.py
```

响应格式：

```json
{
  "text": "转写的文本内容",
  "segments": [
    {"text": "分段文本", "start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

### 查看已加载模型

```bash
curl http://localhost:10241/v1/models
```

## 配置

环境变量在 `.env` 文件中配置（参考 `.env.example`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `mlx-community/Qwen3-ASR-1.7B-8bit` | HuggingFace 模型名 |
| `PORT` | `10241` | 服务端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `HF_HUB_CACHE` | `./models` | 模型缓存目录 |
| `REPETITION_PENALTY` | `1.2` | 重复惩罚系数（1.0=禁用） |
| `WESPEAKER_CACHE` | `./models/wespeaker` | CoreML speaker 模型目录 |
| `DIARIZE_THRESHOLD` | `0.7` | 聚类阈值（越小→越多 speaker） |

### 可选模型

| 模型 | 大小 | 适用场景 |
|------|------|----------|
| `mlx-community/Qwen3-ASR-0.6B-8bit` | ~1GB | 低内存设备，速度优先 |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | ~2.3GB | 推荐，精度与速度平衡 |

## 支持的音频格式

mlx-audio 通过 ffmpeg 解码，支持常见音频格式：WAV、MP3、FLAC、M4A、OGG 等。

## 测试

```bash
# 端到端测试（需要服务运行中）
pytest tests/ -v

# 发送测试请求
python scripts/send_test_request.py audio.wav
```

## 常见问题

**Q: 首次请求很慢？**
A: 首次请求触发模型加载到 GPU 内存（约 10-30s）。`start_service.sh` 包含自动预热，后续请求正常速度。

**Q: 如何切换模型？**
A: 修改 `.env` 中的 `MODEL` 变量，重启服务即可。

**Q: 内存不够？**
A: 1.7B-8bit 需要约 4GB 可用内存。如果不够，换用 0.6B 模型。

**Q: 模型下载失败？**
A: 检查网络连接。如需代理，设置 `HTTPS_PROXY` 环境变量。也可使用 HuggingFace 镜像：`HF_ENDPOINT=https://hf-mirror.com`。

详细运维指南见 [docs/operations.md](docs/operations.md)。
