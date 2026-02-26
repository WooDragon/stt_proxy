# 运维手册

## 服务架构

单进程架构，mlx-audio server 直接提供 OpenAI 兼容的 ASR API。

```
Client → mlx-audio server:10241 (Qwen3-ASR, 直接推理)
```

项目包含本地 monkey-patch（`patches/`）修复 mlx-audio 已知 bug，通过 `run_server.py` 在启动时注入。

## 部署方式

### 方式一：手动启动（开发和临时使用）

```bash
./start_service.sh   # 后台启动 + 健康检查 + 模型预热
./stop_service.sh    # 优雅停止
```

启动脚本逻辑：
1. 读取 `.env` 配置
2. 检查端口占用
3. `nohup` 后台启动 mlx-audio server
4. 轮询 `/v1/models` 等待就绪（最多 60s）
5. 发送静音 WAV 预热模型
6. PID 写入 `logs/asr_service.pid`

日志位置：`logs/asr_service.log`

### 方式二：macOS LaunchDaemon（系统服务）

安装：

```bash
cp deployment/com.sttproxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.sttproxy.plist
```

卸载：

```bash
launchctl unload ~/Library/LaunchAgents/com.sttproxy.plist
rm ~/Library/LaunchAgents/com.sttproxy.plist
```

特性：
- 开机自启（`RunAtLoad`）
- 崩溃自动重启（`KeepAlive`）
- 通过 `scripts/run_server.sh` wrapper 启动，解决 launchd 极简 PATH 问题

**重要：手动启动和 launchd 互斥，不要同时使用。**

启动后手动预热模型：

```bash
./scripts/warmup.sh
```

## 模型管理

### 缓存位置

通过 `.env` 中的 `HF_HUB_CACHE=./models` 配置，模型缓存在项目本地 `models/` 目录下（已 gitignore）。

模型子目录命名格式：`models--mlx-community--Qwen3-ASR-1.7B-8bit`

### 模型选择

| 模型 | 磁盘大小 | 内存需求 | 适用场景 |
|------|----------|----------|----------|
| Qwen3-ASR-0.6B-8bit | ~1GB | ~2GB | 低内存设备 |
| Qwen3-ASR-1.7B-8bit | ~2.3GB | ~4GB | 推荐，精度与速度平衡 |

切换模型：修改 `.env` 中的 `MODEL` 变量，重启服务。

### 手动下载模型

如需预下载模型（离线环境）：

```bash
HF_HUB_CACHE=./models huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
```

## 监控与健康检查

### 可用性检查

```bash
curl -sf http://localhost:10241/v1/models && echo "OK" || echo "FAIL"
```

### 日志

- 手动启动日志：`logs/asr_service.log`
- LaunchDaemon 日志：
  - stdout: `logs/asr_service.log`
  - stderr: `logs/asr_service_error.log`

## 故障排查

### 端口占用

```bash
lsof -i :10241 -sTCP:LISTEN
```

### 模型下载失败

1. 检查网络连接
2. 设置代理：`HTTPS_PROXY=http://proxy:port`
3. 使用 HuggingFace 镜像：`HF_ENDPOINT=https://hf-mirror.com`

### 内存不足

Apple Silicon 使用统一内存，GPU 和 CPU 共享。1.7B-8bit 模型运行时约需 4GB 可用内存。

症状：服务启动后立即崩溃或首次推理时 OOM。

解决：
1. 关闭其他占用内存的应用
2. 换用 0.6B 模型

### 服务启动超时

`start_service.sh` 等待 60 秒。如果超时：
1. 检查日志 `logs/asr_service.log`
2. 确认 Python 环境正确（`which python` 应指向安装了 mlx-audio 的环境）
3. 手动启动测试：`python run_server.py --host 0.0.0.0 --port 10241 --workers 1`

## 性能参考

Apple Silicon 各芯片推理速度预期（Qwen3-ASR-1.7B-8bit）：

| 芯片 | 预期实时倍率 | 说明 |
|------|-------------|------|
| M1 | ~5-8x | 基础性能 |
| M1 Pro/Max | ~8-12x | 更多 GPU 核心 |
| M2/M3 | ~10-15x | 架构改进 |
| M4 Pro/Max | ~15-20x | 最新架构 |

实时倍率 = 音频时长 / 处理时间。10x 表示 10 秒音频约 1 秒处理完。

## 本地 Patches

`patches/` 目录包含对 mlx-audio 的 monkey-patch，通过 `run_server.py` 在启动时注入：

| Patch | 修复 | 影响版本 | 移除条件 |
|-------|------|----------|----------|
| `audio_io_fix.py` | M4A/AAC 文件 pipe stdin 不可 seek 导致 ffmpeg 解码 0 采样点 | <=0.3.1 | 上游 `_decode_ffmpeg` 改为 tempfile |
| `generate_enhance.py` | language 默认 Chinese（原 English）；注入 repetition_penalty 抑制重复幻觉；回填 STTOutput.language | <=0.3.1 | 上游修正默认 language 且回填 output.language |
| `vad_chunking.py` | 用 webrtcvad 在 >=500ms 静音处切分音频（原能量最低点切割常把句子切断） | <=0.3.1 | 上游 `split_audio_into_chunks` 采用 VAD 分段 |

**Patch 管理**：`patches/__init__.py` 的 `apply_all()` 统一加载所有 patch，`run_server.py` 调用后再启动 server。各 patch 目标不同模块，加载顺序无关。

**启动链路**：`start_service.sh` / `scripts/run_server.sh` → `run_server.py`（`patches.apply_all()`）→ `mlx_audio.server`

**`--workers 1` 约束**：mlx-audio 默认 `workers=2`，但 macOS Python 的 `multiprocessing` 使用 spawn 模式（非 fork），子进程不继承 monkey-patch。单 GPU 推理也无需多 worker，因此强制 `--workers 1`。

### generate_enhance.py 运维

`REPETITION_PENALTY` 环境变量控制重复抑制强度（`.env` 中配置）：

| 值 | 效果 |
|----|------|
| `1.0` | 禁用（不做惩罚） |
| `1.1` | 轻度抑制（对高密度对话效果有限） |
| `1.2` | 推荐值（单字幻觉大幅抑制，正常叠词不受影响） |
| `1.3+` | 激进抑制（未充分测试，可能影响正常重复） |

修改后重启服务生效。

### vad_chunking.py 运维

VAD 分段参数为代码内置常量（无需运维调参）：

| 参数 | 值 | 说明 |
|------|----|------|
| VAD 模式 | 3（最激进） | webrtcvad 0-3，越高越激进判断非语音 |
| 帧长 | 30ms | webrtcvad 标准帧长 |
| 静音阈值 | 500ms | >= 500ms 连续非语音才视为切分点 |
| 采样率 | 16kHz | VAD 分析固定 16kHz（非 16kHz 输入自动重采样） |

chunk_duration（HTTP API 参数，server 默认 30s）作为单段绝对上限，VAD 在其范围内找最近静音切点。无静音切点时回退到原始能量分段。

### 已知上游默认值限制

`server.py` HTTP endpoint 默认 `max_tokens=128`（`generate()` 本身默认 8192）。对长段转写可能截断。无法在 patch 中安全覆盖（无法区分用户显式传值 vs server 默认）。如遇截断，客户端请求时显式传 `max_tokens=2048` 或更高。

## 升级流程

```bash
# 1. 停止服务
./stop_service.sh

# 2. 升级 mlx-audio
pip install -r requirements.txt --upgrade

# 3. 检查 patches 是否仍需要（查看 CHANGELOG 或测试 m4a 文件）
# 如上游已修复，可移除 patches/ 和 run_server.py，恢复 python -m mlx_audio.server

# 4. 启动服务
./start_service.sh
```

模型权重缓存在 HuggingFace cache 中，升级 mlx-audio 不会重新下载模型（除非模型本身有更新）。
