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

### Apple Silicon 硬件监控

#### 快速状态（无需 sudo）

```bash
# 进程 CPU/内存
ps -p $(pgrep -f run_server.py) -o pid,pcpu,pmem,rss=

# GPU Metal 利用率（注意：只反映渲染管线，不含 MLX compute，见下方说明）
python3 -c "
import subprocess, re
r = subprocess.run(['ioreg', '-r', '-c', 'IOAccelerator', '-d', '1'], capture_output=True, text=True)
m = re.search(r'\"PerformanceStatistics\" = \{(.+?)\}', r.stdout)
if m:
    s = {k.strip('\" '): v.strip('\" ') for i in m.group(1).split(',') for k,v in [i.split('=',1)] if '=' in i}
    print(f'Device={s.get(\"Device Utilization %\")}%  VRAM={int(s.get(\"In use system memory\",0))//1024//1024}MB')
"

# ANE（Apple Neural Engine）激活状态
python3 -c "
import subprocess, re
r = subprocess.run(['ioreg', '-r', '-c', 'H11ANEIn', '-d', '1'], capture_output=True, text=True)
m = re.search(r'\"DevicePowerState\"=(\d+)', r.stdout)
print('ANE:', 'ACTIVE' if m and int(m.group(1)) > 0 else 'idle')
"
```

#### 持续监控脚本（60秒采样，200ms 间隔）

```bash
python3 - $(pgrep -f run_server.py) << 'EOF'
import subprocess, re, time, sys
PID = int(sys.argv[1])

def gpu():
    r = subprocess.run(['ioreg','-r','-c','IOAccelerator','-d','1'], capture_output=True, text=True)
    m = re.search(r'"PerformanceStatistics" = \{(.+?)\}', r.stdout)
    if not m: return 0, 0
    s = {k.strip(' "'): v.strip(' "') for i in m.group(1).split(',') for k,v in [i.split('=',1)] if '=' in i}
    return int(s.get('Device Utilization %',0)), int(s.get('In use system memory',0))//1024//1024

def ane():
    r = subprocess.run(['ioreg','-r','-c','H11ANEIn','-d','1'], capture_output=True, text=True)
    m = re.search(r'"DevicePowerState"=(\d+)', r.stdout)
    return int(m.group(1)) > 0 if m else False

def proc(pid):
    r = subprocess.run(['ps','-p',str(pid),'-o','pcpu=,pmem=,rss='], capture_output=True, text=True)
    p = r.stdout.strip().split()
    return (float(p[0]), float(p[1]), int(p[2])//1024) if p else (0,0,0)

print(f"{'时间':8s} {'GPU%':5s} {'VRAM':6s} {'ANE':6s} {'CPU%':6s} {'RSS MB':7s}")
for _ in range(300):
    dev, vram = gpu()
    ane_on = ane()
    cpu, mem, rss = proc(PID)
    flag = "← GPU" if dev > 0 else ("← ANE" if ane_on else "")
    print(f"{time.strftime('%H:%M:%S')} {dev:5d} {vram:5d}MB {'ACTIVE' if ane_on else 'idle':6s} {cpu:6.1f} {rss:7d}  {flag}")
    sys.stdout.flush()
    time.sleep(0.2)
EOF
```

#### GPU compute 利用率（需要 sudo）

`ioreg` 的 `Device Utilization %` 仅追踪 Metal **渲染管线**（图形），MLX 使用 Metal **Compute Shader**，不计入该指标。真实 GPU compute 占用需要：

```bash
# 需要 sudo，每秒采样一次，共 5 次
sudo powermetrics --samplers gpu_power -n 5 -i 1000
```

或打开 **Activity Monitor → Window → GPU History** 用 GUI 观察。

#### 硬件分工说明

| 组件 | 硬件 | 监控方式 |
|------|------|---------|
| Qwen3-ASR MLX 推理 | Metal GPU（所有 GPU 核心） | `sudo powermetrics` 或 Activity Monitor |
| WeSpeaker CoreML embedding | ANE（Apple Neural Engine） | `ioreg H11ANEIn DevicePowerState` |
| VAD / fbank / AHC 聚类 | CPU | `ps -o pcpu` |

**重要**：MLX 推理调度方式是单次推理横向铺满所有 GPU 核心，不支持多任务并行占不同核心。并发请求在 GPU 层串行执行，workers=1 是正确配置（详见并发限制章节）。

### 并发能力与扩容限制

**M1 8GB 配置下，单 worker 是客观最优：**

| 场景 | 分析 |
|------|------|
| workers=1（当前） | 请求串行处理，GPU 100% 利用率，内存可控 |
| workers=2（假设） | 需要 2×5-6GB RSS ≈ 12GB，超出 M1 8GB 物理内存上限，OOM |
| GPU 核心分配 | MLX 每次推理占满全部 GPU 核（Metal 横向 dispatch），不是"1核/任务" |
| Metal 并发 | 多进程争用 GPU 时 Metal command queue 串行执行，速度不提升 |

要支持多路并发转写，瓶颈是**内存**，不是 GPU 核心数：
- M1 Pro 16GB：理论支持 2 worker（各用 ~6GB）
- M1 Max 32GB / M1 Ultra 64GB：可支持 4-8 worker
- 即使内存充足，Metal GPU 命令队列仍串行，实际提速受限于 GPU 总吞吐

## HTTP 错误码参考

| 状态码 | 含义 | 客户端处理 | 示例场景 |
|--------|------|-----------|----------|
| 200 | 成功 | 正常处理 | — |
| 400 | 请求参数冲突 | 修正请求 | `stream=true` + `diarize=true` |
| 413 | 上传文件超限 | 分段或压缩 | 文件大小超过 `MAX_UPLOAD_BYTES`（默认 500MB） |
| 422 | 请求格式/内容无效 | 跳过该文件 | 缺少 `model` 参数；坏音频文件（格式不支持、解码失败） |
| 500 | 服务内部错误 | 等待后重试 | ffmpeg 未安装、模型加载失败、未知异常 |

422 中的音频解码错误 detail 以 `"Audio decode failed: "` 开头，包含具体异常信息。批量处理客户端可据此区分"跳过坏文件"和"等待服务恢复"。

413 的 detail 格式为 `"Upload exceeds N bytes limit"`。客户端应在上传前检查文件大小，或对超长录音做分段处理。

## 大文件转写最佳实践

### stream=false 的超时风险

`stream=false`（默认）在整个推理完成前不发送任何数据。254MB M4A（3-4 小时音频）推理需 26-42 分钟，期间客户端收不到任何字节。若客户端配置了 read timeout，连接会在推理完成前被客户端关闭，服务端随后发响应时遇 broken pipe——客户端显示 "Server disconnected"，access log 记录 200（服务端逻辑完全正确）。

**解决方案**：

```bash
# 推荐：stream=true（逐 chunk 接收，无超时风险）
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F "file=@long_audio.m4a" \
  -F "model=mlx-community/Qwen3-ASR-1.7B-8bit" \
  -F "stream=true" \
  --no-buffer

# 备选：stream=false + 足够长的 --max-time（秒）
# 3-4 小时音频建议至少 7200s（2 小时）余量
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F "file=@long_audio.m4a" \
  -F "model=mlx-community/Qwen3-ASR-1.7B-8bit" \
  --max-time 7200
```

**经验法则**：音频时长超过 10 分钟，优先使用 `stream=true`。

### max_tokens 与转写截断

`server_diarize.py` wrapper 默认 `max_tokens=8192`（与 `generate()` 本身一致），不会截断长音频。若使用旧版本服务（默认 128），长音频每 chunk 最多输出 128 token，会明显截断——升级即可解决。

## 故障排查

### Server disconnected（客户端侧）

**症状**：客户端报 "Server disconnected" / "Connection reset"，服务端 access log 显示 200。

**根因**：客户端 read timeout 触发 TCP 关闭，服务端推理完成后发响应时遇 broken pipe。服务端逻辑完全正确。

**解决**：
1. 改用 `stream=true`（推荐，逐 chunk 接收，无超时风险）
2. 增大客户端 read timeout / `--max-time`（至少覆盖估计推理时长）

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

#### 大文件内存估算

上传文件经 tempfile spool → 单次 `model.generate(path)` 解码（不再有 double-decode）：

| 阶段 | 内存占用 | 说明 |
|------|---------|------|
| Upload spool | ~0（磁盘 I/O） | 64KB 分块写 tempfile，不在内存中累积 |
| ffmpeg decode 瞬时 | ~文件时长×SR×8B | `capture_output=True` + float64 转换 |
| 推理稳态 | ~2.5GB | model 1.7GB + 单 chunk PCM + KV cache |

200MB M4A（约 3.5h）示例：瞬时峰值 ~6.6GB（ffmpeg capture_output），稳态 ~2.5GB。8GB Mac 可正常运行。

**历史对比**：v0.3.x 以前，原始 handler 做 decode→reencode→decode（double-decode），200MB M4A 峰值 ~13GB，8GB Mac 必定 OOM。

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

### speaker_diarization.py + server_diarize.py 运维

Speaker diarization 基于 CoreML WeSpeaker ResNet34 embedding + AHC 聚类，在 Apple Silicon GPU/ANE 上运行。

#### 前置条件：模型转换

首次部署或 CoreML 模型丢失时，需运行离线转换脚本：

```bash
# 安装构建时依赖（不在 requirements.txt 中，仅转换用）
pip install onnx onnx2torch

# 运行转换（下载 ONNX → PyTorch → CoreML FP16）
python scripts/convert_wespeaker.py

# 验证
ls -la models/wespeaker/wespeaker_resnet34_LM_fp16.mlpackage
# 预期 ~12.7MB
```

转换完成后 `onnx` 和 `onnx2torch` 可卸载。运行时仅需 `coremltools`（已在环境中）。

#### API 用法

```bash
# 常规转写（不变）
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/Qwen3-ASR-1.7B-8bit"

# 带 speaker diarization
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/Qwen3-ASR-1.7B-8bit" \
  -F "diarize=true"

# 指定说话人数
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/Qwen3-ASR-1.7B-8bit" \
  -F "diarize=true" \
  -F "num_speakers=2"
```

Diarization 参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `diarize` | str | `"false"` | `"true"` 或 `"1"` 启用 |
| `num_speakers` | int | null | 精确说话人数（覆盖自动检测） |
| `min_speakers` | int | null | 说话人数下限 |
| `max_speakers` | int | null | 说话人数上限 |

**限制**：`stream=true` 与 `diarize=true` 不兼容（返回 HTTP 400）。Diarization 需要完整 ASR 结果后才能做时间对齐。

#### 响应格式

成功（diarize=true）：
```json
{
    "text": "完整转写文本",
    "segments": [
        {"text": "...", "start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
        {"text": "...", "start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"}
    ],
    "language": "Chinese",
    "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

降级（diarization 失败时返回原始 ASR + 错误字段）：
```json
{
    "text": "...",
    "segments": [{"text": "...", "start": 0.0, "end": 2.5}],
    "diarization_error": "CoreML model not found..."
}
```

#### 配置

`.env` 中的 diarization 配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WESPEAKER_CACHE` | `./models/wespeaker` | CoreML 模型目录 |
| `DIARIZE_THRESHOLD` | `0.75` | AHC cosine distance 停止阈值（越小越倾向多 speaker） |
| `MAX_UPLOAD_BYTES` | `524288000`（500MB） | 单次上传文件大小上限，超限返回 HTTP 413 |
| `PYTHON_BIN` | （空，使用 PATH） | Python 解释器绝对路径（launchd/conda 环境下建议设置） |
| `EXTRA_PATH` | `/opt/homebrew/bin:/usr/local/bin` | 额外 PATH 前缀（Homebrew ffmpeg 所需） |

阈值调参指南（cosine distance = 1 - cosine_similarity）：

| cosine distance | cosine similarity | 效果 |
|:---:|:---:|:---|
| `0.5` | `0.5` | 严格分裂（干净音频、多说话人场景） |
| `0.75` | `0.25` | 推荐默认值（电话录音、2-5 人） |
| `0.9` | `0.1` | 宽松合并（可能漏分 speaker） |

#### 调参经验

电话录音（窄带、压缩）的 WeSpeaker 1.5s 窗口 embedding cosine similarity 中位数仅 **0.24**，远低于干净音频。intra-speaker 和 inter-speaker 的 distance 分布严重重叠，单纯依赖 threshold 难以精确分离。

**推荐策略**：

| 场景 | 推荐配置 |
|------|---------|
| 电话录音（2 人） | 默认 `0.75` + 传 `max_speakers=2` |
| 会议录音（干净多人） | `DIARIZE_THRESHOLD=0.50`~`0.60` |
| 已知说话人数 | 传 `num_speakers=N`（跳过 threshold，走精确聚类） |

**诊断**：如 speaker 数量异常，用 embedding pairwise cosine distance 分布分析定位阈值分界点（参考 `_cluster()` 中的 AHC 参数）。

#### 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| 启动时 "CoreML model not found" | 模型未转换 | `python scripts/convert_wespeaker.py` |
| diarization_error 字段出现 | CoreML 推理异常 | 检查日志，验证模型完整性 |
| speaker 数量不准 | 阈值不适配 | 调整 `DIARIZE_THRESHOLD` 或传 `num_speakers` |
| speaker 数量远超实际（如 40+） | `DIARIZE_THRESHOLD` 过低 | 调高阈值（如 0.5→0.7）；或传 `max_speakers` 设上限 |
| 首次 diarize 请求慢 | CoreML 模型懒加载 | 正常行为，后续请求不受影响 |

### max_tokens 默认值说明

`server.py` 原始 endpoint 默认 `max_tokens=128`，但 `server_diarize.py` wrapper 已覆盖为 `max_tokens=8192`（与 `generate()` 本身一致），长音频不会因 token 上限截断。若客户端显式传 `max_tokens`，以客户端传值为准。

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
