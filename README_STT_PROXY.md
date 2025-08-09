# STT代理服务使用说明

## 简介

STT代理服务是一个轻量级的代理层，用于为MLX Omni Server添加默认参数支持，特别是为Whisper语音转文字服务提供默认的prompt，以提高转写准确性。

## 功能特性

1. **默认参数增强**：通过配置文件为STT请求自动添加默认参数
2. **透明代理**：对客户端完全透明，无需修改客户端代码
3. **性能优化**：使用连接池和HTTP/2优化性能
4. **监控统计**：提供健康检查和请求统计功能
5. **灵活配置**：支持通过配置文件自定义默认参数

## 部署架构

```
客户端 -> STT代理服务(10241) -> MLX Omni Server(10240)
```

## 快速开始

### 1. 启动MLX Omni Server

```bash
# 在端口10240上启动MLX Omni Server
mlx-omni-server --port 10240
```

### 2. 启动STT代理服务

```bash
# 在端口10241上启动代理服务
python stt_proxy.py
```

### 3. 使用代理服务

将客户端请求发送到代理服务端口(10241)，而不是直接发送到MLX Omni Server(10240)。

## 配置文件

### stt_config.json

```json
{
  "language": "zh",
  "temperature": 0.2,
  "prompt": "以下是普通话的会议记录。",
  "response_format": "srt"
}
```

当客户端未提供相应参数时，代理服务会自动使用配置文件中的默认值。

## 命令行参数

```bash
python stt_proxy.py --help
```

支持的参数：
- `--host`: 代理服务监听的主机地址 (默认: 0.0.0.0)
- `--port`: 代理服务监听的端口 (默认: 10241)
- `--target-host`: 目标MLX Omni Server主机地址 (默认: localhost)
- `--target-port`: 目标MLX Omni Server端口 (默认: 10240)
- `--config`: 配置文件路径 (默认: stt_config.json)

## 环境变量

- `STT_CONFIG_PATH`: 配置文件路径
- `TARGET_HOST`: 目标服务主机地址
- `TARGET_PORT`: 目标服务端口

## 监控端点

- `/health`: 健康检查端点
- `/stats`: 请求统计信息端点

## 性能优化

1. **连接池**: 使用httpx连接池复用连接
2. **HTTP/2**: 启用HTTP/2以提高性能
3. **Keep-Alive**: 配置keep-alive连接保持
4. **禁用文档**: 禁用FastAPI自动生成的文档页面以减少资源消耗

## 使用示例

### Python客户端示例

```python
import httpx

# 使用代理服务而不是直接连接到MLX Omni Server
async with httpx.AsyncClient() as client:
    with open("audio.wav", "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {
            "model": "whisper-tiny",
            # 注意：不需要提供prompt，代理会自动添加默认prompt
        }
        
        response = await client.post(
            "http://localhost:10241/v1/audio/transcriptions",
            files=files,
            data=data
        )
```

### curl示例

```bash
# 使用代理服务
curl -X POST http://localhost:10241/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-tiny"
```

## 参数处理规则

1. **配置文件中存在的字段**：强制覆盖客户端发送的值
2. **配置文件中不存在的字段**：直接透传客户端参数
3. **客户端显式提供的参数**：对于配置文件中未定义的字段，保持客户端参数不变

## 更新和维护

由于代理服务是独立于MLX Omni Server的，因此：
1. 可以独立更新代理服务而不影响原始服务
2. MLX Omni Server可以正常更新而无需修改
3. 配置文件可以随时修改以调整默认行为

## 故障排除

1. **代理服务无法启动**：
   - 检查端口是否被占用
   - 检查配置文件格式是否正确
   - 检查目标服务是否可访问

2. **STT请求失败**：
   - 检查MLX Omni Server是否正常运行
   - 检查配置文件中的模型是否存在
   - 查看日志获取更多信息

3. **性能问题**：
   - 检查网络连接
   - 调整连接池参数
   - 检查目标服务性能