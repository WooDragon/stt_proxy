#!/usr/bin/env python3
"""
STT代理服务，为MLX Omni Server添加默认参数支持
"""

import argparse
import json
import logging
import os
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="STT Proxy Service",
    docs_url=None,  # 禁用文档页面以提高性能
    redoc_url=None  # 禁用Redoc页面以提高性能
)

# 全局变量
config: Dict = {}
http_client: Optional[httpx.AsyncClient] = None
target_base_url: str = ""

# 简单的缓存机制（用于缓存配置等）
cache: Dict = {}

# 请求统计（用于监控）
request_stats = {
    "total_requests": 0,
    "stt_requests": 0,
    "forwarded_requests": 0
}


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_path} 未找到，使用空配置")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"配置文件 {config_path} 格式错误: {e}")
        return {}


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global http_client, target_base_url, config
    
    # 获取配置文件路径
    config_path = os.environ.get("STT_CONFIG_PATH", "config/stt_config.json")
    config = load_config(config_path)
    logger.info(f"加载配置: {config}")
    
    # 初始化HTTP客户端（连接池优化）
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),  # 5分钟超时
        limits=httpx.Limits(
            max_keepalive_connections=10, 
            max_connections=20,
            keepalive_expiry=300.0  # 5分钟keepalive
        ),
        # 启用HTTP/2以提高性能
        http2=True
    )
    
    # 设置目标服务URL
    target_host = os.environ.get("TARGET_HOST", "localhost")
    target_port = os.environ.get("TARGET_PORT", "10240")
    target_base_url = f"http://{target_host}:{target_port}"
    logger.info(f"目标服务URL: {target_base_url}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global http_client
    if http_client:
        await http_client.aclose()


async def forward_request(request: Request, modified_form_data: dict = None, files: dict = None) -> Response:
    """转发请求到目标服务"""
    global http_client, target_base_url
    
    # 构建目标URL
    target_url = f"{target_base_url}{request.url.path}"
    
    # 获取原始请求头，但移除可能冲突的头
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    try:
        if request.method == "POST" and modified_form_data is not None:
            # 添加调试日志
            logger.info(f"转发POST请求到: {target_url}")
            logger.info(f"发送的表单数据: {json.dumps(modified_form_data, indent=2, ensure_ascii=False)}")
            if files:
                logger.info(f"发送的文件数量: {len(files)}")
            
            # 对于multipart/form-data请求，移除Content-Type头让httpx自动设置
            # 这样可以确保正确的boundary参数
            multipart_headers = headers.copy()
            multipart_headers.pop("content-type", None)
            
            # 发送修改后的表单数据和文件
            response = await http_client.post(
                target_url,
                data=modified_form_data,
                files=files,
                headers=multipart_headers
            )
        else:
            # 直接转发请求
            logger.info(f"直接转发{request.method}请求到: {target_url}")
            # 读取请求体
            body = await request.body()
            if body:
                logger.info(f"转发的请求体: {body}")
            
            # 转发请求
            response = await http_client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers
            )
        
        # 返回响应
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    except Exception as e:
        logger.error(f"转发请求失败: {e}")
        return Response(
            content=json.dumps({"error": f"转发请求失败: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "STT Proxy Service"}


@app.get("/stats")
async def get_stats():
    """获取请求统计信息"""
    global request_stats
    return request_stats


# 更新请求统计的函数
def update_stats(request_type: str):
    """更新请求统计"""
    global request_stats
    request_stats["total_requests"] += 1
    if request_type == "stt":
        request_stats["stt_requests"] += 1
    elif request_type == "forwarded":
        request_stats["forwarded_requests"] += 1


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_handler(request: Request, path: str):
    """通用代理处理器"""
    # 更新请求统计
    update_stats("forwarded")
    
    # 特殊处理STT转写请求
    if path.endswith("/audio/transcriptions") and request.method == "POST":
        update_stats("stt")
        return await handle_stt_request(request)
    
    # 其他请求直接转发
    return await forward_request(request)


async def handle_stt_request(request: Request) -> Response:
    """处理STT转写请求，添加默认参数"""
    global config
    
    # 解析表单数据
    try:
        form = await request.form()
        form_data = dict(form)
        
        # 分离文件和普通表单数据
        files = {}
        if "file" in form:
            file_obj = form["file"]
            # 正确处理UploadFile对象
            file_content = await file_obj.read()
            # 重置文件指针
            await file_obj.seek(0)
            # 构造文件元组 (filename, file_content, content_type)
            files["file"] = (file_obj.filename, file_content, file_obj.content_type)
            # 从表单数据中移除文件
            form_data.pop("file", None)
        
        # 强制覆盖配置文件中定义的字段
        for key, default_value in config.items():
            # 强制使用配置文件中的值，不论客户端是否提供
            old_value = form_data.get(key, "未提供")
            # 确保所有值都转换为字符串（multipart/form-data要求）
            form_data[key] = str(default_value)
            logger.info(f"强制设置参数 {key}: {old_value} -> {default_value} (类型: {type(default_value).__name__} -> str)")
        
        # 添加详细的调试日志
        logger.info(f"=== STT请求参数处理完成 ===")
        logger.info(f"配置文件强制覆盖的参数数量: {len(config)}")
        logger.info(f"最终表单数据字段数量: {len(form_data)}")
        logger.info(f"文件数据: {len(files)} 个文件")
        
        # 分别显示配置参数和客户端参数
        config_params = {k: v for k, v in form_data.items() if k in config}
        client_params = {k: v for k, v in form_data.items() if k not in config}
        
        logger.info(f"配置参数 (强制覆盖): {json.dumps(config_params, indent=2, ensure_ascii=False)}")
        if client_params:
            logger.info(f"客户端参数 (保持原样): {json.dumps(client_params, indent=2, ensure_ascii=False)}")
        
        logger.info(f"发送到目标服务的完整表单数据: {json.dumps(form_data, indent=2, ensure_ascii=False)}")
        logger.info(f"=== 开始转发请求到目标服务 ===")
        
        # 转发请求
        return await forward_request(request, form_data, files)
        
    except Exception as e:
        logger.error(f"处理STT请求失败: {e}")
        return Response(
            content=json.dumps({"error": f"处理STT请求失败: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )


def build_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="STT Proxy Service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the proxy service to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10241,
        help="Port to bind the proxy service to, defaults to 10241",
    )
    parser.add_argument(
        "--target-host",
        type=str,
        default="localhost",
        help="Target MLX Omni Server host, defaults to localhost",
    )
    parser.add_argument(
        "--target-port",
        type=str,
        default="10240",
        help="Target MLX Omni Server port, defaults to 10240",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/stt_config.json",
        help="Path to STT config file, defaults to config/stt_config.json",
    )
    return parser


def main():
    """主函数"""
    parser = build_parser()
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["TARGET_HOST"] = args.target_host
    os.environ["TARGET_PORT"] = args.target_port
    os.environ["STT_CONFIG_PATH"] = args.config
    
    # 启动服务
    import uvicorn
    uvicorn.run(
        "stt_proxy:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()