"""
Xcode AI Proxy - Python 版本
使用 FastAPI 重写的 AI 代理服务，支持智谱 GLM、千问 Qwen、Kimi 和 DeepSeek 模型
根据环境变量动态加载可用模型
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Union
import json
import argparse

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import uvicorn

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 服务器配置
PORT = int(os.getenv("PORT", 3000))
HOST = os.getenv("HOST", "127.0.0.1")

# 重试配置
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 1000)) / 1000  # 转换为秒
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 60000)) / 1000  # 转换为秒

# 检查必需的环境变量
REQUIRED_ENV_VARS = {
    "ZHIPU_API_KEY": "GLM 模型",
    "KIMI_API_KEY": "Kimi 模型",
    "DEEPSEEK_API_KEY": "DeepSeek 模型",
    "QWEN_API_KEY": "Qwen 模型",
}

# 检查所有环境变量，但只给出警告而不退出
for env_var, model_name in REQUIRED_ENV_VARS.items():
    if not os.getenv(env_var):
        logger.warning(f"⚠️ 缺少环境变量 {env_var} (用于 {model_name})，该模型将不可用")

# API 配置 - 根据环境变量动态添加模型
API_CONFIGS = {}

# 如果有智谱 API 密钥，则添加智谱模型配置
if os.getenv("ZHIPU_API_KEY"):
    API_CONFIGS.update(
        {
            "glm-4.6": {
                "api_url": "https://open.bigmodel.cn/api/paas/v4",
                "api_key": os.getenv("ZHIPU_API_KEY"),
                "type": "zhipu",
                "name": "GLM-4.6",
            },
            "glm-4.7": {
                "api_url": "https://open.bigmodel.cn/api/paas/v4",
                "api_key": os.getenv("ZHIPU_API_KEY"),
                "type": "zhipu",
                "name": "GLM-4.7",
            },
            "glm-5": {
                "api_url": "https://open.bigmodel.cn/api/paas/v4",
                "api_key": os.getenv("ZHIPU_API_KEY"),
                "type": "zhipu",
                "name": "GLM-5",
            },
        }
    )

# 如果有 Kimi API 密钥，则添加 Kimi 模型配置
if os.getenv("KIMI_API_KEY"):
    API_CONFIGS["kimi-k2-0905-preview"] = {
        "api_url": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("KIMI_API_KEY"),
        "type": "kimi",
        "name": "Kimi K2",
    }

# 如果有 DeepSeek API 密钥，则添加 DeepSeek 模型配置
if os.getenv("DEEPSEEK_API_KEY"):
    API_CONFIGS.update(
        {
            "deepseek-reasoner": {
                "api_url": "https://api.deepseek.com/v1",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "type": "deepseek",
                "name": "DeepSeek Reasoner",
            },
            "deepseek-chat": {
                "api_url": "https://api.deepseek.com/v1",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "type": "deepseek",
                "name": "DeepSeek Chat",
            },
        }
    )

# 如果有千问 API 密钥，则添加千问模型配置
if os.getenv("QWEN_API_KEY"):
    API_CONFIGS.update(
        {
            "qwen3.5-plus": {
                "api_url": "https://coding.dashscope.aliyuncs.com/v1",
                "api_key": os.getenv("QWEN_API_KEY"),
                "type": "qwen",
                "name": "Qwen 3.5 Plus",
            },
            "qwen3-coder-next": {
                "api_url": "https://coding.dashscope.aliyuncs.com/v1",
                "api_key": os.getenv("QWEN_API_KEY"),
                "type": "qwen",
                "name": "Qwen 3 Coder Next",
            },
        }
    )

if not API_CONFIGS:
    logger.error("❌ 未配置任何模型API密钥，请至少设置一个环境变量:")
    for env_var, model_name in REQUIRED_ENV_VARS.items():
        logger.error(f"   - {env_var} (用于 {model_name})")
    logger.error("请设置相应的环境变量后重新启动服务")
    sys.exit(1)

logger.info("📋 已加载模型配置:")
for model_id, config in API_CONFIGS.items():
    logger.info(f"   ✅ {model_id} ({config['name']}) - 已配置")

# FastAPI 应用初始化
app = FastAPI(
    title="Xcode AI Proxy",
    description="AI 代理服务，支持智谱 GLM、千问 Qwen、Kimi 和 DeepSeek 模型",
    version="1.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 通用重试装饰器
async def with_retry(operation, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY):
    """通用异步重试函数"""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 第{attempt}次尝试")
            return await operation()
        except Exception as error:
            last_error = error
            logger.error(f"❌ 第{attempt}次尝试失败: {str(error)}")

            if attempt < max_retries:
                delay = base_delay * attempt  # 递增延迟
                logger.info(f"⏳ {delay}秒后重试...")
                await asyncio.sleep(delay)

    logger.error(f"❌ 所有{max_retries}次重试都失败了")
    # 如果没有捕获到具体异常，避免 raise None，提供一个明确的回退错误
    if last_error:
        raise last_error
    else:
        raise RuntimeError("Operation failed after retries with no exception captured")


# 模型列表
@app.get("/v1/models")
async def list_models():
    """返回支持的模型列表"""
    logger.info("📋 返回模型列表")

    model_list = [
        {
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": config["type"],
            "name": config.get("name", model_id),
        }
        for model_id, config in API_CONFIGS.items()
    ]

    return {"object": "list", "data": model_list}


# 智谱 API 处理
async def handle_zhipu_request(request_body: dict) -> Union[dict, StreamingResponse]:
    """处理智谱 API 请求"""
    model = request_body.get("model", "glm-4.6")
    logger.info(f"📡 路由到智谱API (模型: {model})")

    async def make_request():
        config = API_CONFIGS[model]

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{config['api_url']}/chat/completions",
                json={**request_body, "model": model},
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ 智谱API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回智谱流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回智谱非流式响应")
        return response.json()


# Kimi API 处理
async def handle_kimi_request(request_body: dict) -> Union[dict, StreamingResponse]:
    """处理 Kimi API 请求"""
    logger.info("📡 路由到Kimi API")

    async def make_request():
        config = API_CONFIGS["kimi-k2-0905-preview"]

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{config['api_url']}/chat/completions",
                json={**request_body, "model": "kimi-k2-0905-preview"},
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ Kimi API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回Kimi流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回Kimi非流式响应")
        return response.json()


# DeepSeek API 处理
async def handle_deepseek_request(request_body: dict) -> Union[dict, StreamingResponse]:
    """处理 DeepSeek API 请求（OpenAI 兼容模式）"""
    model = request_body.get("model", "deepseek-reasoner")
    logger.info(f"📡 路由到DeepSeek API (模型: {model})")

    async def make_request():
        config = API_CONFIGS[model]

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{config['api_url']}/chat/completions",
                json={**request_body, "model": model},
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ DeepSeek API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回DeepSeek流式响应")

        response_headers = dict(response.headers)
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回DeepSeek非流式响应")
        return response.json()


# 千问 API 处理
async def handle_qwen_request(request_body: dict) -> Union[dict, StreamingResponse]:
    """处理千问 API 请求"""
    model = request_body.get("model", "qwen3.5-plus")
    logger.info(f"📡 路由到千问API (模型: {model})")

    async def make_request():
        config = API_CONFIGS[model]

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{config['api_url']}/chat/completions",
                json={**request_body, "model": model},
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ 千问API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回千问流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)

        async def generate():
            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回千问非流式响应")
        return response.json()


async def handle_proxy(request_data: dict):
    """处理代理请求"""
    try:
        model = request_data.get("model")
        logger.info(f"🎯 请求模型: {model}")
        logger.info(f'🔍 是否流式: {request_data.get("stream", False)}')

        if not model or model not in API_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"不支持的模型: {model}。支持的模型: {', '.join(API_CONFIGS.keys())}",
                        "type": "invalid_request_error",
                    }
                },
            )

        config = API_CONFIGS[model]

        if config["type"] == "zhipu":
            return await handle_zhipu_request(request_data)
        elif config["type"] == "kimi":
            return await handle_kimi_request(request_data)
        elif config["type"] == "deepseek":
            return await handle_deepseek_request(request_data)
        elif config["type"] == "qwen":
            return await handle_qwen_request(request_data)
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"未知的模型类型: {config['type']}",
                        "type": "internal_error",
                    }
                },
            )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as error:
        logger.error(
            f"❌ HTTP 状态错误: {error.response.status_code} - {error.response.text}"
        )
        raise HTTPException(
            status_code=error.response.status_code,
            detail={
                "error": {
                    "message": f"API 请求失败: {error.response.status_code} - {error.response.text}",
                    "type": "api_error",
                }
            },
        )
    except httpx.RequestError as error:
        logger.error(f"❌ 请求错误: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"网络请求失败: {str(error)}",
                    "type": "network_error",
                }
            },
        )
    except Exception as error:
        logger.error(f"❌ 代理请求失败: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(error), "type": "proxy_error"}},
        )


# Chat Completions 接口
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容的聊天完成接口"""
    try:
        body = await request.json()
        logger.info(f"请求体: {body}")

        # 验证必需字段
        if "model" not in body:
            logger.error("请求体缺少 'model' 字段")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Missing required field: 'model'",
                        "type": "invalid_request_error",
                    }
                },
            )

        if "messages" not in body:
            logger.error("请求体缺少 'messages' 字段")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Missing required field: 'messages'",
                        "type": "invalid_request_error",
                    }
                },
            )

        return await handle_proxy(body)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解析请求体失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Invalid request body: {str(e)}",
                    "type": "invalid_request_error",
                }
            },
        )


# 启动函数
def main(port=PORT, host=HOST):
    """启动服务器"""
    logger.info("🚀 Xcode AI 代理服务已启动")
    logger.info(f"📡 监听地址: http://{host}:{port}")
    logger.info("🎯 当前可用的模型:")
    for model, config in API_CONFIGS.items():
        logger.info(f"   ✅ {model} ({config.get('name', config['type'])})")

    if not API_CONFIGS:
        logger.error("❌ 没有可用的模型，请检查环境变量配置")
        return

    logger.info("⚙️ 重试配置:")
    logger.info(f"   最大重试次数: {MAX_RETRIES}")
    logger.info(f"   重试延迟: {int(RETRY_DELAY * 1000)}ms (递增)")
    logger.info(f"   请求超时: {int(REQUEST_TIMEOUT * 1000)}ms")

    logger.info("📋 配置 Xcode:")
    logger.info(f"   ANTHROPIC_BASE_URL: http://localhost:{port}")
    logger.info("   ANTHROPIC_AUTH_TOKEN: any-string-works")
    logger.info("🔧 功能: 智谱/Kimi/DeepSeek代理，流式响应，动态配置，智能重试")

    uvicorn.run("server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xcode AI Proxy CLI - 启动 AI 代理服务"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help="服务监听端口 (默认: 8899)"
    )
    parser.add_argument(
        "--host", type=str, default=HOST, help="服务监听地址 (默认: 127.0.0.1)"
    )
    args = parser.parse_args()
    main(port=args.port, host=args.host)
