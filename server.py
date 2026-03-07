"""
Xcode AI Proxy - Python 版本
使用 FastAPI 重写的 AI 代理服务，支持智谱 GLM-4.6、Kimi 和 DeepSeek 模型
根据环境变量动态加载可用模型
"""

import os
import sys
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
import json
import argparse

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
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
HOST = os.getenv("HOST", "0.0.0.0")

# 重试配置
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 1000)) / 1000  # 转换为秒
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 60000)) / 1000  # 转换为秒

# 检查必需的环境变量
REQUIRED_ENV_VARS = {
    "ZHIPU_API_KEY": "GLM-4.6 模型",
    "KIMI_API_KEY": "Kimi 模型",
    "DEEPSEEK_API_KEY": "DeepSeek 模型",
}

# 检查所有环境变量，但只给出警告而不退出
for env_var, model_name in REQUIRED_ENV_VARS.items():
    if not os.getenv(env_var):
        logger.warning(f"⚠️ 缺少环境变量 {env_var} (用于 {model_name})，该模型将不可用")

# API 配置 - 根据环境变量动态添加模型
API_CONFIGS = {}

# 如果有智谱 API 密钥，则添加智谱模型配置
if os.getenv("ZHIPU_API_KEY"):
    API_CONFIGS.update({
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
    })

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
                "api_url": "https://api.deepseek.com",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "type": "deepseek",
                "name": "DeepSeek Reasoner",
            },
            "deepseek-chat": {
                "api_url": "https://api.deepseek.com",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "type": "deepseek",
                "name": "DeepSeek Chat",
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
    description="AI 代理服务，支持智谱 GLM-4.6、Kimi 和 DeepSeek 模型",
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


# 请求模型
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


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


# 中间件：请求日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"{start_time.isoformat()} - {request.method} {request.url.path}")

    # 记录请求头
    logger.info(f"请求头: {dict(request.headers)}")

    response = await call_next(request)

    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"请求处理时间: {process_time:.3f}秒")
    logger.info(f"响应状态码: {response.status_code}")

    return response


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# 调试端点
@app.get("/debug/config")
async def debug_config():
    """调试配置信息"""
    return {
        "available_models": list(API_CONFIGS.keys()),
        "config_summary": {
            model_id: {
                "name": config["name"],
                "type": config["type"],
                "api_url": config["api_url"],
                "has_api_key": bool(config.get("api_key")),
            }
            for model_id, config in API_CONFIGS.items()
        },
    }


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


# 新增：清洗 messages，确保每条 message['content'] 为字符串
def sanitize_messages(messages):
    """
    确保 messages 是 list，每个 message 为 dict 且 message['content'] 为字符串。
    - 如果 message 是字符串 -> 转为 {'role':'user','content': str}
    - 如果 content 是 list -> 将元素 join（非字符串元素 json.dumps）
    - 其他非字符串 -> json.dumps
    """
    import json

    if not isinstance(messages, list):
        logger.warning("messages 不是列表，已尝试转换为单项列表")
        return [{"role": "user", "content": str(messages)}]

    sanitized = []
    for idx, m in enumerate(messages):
        # 字符串形式的 message，视为 user
        if isinstance(m, str):
            sanitized.append({"role": "user", "content": m})
            continue

        if not isinstance(m, dict):
            # 无法识别的类型，序列化为字符串
            sanitized.append(
                {"role": "user", "content": json.dumps(m, ensure_ascii=False)}
            )
            continue

        content = m.get("content", "")
        if isinstance(content, str):
            s = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            s = "\n".join(parts)
        else:
            s = json.dumps(content, ensure_ascii=False)

        new_m = {**m, "content": s}
        sanitized.append(new_m)

    return sanitized


async def parse_sse_stream(resp: httpx.Response) -> str:
    """解析 response 的 SSE 流，并且把解析的结果暂时存到本地字符串中"""
    buffer = ""
    fragments = []

    async for chunk in resp.aiter_text(chunk_size=8192):
        buffer += chunk

        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            if not event.strip():
                continue

            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return "".join(fragments)

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    fragments.append(data)
                    continue

                if isinstance(payload, dict):
                    choices = payload.get("choices") or []
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        for block in (delta, message):
                            content_piece = block.get("content")
                            if content_piece:
                                fragments.append(content_piece)

                    if not choices and payload.get("content"):
                        content_value = payload["content"]
                        if isinstance(content_value, str):
                            fragments.append(content_value)
                        else:
                            fragments.append(json.dumps(content_value, ensure_ascii=False))
                else:
                    fragments.append(str(payload))

    return "".join(fragments)


def process_parsed_stream_cache(parsed_stream_cache: str) -> str:
    """对 parsed_stream_cache 进行处理"""
    try:
        payload = json.loads(parsed_stream_cache)
    except json.JSONDecodeError:
        return parsed_stream_cache

    try:
        json.loads(payload.get("text", ""))
        return process_parsed_stream_cache(payload.get("text", ""))
    except (json.JSONDecodeError, AttributeError):
        return payload.get("text", "")


# DeepSeek API 处理
async def handle_deepseek_request(request_body: dict) -> Union[dict, StreamingResponse]:
    """处理 DeepSeek API 请求"""
    logger.info("📡 路由到DeepSeek API")
    
    request_body['messages'] = sanitize_messages(request_body['messages'])
    logger.info('🧹 在 handle_proxy 中已清洗 messages')
    
    model = request_body.get("model", "deepseek-reasoner")
    logger.info(f"🔍 使用 DeepSeek 模型: {model}")

    async def make_request():
        config = API_CONFIGS[model]

        # 过滤 DeepSeek API 支持的参数
        supported_params = {
            "model",
            "messages",
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        }

        # 构建清理后的请求数据
        request_data = {
            key: value for key, value in request_body.items() if key in supported_params
        }

        # 确保模型名称正确
        request_data["model"] = model

        # 移除空的数组参数
        if "tools" in request_body and not request_body["tools"]:
            logger.info("🧹 移除空的 tools 参数")

        # 记录过滤的参数
        filtered_params = set(request_body.keys()) - set(request_data.keys())
        if filtered_params:
            logger.info(f"🧹 已过滤不支持的参数: {filtered_params}")

        logger.info(f'📤 发送到 DeepSeek API: {config["api_url"]}/chat/completions')
        logger.info(f"📋 请求参数: {list(request_data.keys())}")

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{config['api_url']}/chat/completions",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
            )

            # 记录响应状态和错误信息
            logger.info(f"📥 DeepSeek API 响应状态: {response.status_code}")
            if response.status_code != 200:
                response_text = response.text
                logger.error(f"❌ DeepSeek API 错误响应: {response_text}")

            # 非 2xx 状态会触发 raise_for_status() 抛出 HTTPStatusError
            response.raise_for_status()
            return response

    response = await with_retry(make_request)
    logger.info(f"✅ DeepSeek API响应状态: {response.status_code}")

    if request_body.get("stream", False):
        logger.info("🔄 返回DeepSeek流式响应")

        # 直接返回原始流式响应，不修改任何内容
        response_headers = dict(response.headers)
        # 移除可能引起问题的头部
        response_headers.pop("content-length", None)
        response_headers.pop("content-encoding", None)
        response_headers["content-type"] = "text/event-stream; charset=utf-8"

        # 解析 response 的 SSE 流，并且把解析的结果暂时存到本地字符串中
        parsed_stream_cache = await parse_sse_stream(response)
        logger.info(f"🧩 DeepSeek流式缓存解析结果: {parsed_stream_cache!r}")

        # 对 parsed_stream_cache 进行处理。
        parsed_stream_cache = process_parsed_stream_cache(parsed_stream_cache)
        logger.info(f"🧩 DeepSeek流式缓存处理后结果: {parsed_stream_cache!r}")

        async def generate():
            # 将解析后的文本拆分为多个 SSE 块并逐个推送
            chunk_size = 1024
            text = parsed_stream_cache or ""
            stream_id = str(uuid.uuid4())
            system_fingerprint = "fp_proxy_stream"
            for index, start in enumerate(range(0, len(text), chunk_size)):
                segment = text[start : start + chunk_size]
                payload = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "system_fingerprint": system_fingerprint,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": segment},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                sse_chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                logger.debug(f"🔀 发送SSE块(index={index}): {sse_chunk!r}")
                yield sse_chunk.encode("utf-8")
                await asyncio.sleep(0)

            # 发送结束块，指示完成
            finish_payload = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "system_fingerprint": system_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }
            finish_chunk = f"data: {json.dumps(finish_payload, ensure_ascii=False)}\n\n"
            logger.debug(f"🏁 发送SSE结束块: {finish_chunk!r}")
            yield finish_chunk.encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            generate(), status_code=response.status_code, headers=response_headers
        )
    else:
        logger.info("📦 返回DeepSeek非流式响应")
        return response.json()  # 代理处理函数


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


@app.post("/api/v1/chat/completions")
async def api_chat_completions(request: Request):
    """备用聊天完成接口"""
    try:
        body = await request.json()
        logger.info(f"API接口请求体: {body}")
        return await handle_proxy(body)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API接口解析请求体失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Invalid request body: {str(e)}",
                    "type": "invalid_request_error",
                }
            },
        )


@app.post("/v1/messages")
async def messages(request: Request):
    """消息接口"""
    try:
        body = await request.json()
        logger.info(f"消息接口请求体: {body}")
        return await handle_proxy(body)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"消息接口解析请求体失败: {str(e)}")
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

    uvicorn.run(
        "server:app", host=host, port=port, reload=False, log_level="info"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xcode AI Proxy CLI - 启动 AI 代理服务"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help="服务监听端口 (默认: 8899)"
    )
    parser.add_argument(
        "--host", type=str, default=HOST, help="服务监听地址 (默认: 0.0.0.0)"
    )
    args = parser.parse_args()
    main(port=args.port, host=args.host)
