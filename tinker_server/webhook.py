"""Webhook 回调模块 - 向 mint-platform 发送事件通知"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlparse

import requests

from .logging_context import classify_failure_reason, get_otel_tracer

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """事件类型"""
    # 任务事件
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Session 事件
    SESSION_CREATED = "session.created"
    SESSION_CLOSED = "session.closed"

    # 采样事件
    SAMPLE_COMPLETED = "sample.completed"


@dataclass
class WebhookEvent:
    """Webhook 事件"""
    event_type: EventType
    user_id: str
    session_id: str
    data: dict
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class WebhookClient:
    """Webhook 客户端 - 异步发送事件到 mint-platform"""

    def __init__(self, timeout: int = 5, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    def send_event(self, webhook_url: str, event: WebhookEvent, secret_key: str = None) -> bool:
        """
        同步发送 webhook 事件

        Args:
            webhook_url: 回调 URL
            event: 事件对象
            secret_key: 可选的签名密钥

        Returns:
            是否发送成功
        """
        if not webhook_url:
            logger.debug("No webhook URL configured, skipping callback")
            return False

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event.event_type.value,
        }

        if secret_key:
            # 可以添加签名验证
            headers["X-Webhook-Secret"] = secret_key

        payload = event.to_dict()
        tracer = get_otel_tracer()

        for attempt in range(self.max_retries + 1):
            try:
                if tracer is not None:
                    try:
                        from opentelemetry.trace import SpanKind, Status, StatusCode
                    except Exception:
                        response = requests.post(
                            webhook_url,
                            json=payload,
                            headers=headers,
                            timeout=self.timeout,
                        )
                    else:
                        with tracer.start_as_current_span("webhook.send", kind=SpanKind.CLIENT) as span:
                            span.set_attribute("component", "webhook")
                            span.set_attribute("event_type", str(event.event_type.value))
                            span.set_attribute("attempt", int(attempt + 1))
                            parsed = urlparse(str(webhook_url))
                            span.set_attribute("peer.host", str(parsed.netloc))
                            span.set_attribute("peer.scheme", str(parsed.scheme))
                            response = requests.post(
                                webhook_url,
                                json=payload,
                                headers=headers,
                                timeout=self.timeout,
                            )
                            span.set_attribute("http.status_code", int(response.status_code))
                            if int(response.status_code) >= 500:
                                span.record_exception(RuntimeError(f"webhook status={response.status_code}"))
                                span.set_status(Status(StatusCode.ERROR, f"http.status_code={response.status_code}"))
                else:
                    response = requests.post(
                        webhook_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )

                if response.status_code == 200:
                    logger.info(
                        "[webhook.send] completed event_type=%s attempt=%s status_code=%s",
                        str(event.event_type.value),
                        int(attempt + 1),
                        int(response.status_code),
                    )
                    return True
                else:
                    logger.warning(
                        "[webhook.send] non_200 event_type=%s attempt=%s status_code=%s response_snippet=%s",
                        str(event.event_type.value),
                        int(attempt + 1),
                        int(response.status_code),
                        str(response.text[:200]),
                    )
            except requests.exceptions.Timeout:
                logger.warning(
                    "[webhook.send] timeout event_type=%s attempt=%s timeout_s=%s",
                    str(event.event_type.value),
                    int(attempt + 1),
                    int(self.timeout),
                )
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "[webhook.send] request_error event_type=%s attempt=%s error_type=%s failure_reason=%s",
                    str(event.event_type.value),
                    int(attempt + 1),
                    type(e).__name__,
                    classify_failure_reason(e),
                )

        return False

    def send_event_async(self, webhook_url: str, event: WebhookEvent, secret_key: str = None):
        """
        异步发送 webhook 事件 (不阻塞主线程)
        """
        thread = threading.Thread(
            target=self.send_event,
            args=(webhook_url, event, secret_key),
            daemon=True,
        )
        thread.start()


# 全局单例
_webhook_client: WebhookClient | None = None


def get_webhook_client() -> WebhookClient:
    """获取全局 WebhookClient 实例"""
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = WebhookClient()
    return _webhook_client


def send_task_event(
    webhook_url: str,
    event_type: EventType,
    user_id: str,
    session_id: str,
    task_name: str,
    task_type: str,
    model_name: str = None,
    error: str = None,
    result: dict = None,
    config: dict = None,
):
    """
    发送任务相关事件的便捷方法

    Args:
        webhook_url: 回调 URL
        event_type: 事件类型
        user_id: 用户 ID
        session_id: 会话 ID
        task_name: 任务名称
        task_type: 任务类型 (training/sampling/evaluation)
        model_name: 模型名称
        error: 错误信息 (仅 TASK_FAILED)
        result: 结果数据 (仅 TASK_COMPLETED)
        config: 任务配置
    """
    data = {
        "task_name": task_name,
        "task_type": task_type,
    }

    if model_name:
        data["model_name"] = model_name
    if error:
        data["error"] = error
    if result:
        data["result"] = result
    if config:
        data["config"] = config

    event = WebhookEvent(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        data=data,
    )

    client = get_webhook_client()
    client.send_event_async(webhook_url, event)


def send_notification_event(
    webhook_url: str,
    user_id: str,
    title: str,
    message: str,
    notification_type: str = "info",
    session_id: str = None,
):
    """
    发送通知事件的便捷方法

    Args:
        webhook_url: 回调 URL
        user_id: 用户 ID
        title: 通知标题
        message: 通知内容
        notification_type: 通知类型 (info/success/warning/error)
        session_id: 关联的会话 ID
    """
    # 根据通知类型决定事件类型
    if notification_type == "success":
        event_type = EventType.TASK_COMPLETED
    elif notification_type == "error":
        event_type = EventType.TASK_FAILED
    else:
        event_type = EventType.TASK_CREATED

    event = WebhookEvent(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id or "system",
        data={
            "notification": {
                "title": title,
                "message": message,
                "type": notification_type,
            }
        },
    )

    client = get_webhook_client()
    client.send_event_async(webhook_url, event)
