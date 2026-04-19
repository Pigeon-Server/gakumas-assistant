import asyncio
import json
from typing import Optional

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from src.constants.websocket_actions import WebsocketActions
from src.entity.Base import SingletonMeta
from src.entity.WebSocketData import WebSocketData
from src.utils.json_encoder import ComplexEncoder
from src.utils.logger import logger


class WebSocketManager(metaclass=SingletonMeta):
    _active_connections: list[WebSocket]
    _loop: asyncio.AbstractEventLoop
    _init: bool = False

    def __init__(self):
        if self._init:
            return
        self.active_connections: list[WebSocket] = []
        # 每个连接独立的发送锁，防止并发 send 触发 websockets 断言
        self._connection_locks: dict[WebSocket, asyncio.Lock] = {}
        self._loop = asyncio.get_event_loop()  # 事件循环
        self._init = True
        logger.add(sink=self._broadcast_log)

    @staticmethod
    def _should_broadcast_log(record: dict) -> bool:
        module_name = record.get("name") or ""
        return module_name not in {
            "src.core.web.websocket",
            "src.core.web.routers",
        }

    def _broadcast_log(self, message):
        record = message.record
        if not self._should_broadcast_log(record):
            return
        data = WebSocketData(message={
            "time": record["time"],
            "level": record["level"].name,
            "message": record.get("message"),
        })
        self.broadcast_action_sync(WebsocketActions.Logger.BroadcastLog, data)

    def set_fastapi_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        """接收新的 WebSocket 连接并接受消息"""
        await websocket.accept()
        self.active_connections.append(websocket)
        # 为该连接创建独立锁，确保同一连接的发送操作串行执行
        self._connection_locks[websocket] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket):
        """移除 WebSocket 连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._connection_locks.pop(websocket, None)

    @staticmethod
    def _is_disconnected(websocket: WebSocket) -> bool:
        return (
            websocket.client_state == WebSocketState.DISCONNECTED
            or websocket.application_state == WebSocketState.DISCONNECTED
        )

    @staticmethod
    async def send_message(data: WebSocketData, websocket: WebSocket):
        """向特定 WebSocket 发送消息"""
        if msg := data.message:
            await websocket.send_text(json.dumps(msg, cls=ComplexEncoder))
        if data := data.data:
            await websocket.send_bytes(data)

    async def _send_message_safely(self, data: WebSocketData, websocket: WebSocket) -> bool:
        # 获取连接锁，串行化同一连接的发送操作，防止 websockets 库并发 drain 断言失败
        lock = self._connection_locks.get(websocket)
        if lock is None:
            return False
        try:
            async with lock:
                await self.send_message(data, websocket)
            return True
        except WebSocketDisconnect:
            self.disconnect(websocket)
            return False
        except RuntimeError as exc:
            self.disconnect(websocket)
            if self._is_disconnected(websocket):
                logger.debug("WebSocket disconnected during send [{}]: {}", type(exc).__name__, exc)
            else:
                logger.warning("WebSocket send failed, disconnecting client [{}]: {}", type(exc).__name__, exc)
            return False
        except Exception as exc:
            self.disconnect(websocket)
            logger.exception("WebSocket send failed, disconnecting client [{}]: {}", type(exc).__name__, exc)
            return False

    async def broadcast(self, data: WebSocketData):
        """向所有 WebSocket 连接发送消息"""
        for connection in list(self.active_connections):
            await self._send_message_safely(data, connection)

    @staticmethod
    def _consume_future_exception(future) -> None:
        try:
            future.result()
        except Exception as exc:
            logger.debug("WebSocket background task ended [{}]: {}", type(exc).__name__, exc)

    def _submit_coroutine(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        future.add_done_callback(self._consume_future_exception)
        return future

    def send_message_sync(self, data: WebSocketData, websocket: WebSocket):
        """供后台线程调用的同步方法，发送消息"""
        self._submit_coroutine(self._send_message_safely(data, websocket))

    def broadcast_sync(self, data: WebSocketData):
        """供后台线程调用的同步方法，广播消息"""
        self._submit_coroutine(self.broadcast(data))

    async def send_action(self, websocket: WebSocket, action: str, data: Optional[WebSocketData] = None) -> bool:
        """发送动作数据到指定客户端"""
        return await self._send_message_safely(WebSocketData(message={
            "action": WebsocketActions.BaseActionFlag+":"+action,
            "data": {} if data is None else data.message
        }), websocket)

    def send_action_sync(self, websocket: WebSocket, action: str, data: Optional[WebSocketData] = None):
        """发送动作数据到指定客户端"""
        self.send_message_sync(WebSocketData(message={
            "action": WebsocketActions.BaseActionFlag+":"+action,
            "data": {} if data is None else data.message
        }), websocket)

    def broadcast_action_sync(self, action: str, data: Optional[WebSocketData] = None):
        """
        广播动作数据给所有客户端
        :param action: 动作名
        :param data: 数据
        """
        self.broadcast_sync(WebSocketData(message={
            "action": WebsocketActions.BaseActionFlag+":"+action,
            "data": {} if data is None else data.message
        }))
