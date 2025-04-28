import asyncio

from starlette.websockets import WebSocket

from src.entity.WebSocket_Data import WebSocket_Data


class WebSocketManager:
    active_connections: list[WebSocket]
    _loop: asyncio.AbstractEventLoop

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop = asyncio.get_event_loop()  # 事件循环

    async def connect(self, websocket: WebSocket):
        """接收新的 WebSocket 连接并接受消息"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """移除 WebSocket 连接"""
        self.active_connections.remove(websocket)

    async def send_message(self, data: WebSocket_Data, websocket: WebSocket):
        """向特定 WebSocket 发送消息"""
        if msg := data.message:
            await websocket.send_text(msg)
        if data := data.data:
            await websocket.send_bytes(data)

    async def broadcast(self, data: WebSocket_Data):
        """向所有 WebSocket 连接发送消息"""
        for connection in self.active_connections:
            await self.send_message(data, connection)

    def send_message_sync(self, data: WebSocket_Data, websocket: WebSocket):
        """供后台线程调用的同步方法，发送消息"""
        asyncio.run_coroutine_threadsafe(self.send_message(data, websocket), self._loop)

    def broadcast_sync(self, data: WebSocket_Data):
        """供后台线程调用的同步方法，广播消息"""
        asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)