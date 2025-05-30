from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.core.Web.websocket import WebSocketManager
from time import sleep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app import AppProcessor

def register_routes(app: FastAPI, processor: "AppProcessor", ws_manager: WebSocketManager):

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    @app.get("/start")
    def start_inference():
        processor.exec_task()
        return {"message": "Inference started"}

    @app.get("/stop")
    def stop_inference():
        processor.task_queue.stop()
        return {"message": "Inference stopped"}

    @app.get("/status")
    def get_status():
        return {'status': processor.running}

    @app.get("/get_registered_tasks")
    def get_registered_tasks():
        return processor.task_queue.get_task_list()

    @app.post("/disable_task/{task_name:str}")
    def disable_task(task_name):
        return processor.task_queue.disable_task(task_name)

    @app.get("/switch_yolo_model/base_ui")
    def switch_yolo_model__base_ui():
        processor.load_model("BASE_UI")
        return {"status": True}

    @app.get("/switch_yolo_model/producer")
    def switch_yolo_model__producer():
        processor.load_model("PRODUCER")
        return {"status": True}
