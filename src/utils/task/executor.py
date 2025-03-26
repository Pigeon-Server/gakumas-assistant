from threading import Thread


class InferenceThread(Thread):
    def __init__(self, app_processor, data_store):
        super().__init__(daemon=True)
        self.processor = app_processor
        self.data = data_store
        self.running = True

    def run(self):
        while self.running:
            frame = self.processor.app.capture()
            if frame is None or frame.size == 0:
                continue

            results = self.processor.model(frame, verbose=False)
            boxes = self.processor._parse_boxes(results[0], frame)

            self.data.update(boxes, frame, results)

            # 控制推理频率
            time.sleep(0.05)  # 20fps

    def stop(self):
        self.running = False
