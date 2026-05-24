from dataclasses import dataclass


from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Box
from src.utils.opencv_tools import check_status_detection

ocr_service = OCRService()

@dataclass
class CheckBox(Yolo_Box):
    checked: bool
    _text: str

    def __init__(self, element: Yolo_Box):
        super().__init__(element.x, element.y, element.w, element.h, "CheckBox", element.frame)
        self.checked = check_status_detection(element.frame).status
        self._text = ""

    @property
    def text(self):
        if self._text:
            return self._text
        ocr_result = ocr_service.ocr(self.frame)
        self._text = "".join([res.text for res in ocr_result.results])
        return self._text