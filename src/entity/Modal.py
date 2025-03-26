from dataclasses import dataclass

from src.entity import Button


@dataclass
class Modal:
    modal_title: str
    modal_body: str
    confirm_button: Button
    cancel_button: Button
    def __init__(self, modal_title: str, modal_body: str = None, confirm_button: Button =None, cancel_button: Button =None):
        self.modal_title = modal_title
        self.modal_body = modal_body
        self.confirm_button = confirm_button
        self.cancel_button = cancel_button