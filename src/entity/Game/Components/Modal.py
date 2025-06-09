from dataclasses import dataclass

import numpy as np

from src.entity.Game.Components import Button


@dataclass
class Modal:
    modal_title: str
    modal_body: np.array
    modal_body_text: str
    confirm_button: Button = None
    cancel_button: Button = None