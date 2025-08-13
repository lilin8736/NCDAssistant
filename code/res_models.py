from typing import Any
from pydantic import BaseModel


class Res(BaseModel):
    code: int
    msg: str
    data: Any = None


class ChatVO(BaseModel):
    id: int
    base_information: str
    content: str
    user_id: int
    with_history:int

