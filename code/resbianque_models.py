from typing import Any
from pydantic import BaseModel


class Res(BaseModel):
    code: int
    msg: str
    data: Any = None


class ChatVO(BaseModel):
    content: str

