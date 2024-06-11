from pydantic import BaseModel
from datetime import datetime

class UserBase(BaseModel):
    id: int
    username: str
    password: str

class ResourceBase(BaseModel):
    id: int
    name: str
    type: str
    recid: str
    quizid: int

class RecommendedResource(BaseModel):
    id: int
    name: str
    type: str 

class InteractionBase(BaseModel):
    id: int
    user_id: int
    resource_id: int
    timestamp: datetime

class CreateInteraction(BaseModel):
    userid: int
    cmid: int

class CreateUser(BaseModel):
    username: str
    password: str
    email: str