from pydantic import BaseModel

class Resource(BaseModel):
    id: int
    name: str
    type: str

class Interaction(BaseModel):
    userId: int
    resourceId: int
    state: int