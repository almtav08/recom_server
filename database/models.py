from typing import Optional
from sqlmodel import Field, Relationship, SQLModel

class Interaction(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=None, foreign_key="user.id", primary_key=False)
    resource_id: int = Field(default=None, foreign_key="resource.id", primary_key=False)
    timestamp: str
    passed: bool = Field(default=True)

    def __hash__(self):
        return hash(self.id)


class User(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    password: str
    email: str
    resources: list["Resource"] = Relationship(back_populates="users", link_model=Interaction, sa_relationship_kwargs={'lazy': 'selectin'})
    is_pass: bool = Field(default=False)

    def __hash__(self):
        return hash(self.id)

class Resource(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    type: str
    quizid: int | None
    recid: int
    users: list["User"] = Relationship(back_populates="resources", link_model=Interaction, sa_relationship_kwargs={'lazy': 'selectin'})

    def __hash__(self):
        return hash(self.id)
