import os
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from .db_schema import Interaction, Resource, User

load_dotenv(override=True)


class AsyncQueryGenerator:
    def __init__(self):
        self.engine: Engine = None

    def connect(self):
        self.engine = create_async_engine(os.getenv("DATABASE_ASYNC_URL"), echo=False)
        self.session = AsyncSession(self.engine)

    def is_connected(self):
        return self.engine is not None

    async def disconnect(self):
        await self.session.close()
        await self.engine.dispose()

    def select_user(self, user_id: int) -> User:
        statement = select(User).where(User.id == user_id)
        user = self.session.exec(statement).first()
        return user

    async def select_user_by_username(self, username: str) -> User:
        statement = select(User).where(User.username == username)
        user = self.session.exec(statement).first()
        return user

    async def select_resource(self, resource_id: int) -> Resource:
        statement = select(Resource).where(Resource.id == resource_id)
        resource = self.session.exec(statement).first()
        return resource

    async def select_interactions_by_user(self, user_id: int) -> list[Interaction]:
        statement = select(Interaction).where(Interaction.user_id == user_id)
        interactions = self.session.exec(statement).all()
        return interactions

    async def select_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = self.session.exec(statement).first()
        return interaction

    async def insert_interaction(self, interaction: Interaction) -> Interaction:
        self.session.add(interaction)
        self.session.commit()
        self.session.refresh(interaction)
        return interaction

    async def insert_interactions(
        self, interactions: list[Interaction]
    ) -> list[Interaction]:
        for interaction in interactions:
            self.session.add(interaction)
        self.session.commit()
        for interaction in interactions:
            self.session.refresh(interaction)
        return interactions

    async def delete_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = self.session.exec(statement).first()
        self.session.delete(interaction)
        self.session.commit()
        return interaction

    async def list_resources(self) -> list[Resource]:
        statement = select(Resource).order_by(Resource.recid.asc())
        resources = self.session.exec(statement).all()
        return resources

    async def list_users(self) -> list[User]:
        statement = select(User).order_by(User.id.asc())
        users = self.session.exec(statement).all()
        return users

    async def list_interactions(self) -> list[Interaction]:
        statement = select(Interaction)
        interactions = self.session.exec(statement).all()
        return interactions

    async def create_user(self, user: User) -> User:
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    async def create_users(self, users: list[User]) -> list[User]:
        for user in users:
            self.session.add(user)
        self.session.commit()
        for user in users:
            self.session.refresh(user)
        return users

    async def create_resource(self, resource: Resource) -> Resource:
        self.session.add(resource)
        self.session.commit()
        self.session.refresh(resource)
        return resource

    async def create_resources(self, resources: list[Resource]) -> list[Resource]:
        for resource in resources:
            self.session.add(resource)
        self.session.commit()
        for resource in resources:
            self.session.refresh(resource)
        return resources
