import os
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlmodel import SQLModel, select
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
    
    async def select_user(self, user_id: int) -> User:
        statement = select(User).where(User.id == user_id)
        user = await self.session.exec(statement)
        return user.first()
    
    async def select_user_by_username(self, username: str) -> User:
        statement = select(User).where(User.username == username)
        user = await self.session.exec(statement)
        return user.first()

    async def select_resource(self, resource_id: int) -> Resource:
        statement = select(Resource).where(Resource.id == resource_id)
        resource = await self.session.exec(statement)
        return resource.first()

    async def select_interactions_by_user(self, user_id: int) -> list[Interaction]:
        statement = select(Interaction).where(Interaction.user_id == user_id)
        interactions = await self.session.exec(statement)
        return interactions.all()
    
    async def select_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = await self.session.exec(statement)
        return interaction.first()

    async def insert_interaction(self, interaction: Interaction) -> Interaction:
        self.session.add(interaction)
        await self.session.commit()
        await self.session.refresh(interaction)
        return interaction

    async def delete_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = await self.session.exec(statement)
        interaction = interaction.first()
        await self.session.delete(interaction)
        await self.session.commit()
        return interaction

    async def list_resources(self) -> list[Resource]:
        statement = select(Resource)
        resources = await self.session.exec(statement)
        return resources.all()
    
    async def list_users(self) -> list[User]:
        statement = select(User)
        users = await self.session.exec(statement)
        return users.all()
    
    async def list_interactions(self) -> list[Interaction]:
        statement = select(Interaction)
        interactions = await self.session.exec(statement)
        return interactions.all()
    
    async def create_user(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def create_resource(self, resource: Resource) -> Resource:
        self.session.add(resource)
        await self.session.commit()
        await self.session.refresh(resource)
        return resource