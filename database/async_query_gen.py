import os
from dotenv import load_dotenv
from sqlalchemy.orm import registry
from sqlalchemy import Engine, MetaData
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from database.models import Interaction, Resource, User

load_dotenv(override=True)


class AsyncQueryGenerator:
    def __init__(self):
        self.engine: Engine = None

    def connect(self):
        self.engine = create_async_engine(os.getenv("DATABASE_ASYNC_URL"), echo=False)
        self.session = AsyncSession(self.engine)

    def clear(self):
        SQLModel.metadata.drop_all(self.engine)
        SQLModel.metadata.clear()
        SQLModel.metadata = MetaData()
        SQLModel.registry = registry()

    def is_connected(self):
        return self.engine is not None

    async def disconnect(self):
        await self.session.close()
        await self.engine.dispose()

    async def select_user(self, user_id: int) -> User:
        statement = select(User).where(User.id == user_id)
        user = await self.session.exec(statement)
        user = user.first()
        return user

    async def select_user_by_username(self, username: str) -> User:
        statement = select(User).where(User.username == username)
        user = await self.session.exec(statement)
        user = user.first()
        return user

    async def select_users_by_pass(self, is_pass: bool = True) -> list[User]:
        statement = select(User).where(User.is_pass == is_pass)
        users = await self.session.exec(statement)
        users = users.all()
        return users

    async def select_users_by_resource_and_passed(
        self, res_id: int, passed: bool = True
    ):
        statement = (
            select(User)
            .join(Interaction)
            .where(
                User.is_pass == True,
                Interaction.resource_id == res_id,
                Interaction.passed == passed,
            )
        )
        users = await self.session.exec(statement)
        users = users.all()
        return users

    async def select_resource(self, resource_id: int) -> Resource:
        statement = select(Resource).where(Resource.id == resource_id)
        resource = await self.session.exec(statement)
        resource = resource.first()
        return resource
    
    async def select_resource_by_recid(self, resource_id: int) -> Resource:
        statement = select(Resource).where(Resource.recid == resource_id)
        resource = await self.session.exec(statement)
        resource = resource.first()
        return resource

    async def select_interactions_by_user(self, user_id: int) -> list[Interaction]:
        statement = select(Interaction).where(Interaction.user_id == user_id)
        interactions = await self.session.exec(statement)
        interactions = interactions.all()
        return interactions

    async def select_interactions_by_user_and_resource(
        self, user_id: int, resource_id: int
    ) -> list[Interaction]:
        statement = select(Interaction).where(
            Interaction.user_id == user_id, Interaction.resource_id == resource_id
        )
        interactions = await self.session.exec(statement)
        interactions = interactions.all()
        return interactions

    async def select_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = await self.session.exec(statement)
        interaction = interaction.first()
        return interaction

    async def insert_interaction(self, interaction: Interaction) -> Interaction:
        self.session.add(interaction)
        await self.session.commit()
        await self.session.refresh(interaction)
        return interaction

    async def insert_interactions(
        self, interactions: list[Interaction]
    ) -> list[Interaction]:
        for interaction in interactions:
            self.session.add(interaction)
        await self.session.commit()
        for interaction in interactions:
            await self.session.refresh(interaction)
        return interactions

    async def delete_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = await self.session.exec(statement)
        interaction = interaction.first()
        await self.session.delete(interaction)
        await self.session.commit()
        return interaction

    async def list_resources(self) -> list[Resource]:
        statement = select(Resource).order_by(Resource.recid.asc())
        resources = await self.session.exec(statement)
        resources = resources.all()
        return resources

    async def list_users(self) -> list[User]:
        statement = select(User).order_by(User.id.asc())
        users = await self.session.exec(statement)
        users = users.all()
        return users

    async def list_interactions(self) -> list[Interaction]:
        statement = select(Interaction)
        interactions = await self.session.exec(statement)
        interactions = interactions.all()
        return interactions

    async def create_user(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def create_users(self, users: list[User]) -> list[User]:
        for user in users:
            self.session.add(user)
        await self.session.commit()
        for user in users:
            await self.session.refresh(user)
        return users

    async def create_resource(self, resource: Resource) -> Resource:
        self.session.add(resource)
        await self.session.commit()
        await self.session.refresh(resource)
        return resource

    async def create_resources(self, resources: list[Resource]) -> list[Resource]:
        for resource in resources:
            self.session.add(resource)
        await self.session.commit()
        for resource in resources:
            await self.session.refresh(resource)
        return resources
