import os
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlmodel import SQLModel, Session, select, create_engine
from .db_schema import Interaction, Resource, User

load_dotenv(override=True)

class QueryGenerator:
    def __init__(self):
        self.engine: Engine = None

    def connect(self):
        self.engine = create_engine(os.getenv("DATABASE_URL"), echo=False)
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def is_connected(self):
        return self.engine is not None

    def disconnect(self):
        self.session.close()
        self.engine.dispose()
    
    def select_user(self, user_id: int) -> User:
        statement = select(User).where(User.id == user_id)
        user = self.session.exec(statement).first()
        return user
    
    def select_user_by_username(self, username: str) -> User:
        statement = select(User).where(User.username == username)
        user = self.session.exec(statement).first()
        return user

    def select_resource(self, resource_id: int) -> Resource:
        statement = select(Resource).where(Resource.id == resource_id)
        resource = self.session.exec(statement).first()
        return resource

    def select_interactions_by_user(self, user_id: int) -> list[Interaction]:
        statement = select(Interaction).where(Interaction.user_id == user_id)
        interactions = self.session.exec(statement).all()
        return interactions
    
    def select_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = self.session.exec(statement).first()
        return interaction

    def insert_interaction(self, interaction: Interaction) -> Interaction:
        self.session.add(interaction)
        self.session.commit()
        self.session.refresh(interaction)
        return interaction

    def delete_interaction(self, interaction_id: int) -> Interaction:
        statement = select(Interaction).where(Interaction.id == interaction_id)
        interaction = self.session.exec(statement).first()
        self.session.delete(interaction)
        self.session.commit()
        return interaction

    def list_resources(self) -> list[Resource]:
        statement = select(Resource).order_by(Resource.recid.asc())
        resources = self.session.exec(statement).all()
        return resources
    
    def list_users(self) -> list[User]:
        statement = select(User).order_by(User.id.asc())
        users = self.session.exec(statement).all()
        return users
    
    def list_interactions(self) -> list[Interaction]:
        statement = select(Interaction)
        interactions = self.session.exec(statement).all()
        return interactions
    
    def create_user(self, user: User) -> User:
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user
    
    def create_resource(self, resource: Resource) -> Resource:
        self.session.add(resource)
        self.session.commit()
        self.session.refresh(resource)
        return resource