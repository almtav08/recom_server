from typing import List
from fastapi import FastAPI
from contextlib import asynccontextmanager
from prisma import Prisma
from prisma.models import User as UserBase
from models import Resource, Interaction

client = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await client.connect()
    yield
    if client.is_connected():
        await client.disconnect()

app = FastAPI(title="e-DIPLOMA Recomendation API", version="0.1.0", lifespan_mode="async", lifespan=lifespan)

@app.get("/recomendation", response_model=List[Resource], tags=["Recomendation"])
async def recomendation(user_id: int):
    user = await client.user.find_first(where={"id": user_id})
    # Calc recomendation
    recomendations = await recomend(user)
    return recomendations

@app.post("/interaction", response_model=Resource, tags=["Interaction"])
async def interaction(interaction: Interaction):
    user = await client.user.find_first(where={"id": interaction.userId})
    resource = await client.resource.find_first(where={"id": interaction.resourceId})
    # Create interaction
    await client.resourceinteraction.create({"userId": user.id, "resourceId": resource.id, "state": interaction.state})
    return resource

@app.get("/notinteracted", response_model=List[Resource], tags=["Resources"])
async def not_interacted(user_id: int):
    user = await client.user.find_first(where={"id": user_id})
    # Get not interacted resources
    resources = await not_interacted(user)
    return resources

# uvicorn server:app --reload

async def not_interacted(user: UserBase) -> List[Resource]:
    non_interacted =  await client.resource.find_many(
        where={
            "NOT": {
                "resourceInteractions": {
                    "some": {
                        "userId": user.id
                    }
                }
            }
        }
    )
    return non_interacted

async def recomend(user: UserBase) -> List[Resource]:
    non_interacted =  await not_interacted(user)

    test_done = await client.resourceinteraction.find_first(
        where={"resourceId": 3, "userId": user.id}
    )

    if test_done is None:
        return [non_interacted[0]]

    if test_done.state == 1:
        return [non_interacted[0]]
    else:
        return [non_interacted[1]]