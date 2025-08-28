import time
import secrets
from typing import List
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from database.api_models import CreateInteraction, CreateUser, RecommendedResource
from database.async_query_gen import AsyncQueryGenerator
from database.models import Interaction, Resource, User
from moodle.api_data import get_message_recommendation, send_moodle_request
from recommender.modules.recommender.hybrid import HybridRecommender

client = AsyncQueryGenerator()
recom = HybridRecommender()

@asynccontextmanager
async def lifespan(app: FastAPI):
    client.connect()
    await recom.load(client)
    yield
    if client.is_connected():
        await client.disconnect()
    recom.unload()

app = FastAPI(title="e-DIPLOMA Recomendation API", version="0.1.0", lifespan_mode="async", lifespan=lifespan, docs_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir acceso desde cualquier origen
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Permitir los métodos HTTP especificados
    allow_headers=["*"],  # Permitir todos los encabezados
)

security = HTTPBasic()

async def check_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    user = await client.select_user_by_username(credentials.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    correct_username = secrets.compare_digest(credentials.username, user.username)
    correct_password = secrets.compare_digest(credentials.password, user.password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/docs", include_in_schema=False)
async def get_documentation(username: str = Depends(check_credentials)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

@app.get("/openapi.json", include_in_schema=False)
async def openapi(username: str = Depends(check_credentials)):
    return get_openapi(title = "FastAPI", version="0.1.0", routes=app.routes)

@app.get("/recommendations/{user_id}/{top}", tags=["Recommendations"], summary="Get recommendations for a user")
async def recommend(user_id: int, top: int) -> List[RecommendedResource]:
    db_user = await client.select_user(user_id)
    if db_user is None:
        return {"error": "User not found"}
    
    recommendations = await recom.recommend(db_user, top, client)

    return recommendations

@app.get("/resources", tags=["Resources"], summary="Get all resources")
async def list_resources() -> List[Resource]:
    db_resources = await client.list_resources()
    return db_resources

@app.get("/resource/{resource_id}", tags=["Resources"], summary="Get a resource by ID")
async def get_resource(resource_id: int) -> Resource:
    db_resource = await client.select_resource(resource_id)
    if db_resource is None:
        return {"error": "Resource not found"}
    return db_resource

@app.get("/interactions/{user_id}", tags=["Users"], summary="Get all interactions from a user")
async def get_interaction_by_user(user_id: int) -> List[Interaction]:
    db_user = await client.select_user(user_id)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
            headers={"WWW-Authenticate": "Basic"},
        )
    db_interactions = await client.select_interactions_by_user(user_id)
    return db_interactions

@app.post("/append_log", tags=["Users"], summary="Create a new interaction for a user")
async def post_interaction(interaction: CreateInteraction) -> Interaction:
    db_user = await client.select_user(interaction.userid)
    if db_user is None:
        return {"error": "User not found"}
    db_resource = await client.select_resource(interaction.cmid)
    if db_resource is None:
        return {"error": "Resource not found"}
    # Add time as a the current timestamp
    interaction_time = str(time.time())
    return await client.insert_interaction(Interaction(user_id=interaction.userid, resource_id=interaction.cmid, timestamp=interaction_time, passed=interaction.passed))

@app.post("/create", tags=["Users"], summary="Create a new user")
async def post_interaction(user: CreateUser) -> User:
    return await client.create_user(User(username=user.username, password=user.password, email=user.email))

@app.delete("/interaction/{interaction_id}", tags=["Users"], summary="Delete an interaction")
async def remove_intearction(interaction_id: int):
    db_interaction = await client.select_interaction(interaction_id)
    if db_interaction is None:
        return {"error": "Interaction not found"}
    return await client.delete_interaction(interaction_id)

@app.get("/train", tags=["Retrain"], summary="Retrain the recommendation model")
async def train():
    recom.train()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)