import os
import time
import secrets
from typing import List
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from database.api_models import CreateInteraction, CreateUser, RecommendedResource
from database.async_query_gen import AsyncQueryGenerator
from database.models import Interaction, Resource, User
from moodle.api_data import get_message_recommendation, send_moodle_request
from recommender.modules.recommender.hybrid import HybridRecommender

client = AsyncQueryGenerator()
recom = HybridRecommender()

load_dotenv(override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await client.connect()
    await recom.load(client)
    yield
    if client.is_connected():
        await client.disconnect()
    recom.unload()

app = FastAPI(title="e-DIPLOMA Recomendation API", version="0.1.0", lifespan_mode="async", lifespan=lifespan, docs_url=None)

# Configure CORS from env ALLOWED_ORIGINS (comma separated) or default to localhost dev
allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
if allowed.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Global middleware that enforces presence of a valid API key on every request,
# but allows configurable public paths to bypass authentication.
PUBLIC_PATHS = [p.strip() for p in os.getenv("PUBLIC_PATHS", "/docs,/openapi.json,/health,/ ").split(",") if p.strip()]


@app.middleware("http")
async def require_api_key_middleware(request: Request, call_next):
    path = request.url.path or "/"

    # Allow public paths (prefix match)
    for public in PUBLIC_PATHS:
        if public and path.startswith(public):
            return await call_next(request)

    auth_header = request.headers.get("authorization")
    xkey = request.headers.get("x-api-key")
    token = None
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    if not token and xkey:
        token = xkey
    print(f"Auth token: {token}")

    app_token = os.getenv("APP_API_TOKEN") or os.getenv("USER_API_PASS")
    if not app_token or not token or not secrets.compare_digest(token, app_token):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token"})

    response = await call_next(request)
    return response

# Provide both legacy HTTPBasic and a preferred Bearer token check.
security_basic = HTTPBasic()
security_bearer = HTTPBearer()


async def check_basic_credentials(credentials: HTTPBasicCredentials = Depends(security_basic)):
    user = await client.select_user_by_username(credentials.username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials")
    correct_username = secrets.compare_digest(credentials.username, user.username)
    correct_password = secrets.compare_digest(credentials.password, user.password)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials")
    return credentials.username


async def check_api_token(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    # Token is fetched from env APP_API_TOKEN or fallback to USER_API_PASS for compat.
    token = credentials.credentials
    app_token = os.getenv("APP_API_TOKEN") or os.getenv("USER_API_PASS")
    if not app_token or not secrets.compare_digest(token, app_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token")
    return True

@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/openapi.json", include_in_schema=False)
async def openapi():
    return get_openapi(title = "FastAPI", version="0.1.0", routes=app.routes)

@app.get("/recommendations/{user_id}/{top}", tags=["Recommendations"], summary="Get recommendations for a user")
async def recommend(user_id: int, top: int, token_ok: bool = Depends(check_api_token)) -> List[RecommendedResource]:
    db_user = await client.select_user(user_id)
    if db_user is None:
        return {"error": "User not found"}
    
    recommendations = await recom.recommend(db_user, top, client)

    return recommendations

@app.get("/resources", tags=["Resources"], summary="Get all resources")
async def list_resources(token_ok: bool = Depends(check_api_token)) -> List[Resource]:
    db_resources = await client.list_resources()
    return db_resources

@app.get("/resource/{resource_id}", tags=["Resources"], summary="Get a resource by ID")
async def get_resource(resource_id: int, token_ok: bool = Depends(check_api_token)) -> Resource:
    db_resource = await client.select_resource(resource_id)
    if db_resource is None:
        return {"error": "Resource not found"}
    return db_resource

@app.get("/interactions/{user_id}", tags=["Users"], summary="Get all interactions from a user")
async def get_interaction_by_user(user_id: int, token_ok: bool = Depends(check_api_token)) -> List[Interaction]:
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
async def post_interaction(interaction: CreateInteraction, token_ok: bool = Depends(check_api_token)) -> Interaction:
    if interaction.userid == 1: # Guest User
        return {"error": "Cannot add interactions for the guest user"}
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
async def post_interaction(user: CreateUser, token_ok: bool = Depends(check_api_token)) -> User:
    return await client.create_user(User(id=user.id, username=user.username, password=user.password, email=user.email))

@app.delete("/interaction/{interaction_id}", tags=["Users"], summary="Delete an interaction")
async def remove_intearction(interaction_id: int, token_ok: bool = Depends(check_api_token)):
    db_interaction = await client.select_interaction(interaction_id)
    if db_interaction is None:
        return {"error": "Interaction not found"}
    return await client.delete_interaction(interaction_id)

@app.get("/train", tags=["Retrain"], summary="Retrain the recommendation model")
async def train(token_ok: bool = Depends(check_api_token)):
    recom.train()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)