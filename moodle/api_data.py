import os
import httpx
from typing import List
from dotenv import load_dotenv
from fastapi import HTTPException
from database.models import User, Resource

load_dotenv(override=True)

# Moodle API data
def get_course_logs(course_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_COURSE_LOGS"),
        "moodlewsrestformat": "json",
        "courseid": course_id
    }

def get_enrolled_students(course_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_ENROLLED_STUDENTS"),
        "moodlewsrestformat": "json",
        "courseid": course_id
    }

def get_course_by_id(course_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_COURSE_INFO"),
        "moodlewsrestformat": "json",
        "field": "id",
        "value": course_id
    }

def get_course_info(course_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_COURSE_CONTENTS"),
        "moodlewsrestformat": "json",
        "courseid": course_id
    }

def get_message_recommendation(user: User, recommendations: List[Resource]) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("SEND_MESSAGES_FUNC"),
        "moodlewsrestformat": "json",
        "messages[0][touserid]": user.id,
        "messages[0][text]": prepare_recommendation_msg_content(user.username, recommendations),
        "messages[0][textformat]": 0,
    }

def get_user_attempts(user_id: int, quiz_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_USER_QUIZ_ATTEMPTS"),
        "moodlewsrestformat": "json",
        "userid": user_id,
        "quizid": quiz_id,
        "status": "all"
    }

def get_review_attempt(attempt_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_REVIEW_ATTEMPT"),
        "moodlewsrestformat": "json",
        "attemptid": attempt_id,
    }

def get_user_gradereport(user_id: int, course_id: int) -> dict:
    return {
        "wstoken": os.getenv("PLATFORM_API_KEY"),
        "wsfunction": os.getenv("GET_GRADE_REPORT"),
        "moodlewsrestformat": "json",
        "courseid": course_id,
        "userid": user_id
    }

# Helper functions
async def send_moodle_request(params: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(os.getenv("PLATFORM_URL"), params=params)
    
    if response.status_code != 200 or "exception" in response.json():
        raise HTTPException(status_code=400, detail="Error connecting to Moodle")
    
    return response.json()

def prepare_recommendation_msg_content(user: str, recommendations: List[Resource]) -> str:
    message = f"¡Hola {user.capitalize()}! Aquí tienes algunas recomendaciones de recursos educativos que podrían interesarte:\n\n"
    for i, resource in enumerate(recommendations):
        message += f"{i + 1}. {resource.name}\n"
    return message