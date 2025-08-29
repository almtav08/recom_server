import asyncio
import time
import sys
import os
import json
from dotenv import load_dotenv

sys.path.append("./")
from async_query_gen import AsyncQueryGenerator
from database.models import Interaction, Resource, User
from utils.commons import load_grades

load_dotenv(override=True)

client = AsyncQueryGenerator()

async def main() -> None:
    await client.clear()
    await client.create()
    await client.connect()

    # Add users data
    with open('database/students.json') as f:
        students = json.load(f)

    for student in students:
        try:
            grades = await load_grades(student['id'], 4)
        except Exception as e:
            # If Moodle is unreachable or the request fails, continue with
            # an empty grades list so we still create the user records.
            print(f"Warning: could not load grades for user {student['id']}: {e}")
            grades = []

        avg_grade = sum(grades) / len(grades) if grades else 0
        is_pass = avg_grade >= 5.0
        await client.create_user(User(id=student['id'], username=student['username'], email=student['email'], password=os.getenv("USER_API_PASS"), is_pass=is_pass))

    # Add items data
    with open('database/resources.json') as f:
        items = json.load(f)

    for item in items:
        if item['type'] == 'quiz':
            await client.create_resource(Resource(id=item['id'], recid=item['recid'], name=item['name'], type=item['type'], quizid=item['quizid']))
        else:
            await client.create_resource(Resource(id=item['id'], recid=item['recid'] , name=item['name'], type=item['type']))

    # Add interactions data
    with open('database/logs.json') as f:
        logs = json.load(f)

    for student in students:
        await client.insert_interaction(Interaction(timestamp=str(time.time()), user_id=student['id'], resource_id=0, passed=True))

    for log in logs:
        await client.insert_interaction(Interaction(timestamp=log['timestamp'], user_id=log['user_id'], resource_id=log['resource_id'], passed=log['passed']))

    await client.disconnect()
    # os.remove('database/students.json')
    # os.remove('database/resources.json')
    # os.remove('database/logs.json')

if __name__ == "__main__":
    asyncio.run(main())
