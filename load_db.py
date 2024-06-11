import os
import json
from datetime import datetime
from dotenv import load_dotenv
from db import QueryGenerator, Interaction, Resource, User

load_dotenv(override=True)

client = QueryGenerator()

def main() -> None:
    client.connect()

    # Add users data
    with open('database/students.json') as f:
        students = json.load(f)

    for student in students:
        client.create_user(User(id=student['id'], username=student['username'], email=student['email'], password=os.getenv("USER_API_PASS")))

    # Add items data
    with open('database/resources.json') as f:
        items = json.load(f)

    for item in items:
        if item['type'] == 'quiz':
            client.create_resource(Resource(id=item['id'], recid=item['recid'], name=item['name'], type=item['type'], quizid=item['quizid']))
        else:
            client.create_resource(Resource(id=item['id'], recid=item['recid'] , name=item['name'], type=item['type']))

    # Add interactions data
    with open('database/logs.json') as f:
        logs = json.load(f)

    for student in students:
        client.insert_interaction(Interaction(timestamp=logs[0]['timestamp'], user_id=student['id'], resource_id=0))

    # for log in logs:
    #     client.insert_interaction(Interaction(timestamp=log['timestamp'], user_id=log['user_id'], resource_id=log['resource_id']))

    os.remove('database/students.json')
    os.remove('database/resources.json')
    os.remove('database/logs.json')

if __name__ == "__main__":
    main()