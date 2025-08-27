import sys
import asyncio
import json

sys.path.append('./')
from api_data import send_moodle_request, get_course_logs

async def load_moodle_info():
    course_id = ''
    while not course_id:
        course_id = input('Please provide the moodle course id: ')

    logs = await send_moodle_request(get_course_logs(course_id))
    logs_info = []
    students_last_interaction = {}
    for log in logs:
        if log['userid'] in students_last_interaction and log['contextinstanceid'] == students_last_interaction[log['userid']]:
            continue
        logs_info.append({'resource_id': log['contextinstanceid'], 'user_id': log['userid'], 'timestamp': log['timecreated']})
        students_last_interaction[log['userid']] = log['contextinstanceid']

    with open('database/logs.json', 'w', encoding='utf-8') as json_file:
        json.dump(logs_info, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(load_moodle_info())