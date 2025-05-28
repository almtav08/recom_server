import asyncio
import json
from moodle import send_moodle_request, get_enrolled_students

async def load_moodle_info():
    course_id = ''
    while not course_id:
        course_id = input('Please provide the moodle course id: ')

    students = await send_moodle_request(get_enrolled_students(course_id))
    students_info = []
    for student in students:
        students_info.append({'id': student['id'], 'username': student['username'], 'email': student['email']})

    with open('database/students.json', 'w', encoding='utf-8') as json_file:
        json.dump(students_info, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(load_moodle_info())