import asyncio
import json
import re
import tkinter as tk
from tkinter import messagebox
from pick import pick
from moodle import get_course_info, send_moodle_request, get_course_by_id

def on_submit(course_name, var_list, course_data, root: tk.Tk):
    selected_modules = [module for module, var in zip(course_data, var_list) if var.get()]
    selected_modules_names = [module["name"] for module in selected_modules]

    recid = 0
    selected_modules_info = []
    selected_modules_info.append({'id': 0, 'recid': recid, 'name': course_name, 'type': 'course'})
    for module in selected_modules:
        recid += 1
        if module['modulename'] == 'quiz':
            selected_modules_info.append({'id': module['cmid'], 'recid': recid, 'name': module['name'], 'type': module['modulename'], 'quizid': module['instance']})
        else:
            selected_modules_info.append({'id': module['cmid'], 'recid': recid, 'name': module['name'], 'type': module['modulename']})
    
    # Crear un archivo JSON con los módulos seleccionados
    with open('database/resources.json', 'w', encoding='utf-8') as json_file:
        json.dump(selected_modules_info, json_file, ensure_ascii=False, indent=4)

    messagebox.showinfo("Selected Modules", "You have selected:\n" + "\n".join(selected_modules_names))
    root.destroy()

async def load_moodle_info():
    course_id = ''
    while not course_id:
        course_id = input('Please provide the moodle course id: ')

    course = await send_moodle_request(get_course_by_id(course_id))
    course_name = course['courses'][0]['fullname']

    course_data = await send_moodle_request(get_course_info(course_id))

    root = tk.Tk()
    root.title("Select Course Modules")

    tk.Label(root, text=f"Select the modules from course {course_name}:").pack(pady=10)
    
    var_list = []
    for module in course_data:
        var = tk.BooleanVar()
        chk = tk.Checkbutton(root, text=f"Moodle ID: {module['cmid']}, Name: {module['name']}, Module Type: {module['modulename']}", variable=var)
        chk.pack(anchor='w', padx=20)
        var_list.append(var)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit(course_name, var_list, [module for module in course_data], root))
    submit_button.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    asyncio.run(load_moodle_info())