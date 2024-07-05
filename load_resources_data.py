import asyncio
import json
import tkinter as tk
from tkinter import messagebox
from moodle import get_course_info, send_moodle_request, get_course_by_id

def on_submit(course_name, var_list, course_data):
    selected_modules = [module for module, var in zip(course_data, var_list) if var.get()]
    selected_modules_names = [module["name"] for module in selected_modules]

    recid = 0
    selected_modules_info = []
    selected_modules_info.append({'id': 0, 'recid': recid, 'name': course_name, 'type': 'course'})
    for module in selected_modules:
        recid += 1
        if module['modname'] == 'quiz':
            selected_modules_info.append({'id': module['id'], 'recid': recid, 'name': str(module['name']).split('**')[0].strip(), 'type': module['modname'], 'quizid': module['instance']})
        else:
            selected_modules_info.append({'id': module['id'], 'recid': recid, 'name': str(module['name']).split('**')[0].strip(), 'type': module['modname']})
    
    # Crear un archivo JSON con los módulos seleccionados
    with open('database/resources.json', 'w', encoding='utf-8') as json_file:
        json.dump(selected_modules_info, json_file, ensure_ascii=False, indent=4)

    messagebox.showinfo("Selected Modules", "You have selected:\n" + "\n".join(selected_modules_names))

async def load_moodle_info():
    course_id = input('Please provide the moodle course id: ')

    course = await send_moodle_request(get_course_by_id(course_id))
    course_name = course['courses'][0]['fullname']

    course_data = await send_moodle_request(get_course_info(course_id))

    root = tk.Tk()
    root.title("Select Course Modules")

    tk.Label(root, text=f"Select the modules from course {course_name}:").pack(pady=10)
    
    var_list = []
    for section in course_data:
        for module in section['modules']:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(root, text=f"Moodle ID: {module['id']}, Name: {str(module['name']).split('**')[0].strip()}, Module Type: {module['modname']}", variable=var)
            chk.pack(anchor='w', padx=20)
            var_list.append(var)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit(course_name, var_list, [module for section in course_data for module in section['modules']]))
    submit_button.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    asyncio.run(load_moodle_info())