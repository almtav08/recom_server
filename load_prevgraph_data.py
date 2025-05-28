import asyncio
import json
import tkinter as tk
from tkinter import messagebox
from db import AsyncQueryGenerator

def save_relations(course_modules, all_entries):
    prev_graph = {}
    for module, entry in zip(course_modules, all_entries):
        if entry.get() == "":
            continue
        relations = [int(x.strip()) for x in entry.get().split(",")]
        prev_graph[module.id] = relations
        print(prev_graph)
    with open('./moodle/data/prev_graph.json', 'w') as og:
        json.dump(prev_graph, og)

    messagebox.showinfo("Relations", "The relations have been created")

async def load_moodle_info():
    client = AsyncQueryGenerator()
    client.connect()
    course_modules = await client.list_resources()
    await client.disconnect()
    course_modules = course_modules[1:]

    root = tk.Tk()
    root.title("Indicate resources previous relations separating the Ids by commas")

    all_entries = []

    for module in course_modules:
        frame = tk.Frame(root)
        frame.pack()

        label = tk.Label(frame, text=f"ID: {module.id}, Name: {module.name}")
        label.pack(side="left")

        entry = tk.Entry(frame)
        entry.pack(side="left")

        all_entries.append(entry)

    button = tk.Button(root, text="Save relations", command=lambda: save_relations(course_modules, all_entries))
    button.pack()

    root.mainloop()


if __name__ == "__main__":
    asyncio.run(load_moodle_info())