from __future__ import annotations

import sys
import asyncio
import json
import os
from typing import Any, List
import dearpygui.dearpygui as dpg

sys.path.append("./")
from api_data import get_course_info, send_moodle_request, get_course_by_id

# ---------------- State -----------------
_course_modules: List[dict[str, Any]] = []
_course_name: str = ""
_checkbox_tags: list[str] = []
RESOURCES_PATH = "database/resources.json"


def _log(msg: str, color=(200, 200, 200, 255)):
    if dpg.does_item_exist("log_text"):
        dpg.configure_item("log_text", default_value=msg, color=color)


def _ensure_dirs():
    os.makedirs(os.path.dirname(RESOURCES_PATH), exist_ok=True)


def _build_modules_table():
    if not _course_modules:
        return
    if dpg.does_item_exist("modules_table"):
        dpg.delete_item("modules_table")
    with dpg.table(
        tag="modules_table",
        parent="modules_region",
        header_row=True,
        resizable=True,
        policy=dpg.mvTable_SizingStretchProp,
        borders_innerH=True,
        borders_innerV=True,
        borders_outerH=True,
        borders_outerV=True,
        scrollY=True,
        height=-1,
    ):
        dpg.add_table_column(label="Select")
        dpg.add_table_column(label="cmid")
        dpg.add_table_column(label="Name")
        dpg.add_table_column(label="Type")
        dpg.add_table_column(label="Quiz ID")
        _checkbox_tags.clear()
        for m in _course_modules:
            with dpg.table_row():
                chk_tag = f"chk_{m['cmid']}"
                dpg.add_checkbox(tag=chk_tag)
                _checkbox_tags.append(chk_tag)
                dpg.add_text(str(m["cmid"]))
                dpg.add_text(m["name"])
                dpg.add_text(m["modulename"])
                dpg.add_text(
                    str(m.get("instance", "")) if m["modulename"] == "quiz" else ""
                )


def _collect_selection() -> list[dict[str, Any]]:
    selected = []
    for m in _course_modules:
        if dpg.get_value(f"chk_{m['cmid']}"):
            selected.append(m)
    return selected


def _save_selected(selected: list[dict[str, Any]]):
    recid = 0
    resources = [{"id": 0, "recid": recid, "name": _course_name, "type": "course"}]
    for m in selected:
        recid += 1
        base = {
            "id": m["cmid"],
            "recid": recid,
            "name": m["name"],
            "type": m["modulename"],
        }
        if m["modulename"] == "quiz":
            base["quizid"] = m["instance"]
        resources.append(base)
    _ensure_dirs()
    with open(RESOURCES_PATH, "w", encoding="utf-8") as f:
        json.dump(resources, f, ensure_ascii=False, indent=4)
    return resources


# ---------------- Async Ops -----------------
async def _fetch_course_and_modules(course_id: str):
    global _course_modules, _course_name
    course_raw = None
    modules_raw = None
    try:
        course_raw = await send_moodle_request(get_course_by_id(course_id))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed getting course metadata for id {course_id}: {e}"
        ) from e
    try:
        _course_name = course_raw["courses"][0]["fullname"]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Unexpected course JSON structure: {course_raw}") from e
    try:
        modules_raw = await send_moodle_request(get_course_info(course_id))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed getting course modules for id {course_id}: {e}"
        ) from e
    if not isinstance(modules_raw, list):
        raise RuntimeError(f"Modules JSON not a list: {modules_raw}")
    # Filtrar entradas vacías o sin campos claves
    filtered = [m for m in modules_raw if isinstance(m, dict) and "cmid" in m]
    _course_modules = filtered


def _load_course_callback():
    course_id = dpg.get_value("course_id_input").strip()
    if not course_id:
        _log("Please provide a course ID", (255, 120, 120, 255))
        return
    _log("Loading course data...")

    async def runner():
        try:
            await _fetch_course_and_modules(course_id)
        except Exception as e:  # noqa: BLE001
            # Mostrar error detallado y guardar en archivo para depurar
            _log(f"Error: {e}", (255, 120, 120, 255))
            try:
                with open("load_error.log", "a", encoding="utf-8") as lf:
                    lf.write(f"CourseID={course_id} ERROR={e}\n")
            except Exception:
                pass
            return
        # Update UI after fetch
        dpg.configure_item("course_title", default_value=f"Course: {_course_name}")
        _build_modules_table()
        _log("Modules loaded", (120, 220, 120, 255))

    # Run coroutine (blocking is acceptable here; network time halts UI briefly)
    asyncio.run(runner())


def _submit_callback():
    selected = _collect_selection()
    if not selected:
        _log("No modules selected", (255, 180, 120, 255))
        return
    resources = _save_selected(selected)
    _log(f"Saved {len(resources)-1} modules to {RESOURCES_PATH}", (120, 220, 120, 255))


def _select_all_callback():
    for tag in _checkbox_tags:
        dpg.set_value(tag, True)
    _log("All modules selected")


def _clear_selection_callback():
    for tag in _checkbox_tags:
        dpg.set_value(tag, False)
    _log("Selection cleared")


def _quit_callback():
    dpg.stop_dearpygui()


# ---------------- UI -----------------
def _build_ui():
    with dpg.window(
        tag="main_window", label="Moodle Module Loader", width=900, height=600
    ):
        dpg.add_text("Enter Moodle course ID and load modules to select")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="course_id_input", hint="Course ID", width=120)
            dpg.add_button(label="Load", callback=_load_course_callback)
            dpg.add_button(label="Submit", callback=_submit_callback)
            dpg.add_button(label="Select All", callback=_select_all_callback)
            dpg.add_button(label="Clear", callback=_clear_selection_callback)
            dpg.add_button(label="Quit", callback=_quit_callback)
        dpg.add_spacer(height=4)
        dpg.add_text("Course: (none)", tag="course_title")
        dpg.add_spacer(height=4)
        dpg.add_text("", tag="log_text")
        dpg.add_spacer(height=6)
        with dpg.child_window(
            tag="modules_region", border=True, autosize_x=True, autosize_y=True
        ):
            dpg.add_text("No modules loaded")


def main():
    dpg.create_context()
    dpg.create_viewport(title="Moodle Module Loader", width=930, height=640)
    _build_ui()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
