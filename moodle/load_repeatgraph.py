import os
import sys
import json
from typing import Dict, List
import dearpygui.dearpygui as dpg
sys.path.append('./')
from database.query_gen import QueryGenerator

course_modules = []  # loaded at runtime
GRAPH_PATH = "./moodle/data/repeat_graph.json"
USE_RECID = True  # now relationships are based on recid instead of id


# ----------------- Utilities -----------------
def _log(msg: str, color=(200, 200, 200, 255)):
    if dpg.does_item_exist("log_text"):
        dpg.configure_item("log_text", default_value=msg, color=color)


def _read_inputs() -> Dict[int, List[int]]:
    """Read user-entered repetition relations keyed by recid (or id if toggled)."""
    graph: Dict[int, List[int]] = {}
    for m in course_modules:
        key_val = m.recid if USE_RECID else m.id
        tag = f"entry_{key_val}"
        if not dpg.does_item_exist(tag):
            continue
        raw = dpg.get_value(tag).strip()
        if not raw:
            continue
        try:
            rels = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            ident = f"recid {m.recid}" if USE_RECID else f"id {m.id}"
            raise ValueError(
                f"Invalid format for resource {ident}. Use comma separated integers."
            )
        graph[key_val] = rels
    return graph


def _validate_graph(graph: Dict[int, List[int]]):
    existing_keys = {m.recid if USE_RECID else m.id for m in course_modules}
    key_label = "RecID" if USE_RECID else "ID"
    errors = []
    # 1. Non existing keys & self refs
    for src, targets in graph.items():
        for d in targets:
            if d not in existing_keys:
                errors.append(f"{key_label} {d} does not exist (referenced by {src})")
            if d == src:
                errors.append(f"Resource {src} references itself")
    # 2. Duplicates
    for src, targets in graph.items():
        if len(targets) != len(set(targets)):
            errors.append(f"Duplicate {key_label}s in repetition of {src}")
    # 3. Cycle detection - simple DFS
    visit = {}

    def dfs(n, stack):
        state = visit.get(n, 0)
        if state == 1:
            errors.append(f"Cycle detected: {' -> '.join(map(str, stack + [n]))}")
            return
        if state == 2:
            return
        visit[n] = 1
        for nxt in graph.get(n, []):
            dfs(nxt, stack + [n])
        visit[n] = 2

    for node in graph.keys():
        if visit.get(node, 0) == 0:
            dfs(node, [])
    return errors


def _load_existing_graph():
    if not os.path.exists(GRAPH_PATH):
        _log("repeat_graph.json does not exist, starting empty.")
        return
    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Heuristic: if keys don't match recids but match ids, offer conversion
        recids = {m.recid for m in course_modules}
        ids = {m.id for m in course_modules}
        keys = set(int(k) for k in data.keys()) if isinstance(data, dict) else set()
        using_recid_file = keys.issubset(recids)
        using_id_file = keys.issubset(ids) and not using_recid_file
        if USE_RECID and using_id_file:
            # Convert id-based graph to recid-based using mapping
            id_to_recid = {m.id: m.recid for m in course_modules}
            converted = {}
            for old_id, rels in data.items():
                new_key = id_to_recid.get(int(old_id))
                if new_key is None:
                    continue
                converted[new_key] = [id_to_recid.get(r, r) for r in rels]
            data = converted
            _log("Loaded old ID-based graph; converted to RecID.", (200, 200, 120, 255))
        for key, rels in data.items():
            tag = f"entry_{key}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, ",".join(str(r) for r in rels))
        _log("Repetition graph loaded.", (120, 220, 120, 255))
    except Exception as e:
        _log(f"Error loading file: {e}", (255, 120, 120, 255))


# ----------------- Callbacks -----------------
def save_relations_callback():
    try:
        graph = _read_inputs()
    except ValueError as e:
        _log(str(e), (255, 120, 120, 255))
        return
    errors = _validate_graph(graph)
    if errors:
        _log(f"Not saved. Errors: {' | '.join(errors)}", (255, 180, 120, 255))
        return
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    _log("Relations saved successfully (key=RecID).", (120, 220, 120, 255))


def validate_callback():
    try:
        graph = _read_inputs()
    except ValueError as e:
        _log(str(e), (255, 120, 120, 255))
        return
    errors = _validate_graph(graph)
    if errors:
        _log(f"Errors: {' | '.join(errors)}", (255, 180, 120, 255))
    else:
        _log("Validation OK (no errors).", (120, 220, 120, 255))


def clear_callback():
    for m in course_modules:
        tag = f"entry_{m.recid if USE_RECID else m.id}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, "")
    _log("Fields cleared.")


def load_callback():
    _load_existing_graph()


# ----------------- UI Construction -----------------
def build_ui():
    with dpg.window(
        tag="main_window", label="Moodle Repetitions", width=900, height=600
    ):
        dpg.add_text("Define repetition relations (IDs separated by commas)")
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save", callback=save_relations_callback)
            dpg.add_button(label="Validate", callback=validate_callback)
            dpg.add_button(label="Load", callback=load_callback)
            dpg.add_button(label="Clear", callback=clear_callback)
        dpg.add_spacer(height=4)
        dpg.add_text("", tag="log_text")
        dpg.add_spacer(height=6)
        # Resources table
        with dpg.child_window(border=True, autosize_x=True, autosize_y=True):
            with dpg.table(
                header_row=True,
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                scrollX=True,
                scrollY=True,
                freeze_rows=1,
            ):
                dpg.add_table_column(label="RecID")
                dpg.add_table_column(label="Name")
                dpg.add_table_column(label="Type")
                dpg.add_table_column(label="Repetitions (RecIDs)")
                for m in course_modules:
                    with dpg.table_row():
                        recid = getattr(m, "recid", None)
                        dpg.add_text(str(recid) if recid is not None else "")
                        dpg.add_text(m.name)
                        dpg.add_text(getattr(m, "type", ""))
                        dpg.add_input_text(
                            tag=f"entry_{recid if USE_RECID else m.id}",
                            hint="1,2,3",
                            width=250,
                        )


def main():
    global course_modules
    client = QueryGenerator()
    client.connect()
    course_modules = client.list_resources()
    client.disconnect()

    dpg.create_context()
    dpg.create_viewport(title="Moodle Repetitions", width=930, height=640)
    build_ui()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    _load_existing_graph()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
