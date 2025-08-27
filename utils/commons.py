from itertools import groupby
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from database.models import User
from moodle.api_data import get_user_gradereport, send_moodle_request, get_user_attempts, get_review_attempt


def torch_cosine_similarity(embedding1, embedding2):
    dot_product = torch.dot(embedding1, embedding2)
    norm_embedding1 = torch.linalg.norm(embedding1)
    norm_embedding2 = torch.linalg.norm(embedding2)
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    return similarity.cpu().numpy()[()]


def calc_user_path(user: User) -> torch.tensor:
    # path = []
    # for resource in user.resources:
    #     if len(path) == 0 or path[-1] != resource.recid:
    #         path.append(resource.recid)
    # return path
    return torch.tensor([recid for recid, _ in groupby(map(lambda r: r.recid, user.resources))], dtype=torch.long)


async def load_grades(user_id: int, course_id: int) -> List[float]:
    response = await send_moodle_request(get_user_gradereport(user_id, course_id))
    res = response['usergrades'][0]['gradeitems']
    max_grade = []
    scores = []
    for item in res:
        max_grade.append(item['grademax'] if item['grademax'] is not None else 0)
        scores.append(item['graderaw'] if item['graderaw'] is not None else 0)
    real = []
    for item in range(len(scores)):
        if max_grade[item] > 0:
            real.append(scores[item] / max_grade[item] * 10)
        else:
            real.append(0)
    return real


async def check_pass_quiz(user_id: int, quiz_id: int) -> Tuple[bool, List[bool]]:
        # Obtain last attempt data
        attempts = await send_moodle_request(get_user_attempts(user_id, quiz_id))
        last_attempt_id = int(attempts['attempts'][-1]['id'])

        # Check if the last attempt was successful
        result = await send_moodle_request(get_review_attempt(last_attempt_id))
        questions_grades = []
        maxmark, mark = 0, 0
        for question in result['questions']:
            maxmark += float(question['maxmark'])
            mark += float(question['mark'])
            questions_grades.append(mark == maxmark)
        return (mark >= maxmark * 0.5, [])