from .g1.g1 import G1
from .go2.go2 import Go2

ROBOT_CLASS_MAP = {
    "go2": Go2,
    "g1": G1,
}


def resolve_robot(task: str, logger):
    for key in ROBOT_CLASS_MAP:
        if key in task.lower():
            return ROBOT_CLASS_MAP[key](task=task, logger=logger)
    raise ValueError(f"Unknown robot type in task: {task}")
