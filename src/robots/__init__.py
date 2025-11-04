from .g1.g1 import G1, G1Fixed, G1Lower, G1Upper
from .go2.go2 import Go2

ROBOT_CLASS_MAP = {
    "go2": Go2,
    "g1_upper": G1Upper,
    "g1_fixed": G1Fixed,
    "g1_lower": G1Lower,
    "g1": G1,
}


def resolve_robot(task: str, logger, device="cuda:0"):
    for key in ROBOT_CLASS_MAP:
        if key == task.split("-")[-1].lower():
            return ROBOT_CLASS_MAP[key](task=task, logger=logger, device=device)
    raise ValueError(f"Unknown robot type in task: {task}")
