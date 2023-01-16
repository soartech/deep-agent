from typing import NamedTuple, Tuple


class GoalAreaData(NamedTuple):
    location: Tuple[int, int]
    size: Tuple[int, int]


sc2_map_goal_areas = {
    "canyon_001": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "canyon_001_light": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "test_logans_run_004": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "test_logans_run_004_melee": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "test_logans_run_004_melee_switched": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "test_logans_run_004_melee_6": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "test_logans_run_004_melee_6_switched": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))],
    "test_logans_run_004_2xV6x_switched": [GoalAreaData((88, 22), (20, 15)), GoalAreaData((138, 60), (12, 12))]
}