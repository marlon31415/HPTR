import os
import sys
import unittest

sys.path.append(".")

print("inside test")

from src.pack_h5_nuplan import (
    create_planner_input_from_scenario,
    collate_agent_features,
    collate_map_features,
    collate_route_features,
    collate_tl_features,
)
from src.utils.pack_h5_nuplan_utils import (
    get_nuplan_scenarios,
    FixedLengthDict,
)

NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")


class TestLoadDatasets(unittest.TestCase):
    def setUp(self):
        data_root = os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1/splits/mini")
        self.scenarios = get_nuplan_scenarios(
            data_root, NUPLAN_MAPS_ROOT, logs=["2021.07.16.20.45.29_veh-35_01095_01486"]
        )
        self.scenario = self.scenarios[0]

    def test_define_interesting_agents(self):
        pass

    def test_fixed_length_dict(self):
        max_length = 3
        fixed_length_dict = FixedLengthDict(max_length)
        self.assertEqual(fixed_length_dict.max_length, max_length)


if __name__ == "__main__":
    unittest.main()
