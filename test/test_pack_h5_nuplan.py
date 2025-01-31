import os
import unittest
import numpy as np

from pack_h5_nuplan import (
    N_STEP,
    STEP_CURRENT,
    PL_TYPES,
    N_AGENT_PRED_CHALLENGE,
    N_AGENT_INTERACT_CHALLENGE,
    create_planner_input_from_scenario,
    collate_agent_features,
    collate_map_features,
    collate_route_features,
    collate_tl_features,
)
from utils.nuplan.pack_h5_nuplan_utils import get_nuplan_scenarios
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)

NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")

# DO NOT CHANGE -> TESTS DEPEND ON IT!
DATA_LOGS = ["2021.07.16.20.45.29_veh-35_01095_01486"]


class TestPackNuplan(unittest.TestCase):
    def setUp(self):
        data_root = os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1/splits/mini")
        # Load scenarios from dataset sample
        scenarios = get_nuplan_scenarios(data_root, NUPLAN_MAPS_ROOT, logs=DATA_LOGS)
        for scenario in scenarios:
            if scenario.token == "f508606982e65143":  # b4e906c2ed9c597a (no tl data)
                self.scenario = scenario
                break

        self.initialization, self.planner_input = create_planner_input_from_scenario(
            self.scenario, 0
        )
        self.assertIsInstance(self.initialization, PlannerInitialization)
        self.assertIsInstance(self.planner_input, PlannerInput)
        self.map_api = self.initialization.map_api
        self.past_observations = [
            obs.tracked_objects.get_agents()
            for obs in self.planner_input.history.observations
        ]
        self.past_ego_states = [
            ego_state.agent for ego_state in self.planner_input.history.ego_states
        ]
        self.scenario_center = self.planner_input.history.ego_states[-1].center.point

    def test_collate_agent_features(self):
        only_agents = True
        agent_id, agent_type, agent_states, agent_role = collate_agent_features(
            self.scenario_center,
            self.past_ego_states,
            self.past_observations,
            N_STEP,
            only_agents=only_agents,
        )
        self.assertEqual(len(agent_id), len(agent_type))
        self.assertEqual(len(agent_id), len(agent_states))
        self.assertEqual(len(agent_id), len(agent_role))
        self.assertEqual(len(agent_states[0]), N_STEP)
        # Test if agent_role aligns with prediction and interaction challenge
        n_predict = np.array(agent_role)[:, 2].sum()
        if len(agent_id) >= N_AGENT_PRED_CHALLENGE:
            self.assertEqual(n_predict, N_AGENT_PRED_CHALLENGE)
        else:
            self.assertEqual(n_predict, len(agent_id))
        n_interact = np.array(agent_role)[:, 1].sum()
        if len(agent_id) >= N_AGENT_INTERACT_CHALLENGE:
            self.assertEqual(n_interact, N_AGENT_INTERACT_CHALLENGE)
        else:
            self.assertEqual(n_interact, len(agent_id))
        n_ego = np.array(agent_role)[:, 0].sum()
        self.assertEqual(n_ego, 1)
        # Test if only vehicles (0), pedestrians (1) and bicycles (2) are present
        if only_agents:
            for type in agent_type:
                self.assertLessEqual(type, 2)

    def test_collate_map_features(self):
        mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(
            self.map_api, self.scenario_center
        )
        self.assertEqual(len(mf_id), len(mf_xyz))
        self.assertEqual(len(mf_id), len(mf_type))

    def test_collate_route_features(self):
        sdc_id, sdc_route_id, sdc_route_type, sdc_route_xyz, sdc_route_goal = (
            collate_route_features(
                self.map_api,
                self.scenario_center,
                self.initialization.route_roadblock_ids,
                self.initialization.mission_goal,
            )
        )
        self.assertEqual(sdc_id[0], -1)
        self.assertEqual(len(sdc_route_id), len(sdc_route_xyz))
        self.assertEqual(len(sdc_route_id), len(sdc_route_type))
        for type in sdc_route_type[0]:
            self.assertEqual(type, 7)

    def test_collate_tl_features(self):
        tl_lane_state, tl_lane_id, tl_stop_point = collate_tl_features(
            self.map_api,
            self.scenario_center,
            self.planner_input.traffic_light_data,
            N_STEP,
            STEP_CURRENT,
        )
        print(tl_lane_state)
        print(tl_lane_id)
        print(tl_stop_point)
        self.assertEqual(len(tl_lane_state), N_STEP)
        self.assertEqual(len(tl_lane_id), N_STEP)
        self.assertEqual(len(tl_stop_point), N_STEP)
        for i in range(N_STEP):
            self.assertEqual(len(tl_lane_state[i]), len(tl_lane_id[i]))
            self.assertEqual(len(tl_lane_state[i]), len(tl_stop_point[i]))
            # No traffic light data besides in current step
            if i == STEP_CURRENT:
                self.assertGreaterEqual(len(tl_lane_state[i]), 0)
            else:
                self.assertEqual(len(tl_lane_state[i]), 0)

    def test_pl_types(self):
        # If test fails, PL_TYPES has been modified:
        # Note that this requires adaptions in pack_h5_nuplan.py
        self.assertEqual(len(PL_TYPES), 8)
        self.assertEqual(PL_TYPES["INTERSECTION"], 0)
        self.assertEqual(PL_TYPES["STOP_LINE"], 1)
        self.assertEqual(PL_TYPES["CROSSWALK"], 2)
        self.assertEqual(PL_TYPES["WALKWAYS"], 3)
        self.assertEqual(PL_TYPES["BOUNDARIES"], 4)
        self.assertEqual(PL_TYPES["CARPARK_AREA"], 5)
        self.assertEqual(PL_TYPES["CENTERLINE"], 6)
        self.assertEqual(PL_TYPES["ROUTE"], 7)


if __name__ == "__main__":
    unittest.main()
