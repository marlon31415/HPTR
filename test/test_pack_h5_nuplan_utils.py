import os
import unittest
import numpy as np

from pack_h5_nuplan import create_planner_input_from_scenario
from utils.nuplan.pack_h5_nuplan_utils import (
    FixedLengthDict,
    get_nuplan_scenarios,
    nuplan_to_centered_vector,
    resample_polyline,
    normal_distance_2d_with_angle,
    create_rectangle_from_points,
    is_point_in_rectangle,
    extract_centerline,
    get_points_from_boundary,
    mock_2d_to_3d_points,
    get_route_lane_polylines_from_roadblock_ids,
    get_scenario_start_iter_tuple,
)

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")

# DO NOT CHANGE -> TESTS DEPEND ON IT!
DATA_LOGS = ["2021.07.16.20.45.29_veh-35_01095_01486"]


class TestPackNuplanUtils(unittest.TestCase):
    def setUp(self):
        data_root = os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1/splits/mini")
        # Load scenarios from dataset sample
        scenarios = get_nuplan_scenarios(data_root, NUPLAN_MAPS_ROOT, logs=DATA_LOGS)
        self.assertIsInstance(scenarios, list)
        for scenario in scenarios:
            if scenario.token == "b4e906c2ed9c597a":
                self.scenario = scenario
                break
        self.assertIsInstance(self.scenario, NuPlanScenario)

        self.initialization, self.planner_input = create_planner_input_from_scenario(
            self.scenario, 0
        )
        self.map_api = self.initialization.map_api
        self.scenario_center = self.planner_input.history.ego_states[-1].center.point

    def test_nuplan_to_centered_vector(self):
        vector = [1, 2]
        center = [1, 2]
        centered_vector = nuplan_to_centered_vector(vector=vector, nuplan_center=center)
        self.assertEqual(centered_vector[0], 0)
        self.assertEqual(centered_vector[1], 0)

    def test_resample_polyline(self):
        semantic_map_layers = self.map_api.get_proximal_map_objects(
            self.scenario_center, 100, [SemanticMapLayer.LANE]
        )
        # Extract lane with 6m length and 26 points
        lane = semantic_map_layers[SemanticMapLayer.LANE][0]
        lane_lenth = lane.baseline_path.length
        resample_distance = 1
        num_points_after_resample = int(lane_lenth) + 1
        resampled_polyline = resample_polyline(
            lane.baseline_path, lane.baseline_path.discrete_path, resample_distance
        )
        # Check if resampled polyline has the expected number of points
        # It can have one more point if mod(num_points_before_resample, resample_distance) == 0
        self.assertIn(
            len(resampled_polyline),
            [num_points_after_resample, num_points_after_resample + 1],
        )

    def test_fixed_length_dict(self):
        # Test if FixedLengthDict has a fixed length
        max_length = 3
        fixed_length_dict = FixedLengthDict(max_length)
        self.assertEqual(fixed_length_dict.max_length, max_length)
        for i in range(max_length + 1):
            fixed_length_dict.add(i, i)
        self.assertEqual(len(fixed_length_dict.data), max_length)
        # Get top k keys and remove them from the dictionary
        top_k = 2
        top_k_keys = fixed_length_dict.get_top_x_keys(top_k)
        self.assertEqual(len(top_k_keys), top_k)
        self.assertEqual(top_k_keys, [max_length, max_length - 1])
        self.assertEqual(len(fixed_length_dict.data), max_length - top_k)

    def test_normal_distance_2d_with_angle(self):
        point = [0, 1]
        point_on_line = [0, 0]
        angle = np.pi / 4
        normal_dist_from_line_to_point = normal_distance_2d_with_angle(
            point, point_on_line, angle
        )
        # sin(angle) = x / 1
        distance = np.sin(angle)
        self.assertAlmostEqual(normal_dist_from_line_to_point, distance)

    def test_create_rectangle_from_points(self):
        points = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0], [0, 0.5]])
        rectangle = create_rectangle_from_points(points)
        self.assertEqual(rectangle["min_x"], -1)
        self.assertEqual(rectangle["max_x"], 1)
        self.assertEqual(rectangle["min_y"], -1)
        self.assertEqual(rectangle["max_y"], 1)

    def test_is_point_in_rectangle(self):
        rectangle = {"min_x": -1, "max_x": 1, "min_y": -1, "max_y": 1}
        inside = is_point_in_rectangle([0, 0], rectangle)
        self.assertTrue(inside)
        outside = is_point_in_rectangle([2, 2], rectangle)
        self.assertFalse(outside)
        on_boundary = is_point_in_rectangle([1, 1], rectangle)
        self.assertTrue(on_boundary)


if __name__ == "__main__":
    unittest.main()
