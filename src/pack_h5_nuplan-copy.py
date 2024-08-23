import sys

sys.path.append(".")

import os
from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import numpy as np
from pathlib import Path

# from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
# from nuplan.planning.scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    get_db_filenames_from_load_path,
    get_scenarios_from_db_file,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
import src.utils.pack_h5 as pack_utils

# onRouteStatus
# OFF_ROUTE = 0
# ON_ROUTE = 1
# UNKNOWN = 2

# mapVectorFeatures
# feature_types:
#     NONE: -1
#     EGO: 0
#     VEHICLE: 1
#     BICYCLE: 2
#     PEDESTRIAN: 3
#     LANE: 4
#     STOP_LINE: 5
#     CROSSWALK: 6
#     LEFT_BOUNDARY: 7
#     RIGHT_BOUNDARY: 8
#     ROUTE_LANES: 9

# trafficLightState
# green = [1, 0, 0, 0]
# yellow = [0, 1, 0, 0]
# red = [0, 0, 1, 0],
# unknown = [0, 0, 0, 1]
N_TL_STATE = 5

N_PL_TYPE = 11
PL_TYPE = {
    "carpark_areas": 0,
    "generic_drivable_areas": 1,
    "lane_connectors": 2,
    "intersections": 3,
    "boundaries": 3,
    "crosswalks": 4,
    "lanes_polygons": 4,
    "road_segments": 5,
    "stop_polygons": 5,
    "traffic_lights": 6,
    "walkways": 6,
    "walkway": 6,
}

AGENT_TYPE = {
    "vehicle": 0,  # Includes all four or more wheeled vehicles, as well as trailers.
    "bicycle": 1,  # Includes bicycles, motorcycles and tricycles.
    "pedestrian": 2,  # All types of pedestrians, incl. strollers and wheelchairs.
    "traffic_cone": 3,  # Cones that are temporarily placed to control the flow of traffic.
    "barrier": 4,  # Solid barriers that can be either temporary or permanent.
    "czone_sign": 5,  # Temporary signs that indicate construction zones.
    "generic_object": 6,  # Animals, debris, pushable/pullable objects, permanent poles.
}

N_PL_MAX = 1500
N_AGENT_MAX = 256

N_PL = 1024
N_AGENT = 64
N_AGENT_NO_SIM = N_AGENT_MAX - N_AGENT

THRESH_MAP = 120
THRESH_AGENT = 120

STEP_CURRENT = 10
N_STEP = 91


def collate_agent_features(
    tracks, sdc_track_index, track_index_predict, object_id_interest
):
    agent_id = []
    agent_type = []
    agent_states = []
    agent_role = []
    # for i, _track in enumerate(tracks):
    #     agent_id.append(_track.id)
    #     agent_type.append(
    #         _track.object_type - 1
    #     )  # [TYPE_VEHICLE=1, TYPE_PEDESTRIAN=2, TYPE_CYCLIST=3] -> [0,1,2]
    #     step_states = []
    #     for s in _track.states:
    #         step_states.append(
    #             [
    #                 s.center_x,
    #                 s.center_y,
    #                 s.center_z,
    #                 s.length,
    #                 s.width,
    #                 s.height,
    #                 s.heading,
    #                 s.velocity_x,
    #                 s.velocity_y,
    #                 s.valid,
    #             ]
    #         )
    #         # This angle is normalized to [-pi, pi). The velocity vector in m/s
    #     agent_states.append(step_states)

    #     agent_role.append([False, False, False])
    #     if i in track_index_predict:
    #         agent_role[-1][2] = True
    #     if _track.id in object_id_interest:
    #         agent_role[-1][1] = True
    #     if i == sdc_track_index:
    #         agent_role[-1][0] = True

    # return agent_id, agent_type, agent_states, agent_role


def collate_tl_features(tl_features):
    tl_lane_state = []
    tl_lane_id = []
    tl_stop_point = []
    # for _step_tl in tl_features:
    #     step_tl_lane_state = []
    #     step_tl_lane_id = []
    #     step_tl_stop_point = []
    #     for _tl in _step_tl.lane_states:
    #         if _tl.state == 0:  # LANE_STATE_UNKNOWN = 0;
    #             tl_state = 0  # LANE_STATE_UNKNOWN = 0;
    #         elif _tl.state in [1, 4]:  # LANE_STATE_ARROW_STOP = 1; LANE_STATE_STOP = 4;
    #             tl_state = 1  # LANE_STATE_STOP = 1;
    #         elif _tl.state in [
    #             2,
    #             5,
    #         ]:  # LANE_STATE_ARROW_CAUTION = 2; LANE_STATE_CAUTION = 5;
    #             tl_state = 2  # LANE_STATE_CAUTION = 2;
    #         elif _tl.state in [3, 6]:  # LANE_STATE_ARROW_GO = 3; LANE_STATE_GO = 6;
    #             tl_state = 3  # LANE_STATE_GO = 3;
    #         elif _tl.state in [
    #             7,
    #             8,
    #         ]:  # LANE_STATE_FLASHING_STOP = 7; LANE_STATE_FLASHING_CAUTION = 8;
    #             tl_state = 4  # LANE_STATE_FLASHING = 4;
    #         else:
    #             assert ValueError

    #         step_tl_lane_state.append(tl_state)
    #         step_tl_lane_id.append(_tl.lane)
    #         step_tl_stop_point.append(
    #             [_tl.stop_point.x, _tl.stop_point.y, _tl.stop_point.z]
    #         )

    #     tl_lane_state.append(step_tl_lane_state)
    #     tl_lane_id.append(step_tl_lane_id)
    #     tl_stop_point.append(step_tl_stop_point)
    # return tl_lane_state, tl_lane_id, tl_stop_point


def collate_map_features(scenario, map_features_input):
    mf_id = []
    mf_xyz = []
    mf_type = []
    mf_edge = []
    # lane_boundary_set = []

    vectorSetMapFB = VectorSetMapFeatureBuilder(
        list(map_features_input.keys()),
        {key: value["max_elements"] for key, value in map_features_input.items()},
        {key: value["max_points"] for key, value in map_features_input.items()},
        50,
        "linear",
    )

    map_features = vectorSetMapFB.get_features_from_scenario(scenario)
    print(map_features)

    # for _id, ped_xing in static_map.vector_pedestrian_crossings.items():
    #     v0, v1 = ped_xing.edge1.xyz
    #     v2, v3 = ped_xing.edge2.xyz
    #     pl_crosswalk = pack_utils.get_polylines_from_polygon(np.array([v0, v1, v3, v2]))
    #     mf_id.extend([_id] * len(pl_crosswalk))
    #     mf_type.extend([PL_TYPE["CROSSWALK"]] * len(pl_crosswalk))
    #     mf_xyz.extend(pl_crosswalk)

    # for _id, lane_segment in static_map.vector_lane_segments.items():
    #     centerline_pts, left_even_pts, right_even_pts = _interpolate_centerline(
    #         lane_segment.left_lane_boundary.xyz, lane_segment.right_lane_boundary.xyz
    #     )

    #     mf_id.append(_id)
    #     mf_xyz.append(centerline_pts)
    #     mf_type.append(PL_TYPE[lane_segment.lane_type])

    #     if (lane_segment.left_lane_boundary not in lane_boundary_set) and not (
    #         lane_segment.is_intersection and lane_segment.left_mark_type in ["NONE", "UNKOWN"]
    #     ):
    #         lane_boundary_set.append(lane_segment.left_lane_boundary)
    #         mf_xyz.append(left_even_pts)
    #         mf_id.append(-2)
    #         mf_type.append(PL_TYPE[lane_segment.left_mark_type])

    #     if (lane_segment.right_lane_boundary not in lane_boundary_set) and not (
    #         lane_segment.is_intersection and lane_segment.right_mark_type in ["NONE", "UNKOWN"]
    #     ):
    #         lane_boundary_set.append(lane_segment.right_lane_boundary)
    #         mf_xyz.append(right_even_pts)
    #         mf_id.append(-2)
    #         mf_type.append(PL_TYPE[lane_segment.right_mark_type])

    #         for _id_exit in lane_segment.successors:
    #             mf_edge.append([_id, _id_exit])
    #     else:
    #         mf_edge.append([_id, -1])

    return mf_id, mf_xyz, mf_type, mf_edge


def main():
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--data-dir",
        default=os.getenv("NUPLAN_DATA_ROOT"),
    )
    parser.add_argument(
        "--map-dir",
        default=os.getenv("NUPLAN_MAPS_ROOT"),
    )
    parser.add_argument("--dataset", default="training")
    parser.add_argument("--out-dir", default="/mrtstorage/datasets_tmp/nuplan_hptr")
    parser.add_argument(
        "--rand-pos", default=50.0, type=float, help="Meter. Set to -1 to disable."
    )
    parser.add_argument(
        "--rand-yaw", default=3.14, type=float, help="Radian. Set to -1 to disable."
    )
    parser.add_argument("--dest-no-pred", action="store_true")
    args = parser.parse_args()

    if "training" in args.dataset:
        pack_all = True  # ["agent/valid"]
        pack_history = False  # ["history/agent/valid"]
        n_step = N_STEP
    elif "validation" in args.dataset:
        pack_all = True
        pack_history = True
        n_step = N_STEP
    elif "testing" in args.dataset:
        pack_all = False
        pack_history = True
        n_step = STEP_CURRENT + 1

    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    out_h5_path = out_path / (args.dataset + ".h5")

    nuPlanScenarioBuilder = NuPlanScenarioBuilder(
        args.data_dir,
        args.map_dir,
        "",
        os.path.join(
            args.data_dir,
            "nuplan-v1.1/splits/mini/2021.05.12.22.28.35_veh-35_00620_01164.db",
        ),
        "nuplan-maps-v1.0",
    )
    nuplan_scenarios = nuPlanScenarioBuilder.get_scenarios(
        ScenarioFilter(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            expand_scenarios=True,
            remove_invalid_goals=True,
            shuffle=True,
        ),
        SingleMachineParallelExecutor(),
    )
    print(len(nuplan_scenarios))

    map_features_input = {
        "LANE": {"max_elements": 30, "max_points": 20},
        "LEFT_BOUNDARY": {"max_elements": 30, "max_points": 20},
        "RIGHT_BOUNDARY": {"max_elements": 30, "max_points": 20},
        "STOP_LINE": {"max_elements": 20, "max_points": 20},
        "CROSSWALK": {"max_elements": 20, "max_points": 20},
        "ROUTE_LANES": {"max_elements": 30, "max_points": 20},
    }

    for scenario in nuplan_scenarios[:1]:
        collate_map_features(scenario, map_features_input)


if __name__ == "__main__":
    main()
