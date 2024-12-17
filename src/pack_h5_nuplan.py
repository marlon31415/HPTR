# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import sys
import os
import copy
from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
from more_itertools import batched

sys.path.append(".")

import src.utils.pack_h5 as pack_utils
from src.utils.pack_h5_nuplan_utils import (
    get_nuplan_scenarios,
    nuplan_to_centered_vector,
    parse_object_state,
    set_light_position,
    get_points_from_boundary,
    extract_centerline,
    mock_2d_to_3d_points,
    get_route_lane_polylines_from_roadblock_ids,
    get_id_and_start_idx_for_scenarios,
    fill_track_with_state,
    calc_velocity_from_positions,
    mining_for_interesting_agents,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)


TL_TYPES = {
    "GREEN": 3,
    "YELLOW": 2,
    "RED": 1,
    "UNKNOWN": 0,
}
N_TL_STATE = len(TL_TYPES)

AGENT_TYPES = {
    "VEHICLE": 0,  # Includes all four or more wheeled vehicles, as well as trailers.
    "PEDESTRIAN": 1,  # All types of pedestrians, incl. strollers and wheelchairs.
    "BICYCLE": 2,  # Includes bicycles, motorcycles and tricycles.
    "TRAFFIC_CONE": 3,  # Cones that are temporarily placed to control the flow of traffic.
    "BARRIER": 3,  # Solid barriers that can be either temporary or permanent.
    "CZONE_SIGN": 3,  # Temporary signs that indicate construction zones.
    "GENERIC_OBJECT": 3,  # Animals, debris, pushable/pullable objects, permanent poles.
    "EGO": 0,  # The ego vehicle.
}
N_AGENT_TYPE = len(set(AGENT_TYPES.values()))

N_PL_MAX = 2000
N_TL_MAX = 40
N_AGENT_MAX = 800
N_PL_ROUTE_MAX = 250

N_PL = 1024
N_TL = 200  # due to polyline splitting this value can be higher than N_TL_MAX
N_AGENT = 64
N_AGENT_NO_SIM = N_AGENT_MAX - N_AGENT
N_PL_ROUTE = N_PL_ROUTE_MAX

THRESH_MAP = 120
THRESH_AGENT = 120

N_STEP = 91
STEP_CURRENT = 10

N_SDC_AGENT = 1
N_AGENT_PRED_CHALLENGE = 8
N_AGENT_INTERACT_CHALLENGE = 2

PL_TYPES = {
    "INTERSECTION": 0,
    "STOP_LINE": 1,
    "CROSSWALK": 2,
    "WALKWAYS": 3,
    "BOUNDARIES": 4,
    "CARPARK_AREA": 5,
    "CENTERLINE": 6,
    "ROUTE": 7,
}
N_PL_TYPE = len(PL_TYPES)
DIM_VEH_LANES = [7]
DIM_CYC_LANES = [4, 7]
DIM_PED_LANES = [2, 3, 4]

LAYER_NAMES = [
    SemanticMapLayer.LANE_CONNECTOR,
    SemanticMapLayer.LANE,
    SemanticMapLayer.CROSSWALK,
    SemanticMapLayer.INTERSECTION,
    SemanticMapLayer.STOP_LINE,
    SemanticMapLayer.WALKWAYS,
    SemanticMapLayer.CARPARK_AREA,
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    # unsupported yet
    # SemanticMapLayer.PUDO,
    # SemanticMapLayer.EXTENDED_PUDO,
    # SemanticMapLayer.SPEED_BUMP,
    # SemanticMapLayer.STOP_SIGN,
    # SemanticMapLayer.DRIVABLE_AREA,
]


def collate_agent_features(
    scenario_center,
    ego_states,
    observation_states,
    n_step,
    only_agents=True,
    interval_length=0.1,
    n_agent_pred_challenge=N_AGENT_PRED_CHALLENGE,
    n_agent_interact_challange=N_AGENT_INTERACT_CHALLENGE,
):
    # Tuple instead of Point2d for compatibility with nuplan pack utils
    scenario_center_tuple = [scenario_center.x, scenario_center.y]

    common_states = [
        [ego] + other for ego, other in zip(ego_states, observation_states)
    ]

    agent_id = []
    agent_type = []
    agent_states = []
    agent_role = []

    default_track = {
        "type": "UNSET",
        "state": {
            "position_x": np.zeros(shape=(n_step,)),
            "position_y": np.zeros(shape=(n_step,)),
            "position_z": np.zeros(shape=(n_step,)),
            "length": np.zeros(shape=(n_step,)),
            "width": np.zeros(shape=(n_step,)),
            "height": np.zeros(shape=(n_step,)),
            "heading": np.zeros(shape=(n_step,)),
            "velocity_x": np.zeros(shape=(n_step,)),
            "velocity_y": np.zeros(shape=(n_step,)),
            "valid": np.zeros(shape=(n_step,)),
        },
        "metadata": {
            "object_id": None,  # itegers defined by the dataset
            "nuplan_id": None,  # hex ids
        },
    }

    tracks = {}
    for i in range(n_step):
        current_observation = common_states[i]
        for obj in current_observation:
            tracked_object_type = obj.tracked_object_type
            if tracked_object_type is None or (
                only_agents and AGENT_TYPES[tracked_object_type.name] > 2
            ):
                continue
            track_token = obj.metadata.track_token
            if track_token not in tracks:
                # add new track with token as key
                tracks[track_token] = copy.deepcopy(default_track)
                tracks[track_token]["metadata"]["nuplan_id"] = track_token
                tracks[track_token]["metadata"]["object_id"] = obj.metadata.track_id
                tracks[track_token]["type"] = AGENT_TYPES[tracked_object_type.name]

            state = parse_object_state(obj, scenario_center_tuple)
            fill_track_with_state(tracks[track_token]["state"], state, i)

    # adapt ego velocity
    calc_velocity_from_positions(tracks["ego"]["state"], interval_length)

    track_ids_predict, track_ids_interact = mining_for_interesting_agents(
        tracks, n_agent_pred_challenge, n_agent_interact_challange
    )

    for nuplan_id, track in tracks.items():
        nuplan_id = track["metadata"]["nuplan_id"]
        agent_role.append([False, False, False])
        if track["type"] in [0, 1, 2]:
            agent_role[-1][2] = True if nuplan_id in track_ids_predict else False
            agent_role[-1][1] = True if nuplan_id in track_ids_interact else False
        if nuplan_id == "ego":
            agent_role[-1] = [True, True, True]

        agent_id.append(track["metadata"]["object_id"])
        agent_type.append(track["type"])
        agent_states_list = np.vstack(list(track["state"].values())).T.tolist()
        agent_states.append(agent_states_list)

    return (
        agent_id,
        agent_type,
        agent_states,
        agent_role,
    )


def collate_tl_features(
    map_api,
    scenario_center,
    traffic_light_data,
    n_step,
    step_current,
):

    scenario_center_tuple = [scenario_center.x, scenario_center.y]

    tl_lane_state = []
    tl_lane_id = []
    tl_stop_point = []

    # PlannerInput only contain TL information for current step
    # -> add empty list for all other steps
    tl_lane_state.extend([] for _ in range(n_step - 1))
    tl_lane_id.extend([] for _ in range(n_step - 1))
    tl_stop_point.extend([] for _ in range(n_step - 1))

    tl_lane_state_current = []
    tl_lane_id_current = []
    for tl in traffic_light_data:
        tl_lane_state_current.append(TL_TYPES[tl.status.name])
        tl_lane_id_current.append(tl.lane_connector_id)

    tl_lane_state.insert(step_current, tl_lane_state_current)
    tl_lane_id.insert(step_current, tl_lane_id_current)

    tl_stop_point_2d = [
        set_light_position(map_api, lane_id, scenario_center_tuple)
        for lane_id in tl_lane_id[step_current]
    ]
    tl_stop_point.insert(step_current, mock_2d_to_3d_points(tl_stop_point_2d))

    return (
        tl_lane_state,
        tl_lane_id,
        tl_stop_point,
    )


def collate_map_features(map_api, scenario_center, radius=200):
    """
    Most parts copied from @metadriverse:
    https://github.com/metadriverse/scenarionet/blob/main/scenarionet/converter/nuplan/utils.py
    """
    # map features
    mf_id = []
    mf_xyz = []
    mf_type = []
    mf_edge = []

    scenario_center_tuple = [scenario_center.x, scenario_center.y]
    semantic_map_layers = map_api.get_proximal_map_objects(
        scenario_center, radius, LAYER_NAMES
    )

    for semantic_layer in [
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]:
        """
        ROADBLOCK and ROADBLOCK_CONNECTOR contain lanes. A ROADBLOCK_CONNECTOR connects two ROADBLOCKS,
        e.g. a ROADBLOCK with 3 parallel lanes and a ROADBLOCK with 2 parallel lanes is connected by a
        ROADBLOCK_CONNECTOR merging the lanes.
        We use the left and right boundaries of the BLOCKS (left boundary of left most lane and right
        boundary of right most lane) as BOUNDARIES.
        """
        for block in semantic_map_layers[semantic_layer]:
            # LANE centerlines from lanes in ROADBLOCKS and ROADBLOCK_CONNECTORS
            # According to the map attributes, lanes are numbered left to right with smaller indices being on the
            # left and larger indices being on the right.
            lanes = sorted(block.interior_edges, key=lambda lane: lane.id)
            for i, lane in enumerate(lanes):
                if not hasattr(lane, "baseline_path"):
                    continue
                # Centerline (as polyline)
                centerline = extract_centerline(lane, scenario_center_tuple, True, 1)
                mf_id.append(int(lane.id))
                mf_type.append(PL_TYPES["CENTERLINE"])
                mf_xyz.append(mock_2d_to_3d_points(centerline))
                # Add successor lanes to edge list
                if len(lane.outgoing_edges) > 0:
                    for successor_lane in lane.outgoing_edges:
                        mf_edge.append([int(lane.id), int(successor_lane.id)])
                else:
                    mf_edge.append([int(lane.id), -1])
                # Add left and right neighbors to edge list
                if i > 0:
                    left_neighbor = lanes[i - 1]
                    mf_edge.append([int(lane.id), int(left_neighbor.id)])
                if i < len(lanes) - 1:
                    right_neighbor = lanes[i + 1]
                    mf_edge.append([int(lane.id), int(right_neighbor.id)])

            # ROADBLOCK boundaries from left and right most lanes
            for boundary_side in ["left", "right"]:
                if boundary_side == "left":
                    boundary = lanes[0].left_boundary
                elif boundary_side == "right":
                    boundary = lanes[-1].right_boundary
                boundary_pl = get_points_from_boundary(
                    boundary, scenario_center_tuple, True, 1
                )
                mf_id.append(int(block.id))
                mf_type.append(PL_TYPES["BOUNDARIES"])
                mf_xyz.append(mock_2d_to_3d_points(boundary_pl))

    for semantic_layer in [
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.INTERSECTION,
    ]:
        """
        These semantic layers are represented as polygons in the map.
        """
        for area in semantic_map_layers[semantic_layer]:
            polygon = area.polygon.exterior.coords
            polygon_centered = nuplan_to_centered_vector(
                np.array(polygon), scenario_center_tuple
            )
            mf_id.append(int(area.id))
            mf_type.append(PL_TYPES[semantic_layer.name])
            mf_xyz.append(mock_2d_to_3d_points(polygon_centered)[::4])

    return mf_id, mf_xyz, mf_type, mf_edge


def collate_route_features(
    map_api, scenario_center, route_roadblock_ids, mission_goal, radius=200
):
    scenario_center_tuple = [scenario_center.x, scenario_center.y]

    # id=-1 is the default nuplan value for the ego; TODO: change this if needed
    sdc_id = [-1]
    sdc_route_type = []
    sdc_route_lane_id = []
    sdc_route_xyz = []
    sdc_route_goal = []

    polylines, route_lane_ids = get_route_lane_polylines_from_roadblock_ids(
        map_api, scenario_center, radius, route_roadblock_ids
    )
    route_lane_polylines = []
    pl_types = []
    for polyline in polylines:
        polyline_centered = nuplan_to_centered_vector(polyline, scenario_center_tuple)
        route_lane_polylines.append(mock_2d_to_3d_points(polyline_centered)[::10])
        pl_types.append(PL_TYPES["ROUTE"])
    sdc_route_lane_id.append(route_lane_ids)
    sdc_route_type.append(pl_types)
    sdc_route_xyz.append(route_lane_polylines)

    mission_goal_centered_with_yaw = np.hstack(
        [
            nuplan_to_centered_vector(
                [mission_goal.x, mission_goal.y], scenario_center_tuple
            ),
            [mission_goal.heading],
        ]
    )
    sdc_route_goal.append(mission_goal_centered_with_yaw)

    return (
        sdc_id,
        sdc_route_lane_id,
        sdc_route_type,
        sdc_route_xyz,
        sdc_route_goal,
    )


def create_planner_input_from_scenario(
    scenario: NuPlanScenario,
    iteration: SimulationIteration,
):
    """
    This function creates a PlannerInput and PlannerInitialization object.
    These objects are used as input arguments within the nuplan-devkit to run the nuPlan simulation.
    To enable using the same functions for creating the dataset as in the simulation environment:
    https://github.com/marlon31415/tuplan_garage/tree/scene-motion, the standardized
    interface (PlannerInput, PlannerInitialization) is used.

    The objects are created as present_timestep is the end of the horizon and all timesteps are past timesteps,
    since the splitting into past and future is done within pack_h5.py.
    """
    virtual_present_timestep = iteration + N_STEP
    present_time_step = iteration + STEP_CURRENT

    route_roadblock_ids = scenario.get_route_roadblock_ids()
    mission_goal = scenario.get_mission_goal()
    map_api = scenario.map_api
    initialization = PlannerInitialization(route_roadblock_ids, mission_goal, map_api)

    history = SimulationHistoryBuffer.initialize_from_scenario(
        buffer_size=N_STEP,
        scenario=scenario,
        observation_type=DetectionsTracks,
        iteration=virtual_present_timestep,
    )
    traffic_light_data = scenario.get_traffic_light_status_at_iteration(
        present_time_step
    )

    planner_input = PlannerInput(virtual_present_timestep, history, traffic_light_data)

    return initialization, planner_input


def convert_nuplan_scenario(
    scenario: NuPlanScenario,
    iteration,
    rand_pos,
    rand_yaw,
    pack_all,
    pack_history,
    dest_no_pred,
    radius,
    split: str = "training",
):
    scenario_log_interval = scenario.database_interval
    assert abs(scenario_log_interval - 0.1) < 1e-3, (
        "Log interval should be 0.1 or Interpolating is required! "
        "By setting NuPlan subsample ratio can address this"
    )

    initialization, current_input = create_planner_input_from_scenario(
        scenario, iteration
    )

    map_api = initialization.map_api
    past_observations = [
        obs.tracked_objects.get_agents() for obs in current_input.history.observations
    ]
    past_ego_states = [
        ego_state.agent for ego_state in current_input.history.ego_states
    ]
    scenario_center = current_input.history.ego_states[-1].center.point

    # agents
    agent_id, agent_type, agent_states, agent_role = collate_agent_features(
        scenario_center,
        past_ego_states,
        past_observations,
        N_STEP,
        only_agents=True,
    )
    # traffic light
    tl_lane_state, tl_lane_id, tl_stop_point = collate_tl_features(
        map_api,
        scenario_center,
        current_input.traffic_light_data,
        N_STEP,
        STEP_CURRENT,
    )
    # map
    mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(
        map_api, scenario_center, radius
    )
    # route
    sdc_id, sdc_route_id, sdc_route_type, sdc_route_xyz, sdc_route_goal = (
        collate_route_features(
            map_api,
            scenario_center,
            initialization.route_roadblock_ids,
            initialization.mission_goal,
            radius,
        )
    )

    episode = {}
    n_pl = pack_utils.pack_episode_map(
        episode=episode,
        mf_id=mf_id,
        mf_xyz=mf_xyz,
        mf_type=mf_type,
        mf_edge=mf_edge,
        n_pl_max=N_PL_MAX,
    )
    n_tl = pack_utils.pack_episode_traffic_lights(
        episode=episode,
        tl_lane_state=tl_lane_state,
        tl_lane_id=tl_lane_id,
        tl_stop_point=tl_stop_point,
        pack_all=pack_all,
        pack_history=pack_history,
        n_tl_max=N_TL_MAX,
        step_current=STEP_CURRENT,
    )
    n_agent = pack_utils.pack_episode_agents(
        episode=episode,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_states=agent_states,
        agent_role=agent_role,
        pack_all=pack_all,
        pack_history=pack_history,
        n_agent_max=N_AGENT_MAX,
        step_current=STEP_CURRENT,
    )
    n_route_pl = pack_utils.pack_episode_route(
        episode=episode,
        sdc_id=sdc_id,
        sdc_route_id=sdc_route_id,
        sdc_route_type=sdc_route_type,
        sdc_route_xyz=sdc_route_xyz,
        sdc_route_goal=sdc_route_goal,
        n_route_pl_max=N_PL_ROUTE_MAX,
    )
    scenario_center, scenario_yaw = pack_utils.center_at_sdc(
        episode, rand_pos, rand_yaw
    )

    episode_reduced = {}
    pack_utils.filter_episode_map(episode, N_PL, THRESH_MAP, thresh_z=3)
    episode_with_map = episode["map/valid"].any(1).sum() > 0
    pack_utils.repack_episode_map(episode, episode_reduced, N_PL, N_PL_TYPE)

    pack_utils.repack_episode_route(episode, episode_reduced, N_PL_ROUTE, N_PL_TYPE)

    pack_utils.filter_episode_traffic_lights(episode)
    pack_utils.repack_episode_traffic_lights(episode, episode_reduced, N_TL, N_TL_STATE)

    if split == "training":
        mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
            episode=episode,
            episode_reduced=episode_reduced,
            n_agent=N_AGENT,
            prefix="",
            dim_veh_lanes=DIM_VEH_LANES,
            dist_thresh_agent=THRESH_AGENT,
            step_current=STEP_CURRENT,
        )
        pack_utils.repack_episode_agents(
            episode=episode,
            episode_reduced=episode_reduced,
            mask_sim=mask_sim,
            n_agent=N_AGENT,
            prefix="",
            dim_veh_lanes=DIM_VEH_LANES,
            dim_cyc_lanes=DIM_CYC_LANES,
            dim_ped_lanes=DIM_PED_LANES,
            dest_no_pred=dest_no_pred,
        )
    elif split == "validation":
        mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
            episode=episode,
            episode_reduced=episode_reduced,
            n_agent=N_AGENT,
            prefix="history/",
            dim_veh_lanes=DIM_VEH_LANES,
            dist_thresh_agent=THRESH_AGENT,
            step_current=STEP_CURRENT,
        )
        pack_utils.repack_episode_agents(
            episode=episode,
            episode_reduced=episode_reduced,
            mask_sim=mask_sim,
            n_agent=N_AGENT,
            prefix="",
            dim_veh_lanes=DIM_VEH_LANES,
            dim_cyc_lanes=DIM_CYC_LANES,
            dim_ped_lanes=DIM_PED_LANES,
            dest_no_pred=dest_no_pred,
        )
        pack_utils.repack_episode_agents(
            episode, episode_reduced, mask_sim, N_AGENT, "history/"
        )
        pack_utils.repack_episode_agents_no_sim(
            episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, ""
        )
        pack_utils.repack_episode_agents_no_sim(
            episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
        )
    elif split == "testing":
        mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
            episode=episode,
            episode_reduced=episode_reduced,
            n_agent=N_AGENT,
            prefix="history/",
            dim_veh_lanes=DIM_VEH_LANES,
            dist_thresh_agent=THRESH_AGENT,
            step_current=STEP_CURRENT,
        )
        pack_utils.repack_episode_agents(
            episode, episode_reduced, mask_sim, N_AGENT, "history/"
        )
        pack_utils.repack_episode_agents_no_sim(
            episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
        )

    n_agent_sim = mask_sim.sum()
    n_agent_no_sim = mask_no_sim.sum()

    if episode_with_map:
        episode_reduced["map/boundary"] = pack_utils.get_map_boundary(
            episode_reduced["map/valid"], episode_reduced["map/pos"]
        )
    else:
        # only in waymo test split.
        assert split == "testing"
        episode_reduced["map/boundary"] = pack_utils.get_map_boundary(
            episode["history/agent/valid"], episode["history/agent/pos"]
        )
        print(
            f"scenario {scenario.log_name} has no map! map boundary is: {episode_reduced['map/boundary']}"
        )

    episode_name = os.path.splitext(scenario.token)[0] + "_" + str(iteration)
    episode_metadata = {
        "scenario_id": episode_name,
        "scenario_center": scenario_center,
        "scenario_yaw": scenario_yaw,
        "with_map": episode_with_map,
    }

    return (
        episode_reduced,
        episode_metadata,
        n_pl,
        n_tl,
        n_agent,
        n_agent_sim,
        n_agent_no_sim,
        n_route_pl,
    )


def wrapper_convert_nuplan_scenario(
    scenario_tuple,
    rand_pos,
    rand_yaw,
    pack_all,
    pack_history,
    dest_no_pred,
    radius,
    split,
):
    scenario, iteration, id = scenario_tuple
    episode, metadata, n_pl, n_tl, n_agent, n_agent_sim, n_agent_no_sim, n_route_pl = (
        convert_nuplan_scenario(
            scenario,
            iteration,
            rand_pos,
            rand_yaw,
            pack_all,
            pack_history,
            dest_no_pred,
            radius,
            split,
        )
    )
    metadata["hf_group_id"] = id
    SCENARIO_QUEUE.put((episode, metadata))
    return (
        episode,
        metadata,
        n_pl,
        n_tl,
        n_agent,
        n_agent_sim,
        n_agent_no_sim,
        n_route_pl,
    )


def write_to_h5_file(h5_file_path, queue, total_items):
    with h5py.File(h5_file_path, "w") as h5_file:
        for _ in range(total_items):
            data, metadata = queue.get()
            # Create a group for each item
            group = h5_file.create_group(str(metadata["hf_group_id"]))
            # Store the transformed data in a dataset within the group
            for k, v in data.items():
                group.create_dataset(
                    k, data=v, compression="gzip", compression_opts=4, shuffle=True
                )
            # Add metadata to the group
            for k, v in metadata.items():
                group.attrs[k] = v


def main():
    # fmt: off
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir", default=os.getenv("NUPLAN_DATA_ROOT"))
    parser.add_argument("--map-dir", default=os.getenv("NUPLAN_MAPS_ROOT"))
    parser.add_argument("--version", "-v", default="v1.1", help="version of the raw data")
    parser.add_argument("--dataset", default="training")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--out-dir", default="/mrtstorage/datasets_tmp/nuplan_hptr")
    parser.add_argument("--rand-pos", default=50.0, type=float, help="Meter. Set to -1 to disable.")
    parser.add_argument("--rand-yaw", default=3.14, type=float, help="Radian. Set to -1 to disable.")
    parser.add_argument("--dest-no-pred", action="store_true")
    parser.add_argument("--radius", default=200, type=int)
    parser.add_argument("--test", action="store_true", help="for test use only. convert one log")
    parser.add_argument("--num-workers", default=32, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    # fmt: on

    if "training" in args.dataset:
        pack_all = True  # ["agent/valid"]
        pack_history = False  # ["history/agent/valid"]
        split = "train"
    elif "validation" in args.dataset:
        pack_all = True
        pack_history = True
        split = "val"
    elif "testing" in args.dataset:
        pack_all = False
        pack_history = True
        split = "test"

    if args.mini:
        split = "mini"

    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    if not args.mini:
        out_h5_path = out_path / (args.dataset + ".h5")
    else:
        out_h5_path = out_path / (args.dataset + "_mini" + ".h5")

    if args.test:
        data_root = os.path.join(args.data_dir, "nuplan-v1.1/splits/mini")
        scenarios = get_nuplan_scenarios(
            data_root, args.map_dir, logs=["2021.07.16.20.45.29_veh-35_01095_01486"]
        )
    else:
        data_root = os.path.join(args.data_dir, "nuplan-v1.1/splits", split)
        scenarios = get_nuplan_scenarios(data_root, args.map_dir)
    print(f"Found {len(scenarios)} nuplan scenarios in the dataset")

    # preprocessing: convert scnearios list to list of tuples (scenario, start_iter)
    scenario_tuples = get_id_and_start_idx_for_scenarios(scenarios, N_STEP)
    print(
        f"Converting {len(scenario_tuples)} subsampled scenarios to {args.dataset} dataset"
    )

    n_pl_max = 0
    n_tl_max = 0
    n_agent_max = 0
    n_agent_sim_max = 0
    n_agent_no_sim_max = 0
    n_route_pl_max = 0

    convert_func = partial(
        wrapper_convert_nuplan_scenario,
        rand_pos=args.rand_pos,
        rand_yaw=args.rand_yaw,
        pack_all=pack_all,
        pack_history=pack_history,
        dest_no_pred=args.dest_no_pred,
        radius=args.radius,
        split=args.dataset,
    )

    # Start the writer thread
    writer_thread = mp.Process(
        target=write_to_h5_file,
        args=(out_h5_path, SCENARIO_QUEUE, len(scenario_tuples)),
    )
    writer_thread.start()
    # Mulitprocessing the data conversion
    for batch in tqdm(list(batched(scenario_tuples, args.batch_size))):
        with mp.Pool(args.num_workers) as pool:
            res = pool.map(convert_func, batch)

        res_reordered_zip = zip(*res)
        res_reordered = list(res_reordered_zip)
        n_pl_max = max(n_pl_max, max(res_reordered[2]))
        n_tl_max = max(n_tl_max, max(res_reordered[3]))
        n_agent_max = max(n_agent_max, max(res_reordered[4]))
        n_agent_sim_max = max(n_agent_sim_max, max(res_reordered[5]))
        n_agent_no_sim_max = max(n_agent_no_sim_max, max(res_reordered[6]))
        n_route_pl_max = max(n_route_pl_max, max(res_reordered[7]))

    writer_thread.join()

    print(f"n_pl_max: {n_pl_max}")
    print(f"n_route_pl_max: {n_route_pl_max}")
    print(f"n_tl_max: {n_tl_max}")
    print(f"n_agent_max: {n_agent_max}")
    print(f"n_agent_sim_max: {n_agent_sim_max}")
    print(f"n_agent_no_sim_max: {n_agent_no_sim_max}")


if __name__ == "__main__":
    SCENARIO_QUEUE = mp.Manager().Queue()
    main()
