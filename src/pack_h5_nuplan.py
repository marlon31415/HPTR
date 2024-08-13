# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import sys

sys.path.append(".")

import os
from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
import matplotlib.pyplot as plt

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    LaneConnectorType,
    StopLineType,
    IntersectionType,
    TrafficLightStatusType,
)
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.state_representation import Point2D
import src.utils.pack_h5 as pack_utils
from src.utils.pack_h5_nuplan_utils import (
    NuPlanEgoType,
    get_nuplan_scenarios,
    nuplan_to_centered_vector,
    parse_object_state,
    set_light_position,
    get_points_from_boundary,
    parse_ego_vehicle_state_trajectory,
    extract_centerline,
    mock_2d_to_3d_points,
)

PL_TYPES = {
    "LANE": 0,
    "INTERSECTION": 1,
    "STOP_LINE": 2,
    # "TURN_STOP": 3,
    "CROSSWALK": 3,
    # "DRIVABLE_AREA": 5,
    # "YIELD": 6,
    # "TRAFFIC_LIGHT": 7,
    # "STOP_SIGN": 8,
    # "EXTENDED_PUDO": 9,
    # "SPEED_BUMP": 10,
    "LANE_CONNECTOR": 4,
    # "BASELINE_PATHS": 12,
    # "BOUNDARIES": 13,
    "WALKWAYS": 5,
    "CARPARK_AREA": 6,
    # "PUDO": 16,
    # "ROADBLOCK": 7,
    # "ROADBLOCK_CONNECTOR": 8,
    "LINE_BROKEN_SINGLE_WHITE": 7,
    "CENTERLINE": 8,
}
N_PL_TYPE = len(PL_TYPES)
DIM_VEH_LANES = [7, 8]
DIM_CYC_LANES = [7, 8]
DIM_PED_LANES = [3, 5]

TL_TYPES = {
    "GREEN": 3,
    "YELLOW": 2,
    "RED": 1,
    "UNKNOWN": 0,
}
N_TL_STATE = len(TL_TYPES)

AGENT_TYPES = {
    "VEHICLE": 0,  # Includes all four or more wheeled vehicles, as well as trailers.
    "PEDESTRIAN": 1,  # Includes bicycles, motorcycles and tricycles.
    "BICYCLE": 2,  # All types of pedestrians, incl. strollers and wheelchairs.
    "TRAFFIC_CONE": 3,  # Cones that are temporarily placed to control the flow of traffic.
    "BARRIER": 3,  # Solid barriers that can be either temporary or permanent.
    "CZONE_SIGN": 3,  # Temporary signs that indicate construction zones.
    "GENERIC_OBJECT": 3,  # Animals, debris, pushable/pullable objects, permanent poles.
    "EGO": 0,  # The ego vehicle.
}
N_AGENT_TYPE = len(set(AGENT_TYPES.values()))

N_PL_MAX = 1500
N_TL_MAX = 40
N_AGENT_MAX = 300

N_PL = 1024
N_TL = 200  # due to polyline splitting this value can be higher than N_TL_MAX
N_AGENT = 300
N_AGENT_NO_SIM = N_AGENT_MAX - N_AGENT

THRESH_MAP = 120
THRESH_AGENT = 120

N_STEP = 91
STEP_CURRENT = 10


def collate_agent_features(
    scenario: NuPlanScenario, center, start_iter, only_agents=True
):
    agent_id = []
    agent_type = []
    agent_states = []
    agent_role = []

    detection_ret = []
    all_obj_ids = set()
    ego_id = scenario.initial_ego_state.scene_object_metadata.track_token  # 'ego'
    # ego_id = "0"
    all_obj_ids.add(ego_id)
    # tracked objects (not ego)
    for frame_data in [
        scenario.get_tracked_objects_at_iteration(i).tracked_objects
        for i in range(start_iter, N_STEP + start_iter)
    ]:
        new_frame_data = {}
        for obj in frame_data:
            new_frame_data[obj.track_token] = obj
            all_obj_ids.add(obj.track_token)
        detection_ret.append(new_frame_data)

    tracks = {
        id: dict(
            type="UNSET",
            state=dict(
                position_x=np.zeros(shape=(N_STEP,)),
                position_y=np.zeros(shape=(N_STEP,)),
                position_z=np.zeros(shape=(N_STEP,)),
                length=np.zeros(shape=(N_STEP,)),
                width=np.zeros(shape=(N_STEP,)),
                height=np.zeros(shape=(N_STEP,)),
                heading=np.zeros(shape=(N_STEP,)),
                velocity_x=np.zeros(shape=(N_STEP,)),
                velocity_y=np.zeros(shape=(N_STEP,)),
                valid=np.zeros(shape=(N_STEP,)),
            ),
            metadata=dict(
                track_length=N_STEP,
                nuplan_type=None,
                object_id=i + 1,  # small integer ids
                nuplan_id=id,  # hex ids
            ),
        )
        for i, id in enumerate(list(all_obj_ids))
    }

    tracks_to_remove = set()

    for frame_idx, frame in enumerate(detection_ret):
        for nuplan_id, obj_state in frame.items():
            assert isinstance(obj_state, Agent) or isinstance(obj_state, StaticObject)
            obj_type = obj_state.tracked_object_type
            if obj_type is None:
                tracks_to_remove.add(nuplan_id)
                continue
            tracks[nuplan_id]["type"] = AGENT_TYPES[f"{obj_type.name}"]
            if tracks[nuplan_id]["metadata"]["nuplan_type"] is None:
                tracks[nuplan_id]["metadata"]["nuplan_type"] = int(
                    obj_state.tracked_object_type
                )

            state = parse_object_state(obj_state, center)
            tracks[nuplan_id]["state"]["position_x"][frame_idx] = state["position"][0]
            tracks[nuplan_id]["state"]["position_y"][frame_idx] = state["position"][1]
            tracks[nuplan_id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[nuplan_id]["state"]["velocity_x"][frame_idx] = state["velocity"][0]
            tracks[nuplan_id]["state"]["velocity_y"][frame_idx] = state["velocity"][1]
            tracks[nuplan_id]["state"]["valid"][frame_idx] = 1
            tracks[nuplan_id]["state"]["length"][frame_idx] = state["length"]
            tracks[nuplan_id]["state"]["width"][frame_idx] = state["width"]
            tracks[nuplan_id]["state"]["height"][frame_idx] = state["height"]

    for track in list(tracks_to_remove):
        tracks.pop(track)

    # ego
    sdc_traj = parse_ego_vehicle_state_trajectory(
        scenario, center, start_iter, start_iter + N_STEP
    )
    ego_track = tracks[ego_id]

    for frame_idx, obj_state in enumerate(sdc_traj):
        obj_type = AGENT_TYPES["EGO"]
        ego_track["type"] = obj_type
        if ego_track["metadata"]["nuplan_type"] is None:
            ego_track["metadata"]["nuplan_type"] = int(NuPlanEgoType)
        state = obj_state
        ego_track["state"]["position_x"][frame_idx] = state["position"][0]
        ego_track["state"]["position_y"][frame_idx] = state["position"][1]
        ego_track["state"]["valid"][frame_idx] = 1
        ego_track["state"]["heading"][frame_idx] = state["heading"]
        # this velocity is in ego car frame, abort
        # ego_track["state"]["velocity"][frame_idx] = state["velocity"]
        ego_track["state"]["length"][frame_idx] = state["length"]
        ego_track["state"]["width"][frame_idx] = state["width"]
        ego_track["state"]["height"][frame_idx] = state["height"]

    # get velocity here
    ego_positions = np.hstack(
        [ego_track["state"]["position_x"], ego_track["state"]["position_y"]]
    )
    vel = (ego_positions[1:] - ego_positions[:-1]) / 0.1
    ego_track["state"]["velocity_x"][:-1] = vel[..., 0]
    ego_track["state"]["velocity_x"][-1] = ego_track["state"]["velocity_x"][-2]
    ego_track["state"]["velocity_y"][:-1] = vel[..., 1]
    ego_track["state"]["velocity_y"][-1] = ego_track["state"]["velocity_y"][-2]

    # check
    assert ego_id in tracks
    for track_id in tracks:
        assert tracks[track_id]["type"] != "UNSET"

    for track in tracks.values():
        # only save vehicles, pedestrians, bicycles
        if only_agents and track["type"] > 2:
            continue
        agent_role.append([False, False, False])
        if track["type"] in [0, 1, 2]:
            agent_role[-1][2] = True
            agent_role[-1][1] = True
        if track["metadata"]["nuplan_type"] == int(NuPlanEgoType):
            agent_role[-1][0] = True
        agent_id.append(track["metadata"]["object_id"])
        agent_type.append(track["type"])
        agent_states_list = np.vstack(list(track["state"].values())).T.tolist()
        agent_states.append(agent_states_list)

    return agent_id, agent_type, agent_states, agent_role


def collate_tl_features(scenario, center, start_iter):
    tl_lane_state = []
    tl_lane_id = []
    tl_stop_point = []

    frames = [
        {
            str(t.lane_connector_id): t.status
            for t in scenario.get_traffic_light_status_at_iteration(i)
        }
        for i in range(start_iter, N_STEP + start_iter)
    ]

    for frame in frames:
        tl_lane_state_frame = [TL_TYPES[status.name] for status in frame.values()]
        tl_lane_state.append(tl_lane_state_frame)
        tl_lane_id.append(list(frame.keys()))
        tl_stop_point_frame = []
        for lane_id in frame.keys():
            light_pos = set_light_position(scenario, lane_id, center)
            tl_stop_point_frame.append(light_pos)
        if tl_stop_point_frame:
            tl_stop_point.append(mock_2d_to_3d_points(tl_stop_point_frame))
        else:
            tl_stop_point.append([])

    return tl_lane_state, tl_lane_id, tl_stop_point


def collate_map_features(map_api, center, radius=200):
    # map features
    mf_id = []
    mf_xyz = []
    mf_type = []
    mf_edge = []

    np.seterr(all="ignore")
    # Center is Important !
    layer_names = [
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
        # SemanticMapLayer.STOP_SIGN,
        # SemanticMapLayer.DRIVABLE_AREA,
    ]
    center_for_query = Point2D(*center)
    nearest_vector_map = map_api.get_proximal_map_objects(
        center_for_query, radius, layer_names
    )
    # BOUNDARIES
    boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)

    # STOP LINES
    # Filter out stop polygons from type turn stop
    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon
            for stop_polygon in stop_polygons
            if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]
        for stop_line_polygon_obj in nearest_vector_map[SemanticMapLayer.STOP_LINE]:
            stop_line_polygon = stop_line_polygon_obj.polygon.exterior.coords
            mf_id.append(stop_line_polygon_obj.id)
            mf_type.append(PL_TYPES["STOP_LINE"])
            polygon_centered = nuplan_to_centered_vector(
                np.array(stop_line_polygon), nuplan_center=[center[0], center[1]]
            )
            mf_xyz.append(mock_2d_to_3d_points(polygon_centered)[::4])

    # ROADBLOCKS (contain lanes)
    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            edges = (
                sorted(block.interior_edges, key=lambda lane: lane.index)
                if layer == SemanticMapLayer.ROADBLOCK
                else block.interior_edges
            )
            for index, lane_meta_data in enumerate(edges):
                if not hasattr(lane_meta_data, "baseline_path"):
                    continue
                # if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                #     boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(
                #         index_parts=True
                #     )
                #     sizes = []
                #     for idx, polygon in enumerate(boundary[0]):
                #         sizes.append(len(polygon.xy[1]))
                #     points = boundary[0][np.argmax(sizes)].xy
                # elif isinstance(lane_meta_data.polygon.boundary, LineString):
                #     points = lane_meta_data.polygon.boundary.xy
                # polygon = nuplan_to_centered_vector(
                #     np.array(points).T, nuplan_center=[center[0], center[1]]
                # )
                # mf_id.append(int(lane_meta_data.id))
                # mf_type.append(PL_TYPES[lane_meta_data.name])
                # mf_xyz.append(mock_2d_to_3d_points(polygon)[::3])
                if len(lane_meta_data.outgoing_edges) > 0:
                    for _out_edge in lane_meta_data.outgoing_edges:
                        mf_edge.append([int(lane_meta_data.id), int(_out_edge.id)])
                else:
                    mf_edge.append([int(lane_meta_data.id), -1])

                # real polylines @TODO: maybe use the following polylines instead of roadblocks BUT maybe for lanes instead of roadblocks
                centerline = extract_centerline(lane_meta_data, center, True, 1)
                mf_id.append(int(lane_meta_data.id))
                mf_type.append(PL_TYPES["CENTERLINE"])
                mf_xyz.append(mock_2d_to_3d_points(centerline))
                # mf_edge.append([])
                # left = lane_meta_data.left_boundary
                # left_polyline = get_points_from_boundary(left, center)
                # mf_id.append(left.id)
                # mf_type.append(PL_TYPES["LINE_BROKEN_SINGLE_WHITE"])
                # mf_xyz.append(mock_2d_to_3d_points(left_polyline)[::2])
                # mf_edge.append(
                #     []
                # )  # @TODO: calculate dir vectors
                # right = lane_meta_data.right_boundary
                # right_polyline = get_points_from_boundary(right, center)
                # mf_id.append(right.id)
                # mf_type.append(PL_TYPES["LINE_BROKEN_SINGLE_WHITE"])
                # mf_xyz.append(mock_2d_to_3d_points(right_polyline)[::2])
                # mf_edge.append([])  # @TODO: calculate dir vectors

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    # ROUTE
    # scenario.get_route_roadblock_ids()

    # WALKWAYS
    for area in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = nuplan_to_centered_vector(
            np.array(points).T, nuplan_center=[center[0], center[1]]
        )
        mf_id.append(int(area.id))
        mf_type.append(PL_TYPES["WALKWAYS"])
        mf_xyz.append(mock_2d_to_3d_points(polygon)[::4])

    # CROSSWALK
    for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = nuplan_to_centered_vector(
            np.array(points).T, nuplan_center=[center[0], center[1]]
        )
        mf_id.append(int(area.id))
        mf_type.append(PL_TYPES["CROSSWALK"])
        mf_xyz.append(mock_2d_to_3d_points(polygon)[::4])

    # INTERSECTION
    interpolygons = [
        block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]
    ]
    boundaries = gpd.GeoSeries(
        unary_union(interpolygons + block_polygons)
    ).boundary.explode(index_parts=True)
    # boundaries.plot()
    # plt.show()
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(
            list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1]))
        )
        block_points = nuplan_to_centered_vector(block_points, center)
        mf_id.append(idx)
        mf_type.append(PL_TYPES["INTERSECTION"])
        mf_xyz.append(mock_2d_to_3d_points(block_points)[::4])

    np.seterr(all="warn")
    return mf_id, mf_xyz, mf_type, mf_edge


def convert_nuplan_scenario(
    h5file: h5py.File,
    i: int,
    scenario: NuPlanScenario,
    rand_pos,
    rand_yaw,
    pack_all,
    pack_history,
    dest_no_pred,
    iteration: int = 0,
    split: str = "training",
):
    scenario_log_interval = scenario.database_interval
    assert abs(scenario_log_interval - 0.1) < 1e-3, (
        "Log interval should be 0.1 or Interpolating is required! "
        "By setting NuPlan subsample ratio can address this"
    )
    # centered all positions to ego car
    state = scenario.get_ego_state_at_iteration(iteration)
    scenario_center = [state.waypoint.x, state.waypoint.y]

    # agents
    agent_id, agent_type, agent_states, agent_role = collate_agent_features(
        scenario, scenario_center, iteration
    )
    # traffic light
    tl_lane_state, tl_lane_id, tl_stop_point = collate_tl_features(
        scenario, scenario_center, iteration
    )
    # map
    mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(
        scenario.map_api, scenario_center
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
    scenario_center, scenario_yaw = pack_utils.center_at_sdc(
        episode, rand_pos, rand_yaw
    )

    episode_reduced = {}
    pack_utils.filter_episode_map(episode, N_PL, THRESH_MAP, thresh_z=3)
    episode_with_map = episode["map/valid"].any(1).sum() > 0
    pack_utils.repack_episode_map(episode, episode_reduced, N_PL, N_PL_TYPE)

    pack_utils.filter_episode_traffic_lights(episode)
    pack_utils.repack_episode_traffic_lights(episode, episode_reduced, N_TL, N_TL_STATE)

    if "training" in split:
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
            f"scenario {i} has no map! map boundary is: {episode_reduced['map/boundary']}"
        )

    hf_episode = h5file.create_group(str(i))
    hf_episode.attrs["scenario_id"] = os.path.splitext(scenario.log_name)[0] + str(
        iteration
    )
    hf_episode.attrs["scenario_center"] = scenario_center
    hf_episode.attrs["scenario_yaw"] = scenario_yaw
    hf_episode.attrs["with_map"] = episode_with_map

    for k, v in episode_reduced.items():
        hf_episode.create_dataset(
            k, data=v, compression="gzip", compression_opts=4, shuffle=True
        )

    return n_pl, n_tl, n_agent, n_agent_sim, n_agent_no_sim


def main():
    # fmt: off
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir", default=os.getenv("NUPLAN_DATA_ROOT"))
    parser.add_argument("--map-dir", default=os.getenv("NUPLAN_MAPS_ROOT"))
    parser.add_argument("--version", "-v", default="v1.1", help="version of the raw data")
    parser.add_argument("--dataset", default="training")
    parser.add_argument("--out-dir", default="/mrtstorage/datasets_tmp/nuplan_hptr")
    parser.add_argument("--rand-pos", default=50.0, type=float, help="Meter. Set to -1 to disable.")
    parser.add_argument("--rand-yaw", default=3.14, type=float, help="Radian. Set to -1 to disable.")
    parser.add_argument("--dest-no-pred", action="store_true")
    parser.add_argument("--test", action="store_true", help="for test use only. convert one log")
    args = parser.parse_args()
    # fmt: on

    if "training" in args.dataset:
        pack_all = True  # ["agent/valid"]
        pack_history = False  # ["history/agent/valid"]
    elif "validation" in args.dataset:
        pack_all = True
        pack_history = True
    elif "testing" in args.dataset:
        pack_all = False
        pack_history = True

    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    out_h5_path = out_path / (args.dataset + "_tmp" + ".h5")

    data_root = os.path.join(args.data_dir, "nuplan-v1.1/splits/mini")

    if args.test:
        scenarios = get_nuplan_scenarios(
            data_root, args.map_dir, logs=["2021.07.16.20.45.29_veh-35_01095_01486"]
        )
    else:
        scenarios = get_nuplan_scenarios(data_root, args.map_dir)

    n_pl_max, n_tl_max, n_agent_max, n_agent_sim, n_agent_no_sim, data_len = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    with h5py.File(out_h5_path, "w") as hf:
        k = 0
        for i, scenario in tqdm(
            enumerate(scenarios),
            total=len(scenarios),
            desc="Converting nuPlan Scenarios",
        ):
            scenario_len_sec = int(scenario.duration_s.time_s)
            # episode_len_iter = scenario.get_number_of_iterations()
            scenario_time_step = scenario.database_interval
            assert scenario_time_step == 0.1, "Only support 0.1s time step"
            for j in range(
                0,
                int((scenario_len_sec - 1) / scenario_time_step) - (N_STEP - 1),
                10,
            ):
                # try:
                (
                    _n_pl_max,
                    _n_tl_max,
                    _n_agent_max,
                    _n_agent_sim,
                    _n_agent_no_sim,
                ) = convert_nuplan_scenario(
                    hf,
                    k,
                    scenario,
                    args.rand_pos,
                    args.rand_yaw,
                    pack_all,
                    pack_history,
                    args.dest_no_pred,
                    j,
                    args.dataset,
                )

                n_pl_max = max(n_pl_max, _n_pl_max)
                n_tl_max = max(n_tl_max, _n_tl_max)
                n_agent_max = max(n_agent_max, _n_agent_max)
                n_agent_sim = max(n_agent_sim, _n_agent_sim)
                n_agent_no_sim = max(n_agent_no_sim, _n_agent_no_sim)
                data_len += 1

                print(f"data_len: {data_len}, dataset_size: {len(scenarios)}")
                print(f"n_pl_max: {n_pl_max}")
                print(
                    f"n_agent_max: {n_agent_max}, n_agent_sim: {n_agent_sim}, n_agent_no_sim: {n_agent_no_sim}"
                )
                hf.attrs["data_len"] = data_len
                k += 1
                # except:
                #     print(f"\nError in scenario {i}: {scenario.log_name}\n")
                #     continue
        hf.attrs["version"] = args.version


if __name__ == "__main__":
    main()
