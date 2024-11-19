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
import multiprocessing as mp
from functools import partial
from more_itertools import batched
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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
    get_route_lane_polylines_from_roadblock_ids,
    get_id_and_start_idx_for_scenarios,
)

SCENARIO_QUEUE = mp.Manager().Queue()

PL_TYPES = {
    # "LANE": 0,
    "INTERSECTION": 0,
    "STOP_LINE": 1,
    # "TURN_STOP": x,
    "CROSSWALK": 2,
    # "DRIVABLE_AREA": x,
    # "YIELD": x,
    # "TRAFFIC_LIGHT": x,
    # "STOP_SIGN": x,
    # "EXTENDED_PUDO": x,
    # "SPEED_BUMP": x,
    # "LANE_CONNECTOR": x,
    # "BASELINE_PATHS": x,
    "BOUNDARIES": 3,
    "WALKWAYS": 4,
    "CARPARK_AREA": 5,
    # "PUDO": x,
    # "ROADBLOCK": x,
    # "ROADBLOCK_CONNECTOR": x,
    "LINE_BROKEN_SINGLE_WHITE": 6,
    "CENTERLINE": 7,
    "ROUTE": 8,
}
N_PL_TYPE = len(PL_TYPES)
DIM_VEH_LANES = [7]
DIM_CYC_LANES = [3, 7]
DIM_PED_LANES = [2, 3, 4]

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
                near_to_ego=False,
                distance_to_ego=1000,
            ),
        )
        for i, id in enumerate(list(all_obj_ids))
    }

    tracks_to_remove = set()
    dists_to_ego = []

    for frame_idx, frame in enumerate(detection_ret):
        for nuplan_id, obj_state in frame.items():
            assert isinstance(obj_state, Agent) or isinstance(obj_state, StaticObject)
            obj_type = obj_state.tracked_object_type
            if obj_type is None:
                tracks_to_remove.add(nuplan_id)
                continue
            tracks[nuplan_id]["type"] = AGENT_TYPES[obj_type.name]
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

            if frame_idx == STEP_CURRENT and tracks[nuplan_id]["type"] < 3:
                x_pos, y_pos = obj_state.center.x, obj_state.center.y
                dist_to_ego = np.linalg.norm(np.array([x_pos, y_pos]) - center)
                dists_to_ego.append(dist_to_ego)
                tracks[nuplan_id]["metadata"]["distance_to_ego"] = dist_to_ego

    dists_to_ego.sort()

    if len(dists_to_ego) > N_AGENT_PRED_CHALLENGE - 1:
        predict_dist = dists_to_ego[N_AGENT_PRED_CHALLENGE - 2]
    else:  # not enough agents
        predict_dist = dists_to_ego[-1] if len(dists_to_ego) > 0 else -1
    interest_dist = (
        dists_to_ego[N_AGENT_INTERACT_CHALLENGE - 2] if len(dists_to_ego) > 1 else -1
    )

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
        _dist_to_ego = track["metadata"]["distance_to_ego"]
        agent_role.append([False, False, False])
        if track["type"] in [0, 1, 2]:
            agent_role[-1][2] = True if _dist_to_ego <= predict_dist else False
            agent_role[-1][1] = True if _dist_to_ego <= interest_dist else False
        if track["metadata"]["nuplan_type"] == int(NuPlanEgoType):
            agent_role[-1][0] = True
            agent_role[-1][1] = True
            agent_role[-1][2] = True
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
            light_pos = set_light_position(scenario.map_api, lane_id, center)
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

    # LANES
    for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        for lane in nearest_vector_map[layer]:
            if not hasattr(lane, "baseline_path"):
                continue
            # Centerline (as polyline)
            centerline = extract_centerline(lane, center, True, 1)
            mf_id.append(int(lane.id))  # using lane ids for centerlines!
            mf_type.append(PL_TYPES["CENTERLINE"])
            mf_xyz.append(mock_2d_to_3d_points(centerline))
            if len(lane.outgoing_edges) > 0:
                for _out_edge in lane.outgoing_edges:
                    mf_edge.append([int(lane.id), int(_out_edge.id)])
            else:
                mf_edge.append([int(lane.id), -1])
            # Left boundary of centerline (as polyline)
            left = lane.left_boundary
            left_polyline = get_points_from_boundary(left, center, True, 1)
            mf_id.append(left.id)
            mf_type.append(PL_TYPES["LINE_BROKEN_SINGLE_WHITE"])
            mf_xyz.append(mock_2d_to_3d_points(left_polyline))
            # right = lane.right_boundary
            # right_polyline = get_points_from_boundary(right, center)
            # mf_id.append(right.id)
            # mf_type.append(PL_TYPES["LINE_BROKEN_SINGLE_WHITE"])
            # mf_xyz.append(mock_2d_to_3d_points(right_polyline)[::2])

    # ROADBLOCKS (contain lanes)
    # Extract neighboring lanes and road boundaries
    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            # roadblock_polygon = block.polygon.boundary.xy
            # polygon = nuplan_to_centered_vector(
            #     np.array(roadblock_polygon).T, nuplan_center=[center[0], center[1]]
            # )

            # According to the map attributes, lanes are numbered left to right with smaller indices being on the
            # left and larger indices being on the right.
            lanes = (
                sorted(block.interior_edges, key=lambda lane: lane.index)
                if layer == SemanticMapLayer.ROADBLOCK
                else block.interior_edges
            )
            for i, lane in enumerate(lanes):
                if not hasattr(lane, "baseline_path"):
                    continue
                if layer == SemanticMapLayer.ROADBLOCK:
                    if i != 0:
                        left_neighbor = lanes[i - 1]
                        mf_edge.append([int(lane.id), int(left_neighbor.id)])
                    if i != len(lanes) - 1:
                        right_neighbor = lanes[i + 1]
                        mf_edge.append([int(lane.id), int(right_neighbor.id)])
                    # if i == 0:  # left most lane
                    #     left = lane.left_boundary
                    #     left_boundary = get_points_from_boundary(left, center, True, 1)
                    #     try:
                    #         idx = mf_id.index(left.id)
                    #         mf_id[idx] = left.id  # use roadblock ids for boundaries
                    #         mf_type[idx] = PL_TYPES["BOUNDARIES"]
                    #         mf_xyz[idx] = mock_2d_to_3d_points(left_boundary)
                    #     except:
                    #         mf_id.append(block.id)  # use roadblock ids for boundaries
                    #         mf_type.append(PL_TYPES["BOUNDARIES"])
                    #         mf_xyz.append(mock_2d_to_3d_points(right_boundary))
                    if i == len(lanes) - 1:  # right most lane
                        right = lane.right_boundary
                        right_boundary = get_points_from_boundary(
                            right, center, True, 1
                        )
                        try:
                            idx = mf_id.index(right.id)
                            mf_id[idx] = right.id  # use roadblock ids for boundaries
                            mf_type[idx] = PL_TYPES["BOUNDARIES"]
                            mf_xyz[idx] = mock_2d_to_3d_points(right_boundary)
                        except:
                            mf_id.append(block.id)  # use roadblock ids for boundaries
                            mf_type.append(PL_TYPES["BOUNDARIES"])
                            mf_xyz.append(mock_2d_to_3d_points(right_boundary))

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

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


def collate_route_features(scenario, center, agent_id, agent_role, radius=200):
    # Note: currently only working for one ego vehicle
    sdc_id = []
    sdc_route_type = []
    sdc_route_lane_id = []
    sdc_route_xyz = []

    sdc_idx = agent_role.index([True, True, True])  # ego
    sdc_id.append(agent_id[sdc_idx])

    route_roadblock_ids = scenario.get_route_roadblock_ids()
    polylines, route_lane_ids = get_route_lane_polylines_from_roadblock_ids(
        scenario.map_api, Point2D(center[0], center[1]), radius, route_roadblock_ids
    )
    route_lane_polylines = []
    pl_types = []
    for polyline in polylines:

        polyline_centered = nuplan_to_centered_vector(polyline, center)
        route_lane_polylines.append(mock_2d_to_3d_points(polyline_centered)[::10])
        pl_types.append(PL_TYPES["ROUTE"])
    sdc_route_xyz.append(route_lane_polylines)
    sdc_route_lane_id.append(route_lane_ids)
    sdc_route_type.append(pl_types)
    return sdc_id, sdc_route_lane_id, sdc_route_type, sdc_route_xyz


def convert_nuplan_scenario(
    scenario: NuPlanScenario,
    iteration,
    rand_pos,
    rand_yaw,
    pack_all,
    pack_history,
    dest_no_pred,
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

    # route
    sdc_id, sdc_route_id, sdc_route_type, sdc_route_xyz = collate_route_features(
        scenario, scenario_center, agent_id, agent_role
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
    scenario_tuple, rand_pos, rand_yaw, pack_all, pack_history, dest_no_pred, split
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
        out_h5_path = out_path / (args.dataset + "_mini_" + ".h5")

    if args.test:
        data_root = os.path.join(args.data_dir, "nuplan-v1.1/splits/mini")
        scenarios = get_nuplan_scenarios(
            data_root, args.map_dir, logs=["2021.07.16.20.45.29_veh-35_01095_01486"]
        )
    else:
        data_root = os.path.join(args.data_dir, "nuplan-v1.1/splits", split)
        scenarios = get_nuplan_scenarios(data_root, args.map_dir)
    print(f"Converting {len(scenarios)} scenarios to {args.dataset} dataset")

    # preprocessing: convert scnearios list to list of tuples (scenario, start_iter)
    scenario_tuples = get_id_and_start_idx_for_scenarios(scenarios, N_STEP)
    print(f"Converting {len(scenario_tuples)} sub-scenarios to {args.dataset} dataset")

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

    # with h5py.File(out_h5_path, "w") as hf:
    #     for batch in tqdm(list(batched(scenario_tuples, args.batch_size))):
    #         with ProcessPoolExecutor(max_workers=args.num_workers) as p:
    #             res = p.map(convert_func, batch)
    #             # alternative: starmap from multiprocessing (with mp.Pool)

    #         for sample in res:
    #             episode, metadata = sample
    #             hf_episode = hf.create_group(str(metadata["id"]))

    #             for k, v in episode.items():
    #                 hf_episode.create_dataset(
    #                     str(k),
    #                     data=v,
    #                     compression="gzip",
    #                     compression_opts=4,
    #                     shuffle=True,  # shuffling saves memory
    #                 )

    print(f"n_pl_max: {n_pl_max}")
    print(f"n_route_pl_max: {n_route_pl_max}")
    print(f"n_tl_max: {n_tl_max}")
    print(f"n_agent_max: {n_agent_max}")
    print(f"n_agent_sim_max: {n_agent_sim_max}")
    print(f"n_agent_no_sim_max: {n_agent_no_sim_max}")


if __name__ == "__main__":
    main()
