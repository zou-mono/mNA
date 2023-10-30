import os, sys
import traceback
from multiprocessing import current_process, Lock
from time import time, strftime

import networkx as nx
from shapely import STRtree
from tqdm import tqdm

from Core.common import resource_path, set_main_path
from Core.log4p import Log, mTqdm
from colorama import Fore

from Core.graph import makeGraph, Direction

log = Log(__name__)
lock = Lock()

def nearest_facilities_from_layer(
        input_network_data,
        input_start_data,
        input_target_data,
        out_path="res",
        travelCost=0.0,
        direction_field="",
        forwardValue="",
        backwardValue="",
        bothValue="",
        distance_tolerance=500,  # 从原始点到网络最近snap点的距离容差，如果超过说明该点无法到达网络，不进行计算
        defaultDirection=Direction.DirectionBoth,
        out_type=0):

    start_time = time()

    log.info("读取网络数据...")


    log.info("计算可达范围内的邻近设施...")

    main_path = resource_path("")
    set_main_path(main_path)
    out_path = os.path.join(main_path, "res")
    # QgsProviderRegistry.instance().setLibraryDirectory(QDir(os.path.join(main_path, r"qgis\plugins")))
    # log.debug("load providers {}".format(QgsProviderRegistry.instance().providerList()))

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    else:
        end_time = time()
        log.info("所有步骤完成,共耗时{}秒".format(str(end_time - start_time)))


def nearest_facilities(mNetwork,
                       mDirector,
                       start_points,
                       target_points,
                       start_points_lst,
                       target_points_lst,
                       start_features,
                       target_features,
                       distance_tolerance,
                       travelCost
                       ):

    pass


def nearest_facilities_from_point(G, start_node, target_df,
                                  travelCost,
                                  bRoutes=True,
                                  bDistances=True):

    match_routes = {}
    match_distances = {}

    # return current_process().name + '-' + str(start_node)

    distances, routes = nx.single_source_dijkstra(G, start_node, weight='length',
                                                  cutoff=travelCost)

    if len(routes) > 1:  # 只有一个是本身，不算入搜索结果
        # 寻找匹配到的目标设施对应的node以及fid
        match_df = target_df[target_df['nodeID'].apply(lambda x: x in routes)]

        # match_nodes = match_df['nodeID']
        for row in match_df.itertuples():
            match_node = row.nodeID
            target_fid = row.fid

            if bRoutes:
                route = routes[match_node]
                target_fid = row.fid
                match_routes[target_fid] = route

            if bDistances:
                dis = distances[match_node]
                match_distances[target_fid] = dis

    if bRoutes and bDistances:
        return match_routes, match_distances

    if bRoutes:
        return match_routes

    if bDistances:
        return match_distances


def nearest_facilities_from_point_worker(connection, shared_custom, lst, travelCost, bRoutes=True,
                                         bDistances=True, ipos=0):
    G, target_df = shared_custom.task()

    nearest_facilities = {}

    # tqdm.set_lock(tqdm.get_lock())
    # ipos = current_process()._identity[0]-1

    # mTqdm.set_lock(mTqdm.get_lock())
    with lock:
        bar = mTqdm(lst, desc=current_process().name, position=ipos, leave=False)
    # with mTqdm(lst, desc=current_process().name, position=ipos, leave=False) as bar:
    for t in lst:
        start_node = t[0]
        fid = t[1]
        distances, routes = nx.single_source_dijkstra(G, start_node, weight='length',
                                                      cutoff=travelCost)

        match_routes = {}
        match_distances = {}

        if len(routes) > 1:  # 只有一个是本身，不算入搜索结果
            # 寻找匹配到的目标设施对应的node以及fid
            match_df = target_df[target_df['nodeID'].apply(lambda x: x in routes)]

            for row in match_df.itertuples():
                match_node = row.nodeID
                target_fid = row.fid

                if bRoutes:
                    route = routes[match_node]
                    target_fid = row.fid
                    match_routes[target_fid] = route

                if bDistances:
                    dis = distances[match_node]
                    match_distances[target_fid] = dis

        if bRoutes and bDistances:
            nearest_facilities[fid] = {
                'routes': match_routes,
                'distance': match_distances
            }

        if bRoutes and not bDistances:
            nearest_facilities[fid] = match_routes

        if bDistances and not bRoutes:
            nearest_facilities[fid] = match_distances

        # bar.update()
        with lock:
            bar.update()

    with lock:
        bar.close()

    if connection is not None:
        connection.send(nearest_facilities)
    else:
        return nearest_facilities


def neareset_route_from_point(graph,
                              tree,  # 最短路径树
                              idxStart,
                              idxTarget,
                              snapped_point_start,
                              travelCost):

    route = []
    cost = 0.0

    # print("start point:{}".format(snappedPoints[i]))
    # print("target point:{}".format(snappedPoints[target_pos]))

    current = idxTarget

    if tree[current] == -1:
        return [], 0.0

    bCan = True
    while current != idxStart:
        cost += graph.edge(tree[current]).cost(0)

        if cost <= travelCost:
            route.append(graph.vertex(graph.edge(tree[current]).toVertex()).point())
            current = graph.edge(tree[current]).fromVertex()
        else:
            bCan = False
            break

    if bCan:
        # target_ids.append(cand_target)
        route.append(snapped_point_start)
        route.reverse()

        if len(route) > 1:
            # routes.append(route)
            # geom_route = QgsGeometry.fromPolylineXY(route)
            # routes.append(route)
            # res_costs.append(cost)
            # target_pts.append(target_points[cand_target])
            return route, cost
    else:
        return [], 0.0


def export_to_file(results, fields, crs, layer_start, out_path, out_type):
    pass


if __name__ == '__main__':
    # QgsApplication.setPrefixPath('', True)
    # app = QgsApplication([], True)
    # app.initQgis()

    nearest_facilities_from_layer(
        r"D:\空间模拟\PublicSupplyDemand\Data\sz_road_cgcs2000_test.shp",
        r"D:\空间模拟\PublicSupplyDemand\Data\building_test.shp",
        r"D:\空间模拟\PublicSupplyDemand\Data\2022年现状幼儿园.shp",
        travelCost=1000,
        out_type=0,
        direction_field="")

    # sys.exit(app.exec_())

