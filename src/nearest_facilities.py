import os, sys
import traceback
from multiprocessing import current_process, Lock
from time import time, strftime

import click
import networkx as nx
from shapely import STRtree
from tqdm import tqdm

from Core.common import resource_path, set_main_path
from Core.log4p import Log, mTqdm
from colorama import Fore

from Core.graph import makeGraph, Direction

log = Log(__name__)
lock = Lock()

@click.command()
@click.option("--network", '-n', type=str, required=True,
              help="输入网络数据, 支持的格式包括ESRI Shapefile, geojson, ESRI FileGDB, Spatilite, "
                   "gpickle, GEXF, GML, GraphML. 必选.")
@click.option("--network-layer", type=str, required=False,
              help="输入网络数据的图层名, 可选. 如果是文件数据库(gdb, sqlite)则必须提供,"
                   "如果是文件(shapefile, geojson)则不需要提供.")
@click.option("--direction-field", "-d", type=str, required=False, default="",
              help="输入网络数据表示方向的字段, 可选."
                   "如果网络数据需要表示有向边,则需要提供. 例如OSM数据中用'oneway'字段表示方向, 则输入'-d 'oneway''.")
@click.option("--forward-value", type=str, required=False, default="",
              help="输入网络数据方向字段中表示正向的值, 可选. "
                   "如果网络数据需要表示有向边,则需要提供. 例如OSM数据中用'F'值表示正向，则输入'--forward-value 'F''.")
@click.option("--backward-value", type=str, required=False, default="",
              help="输入网络数据方向字段中表示反向的值, 可选. "
                   "如果网络数据需要表示有向边,则需要提供. 例如OSM数据中用'T'值表示反向，则输入'--backward-value 'T''.")
@click.option("--both-value", type=str, required=False, default="",
              help="输入网络数据方向字段中表示双向的值, 可选. "
                   "如果网络数据需要表示有向边,则需要提供. 例如OSM数据中用'B'值表示双方向，则输入'--both-value 'B''.")
@click.option("--default-direction", type=int, required=False, default=0,
              help="输入网络数据edge的默认方向, 可选. 当未提供方向字段或者方向字段中的方向值不存在时生效,用于提供edge的默认方向."
                   "0-双向, 1-正向, 2-反向. ")
@click.option("--spath", '-s', type=str, required=True,
              help="输入起始设施数据, 必选.")
@click.option("--spath-layer", type=str, required=False,
              help="输入起始设施数据的图层名, 可选. 如果是文件数据库(gdb, sqlite)则必须提供, "
                   "如果是文件(shapefile, geojson)则不需要提供.")
@click.option("--scapacity-field", type=str, required=True,
              help="输入起始设施数据的容量字段, 必选.")
@click.option("--tpath", '-t', type=str, required=True,
              help="输入目标设施数据, 必选.")
@click.option("--tpath-layer", type=str, required=False, default="",
              help="输入目标设施数据的图层名, 可选. 如果是文件数据库(gdb, sqlite)则必须提供, "
                   "如果是文件(shapefile, geojson)则不需要提供.")
@click.option("--tcapacity-field", type=str, required=True,
              help="输入目标设施数据的容量字段, 必选.")
# @click.option("--sweigth-field", type=str, required=True,
#               help="输入起始设施数据的权重字段, 必选. 用于分配过程中提升设施的选取概率.")
@click.option("--tweight-field", type=str, required=False, default="",
              help="输入目标设施数据的权重字段, 可选. 用于分配过程中提升设施的选取概率;"
                   "如果不提供,则所有目标设施选取权重根据距离的倒数来定义.")
@click.option("--cost", "-c", type=float, required=False, multiple=True, default=[sys.float_info.max],
              help="路径搜索范围, 超过范围的设施不计入搜索结果, 可选. 缺省值会将所有可达设施都加入结果,同时导致搜索速度极大下降, "
                   "建议根据实际情况输入合适的范围."
                   "允许输入多个值，例如'-c 1000 -c 1500'表示同时计算1000和1500两个范围的可达设施.")
@click.option("--distance-tolerance", type=float, required=False, default=500,
              help="定义目标设施到网络最近点的距离容差，如果超过说明该设施偏离网络过远，不参与计算, 可选, 默认值为500.")
@click.option("--out-type", type=click.Choice(['shp', 'geojson', 'filegdb', 'sqlite', 'csv'], case_sensitive=False),
              required=False, default='shp',
              help="输出文件格式, 默认值shp. 支持格式shp-ESRI Shapefile, geojson-geojson, filegdb-ESRI FileGDB, "
                   "sqlite-spatialite, csv-csv.")
@click.option("--out-graph-type", type=click.Choice(['gpickle', 'graphml', 'gml', 'gexf'], case_sensitive=False),
              required=False, default='gpickle',
              help="如果原始网络数据是空间数据(shp, geojson, gdb等), 则需要设置存储的图文件格式, "
                   "默认值gpicke. 支持gpickle, graphml, gml, gexf.")
# 0-ESRI Shapefile, 1-geojson, 2-fileGDB, 3-spatialite, 4-csv
@click.option("--out-path", "-o", type=str, required=False, default="res",
              help="输出目录名, 可选, 默认值为当前目录下的'res'.")
@click.option("--cpu-count", type=int, required=False, default=1,
              help="多进程数量, 可选, 默认为1, 即单进程. 小于0或者大于CPU最大核心数则进程数为最大核心数,否则为输入实际核心数.")
def nearest(network, network_layer, direction_field, forward_value, backward_value, both_value,
            default_direction, cost, out_type, out_graph_type, out_path, cpu_count):
    """网络距离可达设施搜索算法."""
    pass


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
        bar = mTqdm(lst, desc="worker-{}".format(ipos), position=ipos, leave=False)
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

