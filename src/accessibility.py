import os.path
import sys
import traceback
from math import ceil
from multiprocessing import freeze_support, cpu_count, RLock, Pool, Lock
from time import strftime, time

import click
import networkx as nx
import shapely
from osgeo import gdal, ogr
from osgeo.ogr import CreateGeometryFromWkt
from pandas import DataFrame
from shapely import Point, geometry, concave_hull, set_precision, to_wkt
from shapely.ops import linemerge
from tqdm import tqdm

from Core.DataFactory import get_suffix, workspaceFactory, addFeature, creation_options
from Core.check import init_check
from Core.common import get_centerPoints
from Core.core import DataType, QueueManager, DataType_suffix, check_layer_name
from Core.graph import Direction, import_graph_to_network, create_graph_from_file, export_network_to_graph, makeGraph, \
    GraphTransfer
from Core.log4p import Log, mTqdm
from Core.ogrmerge import ogrmerge
from Core.utils_graph import split_line_by_point

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
@click.option("--out-fields", '-f', type=int, required=False, multiple=True, default=[-99],
              help="输出数据保留原目标设施的字段编号, 可选, 默认全部保留, -1表示不保留任何字段, -99表示全部保留. "
                   "允许输入多个值，例如'-f 1 -f 2'表示保留第1和第2个字段.")
@click.option("--cost", "-c", type=float, required=False, multiple=True, default=[sys.float_info.max],
              help="路径搜索范围, 超过范围的设施不计入搜索结果, 可选. 缺省值会将所有可达设施都加入结果,同时导致搜索速度极大下降, "
                   "建议根据实际情况输入合适的范围."
                   "允许输入多个值，例如'-c 1000 -c 1500'表示同时计算1000和1500两个范围的可达设施.")
# @click.option("--include-bounds", '-b', type=bool, required=False, default=False,
#               help="是否输出可达性边界每条边的起点和终点. 可选, 默认值为否.")
@click.option("--concave-hull-ratio", '-r', type=click.FloatRange(0, 1, clamp=False), required=False, default=0.5,
              help="可达性凹包的阈值, 取值范围0-1, 越接近1则凹包越平滑. 可选, 默认值为0.5.")
@click.option("--distance-tolerance", type=float, required=False, default=500,
              help="定义目标设施到网络最近点的距离容差，如果超过说明该设施偏离网络过远，不参与计算. 可选, 默认值为500.")
@click.option("--duplication-tolerance", type=float, required=False, default=10.0,
              help="定义拓扑重复点的容差，取值越大则最后的图节点越少，运行速度越快，但同时会牺牲一定的精度. 可选, 默认值为10.")
@click.option("--out-type", type=click.Choice(['shp', 'geojson', 'filegdb', 'sqlite', 'csv'], case_sensitive=False),
              required=False, default='csv',
              help="输出文件格式, 默认值shp. 支持格式shp-ESRI Shapefile, geojson-geojson, filegdb-ESRI FileGDB, "
                   "sqlite-spatialite, csv-csv.")
@click.option("--out-graph-type", type=click.Choice(['gpickle', 'graphml', 'gml', 'gexf'], case_sensitive=False),
              required=False, default='gpickle',
              help="如果原始网络数据是空间数据(shp, geojson, gdb等), 则需要设置存储的图文件格式, "
                   "默认值gpicke. 支持gpickle, graphml, gml, gexf.")
# 0-ESRI Shapefile, 1-geojson, 2-fileGDB, 3-spatialite, 4-csv
@click.option("--out-path", "-o", type=str, required=False, default="res",
              help="输出目录名, 可选, 默认值为当前目录下的'res'.")
@click.option("--out-nodes", type=bool, required=False, default=False,
              help="是否输出可达最远节点. 可选, 默认不输出.")
@click.option("--out-routes", type=bool, required=False, default=True,
              help="是否输出可达路径. 可选, 默认不输出.")
@click.option("--cpu-count", type=int, required=False, default=1,
              help="多进程数量, 可选, 默认为1, 即单进程. 小于0或者大于CPU最大核心数则进程数为最大核心数,否则为输入实际核心数.")
def accessibility(network, network_layer, direction_field, forward_value, backward_value, both_value,
                  default_direction, spath, spath_layer, out_fields, cost, concave_hull_ratio,
                  distance_tolerance, duplication_tolerance, out_type, out_graph_type, out_path, out_nodes, out_routes, cpu_count):
    """设施可达范围算法"""
    travelCosts = []
    for c in cost:
        if c < 0:
            c = -c
        if c not in travelCosts:
            travelCosts.append(c)

    if not os.path.exists(network):
        log.error("网络数据文件不存在,请检查后重新计算!")
        return

    if not os.path.exists(spath):
        log.error("起始设施数据文件不存在,请检查后重新计算!")
        return

    if out_type.lower() == 'shp':
        out_type = DataType.shapefile.value
    elif out_type.lower() == 'geojson':
        out_type = DataType.geojson.value
    elif out_type.lower() == 'filegdb':
        out_type = DataType.fileGDB.value
    elif out_type.lower() == 'sqlite':
        out_type = DataType.sqlite.value
    elif out_type.lower() == 'csv':
        out_type = DataType.csv.value

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_path = os.path.abspath(os.path.join(out_path, "accessibility_res_{}".format(strftime('%Y-%m-%d-%H-%M-%S'))))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # -99 None 表示全部保留，-1 []表示不保留
    panMap = []
    for field in out_fields:
        if field == -1:
            panMap = []
            break
        if field == -99:
            panMap = None
            break
        if field not in panMap:
            panMap.append(field)

    accessible_from_layer(
        network_path=network,
        network_layer=network_layer,
        start_path=spath,
        start_layer_name=spath_layer,
        travelCosts=travelCosts,
        # include_bounds=include_bounds,
        concave_hull_ratio=concave_hull_ratio,
        panMap=panMap,
        direction_field=direction_field,
        forwardValue=forward_value,
        backwardValue=backward_value,
        bothValue=both_value,
        defaultDirection=default_direction,
        distance_tolerance=distance_tolerance,
        duplication_tolerance=duplication_tolerance,
        out_type=out_type,
        out_graph_type=out_graph_type,
        out_path=out_path,
        out_nodes=out_nodes,
        out_routes=out_routes,
        cpu_core=cpu_count)


def accessible_from_layer(
        network_path,
        network_layer,
        start_path,
        start_layer_name,
        travelCosts=[sys.float_info.max],
        # include_bounds=False,
        concave_hull_ratio=0.3,
        direction_field="",
        forwardValue="",
        backwardValue="",
        bothValue="",
        panMap=None,
        distance_tolerance=500,  # 从原始点到网络最近snap点的距离容差，如果超过说明该点无法到达网络，不进行计算
        duplication_tolerance=10,
        defaultDirection=Direction.DirectionBoth,
        out_type=0,
        out_graph_type='gpickle',
        out_path="res",
        out_nodes=False,
        out_routes=False,
        cpu_core=1):
    start_time = time()

    log.info("读取网络数据...")
    ds_start = None
    layer_start = None
    wks = workspaceFactory()

    if cpu_core <= 0 or cpu_core > cpu_count():
        cpu_core = cpu_count()

    try:
        log.info("读取网络数据, 路径为{}...".format(network_path))

        input_type = get_suffix(network_path)

        if input_type == 'gml' or input_type == 'gexf' or input_type == 'graphml' or input_type == 'gpickle':
            net = import_graph_to_network(network_path, input_type)
            if net is not None:
                log.info(f"从输入文件读取后构建的图共{len(net)}个节点, {len(net.edges)}条边")
        else:
            net = create_graph_from_file(network_path,
                                         network_layer,
                                         direction_field=direction_field,
                                         bothValue=bothValue,
                                         backwardValue=backwardValue,
                                         forwardValue=forwardValue)

            if net is not None:
                export_network_to_graph(out_graph_type, net, out_path)

        if net is None:
            log.error("网络数据存在问题, 无法创建图结构.")
            return

        log.info("读取起始设施数据,路径为{}...".format(start_path))
        ds_start = wks.get_ds(start_path)
        layer_start, start_layer_name = wks.get_layer(ds_start, start_path, start_layer_name)
        bflag, panMap, *_ = init_check(layer_start, None, "起始", panMap=panMap)
        if not bflag:
            return

        #  提取中心点
        log.info("计算设施位置坐标...")
        start_points_df = get_centerPoints(layer_start)

        start_points = start_points_df['geom'].to_list()

        # 这里需不需要对target_point进行空间检索？
        log.info("将目标设施附着到最近邻edge上，并且进行图重构...")
        G, snapped_nodeIDs = makeGraph(net, start_points, distance_tolerance=distance_tolerance,
                                       duplication_tolerance=duplication_tolerance)
        start_points_df["nodeID"] = snapped_nodeIDs

        log.info("重构后的图共有{}条边，{}个节点".format(len(G.edges), len(G)))

        # if G is not None:
        #     export_network_to_graph(out_graph_type, G, out_path, out_file="reconstructed")

        # max_cost = max(travelCosts)
        # 测试用点
        # start_points = [Point([519112.9421, 2505711.571])]
        # start_points = [Point([486769.99534558534,2520799.732169332])]
        # dic = {
        #     'fid': 0,
        #     'geom': start_points[0],
        #     'nodeID': 587455
        # }
        # start_points_df = DataFrame(dic, index=[0])

        log.info("计算起始设施可达范围的目标设施...")

        if out_type == DataType.csv.value:
            pass
            # nearest_facilities = calculate(G, df, start_points_df, max_cost, cpu_core)
            # log.info("开始导出计算结果...")
            # export_to_file(G, out_path, start_points_df, target_points_df, nearest_facilities,
            #                travelCosts, layer_start, "nearest", out_type, cpu_core)
        else:
            out_temp_path = os.path.abspath(os.path.join(out_path, "temp"))
            path_res = calculate(G, start_path, start_layer_name, out_temp_path, start_points_df,
                                 panMap, travelCosts, concave_hull_ratio, out_nodes, out_routes, cpu_core)

            tqdm.write("\r", end="")

            log.info("开始导出计算结果...")
            srs = layer_start.GetSpatialRef()
            # if jsonl_to_file(path_res, layer_start, srs, max_cost, out_path, out_type):
            if len(path_res) > 0:
                if combine_res_files(path_res, travelCosts, out_path, out_type):
                    # remove_temp_folder(out_temp_path)
                    pass
                else:
                    log.error("导出时发生错误, 请检查临时目录:{}".format(out_temp_path))

        end_time = time()
        log.info("计算完成, 结果保存在:{}, 共耗时{}秒".format(os.path.abspath(out_path), str(end_time - start_time)))
    except ValueError as msg:
        log.error(msg)
    except:
        log.error(traceback.format_exc())
        log.error("计算未完成, 请查看日志: {}.".format(os.path.abspath(log.logName)))
    finally:
        del ds_start
        del layer_start
        del wks


def combine_res_files(path_res, costs, out_path, out_type):
    try:
        # srs = in_layer.GetSpatialRef()
        datasetCreationOptions = []
        layerCreationOptions = []

        datasetCreationOptions, layerCreationOptions, out_type_f, out_format = creation_options(out_type)
        out_suffix = DataType_suffix[out_type_f]

        range_paths_dict = {}
        route_paths_dict = {}
        node_paths_dict = {}
        total_dict = {}
        for path in path_res:
            for cost in costs:
                cost_path = path[cost]
                if cost in range_paths_dict:
                    range_paths_dict[cost].extend([cost_path['range']])
                else:
                    range_paths_dict[cost] = [cost_path['range']]

                if cost in route_paths_dict:
                    route_paths_dict[cost].extend([cost_path['routes']])
                else:
                    route_paths_dict[cost] = [cost_path['routes']]

                if cost in node_paths_dict:
                    node_paths_dict[cost].extend([cost_path['nodes']])
                else:
                    node_paths_dict[cost] = [cost_path['nodes']]

                total_dict[cost] = cost_path['count']

        tqdm.set_lock(RLock())
        if len(costs) * 3 > cpu_count():
            process_num = cpu_count()
        else:
            process_num = len(costs) * 3
        pool = Pool(processes=process_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

        input_param = []
        ipos = 0
        for cost in costs:
            range_paths = range_paths_dict[cost]
            route_paths = route_paths_dict[cost]
            node_paths = node_paths_dict[cost]
            total_num = total_dict[cost]

            out_range_layer_name = check_layer_name("accessiblity_range_{}".format(str(cost)))
            out_routes_layer_name = check_layer_name("accessiblity_routes_{}".format(str(cost)))
            out_nodes_layer_name = check_layer_name("accessiblity_nodes_{}".format(str(cost)))

            range_dst_name = os.path.join(out_path, "{}.{}".format(out_range_layer_name, out_suffix))
            routes_dst_name = os.path.join(out_path, "{}.{}".format(out_routes_layer_name, out_suffix))
            nodes_dst_name = os.path.join(out_path, "{}.{}".format(out_nodes_layer_name, out_suffix))

            input_param.append((range_paths, range_dst_name, out_format, True, out_range_layer_name, ogr.wkbPolygon,
                                datasetCreationOptions, layerCreationOptions, total_num, ipos))
            ipos += 1
            input_param.append((route_paths, routes_dst_name, out_format, True, out_routes_layer_name, ogr.wkbMultiLineString,
                                datasetCreationOptions, layerCreationOptions, total_num, ipos))

            ipos += 1
            input_param.append((node_paths, nodes_dst_name, out_format, True, out_nodes_layer_name, ogr.wkbMultiPoint,
                                datasetCreationOptions, layerCreationOptions, total_num, ipos))

        pool.starmap(_ogrmerge, input_param)
        pool.close()
        pool.join()

        tqdm.write("\r", end="")

        return True
    except:
        log.error(traceback.format_exc())
        return False


def _ogrmerge(src_datasets, dst_filename, out_format, single_layer, layer_name_template, dst_geom_type, dsco, lco, total_num, pos):
    with lock:
        bar = mTqdm(range(total_num), desc="worker-{}".format(pos), position=pos, leave=False)
    ogrmerge(src_datasets, dst_filename, out_format, single_layer=single_layer, layer_name_template=layer_name_template,
             dst_geom_type=dst_geom_type, dsco=dsco, lco=lco, progress_callback=progress_callback, progress_arg=(bar, total_num))


def progress_callback(complete, message, cb_data):
    bar = cb_data[0]
    total = cb_data[1]
    if int(complete*100) < 100:
        with lock:
            # bar.update(total * int(complete ))
            bar.update(int(total * 0.01))
    else:
        with lock:
            bar.close()


def calculate(G, start_path, start_layer_name, out_path, start_points_df, panMap, costs, concave_hull_ratio,
              out_nodes, out_routes, cpu_core):
    # line_out_path, line_layer_name, route_out_path, route_layer_name, panMap, out_type_f = \
    #     create_output_file(out_path, in_layer, layer_name, travelCosts, out_type)
    out_line_ds = None
    out_line_layer = None
    out_route_ds = None
    out_route_layer = None
    out_type_f = None

    try:
        QueueManager.register('graphTransfer', GraphTransfer)
        # conn1, conn2 = Pipe()

        path_res = []
        with QueueManager() as manager:
            shared_obj = manager.graphTransfer(G, start_points_df)
            # value = shared_obj.shortest(start_node, travelCost)
            lst = []

            start_nodes = []
            for fid, start_node in zip(start_points_df['fid'], start_points_df['nodeID']):
                if start_node == -1:
                    continue
                start_nodes.append((start_node, fid))

            n = ceil(len(start_nodes) / cpu_core)  # 数据按照CPU数量分块

            tqdm.set_lock(RLock())
            pool = Pool(processes=cpu_core, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

            input_param = []
            ipos = 0
            for i in range(len(start_nodes)):
                lst.append(start_nodes[i])

                if (i + 1) % n == 0 or i == len(start_nodes) - 1:
                    input_param.append((shared_obj, lst, costs, out_path, panMap, start_path, start_layer_name,
                                        concave_hull_ratio, out_nodes, out_routes, ipos))
                    lst = []
                    ipos += 1

            returns = pool.starmap(accessible_geometry_from_point_worker, input_param)
            pool.close()
            pool.join()

            for res in returns:
                if len(res) > 0:
                    path_res.append(res)

        return path_res
    except:
        log.error(traceback.format_exc())
        return None
    finally:
        if out_line_ds is not None:
            out_line_ds.Destroy()
        del out_line_ds

        if out_route_ds is not None:
            out_route_ds.Destroy()
        del out_route_ds

        del out_line_layer
        del out_route_layer


#  多进程导出geojson
def accessible_geometry_from_point_worker(shared_custom, lst, costs, out_path, panMap, start_path, start_layer_name,
                                          concave_hull_ratio, out_nodes, out_routes, ipos=0):
    out_range_ds = None
    out_range_layer = None
    out_route_ds = None
    out_route_layer = None
    out_node_ds = None
    out_node_layer = None
    in_wks = None
    wks = None
    ds_start = None
    in_layer = None

    try:
        G, df = shared_custom.task()

        start_points_dict = df.to_dict()['geom']

        out_type_f = DataType.geojson

        gdal.SetConfigOption('ATTRIBUTES_SKIP', 'NO')
        gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '0')
        gdal.SetConfigOption('OLMD_FID64', 'YES')

        out_path_range = os.path.join(out_path, "range")
        out_path_nodes = os.path.join(out_path, "nodes")
        out_path_routes = os.path.join(out_path, "routes")

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(out_path_range):
            os.mkdir(out_path_range)
        if not os.path.exists(out_path_routes) and out_nodes:
            os.mkdir(out_path_routes)
        if not os.path.exists(out_path_nodes) and out_routes:
            os.mkdir(out_path_nodes)

        datasetCreationOptions = []
        layerCreationOptions = []

        in_wks = workspaceFactory()
        ds_start = in_wks.get_ds(start_path)
        in_layer, start_layer_name = in_wks.get_layer(ds_start, start_path, start_layer_name)
        srs = in_layer.GetSpatialRef()

        out_suffix = DataType_suffix[out_type_f]

        new_fields = []
        new_fields.append(ogr.FieldDefn('s_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('s_POS', ogr.OFTString))

        out_paths_dict = {}  # 返回分块文件路径

        for cost in costs:
            nodes_layer_name = check_layer_name("nodes_{}_{}".format(str(cost), str(ipos)))
            routes_layer_name = check_layer_name("routes_{}_{}".format(str(cost), str(ipos)))
            range_layer_name = check_layer_name("range_{}_{}".format(str(cost), str(ipos)))

            out_file_range = os.path.join(out_path_range, "{}.{}".format(range_layer_name, out_suffix))
            out_file_routes = os.path.join(out_path_routes, "{}.{}".format(routes_layer_name, out_suffix))
            out_file_nodes = os.path.join(out_path_nodes, "{}.{}".format(nodes_layer_name, out_suffix))

            wks = workspaceFactory().get_factory(out_type_f)
            if out_nodes:
                out_node_ds,  out_node_layer = wks.createFromExistingDataSource(in_layer, out_file_nodes, nodes_layer_name, srs,
                                                                                datasetCreationOptions, layerCreationOptions, new_fields,
                                                                                geom_type=ogr.wkbMultiPoint, panMap=panMap, open=True)
            out_range_ds,  out_range_layer = wks.createFromExistingDataSource(in_layer, out_file_range, range_layer_name, srs,
                                                                              datasetCreationOptions, layerCreationOptions, new_fields,
                                                                              geom_type=ogr.wkbPolygon, panMap=panMap, open=True)
            if out_routes:
                out_route_ds, out_route_layer = wks.createFromExistingDataSource(in_layer, out_file_routes, routes_layer_name, srs,
                                                                                 datasetCreationOptions, layerCreationOptions, new_fields,
                                                                                 geom_type=ogr.wkbMultiLineString, panMap=panMap, open=True)

            with lock:
                bar = mTqdm(lst, desc="worker-{}-{}".format(str(cost), ipos), position=ipos, leave=False)

            icount = 0
            for t in lst:
                start_node = t[0]
                start_fid = t[1]
                start_pt = start_points_dict[start_fid]
                in_fea = in_layer.GetFeature(start_fid)

                # if start_node == 585245:
                #     print("debug")

                try:
                    distances, routes = nx.single_source_dijkstra(G, start_node, weight='length',
                                                                  cutoff=cost)
                    # geoms = {}
                    farthest_nodes = {}
                    for end_node, route in routes.items():
                        if len(route) > 1:  # 只有一个是本身，不算入搜索结果
                            distance = distances[end_node]
                            farthest_nodes.update(get_farthest_nodes(G, route, distance, cost))

                            # addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value)
                            # addFeature(in_fea, start_fid, lines, out_route_layer, panMap, out_value)

                    out_value = {
                        's_ID': str(start_fid),
                        's_POS': str("{},{}".format(start_pt.x, start_pt.y))
                    }

                    if len(farthest_nodes) > 0:
                        out_nodes, out_routes, cover_area = out_geometry(farthest_nodes, concave_hull_ratio)
                        if not out_nodes.IsEmpty() and out_nodes:
                            addFeature(in_fea, start_fid, out_nodes, out_node_layer, panMap, out_value)
                        if not out_routes.IsEmpty() and out_routes:
                            addFeature(in_fea, start_fid, out_routes, out_route_layer, panMap, out_value)
                        if not cover_area.IsEmpty():
                            addFeature(in_fea, start_fid, cover_area, out_range_layer, panMap, out_value)

                        icount += 1

                except:
                    log.error("发生错误节点:{}, 位置{}, error:{}".format(str(start_node),
                                                                 "{},{}".format(start_pt.x, start_pt.y),
                                                                 traceback.format_exc()))
                    continue

                with lock:
                    bar.update()

            out_paths_dict[cost] = {
                'range': out_file_range,
                'routes': out_file_routes,
                'nodes': out_file_nodes,
                'count': icount
            }

        with lock:
            bar.close()

        return out_paths_dict
    except:
        log.error(traceback.format_exc())
        return None
    finally:
        with lock:
            if out_range_ds is not None:
                out_range_ds.Destroy()
            del out_range_ds

            if out_route_ds is not None:
                out_route_ds.Destroy()
            del out_route_ds

            if out_node_ds is not None:
                out_node_ds.Destroy()
            del out_node_ds

            del out_range_layer
            del out_route_layer
            del out_node_layer
            del in_wks
            del wks
            del ds_start
            del in_layer


# 整理需要输入的geometry
def out_geometry(farthest_nodes, concave_hull_ratio):
    out_nodes = ogr.Geometry(ogr.wkbMultiPoint)
    out_routes = ogr.Geometry(ogr.wkbMultiLineString)
    cover_area = ogr.Geometry(ogr.wkbPolygon)

    t_routes = []
    # extend_nodes = []
    for end_node, nodes in farthest_nodes.items():
        extend_nodes = nodes['extend_nodes']
        extend_routes = nodes['routes']
        for extend_node in extend_nodes:
            out_nodes.AddGeometry(CreateGeometryFromWkt(extend_node.wkt))
            # t_nodes.append(extend_node)
        for extend_route in extend_routes:
            out_routes.AddGeometryDirectly(CreateGeometryFromWkt(extend_route.wkt))
            t_routes.append(extend_route)
            # o = extend_route.coords
            # t_nodes.extend([Point(pt[0], pt[1]) for pt in extend_route.coords])

    if len(t_routes) >= 2:
        # geom = concave_hull(geometry.MultiPoint(extend_nodes), ratio=concave_hull_ratio)
        geom = concave_hull(geometry.MultiLineString(t_routes), ratio=concave_hull_ratio)
        if geom.geom_type == 'Polygon':
            cover_area = CreateGeometryFromWkt(geom.wkt)

    # cover_area = CreateGeometryFromWkt(concave_hull(geometry.MultiLineString(t_routes), ratio=concave_hull_ratio).wkt)
    # cover_area = out_nodes.ConcaveHull(concave_hull_ratio, False)
    # cover_area = out_nodes.ConvexHull()

    return out_nodes, out_routes, cover_area


def get_farthest_nodes(G, route, distance, cost):
    last_node = route[-1]
    farthest_nodes = {}  # 记录最远的节点编号,Key是最远点的nodeid, value是扩展点信息以及routes
    extend_nodes = []  # 存储扩展点的位置
    out_routes = []

    # lines = ogr.Geometry(ogr.wkbMultiLineString)
    # start_pt = ogr.Geometry(ogr.wkbPoint)
    # start_pt.AddPoint_2D(G.nodes[route[0]]['x'], G.nodes[route[0]]['y'])
    lines = []

    # 原始route的geometry
    for s, t in zip(route[:-1], route[1:]):
        eids = G[s][t]
        minlength = sys.float_info.max
        sel_key = -1
        for key, v in eids.items():
            if v['length'] <= minlength:
                sel_key = key
        if sel_key > -1:
            eid = G[s][t][sel_key]
            lines.append(set_precision(eid['geometry'], 0.001))
        # l = CreateGeometryFromWkt(eid['geometry'].wkt)
        # lines.AddGeometry(l)

    bhas_extend = False

    for successor_node in G.successors(last_node):
        t_eids = G[last_node][successor_node]

        for key, v in t_eids.items():
            t_eid = G[last_node][successor_node][key]
            if v['length'] + distance > cost:  # 到达最远边，需要插入within点
                # bhas_extend = True
                l = t_eid['geometry']

                interpolate_pt = l.interpolate(cost - distance)
                geomColl = split_line_by_point(interpolate_pt, l).geoms
                lines_clone = lines.copy()
                split_geom = set_precision(geomColl[0], 0.001)
                if not split_geom.is_empty:
                    lines_clone.append(split_geom)

                out_route_geom = geometry.MultiLineString(lines_clone)
                out_route = linemerge(out_route_geom)

                if out_route.geom_type == 'LineString':
                    out_routes.append(out_route)
                extend_nodes.append(interpolate_pt)
                bhas_extend = True
                # break

    if bhas_extend:
        farthest_nodes[last_node] = {
            'extend_nodes': extend_nodes,
            'routes': out_routes
        }

    return farthest_nodes


def export_to_file(geoms, fields, crs, source_attributes, out_path, out_type):
    pass


if __name__ == '__main__':
    freeze_support()
    # gdal.SetConfigOption('CPL_LOG', 'NUL')
    accessibility()
