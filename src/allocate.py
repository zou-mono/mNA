import ast
import copy
import csv
import click
import json
import os, sys
import traceback
import random
from math import ceil
from multiprocessing import Pool, freeze_support, RLock
from time import time, strftime

import pandas as pd
from networkx import Graph
from osgeo.ogr import Feature
from tqdm import tqdm
from multiprocessing import cpu_count

from Core.DataFactory import workspaceFactory
from Core.common import check_geom_type, check_field_type, get_centerPoints
from Core.core import DataType, QueueManager
from Core.graph import create_graph_from_file, Direction, makeGraph
from Core.log4p import Log, mTqdm
from nearest_facilities import nearest_facilities_from_point, \
    nearest_facilities_from_point_worker
from person import Person

log = Log(__name__)

@click.command()
@click.option("--network", '-n', type=str, required=True,
              help="输入网络数据, 必选.")
@click.option("--network-layer", type=str, required=False,
              help="输入网络数据的图层名, 可选. 如果是文件数据库(mdb, gdb, sqlite等)则必须提供,"
                   "如果是shapefile, geojson等文件格式则不需要提供.")
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
              help="输入起始设施数据的图层名, 可选. 如果是文件数据库(mdb, gdb, sqlite等)则必须提供, "
                   "如果是shapefile, geojson等文件格式则不需要提供.")
@click.option("--scapacity-field", type=str, required=True,
              help="输入起始设施数据的容量字段, 必选.")
@click.option("--tpath", '-t', type=str, required=True,
              help="输入目标设施数据, 必选.")
@click.option("--tpath-layer", type=str, required=False, default="",
              help="输入目标设施数据的图层名, 可选. 如果是文件数据库(mdb, gdb, sqlite等)则必须提供, "
                   "如果是shapefile, geojson等文件格式则不需要提供.")
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
@click.option("--out-type", type=int, required=False, default=0,
              help="输出文件格式, 默认值0. 0-ESRI Shapefile, 1-geojson, 3-fileGDB, 4-csv, 9-spatialite.")
@click.option("--out-path", "-o", type=str, required=False, default="res",
              help="输出目录名, 可选, 默认值为当前目录下的'res'.")
@click.option("--cpu-count", type=int, required=False, default=1,
              help="多进程数量, 可选, 默认为1, 即单进程. 小于0或者大于CPU最大核心数则进程数为最大核心数,否则为输入实际核心数.")
def allocate(network, network_layer, direction_field, forward_value, backward_value, both_value,
             default_direction, spath, spath_layer, scapacity_field, tpath, tpath_layer, tcapacity_field,
             tweight_field, cost, distance_tolerance, out_type, out_path, cpu_count):
    travelCost = list()
    for c in cost:
        if c in travelCost:
            log.warning("cost参数存在重复值{}, 重复值不参与计算.".format(c))
        else:
            travelCost.append(c)

    allocate_from_layer(network_path=network,
                        start_path=spath,
                        target_path=tpath,
                        start_capacity_field=scapacity_field,
                        target_capacity_field=tcapacity_field,
                        network_layer=network_layer,
                        start_layer_name=spath_layer,
                        target_layer_name=tpath_layer,
                        target_weight_field=tweight_field,
                        travelCost=travelCost,
                        direction_field=direction_field,
                        forwardValue=forward_value,
                        backwardValue=backward_value,
                        bothValue=both_value,
                        distance_tolerance=distance_tolerance,
                        defaultDirection=default_direction,
                        out_type=out_type,
                        out_path=out_path,
                        cpu_core=cpu_count)


def allocate_from_layer(
        network_path,
        start_path,
        target_path,
        start_capacity_field,
        target_capacity_field,
        network_layer="",
        start_layer_name="",
        target_layer_name="",
        start_weight_field="",
        target_weight_field="",
        out_path="res",
        travelCost=[sys.float_info.max],
        direction_field="",
        forwardValue="",
        backwardValue="",
        bothValue="",
        distance_tolerance=500,  # 从原始点到网络最近snap点的距离容差，如果超过说明该点无法到达网络，不进行计算
        defaultDirection=Direction.DirectionBoth,
        out_type=0,
        cpu_core=1):

    start_time = time()

    wks = workspaceFactory()
    ds_start = None
    ds_target = None
    layer_start = None
    layer_target = None

    if cpu_core <= 0 or cpu_core > cpu_count():
        cpu_core = cpu_count()

    try:
        log.info("读取网络数据, 路径为{}...".format(network_path))
        net = create_graph_from_file(network_path,
                                     network_layer,
                                     direction_field=direction_field,
                                     bothValue=bothValue,
                                     backwardValue=backwardValue,
                                     forwardValue=forwardValue)

        if net is None:
            log.error("网络数据存在问题, 无法创建图结构")
            return

        target_capacity_dict = {}  # 存储目标设施实时容量
        start_capacity_dict = {}  # 存储起始设施实时容量
        # nodes, edges = graph_to_gdfs(net)

        log.info("读取起始设施数据,路径为{}...".format(start_path))
        ds_start = wks.get_ds(start_path)
        layer_start, start_layer_name = wks.get_layer(ds_start, start_path, start_layer_name)
        # layer_start = init_check(layer_start, start_capacity_field, "起始")
        bflag, start_capacity, start_capacity_dict, start_capacity_idx = init_check(layer_start, start_capacity_field, "起始")
        if not bflag:
            return

        log.info("读取目标设施数据,路径为{}...".format(target_path))
        ds_target = wks.get_ds(target_path)
        layer_target, target_layer_name = wks.get_layer(ds_target, target_path, target_layer_name)
        bflag, target_capacity, target_capacity_dict, target_capacity_idx = init_check(layer_target, target_capacity_field, "目标")
        if not bflag:
            return

        target_weight_idx = layer_target.FindFieldIndex(target_weight_field, False)

        #  提取中心点
        log.info("计算设施位置坐标...")
        start_points_df = get_centerPoints(layer_start)
        target_points_df = get_centerPoints(layer_target)

        #  测试用点
        # start_points = [Point([519112.9421, 2505711.571])]
        # dic = {
        #     'fid': 599295,
        #     'geom': start_points[0]
        # }
        # start_points_df = DataFrame(dic, index=[0])

        start_points = start_points_df['geom'].to_list()
        target_points = target_points_df['geom'].to_list()

        # 这里需不需要对target_point进行空间检索？
        log.info("将目标设施附着到最近邻edge上，并且进行图重构...")
        all_points = start_points + target_points

        G, snapped_nodeIDs = makeGraph(net, all_points, distance_tolerance=distance_tolerance)
        # G, snapped_target_nodeIDs = makeGraph(net, target_points, distance_tolerance=distance_tolerance)

        # G, snapped_target_nodeIDs = makeGraph(net, all_points, distance_tolerance=distance_tolerance)
        target_points_df["nodeID"] = snapped_nodeIDs[len(start_points):]
        start_points_df["nodeID"] = snapped_nodeIDs[:len(start_points)]

        log.info("重构后的图共有{}条边，{}个节点".format(len(G.edges), len(G)))
        # start_points = [Point([521915.7194, 2509312.8204])]
        # start_points = [Point([520096, 2506194])]
        # start_points = [Point([519112.9421, 2505711.571])]
        df = target_points_df[target_points_df["nodeID"] != -1]

        # edge_geoms = list(nx.get_edge_attributes(G, "geometry").values())
        # rtree = STRtree(edge_geoms)
        # o_max = max(G.nodes)

        nearest_facilities = {}  # 存储起始设施可达的目标设施

        log.info("计算起始设施可达范围的目标设施...")

        max_cost = max(travelCost)

        if cpu_core == 1:
            for fid, start_node in mTqdm(zip(start_points_df['fid'], start_points_df['nodeID']), total=start_points_df.shape[0]):
                if start_node == -1:
                    continue

                nf = nearest_facilities_from_point(G, start_node, df,
                                                                  bRoutes=False,
                                                                  travelCost=max_cost)

                nearest_facilities[fid] = nf
        else:
            QueueManager.register('graphTransfer', GraphTransfer)
            # conn1, conn2 = Pipe()

            with QueueManager() as manager:
                shared_obj = manager.graphTransfer(G, df)
                # value = shared_obj.shortest(start_node, travelCost)
                lst = []
                # processes = []

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
                        input_param.append((None, shared_obj, lst, max_cost, False, True, ipos))
                        # processes.append(Process(target=nearest_facilities_from_point_worker,
                        #                          args=(None, shared_obj, lst, travelCost, False, True)))
                        lst = []
                        ipos += 1

                returns = pool.starmap(nearest_facilities_from_point_worker, input_param)
                for res in returns:
                    nearest_facilities.update(res)

                # for proc in processes:
                #     proc.start()
                # for proc in processes:
                #     nearest_facilities.update(conn2.recv())

                # conn1.close()
                # conn2.close()

        print("\n")
        log.info("加载起始设施的个体数据...")
        persons = load_persons(start_points_df, layer_start, start_capacity_idx, start_capacity)
        log.info("开始将起始设施的个体分配到可达范围的目标设施...")
        for cost in travelCost:
            start_dict = copy.deepcopy(start_capacity_dict)
            target_dict = copy.deepcopy(target_capacity_dict)

            start_res,  target_res = allocate_capacity(persons, nearest_facilities,
                                                       start_dict, target_dict,
                                                       layer_target, target_weight_idx, cost)

            with open('../res/start_capacity_{}.csv'.format(cost), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(start_res.keys())
                writer.writerows([start_res.values()])

            with open('../res/target_capacity_{}.csv'.format(cost), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(target_res.keys())
                writer.writerows([target_res.values()])

        end_time = time()
        log.info("计算完成,共耗时{}秒".format(str(end_time - start_time)))
        print("ok")
    except:
        log.error(traceback.format_exc())
        return
    finally:
        del ds_start
        del ds_target
        del layer_start
        del layer_target
        del wks


def load_persons(start_points_df, layer_start, start_capacity_idx, start_capacity):
    # 把设施的人打乱后重新排序，保证公平性
    start_order_lst = list(range(0, start_capacity))
    random.shuffle(start_order_lst)

    persons = []
    i = 0
    for fid, start_point, start_node in zip(start_points_df['fid'], start_points_df['geom'], start_points_df['nodeID']):
        capacity = layer_start.GetFeature(fid).GetField(start_capacity_idx)
        for p in range(capacity):
            person = Person()
            person.ID = start_order_lst[i]
            person.facility = fid  # 所属设施的ID号
            # person.location = start_point
            # person.facility_order =
            persons.append(person)

            i += 1

    persons = sorted(persons, key=lambda item: item.ID)

    return persons


def allocate_capacity(persons, nearest_facilities, start_capacity_dict, target_capacity_dict,
                      layer_target, target_weight_idx, max_cost):

    for i in mTqdm(range(len(persons))):
        person = persons[i]
        # facility_order = person.facility_order
        facility_id = person.facility

        weights = []
        cand_ids = []  # 候选设施id

        if facility_id not in nearest_facilities:
            continue

        nearest_distances = nearest_facilities[facility_id]

        for target_id, cost in nearest_distances.items():
            feature_target: Feature = layer_target.GetFeature(target_id)
            target_capacity = target_capacity_dict[target_id]

            if cost <= max_cost:
                # 只有当target设施还有余量时才参与选择，否则跳过
                if target_capacity > 0:
                    if feature_target.GetField(target_weight_idx) == "地级教育部门":
                        w = 100
                    elif feature_target.GetField(target_weight_idx) == "事业单位":
                        w = 50
                    else:
                        if cost == 0:
                            cost = 0.1
                        w = 1 / cost

                    weights.append(w)  # 选择概率为花费的倒数， 花费越小则概率越高
                    cand_ids.append(target_id)

        bChoice = False
        choice_id = -1
        if len(cand_ids) > 1:
            bChoice = True
            choice_id = random.choices(cand_ids, weights=weights)[0]
        elif len(cand_ids) == 1:
            bChoice = True
            choice_id = cand_ids[0]

        if bChoice:
            target_capacity_dict[choice_id] = target_capacity_dict[choice_id] - 1
            start_capacity_dict[facility_id] = start_capacity_dict[facility_id] - 1

    return start_capacity_dict, target_capacity_dict


def export_to_file(layer_name: str, capacity_dict: dict, capacity_idx,
                   out_path="res", out_type=DataType.shapefile.value):
    pass


def init_check(layer, capacity_field, suffix=""):
    layer_name = layer.GetName()

    if not check_geom_type(layer):
        log.error("设施数据{}不满足几何类型要求,只允许Polygon,multiPolygon,Point,multiPoint类型".format(suffix, layer_name))
        return None

    capacity_idx = layer.FindFieldIndex(capacity_field, False)
    if capacity_idx == -1:
        log.error("{}设施数据'{}'缺少容量字段{},无法进行后续计算".format(suffix, layer_name, capacity_field))
        return False

    if not check_field_type(layer.GetLayerDefn().GetFieldDefn(capacity_idx)):
        log.error("设施数据'{}'的字段{}不满足类型要求,只允许int, double类型".format(suffix, layer_name, capacity_field))
        return False

    query_str = '''"{}" > 0'''.format(capacity_field)
    layer.SetAttributeFilter(query_str)

    capacity = 0
    capacity_dict = {}
    layer.ResetReading()
    for feature in layer:
        v = feature.GetField(capacity_field)
        capacity_dict[feature.GetFID()] = v
        capacity = capacity + v
    # exec_str = '''SELECT SUM("{}") FROM "{}" WHERE "{}" > 0'''.format(
    #     capacity_field, layer_name, capacity_field)
    # exec_res = ds.ExecuteSQL(exec_str)
    #
    # if exec_res is not None:
    #     capacity = exec_res.GetNextFeature().GetField(0)
    # else:
    #     log.error("无法统计{}设施总容量.".format(suffix))
    #     return False

    log.info("{}设施总容量为{}".format(suffix, capacity))
    # ds.ReleaseResultSet(exec_res)

    return True, capacity, capacity_dict, capacity_idx


class GraphTransfer:
    def __init__(self, G, df):
        self.G = G
        self.df = df

    def task(self):
        return self.G, self.df


if __name__ == '__main__':
    freeze_support()
    allocate()
    # allocate_from_layer(
    #     r"D:\空间模拟\mNA\Data\sz_road_cgcs2000_test.shp",
    #     r"D:\空间模拟\mNA\Data\building_test.shp",
    #     r"D:\空间模拟\mNA\Data\2022年现状幼儿园.shp",
    #     travelCost=(3000),
    #     out_type=0,
    #     start_capacity_field="ZL3_5",
    #     target_capacity_field="学生数",
    #     target_weight_field="学校举",
    #     direction_field="oneway",
    #     cpu_core=-1)

    # allocate_from_layer(
    #     r"D:\空间模拟\PublicSupplyDemand\Data\sz_road_cgcs2000.shp",
    #     r"D:\空间模拟\PublicSupplyDemand\Data\楼栋.shp",
    #     r"D:\空间模拟\PublicSupplyDemand\Data\2022年现状幼儿园.shp",
    #     travelCost=1000,
    #     out_type=0,
    #     start_capacity_field="ZL3_5",
    #     target_capacity_field="学生数",
    #     target_weight_field="学校举",
    #     direction_field="oneway",
    #     cpu_core=-1)

