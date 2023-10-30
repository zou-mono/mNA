import csv
import json
import os, sys
import traceback
import random
from math import ceil
from multiprocessing import Pool, current_process, Manager, Pipe, Process, freeze_support, RLock
from time import time, strftime

import pandas as pd
from networkx import Graph
from osgeo.ogr import Feature
from pandas import DataFrame
from pygments.lexers import math
from tqdm import tqdm
from osgeo import ogr
from shapely import STRtree, Point
from shapely.ops import nearest_points
from multiprocessing import cpu_count

from Core.DataFactory import workspaceFactory
from Core.common import check_geom_type, check_field_type, get_centerPoints
from Core.core import DataType, QueueManager
from Core.graph import create_graph_from_file, Direction, makeGraph
from Core.log4p import Log, mTqdm
from Core.utils_graph import graph_to_gdfs, _is_endpoint_of_edge
from nearest_facilities import nearest_facilities_from_point, \
    nearest_facilities_from_point_worker
from person import Person
import networkx as nx

log = Log(__name__)

seed = 1
m = 1
N = 1340//m
# outer_G = nx.gnm_random_graph(N, int(1.7*N), seed)
outer_G: Graph = None

test_num = 0

# outer_G: Graph = None

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
        travelCost=0.0,
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

        if cpu_core == 1:
            for fid, start_node in mTqdm(zip(start_points_df['fid'], start_points_df['nodeID']), total=start_points_df.shape[0]):
                if start_node == -1:
                    continue

                nf = nearest_facilities_from_point(G, start_node, df,
                                                                  bRoutes=False,
                                                                  travelCost=travelCost)

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
                        input_param.append((None, shared_obj, lst, travelCost, False, True, ipos))
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

        log.info("开始将起始设施的个体分配到可达范围的目标设施...")

        for i in mTqdm(range(len(persons))):
            person = persons[i]
            # facility_order = person.facility_order
            facility_id = person.facility

            idx = 0
            weights = []
            cand_ids = []  # 候选设施id

            if facility_id not in nearest_facilities:
                continue

            nearest_distances = nearest_facilities[facility_id]

            for target_id, cost in nearest_distances.items():
                feature_target: Feature = layer_target.GetFeature(target_id)
                target_capacity = target_capacity_dict[target_id]

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

                idx += 1

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

        with open('../res/start_capacity.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(start_capacity_dict.keys())
            writer.writerows([start_capacity_dict.values()])

        with open('../res/target_capacity.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(target_capacity_dict.keys())
            writer.writerows([target_capacity_dict.values()])

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


# def func(msg, lst):
#     G = lst[0]
#     # return current_process().name + '-' + str(G[2])
#     print(current_process().name + '-' + str(len(G[0].edges())))
#     # return current_process().name + '-' + str(len(G.edges()))
#     # return current_process().name + '-' + str(test_num)


def allocate():
    pass


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
    allocate_from_layer(
        r"D:\空间模拟\mNA\Data\sz_road_cgcs2000_test.shp",
        r"D:\空间模拟\mNA\Data\building_test.shp",
        r"D:\空间模拟\mNA\Data\2022年现状幼儿园.shp",
        travelCost=3000,
        out_type=0,
        start_capacity_field="ZL3_5",
        target_capacity_field="学生数",
        target_weight_field="学校举",
        direction_field="oneway",
        cpu_core=-1)

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

