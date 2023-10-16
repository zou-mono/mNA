import os, sys
import traceback
import random
from time import time, strftime

import pandas as pd
from PyQt5.QtCore import QVariant
from osgeo import ogr
from qgis._analysis import QgsVectorLayerDirector, QgsNetworkDistanceStrategy, QgsGraphBuilder, QgsGraphAnalyzer
from qgis._core import QgsVectorLayer, QgsField, QgsPointXY, QgsWkbTypes, QgsSpatialIndex, QgsRectangle, \
    QgsVectorFileWriter, QgsFeature, QgsProject, QgsGeometry, QgsLineString, QgsProviderRegistry, \
    QgsAggregateCalculator, QgsFeatureRequest, QgsFields, QgsApplication
from shapely import STRtree, Point
from shapely.ops import nearest_points

from Core.DataFactory import workspaceFactory
from Core.common import check_geom_type, check_field_type, get_centerPoints
from Core.core import DataType
from Core.graph import create_graph_from_file, Direction, makeGraph
from Core.log4p import Log, mTqdm
from Core.utils_graph import graph_to_gdfs, _is_endpoint_of_edge
from nearest_facilities import nearest_facilities_from_point
from person import Person
import networkx as nx

log = Log(__name__)


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
        out_type=0):

    start_time = time()

    wks = workspaceFactory()
    ds_start = None
    ds_target = None
    layer_start = None
    layer_target = None

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

        # nodes, edges = graph_to_gdfs(net)

        log.info("读取起始设施数据,路径为{}...".format(start_path))
        ds_start = wks.get_ds(start_path)
        layer_start, start_layer_name = wks.get_layer(ds_start, start_path, start_layer_name)
        # layer_start = init_check(layer_start, start_capacity_field, "起始")
        if not init_check(layer_start, start_capacity_field, "起始"):
            return

        log.info("读取目标设施数据,路径为{}...".format(target_path))
        ds_target = wks.get_ds(target_path)
        layer_target, target_layer_name = wks.get_layer(ds_target, target_path, target_layer_name)
        if not init_check(layer_target, target_capacity_field, "目标"):
            return

        #  提取中心点
        log.info("计算设施位置坐标...")
        start_points_df = get_centerPoints(layer_start)
        target_points_df = get_centerPoints(layer_target)

        # start_points = [Point([520096, 2506194])]
        # start_points = [Point([521915.7194, 2509312.8204])]
        # start_points = start_points_df['geom'].to_list()
        target_points = target_points_df['geom'].to_list()

        # 这里需不需要对target_point进行空间检索？
        log.info("将目标设施附着到最近邻edge上，并且进行图重构...")
        G, snapped_target_nodeIDs = makeGraph(net, target_points, distance_tolerance=distance_tolerance)

        target_points_df["nodeID"] = snapped_target_nodeIDs
        log.info("重构后的图共有{}条边，{}个节点".format(len(G.edges), len(G)))

        # start_points = [Point([521915.7194, 2509312.8204])]
        # start_points = [Point([520096, 2506194])]
        start_points = [Point([519112.9421, 2505711.571])]
        target_df = target_points_df[target_points_df["nodeID"] != -1]

        edge_geoms = list(nx.get_edge_attributes(G, "geometry").values())
        rtree = STRtree(edge_geoms)
        o_max = max(G.nodes)

        res = {}
        for row in mTqdm(start_points_df.itertuples()):
            start_point = row.geom
            nearest_distances = nearest_facilities_from_point(G, start_point, target_df,
                                                              bRoutes=False,
                                                              rtree=rtree,
                                                              o_max=o_max,
                                                              travelCost=travelCost,
                                                              distance_tolerance=distance_tolerance)
            res[row.fid] = nearest_distances

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


def allocate(mNetwork,
             mDirector,
             persons,
             start_points,
             target_points,
             start_features,
             target_features,
             start_capacity_idx,
             target_capacity_idx,
             target_weight_idx,
             distance_tolerance,
             travelCost):

    log.info("构建图结构...")

    mBuilder = QgsGraphBuilder(mNetwork.crs())

    # 将start_points和target_points组成新的节点
    snappedPoints = mDirector.makeGraph(mBuilder, start_points + target_points)
    graph = mBuilder.takeGraph()

    log.info("图共有{}个vertex，{}个edge...".format(graph.vertexCount(), graph.edgeCount()))

    target_capacity_dict = {}  # 存储目标设施实时容量
    start_capacity_dict = {}  # 存储起始设施实时容量
    index = QgsSpatialIndex()
    for data in target_features.values():
        pfea: QgsFeature = data['feature']
        index.addFeature(pfea)
        target_capacity_dict[pfea.id()] = pfea.attributes()[target_capacity_idx]

    for data in start_features.values():
        pfea: QgsFeature = data['feature']
        start_capacity_dict[pfea.id()] = pfea.attributes()[start_capacity_idx]

    result = {}

    log.info("开始将起始设施的个体分配到可达范围的目标设施...")

    for i in mTqdm(range(0, len(persons))):
        person = persons[i]
        facility_order = person.facility_order
        facility_id = person.facility
        snapped_point_start = snappedPoints[facility_order]
        start_point = start_points[facility_order]
        # start_point = person.location

        res = None
        if facility_id in result:
             res = result[facility_id]
        else:
            # origin_pt = start_point

            #  这里才对start点进行过滤
            if snapped_point_start.distance(start_point) > distance_tolerance:
                continue

            # 对target点进行过滤，把距离太远的点过滤掉
            rect = QgsRectangle(snapped_point_start.x() - travelCost, snapped_point_start.y() - travelCost,
                                snapped_point_start.x() + travelCost, snapped_point_start.y() + travelCost)
            candidate_targets_ids = index.intersects(rect)

            candidate_targets = []
            for iid in candidate_targets_ids:
                candidate_targets.append({
                    'target_id': iid,
                    'target_order': target_features[iid]['id'],
                    'point': target_features[iid]['feature'].geometry().asPoint()
                })

            [routes, res_costs, target_ids, target_pts] = nearest_facilities_from_point(
                graph, snapped_point_start, start_points, target_points,
                snappedPoints, candidate_targets, travelCost)

            if len(routes) >= 1:
                # geom_routes = QgsGeometry.fromMultiPolylineXY(routes)
                res = {
                    # "org_pt": origin_pt,
                    # "routes": geom_routes,
                    'target_ids': target_ids,
                    'costs': res_costs,
                    # 'target_pts': target_pts
                }
                result[facility_id] = res

        if res is not None:
            idx = 0
            weights = []
            cand_ids = []
            for target_id in res['target_ids']:
                feature_target: QgsFeature = target_features[target_id]['feature']
                cost = res['costs'][idx]

                target_capacity = target_capacity_dict[target_id]

                # 只有当target设施还有余量时才参与选择，否则跳过
                if target_capacity > 0:
                    if feature_target.attributes()[target_weight_idx] == "地级教育部门":
                        w = 100
                    elif feature_target.attributes()[target_weight_idx] == "事业单位":
                        w = 50
                    else:
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

    return start_capacity_dict, target_capacity_dict


def export_to_file(layer_name: str, capacity_dict: dict, fields: QgsFields, capacity_idx,
                   layer_origin: QgsVectorLayer,
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
    layer.ResetReading()
    for feature in layer:
        capacity += feature.GetField(capacity_field)
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

    return True


if __name__ == '__main__':
    # allocate_from_layer(
    #     r"D:\空间模拟\mNA\Data\sz_road_cgcs2000_test.shp",
    #     r"D:\空间模拟\mNA\Data\building_test.shp",
    #     r"D:\空间模拟\mNA\Data\2022年现状幼儿园.shp",
    #     travelCost=1000,
    #     out_type=0,
    #     start_capacity_field="ZL3_5",
    #     target_capacity_field="学生数",
    #     target_weight_field="学校举",
    #     direction_field="oneway")

    allocate_from_layer(
        r"D:\空间模拟\PublicSupplyDemand\Data\sz_road_cgcs2000.shp",
        r"D:\空间模拟\PublicSupplyDemand\Data\楼栋.shp",
        r"D:\空间模拟\PublicSupplyDemand\Data\2022年现状幼儿园.shp",
        travelCost=1000,
        out_type=0,
        start_capacity_field="ZL3_5",
        target_capacity_field="学生数",
        target_weight_field="学校举",
        direction_field="oneway")

    # sys.exit(app.exec_())
