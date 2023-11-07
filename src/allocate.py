import pickle
import csv
import click
import os, sys
import traceback
import random
from math import ceil
from multiprocessing import Pool, freeze_support, RLock, Lock
from time import time, strftime

from osgeo import ogr, gdal

from tqdm import tqdm
from multiprocessing import cpu_count

from Core import filegdbapi
from Core.DataFactory import workspaceFactory, get_suffix
from Core.common import check_geom_type, check_field_type, get_centerPoints
from Core.core import DataType, QueueManager, check_layer_name
from Core.fgdb import FieldType
from Core.filegdbapi import FieldDef
from Core.graph import create_graph_from_file, Direction, makeGraph, export_network_to_graph, import_graph_to_network
from Core.log4p import Log, mTqdm
from nearest_facilities import nearest_facilities_from_point, \
    nearest_facilities_from_point_worker
from person import Person

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
def allocate(network, network_layer, direction_field, forward_value, backward_value, both_value,
             default_direction, spath, spath_layer, scapacity_field, tpath, tpath_layer, tcapacity_field,
             tweight_field, cost, distance_tolerance, out_type, out_graph_type, out_path, cpu_count):
    """设施容量分配算法"""
    travelCosts = list()
    for c in cost:
        if c in travelCosts:
            log.warning("cost参数存在重复值{}, 重复值不参与计算.".format(c))
        else:
            travelCosts.append(c)

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

    out_path = os.path.abspath(os.path.join(out_path, "allocate_res_{}".format(strftime('%Y-%m-%d-%H-%M-%S'))))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    allocate_from_layer(network_path=network,
                        start_path=spath,
                        target_path=tpath,
                        start_capacity_field=scapacity_field,
                        target_capacity_field=tcapacity_field,
                        network_layer=network_layer,
                        start_layer_name=spath_layer,
                        target_layer_name=tpath_layer,
                        target_weight_field=tweight_field,
                        travelCosts=travelCosts,
                        direction_field=direction_field,
                        forwardValue=forward_value,
                        backwardValue=backward_value,
                        bothValue=both_value,
                        distance_tolerance=distance_tolerance,
                        defaultDirection=default_direction,
                        out_type=out_type,
                        out_graph_type=out_graph_type,
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
        travelCosts=[sys.float_info.max],
        direction_field="",
        forwardValue="",
        backwardValue="",
        bothValue="",
        distance_tolerance=500,  # 从原始点到网络最近snap点的距离容差，如果超过说明该点无法到达网络，不进行计算
        defaultDirection=Direction.DirectionBoth,
        out_type=0,
        out_graph_type='gpickle',
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
        # layer_start = init_check(layer_start, start_capacity_field, "起始")
        bflag, start_capacity, start_capacity_dict, start_capacity_idx, __ = \
            init_check(layer_start, start_capacity_field, "起始")
        if not bflag:
            return

        log.info("读取目标设施数据,路径为{}...".format(target_path))
        ds_target = wks.get_ds(target_path)
        layer_target, target_layer_name = wks.get_layer(ds_target, target_path, target_layer_name)
        target_weight_idx = layer_target.FindFieldIndex(target_weight_field, False)
        bflag, target_capacity, target_capacity_dict, target_capacity_idx, target_weight_dict = \
            init_check(layer_target, target_capacity_field, "目标", target_weight_idx)
        if not bflag:
            return

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
        target_points_df["nodeID"] = snapped_nodeIDs[len(start_points):]
        start_points_df["nodeID"] = snapped_nodeIDs[:len(start_points)]

        log.info("重构后的图共有{}条边，{}个节点".format(len(G.edges), len(G)))
        df = target_points_df[target_points_df["nodeID"] != -1]

        nearest_facilities = {}  # 存储起始设施可达的目标设施

        log.info("计算起始设施可达范围的目标设施...")

        max_cost = max(travelCosts)

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
                pool.close()
                pool.join()

                for res in returns:
                    nearest_facilities.update(res)

                # for proc in processes:
                #     proc.start()
                # for proc in processes:
                #     nearest_facilities.update(conn2.recv())

                # conn1.close()
                # conn2.close()
        # print("\r")
        tqdm.write("\r", end="")
        log.info("加载起始设施的个体数据...")
        persons = load_persons(start_points_df, layer_start, start_capacity_idx, start_capacity)
        log.info("开始将起始设施的个体分配到可达范围的目标设施...")

        # if len(travelCost) == 1:
        #     cost = travelCost[0]
        #     start_dict = copy.deepcopy(start_capacity_dict)
        #     target_dict = copy.deepcopy(target_capacity_dict)
        #
        #     start_res,  target_res = allocate_capacity(persons, nearest_facilities,
        #                                                start_capacity_dict, target_capacity_dict,
        #                                                target_weight_dict, cost)
        #
        #     log.info("开始导出计算结果...")
        #     out_path = os.path.join(out_path, "capacity_{}".format(strftime('%Y-%m-%d-%H-%M-%S')))
        #     export_to_file(out_path, start_res, start_dict, cost, layer_start,
        #                    layer_name="start_capacity", out_type=out_type)
        #     export_to_file(out_path, target_res, target_dict, cost, layer_target,
        #                    layer_name="target_capacity", out_type=out_type)
        #
        # else:
        QueueManager.register('capacityTransfer', CapacityTransfer)
        bflag1 = bflag2 = False
        with QueueManager() as manager:
            shared_obj = manager.capacityTransfer(persons)

            process_num = len(travelCosts)
            tqdm.set_lock(RLock())
            pool = Pool(processes=process_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

            input_param = []
            ipos = 0
            for cost in travelCosts:
                input_param.append((shared_obj, nearest_facilities, start_capacity_dict,
                                    target_capacity_dict, target_weight_dict, cost, ipos))
                ipos += 1

            returns = pool.starmap(allocate_capacity_worker, input_param)
            pool.close()
            pool.join()

            if len(returns) < 1:
                raise ValueError("没有返回计算结果.")

            tqdm.write("\r", end="")
            log.info("开始导出计算结果...")

            for idx, res in enumerate(returns):
                if idx == 0:
                    start_res = res[0]
                    target_res = res[1]
                    for k, v in start_res.items():
                        start_res[k] = [v]
                    for k, v in target_res.items():
                        target_res[k] = [v]
                    continue

                # cost = travelCosts[idx]

                start_return = res[0]
                target_return = res[1]

                for k, v in start_return.items():
                    start_res[k].extend([v])
                for k, v in target_return.items():
                    target_res[k].extend([v])

            log.debug("正在导出起始设施分配结果...")
            bflag1 = export_to_file(start_path, out_path, start_res, start_capacity_dict, travelCosts, layer_start,
                           layer_name="start_capacity", out_type=out_type)
            log.debug("正在导出目标设施分配结果...")
            bflag2 = export_to_file(target_path, out_path, target_res, target_capacity_dict, travelCosts, layer_target,
                           layer_name="target_capacity", out_type=out_type)

        end_time = time()

        if bflag1 and bflag2:
            log.info("计算完成, 结果保存在:{}, 共耗时{}秒".format(os.path.abspath(out_path), str(end_time - start_time)))
        else:
            log.error("计算未完成, 请查看日志: {}.".format(os.path.abspath(log.logName)))
    except ValueError as msg:
        log.error(msg)
    except:
        log.error(traceback.format_exc())
        log.error("计算未完成, 请查看日志: {}.".format(os.path.abspath(log.logName)))
        return
    finally:
        del ds_start
        del ds_target
        del layer_start
        del layer_target
        del wks


def export_csv(out_path, layer_name, res, capacity_dict, costs):
    try:
        for idx, cost in enumerate(costs):
            out_file = os.path.join(out_path, "{}_{}.csv".format(layer_name, cost))
            with open(out_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['fid', 'used', 'remain'])
                for key, value in res.items():
                    capacity = capacity_dict[key]
                    remain = value[idx]
                    writer.writerow([key, capacity - remain, remain])

        return True
    except:
        return False


def export_to_file(in_path, out_path, res, capacity_dict, costs, in_layer=None, layer_name="", out_type=DataType.csv.value):
    out_ds = None
    out_layer = None
    out_type_f = None

    try:
        layer_name = check_layer_name(layer_name)

        if out_type == DataType.csv.value:
            if export_csv(out_path, layer_name, res, capacity_dict, costs):
                return True
            else:
                return False

        if in_layer is not None:
            in_layer.SetAttributeFilter("")
            in_layer.ResetReading()

        out_format = ""
        srs = in_layer.GetSpatialRef()
        inlayername = in_layer.GetName()

        datasetCreationOptions = []
        layerCreationOptions = []
        limit = -1
        if out_type == DataType.shapefile.value:
            out_format = "ESRI Shapefile"
            out_type_f = DataType.shapefile
            layerCreationOptions = ['ENCODING=UTF-8', "2GB_LIMIT=NO"]
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            out_path = os.path.join(out_path, "{}.shp".format(layer_name))
        elif out_type == DataType.geojson.value:
            out_type_f = DataType.geojson
            out_format = "GeoJSON"
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            out_path = os.path.join(out_path, "{}.geojson".format(layer_name))
        elif out_type == DataType.fileGDB.value:
            out_format = "FileGDB"
            out_type_f = DataType.fileGDB
            out_path = os.path.join(out_path, "{}.gdb".format(layer_name))
            layerCreationOptions = ["FID=FID"]
            gdal.SetConfigOption('FGDB_BULK_LOAD', 'YES')
        elif out_type == DataType.sqlite.value:
            out_format = "SQLite"
            out_type_f = DataType.sqlite
            datasetCreationOptions = ['SPATIALITE=YES']
            layerCreationOptions = ['SPATIAL_INDEX=NO']
            gdal.SetConfigOption('OGR_SQLITE_SYNCHRONOUS', 'OFF')
            out_path = os.path.join(out_path, "{}.sqlite".format(layer_name))

        translateOptions = gdal.VectorTranslateOptions(format=out_format, srcSRS=srs, dstSRS=srs, layers=[inlayername],
                                                       accessMode="overwrite", layerName=layer_name,
                                                       datasetCreationOptions=datasetCreationOptions,
                                                       geometryType="PROMOTE_TO_MULTI",
                                                       limit=limit,
                                                       layerCreationOptions=layerCreationOptions)

        if not gdal.VectorTranslate(out_path, in_path, options=translateOptions):
            return False

        if out_type_f == DataType.fileGDB:
            out_type_f = DataType.FGDBAPI

        # out_Driver = ogr.GetDriverByName(out_format)
        wks = workspaceFactory().get_factory(out_type_f)
        out_ds = wks.openFromFile(out_path, 1)
        out_layer = out_ds.GetLayerByName(layer_name)

        remain_field_idx =[]
        for idx, cost in enumerate(costs):
            used_field_name = "used_{}".format(idx)
            used_idx = out_layer.FindFieldIndex(used_field_name, False)
            if used_idx == -1:
                if out_type == DataType.fileGDB.value:
                    new_field = FieldDef()
                    new_field.SetName(used_field_name)
                    new_field.SetType(FieldType.fieldTypeInteger.value)
                else:
                    new_field = ogr.FieldDefn(used_field_name, ogr.OFTInteger64)

                out_layer.CreateField(new_field)
                del new_field
                # used_idx = out_layer.FindFieldIndex(used_field_name, False)

            remain_field_name = "remain_{}".format(idx)
            remain_idx = out_layer.FindFieldIndex(remain_field_name, False)
            if remain_idx == -1:
                if out_type == DataType.fileGDB.value:
                    new_field = FieldDef()
                    new_field.SetName(remain_field_name)
                    new_field.SetType(FieldType.fieldTypeInteger.value)
                else:
                    new_field = ogr.FieldDefn(remain_field_name, ogr.OFTInteger64)
                out_layer.CreateField(new_field)
                remain_idx = out_layer.FindFieldIndex(remain_field_name, False)
                remain_field_idx.append(remain_idx)
                del new_field

        if out_type == DataType.fileGDB.value:
            out_layer.LoadOnlyMode(True)
            out_layer.SetWriteLock()

        total_feature = out_layer.GetFeatureCount()

        out_ds.StartTransaction(force=True)
        # for feature in mTqdm(out_layer, total=total_feature):
        with mTqdm(out_layer, total=total_feature) as bar:
            feature = out_layer.GetNextFeature()
            while feature:
                fid = feature.GetFID()

                for idx, cost in enumerate(costs):
                    if fid in capacity_dict:
                        capacity = capacity_dict[fid]
                    else:
                        capacity = 0

                    if fid in res:
                        remain = res[fid][idx]
                    else:
                        remain = 0

                    feature.SetField(remain_field_idx[idx] - 1, capacity - remain)
                    feature.SetField(remain_field_idx[idx], remain)

                out_layer.SetFeature(feature)
                feature = out_layer.GetNextFeature()
                bar.update()

        out_ds.CommitTransaction()
        out_ds.FlushCache()
        return True
    except:
        log.error(traceback.format_exc())
        return False
    finally:
        if out_type_f == DataType.FGDBAPI:
            if out_layer is not None:
                out_layer.LoadOnlyMode(False)
                out_layer.FreeWriteLock()
            out_ds.CloseTable(out_layer)
            filegdbapi.CloseGeodatabase(out_ds)
        else:
            if out_ds is not None:
                out_ds.Destroy()
        del out_ds
        del out_layer


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
                      target_weight_dict, max_cost):
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
            # feature_target: Feature = layer_target.GetFeature(target_id)
            if cost <= max_cost:
                target_capacity = target_capacity_dict[target_id]
                # 只有当target设施还有余量时才参与选择，否则跳过
                if target_capacity > 0:
                    feature_weight = target_weight_dict[target_id]
                    if feature_weight == "地级教育部门":
                        w = 100
                    elif feature_weight == "事业单位":
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


def allocate_capacity_worker(shared_custom, nearest_facilities, start_capacity_dict, target_capacity_dict,
                             target_weight_dict, max_cost, ipos):
    persons = shared_custom.task()
    # print(len(persons))
    with lock:
        bar = mTqdm(range(len(persons)), desc="worker-{}".format(ipos), position=ipos, leave=False)

    for i in range(len(persons)):
        person = persons[i]
        # facility_order = person.facility_order
        facility_id = person.facility

        weights = []
        cand_ids = []  # 候选设施id

        if facility_id not in nearest_facilities:
            continue

        nearest_distances = nearest_facilities[facility_id]

        for target_id, cost in nearest_distances.items():
            # feature_target: Feature = layer_target.GetFeature(target_id)
            if cost <= max_cost:
                target_capacity = target_capacity_dict[target_id]
                feature_weight = target_weight_dict[target_id]

            # 只有当target设施还有余量时才参与选择，否则跳过
                if target_capacity > 0:
                    if feature_weight == "地级教育部门":
                        w = 100
                    elif feature_weight == "事业单位":
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

        with lock:
            bar.update()

    with lock:
        bar.close()

    return start_capacity_dict, target_capacity_dict


def init_check(layer, capacity_field, suffix="", target_weight_idx=-1):
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
    weight_dict = {}
    layer.ResetReading()
    for feature in layer:
        fid = feature.GetFID()
        v = feature.GetField(capacity_field)
        capacity_dict[fid] = v
        capacity = capacity + v

        if target_weight_idx > 0:
            weight = feature.GetField(target_weight_idx)
            weight_dict[fid] = weight

    log.info("{}设施总容量为{}".format(suffix, capacity))

    return True, capacity, capacity_dict, capacity_idx, weight_dict


class CapacityTransfer:
    def __init__(self, persons):
        self.persons = persons
        # self.nearest_facilities = nearest_facilities
        # self.start_dict = start_dict
        # self.target_dict = target_dict
        # self.target_weight_dict = target_weight_dict
        # self.layer_target = layer_target

    def task(self):
        return self.persons
        # return self.persons, self.nearest_facilities, self.start_dict, self.target_dict, self.target_weight_dict


class GraphTransfer:
    def __init__(self, G, df):
        self.G = G
        self.df = df

    def task(self):
        return self.G, self.df


if __name__ == '__main__':
    freeze_support()
    gdal.SetConfigOption('CPL_LOG', 'NUL')
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

