import asyncio
import csv
import io
import os, sys
import traceback
from math import ceil
from multiprocessing import current_process, Lock, Pool, RLock, cpu_count, freeze_support
from time import time, strftime

import jsonlines

import shutil
import click
import networkx as nx
from osgeo import gdal, ogr
from osgeo.ogr import CreateGeometryFromWkt
from pandas import DataFrame
from shapely import Point, wkt
from tqdm import tqdm

from Core import filegdbapi
from Core.DataFactory import workspaceFactory, get_suffix, addFeature
from Core.check import init_check
from Core.common import resource_path, set_main_path, get_centerPoints, stdout_moveto, stdout_clear
from Core.core import DataType, QueueManager, check_layer_name, DataType_suffix, DataType_dict
from Core.log4p import Log, mTqdm
from colorama import Fore

from Core.graph import makeGraph, Direction, GraphTransfer, import_graph_to_network, create_graph_from_file, \
    export_network_to_graph
from Core.ogrmerge import ogrmerge

log = Log(__name__)
lock = Lock()

concurrence_num = 1

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
# @click.option("--scapacity-field", type=str, required=True,
#               help="输入起始设施数据的容量字段, 必选.")
@click.option("--tpath", '-t', type=str, required=True,
              help="输入目标设施数据, 必选.")
@click.option("--tpath-layer", type=str, required=False, default="",
              help="输入目标设施数据的图层名, 可选. 如果是文件数据库(gdb, sqlite)则必须提供, "
                   "如果是文件(shapefile, geojson)则不需要提供.")
# @click.option("--tcapacity-field", type=str, required=True,
#               help="输入目标设施数据的容量字段, 必选.")
# @click.option("--sweigth-field", type=str, required=True,
#               help="输入起始设施数据的权重字段, 必选. 用于分配过程中提升设施的选取概率.")
# @click.option("--tweight-field", type=str, required=False, default="",
#               help="输入目标设施数据的权重字段, 可选. 用于分配过程中提升设施的选取概率;"
#                    "如果不提供,则所有目标设施选取权重根据距离的倒数来定义.")
@click.option("--cost", "-c", type=float, required=False, multiple=True, default=[sys.float_info.max],
              help="路径搜索范围, 超过范围的设施不计入搜索结果, 可选. 缺省值会将所有可达设施都加入结果,同时导致搜索速度极大下降, "
                   "建议根据实际情况输入合适的范围."
                   "允许输入多个值，例如'-c 1000 -c 1500'表示同时计算1000和1500两个范围的可达设施.")
@click.option("--distance-tolerance", type=float, required=False, default=500,
              help="定义目标设施到网络最近点的距离容差，如果超过说明该设施偏离网络过远，不参与计算, 可选, 默认值为500.")
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
@click.option("--cpu-count", type=int, required=False, default=1,
              help="多进程数量, 可选, 默认为1, 即单进程. 小于0或者大于CPU最大核心数则进程数为最大核心数,否则为输入实际核心数.")
def nearest(network, network_layer, direction_field, forward_value, backward_value, both_value,
            default_direction, spath, spath_layer, tpath, tpath_layer, cost, distance_tolerance,
            out_type, out_graph_type, out_path, cpu_count):
    """网络距离可达设施搜索算法."""
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

    out_path = os.path.abspath(os.path.join(out_path, "nearest_res_{}".format(strftime('%Y-%m-%d-%H-%M-%S'))))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    nearest_facilities_from_layer(network_path=network,
                                  network_layer=network_layer,
                                  start_path=spath,
                                  start_layer_name=spath_layer,
                                  target_path=tpath,
                                  target_layer_name=tpath_layer,
                                  travelCosts=travelCosts,
                                  direction_field=direction_field,
                                  forwardValue=forward_value,
                                  backwardValue=backward_value,
                                  bothValue=both_value,
                                  defaultDirection=default_direction,
                                  distance_tolerance=distance_tolerance,
                                  out_type=out_type,
                                  out_graph_type=out_graph_type,
                                  out_path=out_path,
                                  cpu_core=cpu_count)


def nearest_facilities_from_layer(
        network_path,
        network_layer,
        start_path,
        start_layer_name,
        target_path,
        target_layer_name,
        travelCosts=[sys.float_info.max],
        direction_field="",
        forwardValue="",
        backwardValue="",
        bothValue="",
        distance_tolerance=500,  # 从原始点到网络最近snap点的距离容差，如果超过说明该点无法到达网络，不进行计算
        defaultDirection=Direction.DirectionBoth,
        out_type=0,
        out_graph_type='gpickle',
        out_path="res",
        cpu_core=1):

    start_time = time()

    log.info("读取网络数据...")
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
        if not init_check(layer_start, None, "起始"):
            return

        log.info("读取目标设施数据,路径为{}...".format(target_path))
        ds_target = wks.get_ds(target_path)
        layer_target, target_layer_name = wks.get_layer(ds_target, target_path, target_layer_name)
        if not init_check(layer_start, None, "起始"):
            return

        #  提取中心点
        log.info("计算设施位置坐标...")
        start_points_df = get_centerPoints(layer_start)
        target_points_df = get_centerPoints(layer_target)

        # # 测试用点
        # start_points = [Point([519112.9421, 2505711.571])]
        # dic = {
        #     'fid': 0,
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

        max_cost = max(travelCosts)
        log.info("计算起始设施可达范围的目标设施...")

        if out_type == DataType.csv.value:
            nearest_facilities = calculate(G, df, start_points_df, max_cost, cpu_core)
            log.info("开始导出计算结果...")
            export_to_file(G, out_path, start_points_df, target_points_df, nearest_facilities,
                           travelCosts, layer_start, "nearest", out_type, cpu_core)
        else:
            out_temp_path = os.path.abspath(os.path.join(out_path, "temp"))
            path_res = calculate2(G, start_path, start_layer_name, out_temp_path, layer_start, df, start_points_df, target_points_df, max_cost, cpu_core)

            tqdm.write("\r", end="")

            log.info("开始导出计算结果...")
            srs = layer_start.GetSpatialRef()
            # if jsonl_to_file(path_res, layer_start, srs, max_cost, out_path, out_type):
            if combine_res_files(path_res, srs, max_cost, out_path, out_type):
                # remove_temp_folder(out_temp_path)
                pass
            else:
                log.error("导出时发生错误, 请检查临时目录:{}".format(out_temp_path))

        # export_to_file2(nearest_facilities, out_path, layer_start, "nearest", travelCosts, out_type)
        # export_to_file(G, out_path, start_points_df, target_points_df, nearest_facilities,
        #                travelCosts, layer_start, "nearest", out_type, cpu_core)

        end_time = time()
        log.info("计算完成, 结果保存在:{}, 共耗时{}秒".format(os.path.abspath(out_path), str(end_time - start_time)))
    except ValueError as msg:
        log.error(msg)
    except:
        log.error(traceback.format_exc())
        log.error("计算未完成, 请查看日志: {}.".format(os.path.abspath(log.logName)))
    finally:
        del ds_start
        del ds_target
        del layer_start
        del layer_target
        del wks


# def progress_callback(complete, message, cb_data):
#     '''Emit progress report in numbers for 10% intervals and dots for 3%'''
#     block = u'\u2588\u2588'
#     ncol = 80
#
#     if int(complete*100) % 3 == 0:
#         n = int(ncol * (complete*100 / 3) / 100)
#         with lock:
#             stdout_moveto(cb_data)
#             sys.stdout.write(f'\r {Fore.BLUE}{complete*100:.0f}% {block * n}')
#             sys.stdout.write('')
#             stdout_moveto(-cb_data)
#     if int(complete*100) == 100:
#         # print('\r', end='', flush=True)
#         # sys.stdout.write(f'\r {COLOUR_RESET}')
#         # sys.stdout.write('')
#         with lock:
#             stdout_clear(cb_data)


def remove_temp_folder(in_path):
    try:
        if os.path.exists(in_path):
            # os.remove(in_path)
            shutil.rmtree(in_path, True)
    except:
        log.warning("临时文件夹{}被占用, 无法自动删除, 请手动删除!".format(in_path))


def jsonl_to_file(path_res, in_layer, srs, cost, out_path, out_type):
    layer_name = in_layer.GetName()
    line_layer_name = "lines_{}_{}".format(layer_name, str(cost))
    route_layer_name = "routes_{}_{}".format(layer_name, str(cost))
    line_layer_name = check_layer_name(line_layer_name)
    route_layer_name = check_layer_name(route_layer_name)

    try:
        datasetCreationOptions = []
        layerCreationOptions = []
        if out_type == DataType.shapefile.value:
            out_format = "ESRI Shapefile"
            out_type_f = DataType.shapefile
            layerCreationOptions = ['ENCODING=UTF-8', "2GB_LIMIT=NO"]
        elif out_type == DataType.geojson.value:
            out_type_f = DataType.geojson
            out_format = "GeoJSON"
            gdal.SetConfigOption('ATTRIBUTES_SKIP', 'NO')
            gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '0')
        elif out_type == DataType.fileGDB.value:
            out_format = "FileGDB"
            out_type_f = DataType.fileGDB
            layerCreationOptions = ["FID=FID"]
            gdal.SetConfigOption('FGDB_BULK_LOAD', 'YES')
        elif out_type == DataType.sqlite.value:
            out_format = "SQLite"
            out_type_f = DataType.sqlite
            datasetCreationOptions = ['SPATIALITE=YES']
            layerCreationOptions = ['SPATIAL_INDEX=NO']
            gdal.SetConfigOption('OGR_SQLITE_SYNCHRONOUS', 'OFF')
        else:
            out_type_f = DataType.geojson

        out_suffix = DataType_suffix[out_type_f]

        line_out_path = os.path.join(out_path, "{}.{}".format(line_layer_name, out_suffix))
        route_out_path = os.path.join(out_path, "{}.{}".format(route_layer_name, out_suffix))

        new_fields = []
        new_fields.append(ogr.FieldDefn('s_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('t_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('cost', ogr.OFTReal))

        wks = workspaceFactory().get_factory(out_type_f)
        wks.createFromExistingDataSource(in_layer, line_out_path, line_layer_name, srs,
                                         datasetCreationOptions, layerCreationOptions, new_fields,
                                         geom_type=ogr.wkbLineString, open=False)
        wks.createFromExistingDataSource(in_layer, route_out_path, route_layer_name, srs,
                                         datasetCreationOptions, layerCreationOptions, new_fields,
                                         geom_type=ogr.wkbMultiLineString, open=False)

        total_num = 0
        line_paths = []
        route_paths = []
        for path in path_res:
            total_num += path[2]
            line_paths.append(path[0])
            route_paths.append(path[1])

        tqdm.set_lock(RLock())
        pool = Pool(processes=2, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

        input_param = []
        input_param.append((line_paths, line_out_path, line_layer_name, total_num, out_type, 0))
        input_param.append((route_paths, route_out_path, route_layer_name, total_num, out_type, 1))

        pool.starmap(jsonl_to_file_worker, input_param)
        pool.close()
        pool.join()

        tqdm.write("\r", end="")

        return True
    except:
        log.debug(traceback.format_exc())
        return False


def jsonl_to_file_worker(paths, out_path, out_layer_name, total_num, out_type, ipos):
    wks = workspaceFactory()
    out_ds = wks.get_ds(out_path, access=1)
    out_layer, out_layer_name = wks.get_layer(out_ds, out_path, out_layer_name)

    if out_type == DataType.fileGDB.value:
        gdal.SetConfigOption('FGDB_BULK_LOAD', 'YES')
    elif out_type == DataType.sqlite.value:
        gdal.SetConfigOption('OGR_SQLITE_SYNCHRONOUS', 'OFF')

    wks = workspaceFactory().get_factory(DataType.geojson)

    with lock:
        bar = mTqdm(range(total_num), desc="worker-{}".format(ipos), leave=False, position=ipos)

    poDstFDefn = out_layer.GetLayerDefn()
    nDstFieldCount = poDstFDefn.GetFieldCount()
    panMap = [-1] * nDstFieldCount

    for iField in range(poDstFDefn.GetFieldCount()):
        poDstFieldDefn = poDstFDefn.GetFieldDefn(iField)
        iDstField = poDstFDefn.GetFieldIndex(poDstFieldDefn.GetNameRef())
        if iDstField >= 0:
            panMap[iField] = iDstField

    out_ds.StartTransaction(force=True)
    icount = 1
    for path in paths:
        with jsonlines.open(path, "r") as reader:
            for l in reader:
                # fid = in_fea.GetFID()
                ds = wks.openFromFile(l)
                lyr = ds.GetLayer()
                in_fea = lyr.GetFeature(0)
                # out_layer.CreateFeature(in_fea)
                # out_fea = in_fea.Clone()
                in_fea.SetFID(icount)
                # out_fea = in_fea.Clone()
                # out_fea.SetFID(icount)
                # out_Feature = ogr.Feature(in_fea.GetLayerDefn())
                # out_fea.SetFID(fid)
                # out_Feature.SetFromWithMap(in_fea, 1, panMap)
                # out_Feature.SetGeometry(in_fea.GetGeometryRef())
                out_layer.CreateFeature(in_fea)
                # del out_fea

                icount += 1
                with lock:
                    bar.update()

    out_ds.CommitTransaction()
    # out_ds.SyncToDisk()

    with lock:
        bar.close()

    if out_ds is not None:
        out_ds.Destroy()
    del out_ds
    del out_layer


def combine_res_files(path_res, srs, cost, out_path, out_type):
    try:
        # srs = in_layer.GetSpatialRef()
        datasetCreationOptions = []
        layerCreationOptions = []

        out_line_layer_name = check_layer_name("nearest_line_{}".format(str(cost)))
        out_route_layer_name = check_layer_name("nearest_route_{}".format(str(cost)))

        if out_type == DataType.shapefile.value:
            out_type_f = DataType.shapefile
            layerCreationOptions = ['ENCODING=UTF-8', "2GB_LIMIT=NO"]
            line_dst_name = os.path.join(out_path, "{}.shp".format(out_line_layer_name))
            route_dst_name = os.path.join(out_path, "{}.shp".format(out_route_layer_name))
        elif out_type == DataType.geojson.value:
            out_type_f = DataType.geojson
            gdal.SetConfigOption('ATTRIBUTES_SKIP', 'NO')
            gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '0')
            line_dst_name = os.path.join(out_path, "{}.geojson".format(out_line_layer_name))
            route_dst_name = os.path.join(out_path, "{}.geojson".format(out_route_layer_name))
        elif out_type == DataType.fileGDB.value:
            out_type_f = DataType.fileGDB
            layerCreationOptions = ["FID=FID"]
            gdal.SetConfigOption('FGDB_BULK_LOAD', 'YES')
            line_dst_name = os.path.join(out_path, "{}.gdb".format(out_line_layer_name))
            route_dst_name = os.path.join(out_path, "{}.gdb".format(out_route_layer_name))
        elif out_type == DataType.sqlite.value:
            out_type_f = DataType.sqlite
            datasetCreationOptions = ['SPATIALITE=YES']
            layerCreationOptions = ['SPATIAL_INDEX=NO']
            gdal.SetConfigOption('OGR_SQLITE_SYNCHRONOUS', 'OFF')
            line_dst_name = os.path.join(out_path, "{}.sqlite".format(out_line_layer_name))
            route_dst_name = os.path.join(out_path, "{}.sqlite".format(out_route_layer_name))
        else:
            out_type_f = DataType.geojson

        out_format = DataType_dict[out_type_f]

        line_paths = []
        route_paths = []
        total_num = 0
        for path in path_res:
            line_paths.append(path[0])
            route_paths.append(path[1])
            total_num += path[2]

        tqdm.set_lock(RLock())
        pool = Pool(processes=2, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

        input_param = []
        input_param.append((line_paths, line_dst_name, out_format, True, out_line_layer_name, ogr.wkbLineString,
                            datasetCreationOptions, layerCreationOptions, total_num, 0))
        input_param.append((route_paths, route_dst_name, out_format, True, out_route_layer_name, ogr.wkbMultiLineString,
                            datasetCreationOptions, layerCreationOptions, total_num, 1))

        pool.starmap(_ogrmerge, input_param)
        pool.close()
        pool.join()

        tqdm.write("\r", end="")

        # log.debug("正在导出line图层...")
        # ogrmerge(line_src, line_dst_name, out_format, single_layer=True, layer_name_template=out_line_layer_name,
        #          dst_geom_type=ogr.wkbLineString, progress_callback=progress_callback, progress_arg=".")
        # log.debug("正在导出route图层...")
        # ogrmerge(route_src, route_dst_name, out_format, single_layer=True, layer_name_template=out_route_layer_name,
        #          dst_geom_type=ogr.wkbMultiLineString, progress_callback=progress_callback, progress_arg=".")

        return True
    except:
        log.error(traceback.format_exc())
        return False


def _ogrmerge(line_src, line_dst_name, out_format, single_layer, layer_name_template, dst_geom_type, dsco, lco, total_num, pos):
    with lock:
        bar = mTqdm(range(total_num), desc="worker-{}".format(pos), position=pos, leave=False)
    ogrmerge(line_src, line_dst_name, out_format, single_layer=single_layer, layer_name_template=layer_name_template,
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


def calculate(G, df, start_points_df, max_cost, cpu_core):
    nearest_facilities = {}  # 存储起始设施可达的目标设施

    if cpu_core == 1:
        for fid, start_node in mTqdm(zip(start_points_df['fid'], start_points_df['nodeID']), total=start_points_df.shape[0]):
            if start_node == -1:
                continue

            nf = nearest_facilities_from_point(G, start_node, df, travelCost=max_cost)

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
                    input_param.append((shared_obj, lst, max_cost, True, True, ipos))
                    # processes.append(Process(target=nearest_facilities_from_point_worker,
                    #                          args=(None, shared_obj, lst, travelCost, False, True)))
                    lst = []
                    ipos += 1

            returns = pool.starmap(nearest_facilities_from_point_worker, input_param)
            pool.close()
            pool.join()

            for res in returns:
                nearest_facilities.update(res)

    tqdm.write("\r", end="")

    return nearest_facilities


def calculate2(G, start_path, start_layer_name, out_path, in_layer, df, start_points_df, target_points_df, max_cost, cpu_core):
    # line_out_path, line_layer_name, route_out_path, route_layer_name, panMap, out_type_f = \
    #     create_output_file(out_path, in_layer, layer_name, travelCosts, out_type)
    out_line_ds = None
    out_line_layer = None
    out_route_ds = None
    out_route_layer = None
    out_type_f = None

    try:
        poSrcFDefn = in_layer.GetLayerDefn()
        nSrcFieldCount = poSrcFDefn.GetFieldCount()
        panMap = [-1] * nSrcFieldCount

        for iField in range(poSrcFDefn.GetFieldCount()):
            poSrcFieldDefn = poSrcFDefn.GetFieldDefn(iField)
            iDstField = poSrcFDefn.GetFieldIndex(poSrcFieldDefn.GetNameRef())
            if iDstField >= 0:
                panMap[iField] = iDstField

        start_points_dict = start_points_df.to_dict()['geom']
        target_points_dict = target_points_df.to_dict()['geom']

        QueueManager.register('routesTransfer', routesTransfer)
        # conn1, conn2 = Pipe()

        path_res = []
        with QueueManager() as manager:
            shared_obj = manager.routesTransfer(G, df, start_points_dict, target_points_dict)
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
                    input_param.append((shared_obj, lst, max_cost, out_path, panMap, start_path, start_layer_name, ipos))
                    lst = []
                    ipos += 1

            returns = pool.starmap(nearest_geometry_from_point_worker, input_param)
            pool.close()
            pool.join()

            for res in returns:
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


def create_output_file(out_path, in_layer, layer_name, costs, out_type):
    out_line_ds = None
    out_line_layer = None
    out_route_ds = None
    out_route_layer = None
    out_type_f = None

    try:
        layer_name = check_layer_name(layer_name)
        # if out_type == DataType.csv.value:
        #     if export_csv(out_path, layer_name, res, costs):
        #         return True
        #     else:
        #         return False

        if in_layer is not None:
            in_layer.SetAttributeFilter("")
            in_layer.ResetReading()

        out_format = ""
        srs = in_layer.GetSpatialRef()
        # inlayername = in_layer.GetName()

        datasetCreationOptions = []
        layerCreationOptions = []

        if out_type == DataType.shapefile.value:
            out_type_f = DataType.shapefile
            layerCreationOptions = ['ENCODING=UTF-8', "2GB_LIMIT=NO"]
        elif out_type == DataType.geojson.value:
            out_type_f = DataType.geojson
            gdal.SetConfigOption('ATTRIBUTES_SKIP', 'NO')
            gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '0')
        elif out_type == DataType.fileGDB.value:
            out_type_f = DataType.fileGDB
            layerCreationOptions = ["FID=FID"]
            gdal.SetConfigOption('FGDB_BULK_LOAD', 'YES')
        elif out_type == DataType.sqlite.value:
            out_type_f = DataType.sqlite
            datasetCreationOptions = ['SPATIALITE=YES']
            layerCreationOptions = ['SPATIAL_INDEX=NO']
            gdal.SetConfigOption('OGR_SQLITE_SYNCHRONOUS', 'OFF')

        out_suffix = DataType_suffix[out_type_f]

        new_fields = []
        new_fields.append(ogr.FieldDefn('s_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('t_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('cost', ogr.OFTReal))

        poSrcFDefn = in_layer.GetLayerDefn()
        nSrcFieldCount = poSrcFDefn.GetFieldCount()
        panMap = [-1] * nSrcFieldCount

        for iField in range(poSrcFDefn.GetFieldCount()):
            poSrcFieldDefn = poSrcFDefn.GetFieldDefn(iField)
            iDstField = poSrcFDefn.GetFieldIndex(poSrcFieldDefn.GetNameRef())
            if iDstField >= 0:
                panMap[iField] = iDstField

        cost = max(costs)  # 只导出最大距离，其他的可以从最大距离中筛选，不重复导出

        # for cost in costs:

        line_layer_name = "lines_{}_{}".format(layer_name, str(cost))
        route_layer_name = "routes_{}_{}".format(layer_name, str(cost))
        line_layer_name = check_layer_name(line_layer_name)
        route_layer_name = check_layer_name(route_layer_name)

        line_out_path = os.path.join(out_path, "{}.{}".format(line_layer_name, out_suffix))
        route_out_path = os.path.join(out_path, "{}.{}".format(route_layer_name, out_suffix))

        wks = workspaceFactory().get_factory(out_type_f)
        out_line_ds,  out_line_layer = wks.createFromExistingDataSource(in_layer, line_out_path, line_layer_name, srs,
                                                                        datasetCreationOptions, layerCreationOptions, new_fields,
                                                                        geom_type=ogr.wkbLineString, open=True)
        out_route_ds, out_route_layer = wks.createFromExistingDataSource(in_layer, route_out_path, route_layer_name, srs,
                                                                         datasetCreationOptions, layerCreationOptions, new_fields,
                                                                         geom_type=ogr.wkbMultiLineString, open=True)

        return line_out_path, line_layer_name, route_out_path, route_layer_name, panMap, out_type_f
    except:
        log.error(traceback.format_exc())
        return False
    finally:
        if out_line_ds is not None:
            out_line_ds.Destroy()
        out_line_ds = None

        if out_route_ds is not None:
            out_route_ds.Destroy()
        out_route_ds = None

        del out_line_layer
        del out_route_layer


def export_to_file2(res, in_layer, line_out_path, line_layer_name,  route_out_path, route_layer_name, panMap, out_type_f):
    try:
        wks = workspaceFactory().get_factory(out_type_f)
        out_line_ds = wks.openFromFile(line_out_path, 1)
        out_line_layer = out_line_ds.GetLayerByName(line_layer_name)
        out_route_ds = wks.openFromFile(route_out_path, 1)
        out_route_layer = out_route_ds.GetLayerByName(route_layer_name)

        out_line_ds.StartTransaction(force=True)
        out_route_ds.StartTransaction(force=True)

        for start_fid, geoms_dict in mTqdm(res.items(), total=len(res)):
            in_fea = in_layer.GetFeature(start_fid)

            for target_fid, target_dict in geoms_dict.items():
                out_value = {
                    's_ID': str(start_fid),
                    't_ID': str(target_fid),
                    'cost': target_dict[2]
                }

                line = target_dict[0]
                lines = target_dict[1]

                addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value)
                addFeature(in_fea, start_fid, lines, out_route_layer, panMap, out_value)

        out_line_ds.CommitTransaction()
        out_line_ds.SyncToDisk()
        out_route_ds.CommitTransaction()
        out_route_ds.SyncToDisk()

        return True
    except:
        log.error(traceback.format_exc())
        return False
    finally:
        if out_line_ds is not None:
            out_line_ds.Destroy()
        out_line_ds = None

        if out_route_ds is not None:
            out_route_ds.Destroy()
        out_route_ds = None

        del out_line_layer
        del out_route_layer


def export_to_file(G, out_path, start_points_df, target_points_df,
                   res, costs, in_layer=None, layer_name="", out_type=0, cpu_core=-1):
    out_line_ds = None
    out_line_layer = None
    out_route_ds = None
    out_route_layer = None
    out_type_f = None

    try:
        layer_name = check_layer_name(layer_name)

        if out_type == DataType.csv.value:
            if export_csv(out_path, layer_name, res, costs):
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
        elif out_type == DataType.geojson.value:
            out_type_f = DataType.geojson
            out_format = "GeoJSON"
            gdal.SetConfigOption('ATTRIBUTES_SKIP', 'NO')
            gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '0')
        elif out_type == DataType.fileGDB.value:
            out_format = "FileGDB"
            out_type_f = DataType.fileGDB
            layerCreationOptions = ["FID=FID"]
            gdal.SetConfigOption('FGDB_BULK_LOAD', 'YES')
        elif out_type == DataType.sqlite.value:
            out_format = "SQLite"
            out_type_f = DataType.sqlite
            datasetCreationOptions = ['SPATIALITE=YES']
            layerCreationOptions = ['SPATIAL_INDEX=NO']
            gdal.SetConfigOption('OGR_SQLITE_SYNCHRONOUS', 'OFF')

        out_suffix = DataType_suffix[out_type_f]

        new_fields = []
        new_fields.append(ogr.FieldDefn('s_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('t_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('cost', ogr.OFTReal))

        poSrcFDefn = in_layer.GetLayerDefn()
        nSrcFieldCount = poSrcFDefn.GetFieldCount()
        panMap = [-1] * nSrcFieldCount

        for iField in range(poSrcFDefn.GetFieldCount()):
            poSrcFieldDefn = poSrcFDefn.GetFieldDefn(iField)
            iDstField = poSrcFDefn.GetFieldIndex(poSrcFieldDefn.GetNameRef())
            if iDstField >= 0:
                panMap[iField] = iDstField

        cost = max(costs)  # 只导出最大距离，其他的可以从最大距离中筛选，不重复导出

        # for cost in costs:

        line_layer_name = "lines_{}_{}".format(layer_name, str(cost))
        route_layer_name = "routes_{}_{}".format(layer_name, str(cost))
        line_layer_name = check_layer_name(line_layer_name)
        route_layer_name = check_layer_name(route_layer_name)

        line_out_path = os.path.join(out_path, "{}.{}".format(line_layer_name, out_suffix))
        route_out_path = os.path.join(out_path, "{}.{}".format(route_layer_name, out_suffix))

        wks = workspaceFactory().get_factory(out_type_f)
        out_line_ds,  out_line_layer = wks.createFromExistingDataSource(in_layer, line_out_path, line_layer_name, srs,
                                                                        datasetCreationOptions, layerCreationOptions, new_fields,
                                                                        geom_type=ogr.wkbLineString, open=True)
        out_route_ds, out_route_layer = wks.createFromExistingDataSource(in_layer, route_out_path, route_layer_name, srs,
                                                                         datasetCreationOptions, layerCreationOptions, new_fields,
                                                                         geom_type=ogr.wkbMultiLineString, open=True)

        # if out_type_f == DataType.fileGDB:
        #     out_type_f = DataType.FGDBAPI

        # wks = workspaceFactory().get_factory(out_type_f)
        # out_line_ds = wks.openFromFile(line_out_path, 1)
        # out_line_layer = out_line_ds.GetLayerByName(line_layer_name)
        # out_route_ds = wks.openFromFile(route_out_path, 1)
        # out_route_layer = out_route_ds.GetLayerByName(route_layer_name)

        icount = 0
        # total_features = in_layer.GetFeatureCount()

        start_points_dict = start_points_df.to_dict()['geom']
        target_points_dict = target_points_df.to_dict()['geom']

        # if out_type_f == DataType.FGDBAPI:
        #     out_line_layer.LoadOnlyMode(True)
        #     out_line_layer.SetWriteLock()
        #     out_route_layer.LoadOnlyMode(True)
        #     out_route_layer.SetWriteLock()

        # for in_fea in mTqdm(in_layer, total=total_features):
        # with mTqdm(in_layer, total=total_features) as bar:

        QueueManager.register('routesTransfer', routesTransfer)
        with QueueManager() as manager:
            pool = Pool(processes=cpu_core, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
            shared_obj = manager.routesTransfer(G, res, start_points_dict, target_points_dict)

            n = ceil(len(res) / cpu_core)  # 数据按照CPU数量分块

            tqdm.set_lock(RLock())

            input_param = []
            ipos = 0
            out_geoms = {}
            lst = []
            for start_fid, value in res.items():
                # start_loc = start_points_df.loc[start_points_df['fid']==start_fid]
                # in_fea = in_layer.GetFeature(start_fid)

                # start_pt = start_points_dict[start_fid]
                # target_distances = value['distance']
                # target_routes = value['routes']
                lst.append(start_fid)

                if (icount + 1) % n == 0 or icount == len(res) - 1:
                    # G, start_pt, target_pt, target_fid, target_routes
                    input_param.append((shared_obj, lst, ipos))
                    # processes.append(Process(target=nearest_facilities_from_point_worker,
                    #                          args=(None, shared_obj, lst, travelCost, False, True)))
                    lst = []
                    ipos += 1

                # for target_fid, dis in target_distances.items():
                    # target_pt = target_points_dict[target_fid]
                    # output_data(G, start_fid, target_fid, start_pt, target_pt, target_routes, dis,
                    #             in_fea, out_line_layer, out_route_layer, panMap)
                icount += 1

            returns = pool.starmap(output_geometry_worker, input_param)

            pool.close()
            pool.join()

            for _ in returns:
                out_geoms.update(_)

        tqdm.write("\r", end="")

        out_line_ds.StartTransaction(force=True)
        out_route_ds.StartTransaction(force=True)

        for start_fid, geoms_dict in mTqdm(out_geoms.items(), total=len(out_geoms)):
            in_fea = in_layer.GetFeature(start_fid)

            for target_fid, geom in geoms_dict.items():
                dis = res[start_fid]['distance'][target_fid]

                out_value = {
                    's_ID': str(start_fid),
                    't_ID': str(target_fid),
                    'cost': dis
                }

                line = geom[0]
                lines = geom[1]

                addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value)
                addFeature(in_fea, start_fid, lines, out_route_layer, panMap, out_value)

        out_line_ds.CommitTransaction()
        out_line_ds.SyncToDisk()
        out_route_ds.CommitTransaction()
        out_route_ds.SyncToDisk()

        return True
    except:
        log.error(traceback.format_exc())
        return False
    finally:
        if out_line_ds is not None:
            out_line_ds.Destroy()
        out_line_ds = None

        if out_route_ds is not None:
            out_route_ds.Destroy()
        out_route_ds = None

        del out_line_layer
        del out_route_layer


def output_data(G, start_fid, target_fid, start_pt, target_pt, target_routes, dis, in_fea,
                out_line_layer, out_route_layer, panMap):

    line, lines = output_geometry(G, start_pt, target_pt, target_fid, target_routes)

    out_value = {
        's_ID': str(start_fid),
        't_ID': str(target_fid),
        'cost': dis
    }

    addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value)
    addFeature(in_fea, start_fid, lines, out_route_layer, panMap, out_value)


def output_geometry_worker(shared_custom, lst, ipos):
    G, res, start_points_dict, target_points_dict = shared_custom.task()

    with lock:
        bar = mTqdm(lst, desc="worker-{}".format(ipos), position=ipos, leave=False)

    geoms_dict = {}
    for start_fid in lst:
        geoms = {}
        start_pt = start_points_dict[start_fid]
        match_df = res[start_fid]
        target_routes = match_df['routes']

        for target_fid, dis in target_routes.items():
            target_pt = target_points_dict[target_fid]
            route = target_routes[target_fid]
            line, lines = output_geometry(G, start_pt, target_pt, route)

            geoms[target_fid] = (line, lines)

        geoms_dict[start_fid] = geoms

        with lock:
            bar.update()

    with lock:
        bar.close()

    return geoms_dict


def output_geometry(G, start_pt, target_pt, route):
    # route = target_routes[target_fid]
    # target_pt = target_points_df.loc[target_fid]['geom']
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint_2D(start_pt.x, start_pt.y)
    line.AddPoint_2D(target_pt.x, target_pt.y)

    lines = ogr.Geometry(ogr.wkbMultiLineString)
    # path_graph = nx.path_graph(route)
    for s, t in zip(route[:-1], route[1:]):
        # for ea in path_graph.edges():
        eids = G[s][t]
        minlength = sys.float_info.max
        for key, v in eids.items():
            if v['length'] < minlength:
                sel_key = key
        eid = G[s][t][sel_key]
        l = CreateGeometryFromWkt(eid['geometry'].wkt)
        lines.AddGeometryDirectly(l)

    return line, lines


def export_csv(out_path, layer_name, res, costs):
    csvfile_nearest = None
    csvfile_line = None
    try:
        for idx, cost in enumerate(costs):
            out_nearest_file = os.path.join(out_path, "nearest_{}_{}.csv".format(layer_name, cost))
            out_line_file = os.path.join(out_path, "line_{}_{}.csv".format(layer_name, cost))

            csvfile_nearest = open(out_nearest_file, 'w', newline='')
            csvfile_line = open(out_line_file, 'w', newline='')

            # with open(out_nearest_file, 'w', newline='') as csvfile:
            writer_nearest = csv.writer(csvfile_nearest)
            writer_line = csv.writer(csvfile_line)

            writer_nearest.writerow(['s_fid', 'nearest_fid'])
            writer_line.writerow(['s_fid', 't_fid', 'cost'])
            for key, value in res.items():
                rk_lst = []
                for rkey, dis in value['distance'].items():
                    if dis <= cost:
                        rk_lst.append(str(rkey))
                        writer_line.writerow([key, rkey, dis])

                nr = "[{}]".format(",".join(rk_lst))
                writer_nearest.writerow([key, nr])

        return True
    except:
        return False
    finally:
        if csvfile_line is not None:
            csvfile_line.close()
            del csvfile_line
        if csvfile_nearest is not None:
            csvfile_nearest.close()
            del csvfile_nearest


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
        for match_node, target_fid in zip(match_df["nodeID"], match_df["fid"]):
            if bRoutes:
                route = routes[match_node]
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


def nearest_geometry_from_point_worker(shared_custom, lst, cost, out_path, panMap, start_path, start_layer_name, ipos=0):
    out_line_ds = None
    out_line_layer = None
    out_route_ds = None
    out_route_layer = None
    in_wks = None
    wks = None
    ds_start = None
    in_layer = None

    try:
        G, target_df, start_points_dict, target_points_dict = shared_custom.task()

        out_type_f = DataType.geojson

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        datasetCreationOptions = []
        layerCreationOptions = []

        in_wks = workspaceFactory()
        ds_start = in_wks.get_ds(start_path)
        in_layer, start_layer_name = in_wks.get_layer(ds_start, start_path, start_layer_name)
        srs = in_layer.GetSpatialRef()

        out_suffix = DataType_suffix[out_type_f]

        new_fields = []
        new_fields.append(ogr.FieldDefn('s_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('t_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('cost', ogr.OFTReal))

        line_layer_name = "lines_{}_{}".format(str(cost), str(ipos))
        route_layer_name = "routes_{}_{}".format(str(cost), str(ipos))
        line_layer_name = check_layer_name(line_layer_name)
        route_layer_name = check_layer_name(route_layer_name)

        line_out_path = os.path.join(out_path, "{}.{}".format(line_layer_name, out_suffix))
        route_out_path = os.path.join(out_path, "{}.{}".format(route_layer_name, out_suffix))

        wks = workspaceFactory().get_factory(out_type_f)
        out_line_ds,  out_line_layer = wks.createFromExistingDataSource(in_layer, line_out_path, line_layer_name, srs,
                                                                        datasetCreationOptions, layerCreationOptions, new_fields,
                                                                        geom_type=ogr.wkbLineString, open=True)
        out_route_ds, out_route_layer = wks.createFromExistingDataSource(in_layer, route_out_path, route_layer_name, srs,
                                                                         datasetCreationOptions, layerCreationOptions, new_fields,
                                                                         geom_type=ogr.wkbMultiLineString, open=True)

        with lock:
            bar = mTqdm(lst, desc="worker-{}".format(ipos), position=ipos, leave=False)
        # with mTqdm(lst, desc=current_process().name, position=ipos, leave=False) as bar:
        icount = 0
        for t in lst:
            start_node = t[0]
            start_fid = t[1]
            start_pt = start_points_dict[start_fid]
            in_fea = in_layer.GetFeature(start_fid)

            try:
                distances, routes = nx.single_source_dijkstra(G, start_node, weight='length',
                                                              cutoff=cost)

                # geoms = {}
                if len(routes) > 1:  # 只有一个是本身，不算入搜索结果
                    match_df = target_df[target_df['nodeID'].apply(lambda x: x in routes)]

                    for match_node, target_fid in zip(match_df["nodeID"], match_df["fid"]):
                        target_pt = target_points_dict[target_fid]
                        route = routes[match_node]

                        line, lines = output_geometry(G, start_pt, target_pt, route)
                        # geoms[target_fid] = (line, lines, distances[match_node])

                        out_value = {
                            's_ID': str(start_fid),
                            't_ID': str(target_fid),
                            'cost': distances[match_node]
                        }

                        # ret_json = addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value)
                        addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value)
                        addFeature(in_fea, start_fid, lines, out_route_layer, panMap, out_value)

                        icount += 1
                # nearest_geometry[start_fid] = geoms

                with lock:
                    bar.update()
            except:
                log.error("发生错误节点:{}".format(str(start_node)))
                continue
        with lock:
            bar.close()

        return line_out_path, route_out_path, icount
    except:
        log.error(traceback.format_exc())
        return None
    finally:
        with lock:
            if out_line_ds is not None:
                out_line_ds.Destroy()
            del out_line_ds

            if out_route_ds is not None:
                out_route_ds.Destroy()
            del out_route_ds

            del out_line_layer
            del out_route_layer
            del in_wks
            del wks
            del ds_start
            del in_layer


def nearest_geometry_from_point_worker2(shared_custom, lst, cost, out_path, panMap, start_path, start_layer_name, ipos=0):
    out_line_ds = None
    out_line_layer = None
    out_route_ds = None
    out_route_layer = None
    in_wks = None
    wks = None
    ds_start = None
    in_layer = None
    line_writer = None
    route_writer = None

    try:
        G, target_df, start_points_dict, target_points_dict = shared_custom.task()

        # out_temp_line_path = os.path.abspath(os.path.join(out_path, "line_temp"))
        # out_temp_route_path = os.path.abspath(os.path.join(out_path, "route_temp"))

        # if not os.path.exists(out_temp_line_path):
        #     os.mkdir(out_temp_line_path)
        # if not os.path.exists(out_temp_route_path):
        #     os.mkdir(out_temp_route_path)

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        datasetCreationOptions = []
        layerCreationOptions = []

        in_wks = workspaceFactory()
        ds_start = in_wks.get_ds(start_path)
        in_layer, start_layer_name = in_wks.get_layer(ds_start, start_path, start_layer_name)
        srs = in_layer.GetSpatialRef()

        out_suffix = 'jsonl'

        new_fields = []
        new_fields.append(ogr.FieldDefn('s_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('t_ID', ogr.OFTString))
        new_fields.append(ogr.FieldDefn('cost', ogr.OFTReal))

        line_layer_name = "lines_{}_{}".format(str(cost), str(ipos))
        route_layer_name = "routes_{}_{}".format(str(cost), str(ipos))
        line_layer_name = check_layer_name(line_layer_name)
        route_layer_name = check_layer_name(route_layer_name)

        out_type_f = DataType.memory
        wks = workspaceFactory().get_factory(out_type_f)
        out_line_ds,  out_line_layer = wks.createFromExistingDataSource(in_layer, "", line_layer_name, srs,
                                                                        datasetCreationOptions, layerCreationOptions, new_fields,
                                                                        geom_type=ogr.wkbLineString, open=True)
        out_route_ds, out_route_layer = wks.createFromExistingDataSource(in_layer, "", route_layer_name, srs,
                                                                         datasetCreationOptions, layerCreationOptions, new_fields,
                                                                         geom_type=ogr.wkbMultiLineString, open=True)

        line_out_path = os.path.join(out_path, "{}.{}".format(line_layer_name, out_suffix))
        route_out_path = os.path.join(out_path, "{}.{}".format(route_layer_name, out_suffix))

        line_writer = jsonlines.open(line_out_path, "w")
        route_writer = jsonlines.open(route_out_path, "w")

        with lock:
            bar = mTqdm(lst, desc="worker-{}".format(ipos), position=ipos, leave=False)
        # with mTqdm(lst, desc=current_process().name, position=ipos, leave=False) as bar:

        icount = 0
        for t in lst:
            start_node = t[0]
            start_fid = t[1]
            start_pt = start_points_dict[start_fid]
            in_fea = in_layer.GetFeature(start_fid)

            try:
                distances, routes = nx.single_source_dijkstra(G, start_node, weight='length',
                                                              cutoff=cost)

                # geoms = {}
                if len(routes) > 1:  # 只有一个是本身，不算入搜索结果
                    match_df = target_df[target_df['nodeID'].apply(lambda x: x in routes)]

                    for match_node, target_fid in zip(match_df["nodeID"], match_df["fid"]):
                        target_pt = target_points_dict[target_fid]
                        route = routes[match_node]

                        line, lines = output_geometry(G, start_pt, target_pt, route)

                        out_value = {
                            's_ID': str(start_fid),
                            't_ID': str(target_fid),
                            'cost': distances[match_node]
                        }

                        ret = addFeature(in_fea, start_fid, line, out_line_layer, panMap, out_value, bjson=True)
                        line_writer.write(ret)

                        ret = addFeature(in_fea, start_fid, lines, out_route_layer, panMap, out_value, bjson=True)
                        route_writer.write(ret)

                        icount += 1
                with lock:
                    bar.update()
            except:
                log.error("发生错误节点:{}".format(str(start_node)))
                continue
        with lock:
            bar.close()

        return line_out_path, route_out_path, icount
    except:
        log.error(traceback.format_exc())
        return None
    finally:
        with lock:
            if out_line_ds is not None:
                out_line_ds.Destroy()
            del out_line_ds

            if out_route_ds is not None:
                out_route_ds.Destroy()
            del out_route_ds

            del out_line_layer
            del out_route_layer
            del in_wks
            del wks
            del ds_start
            del in_layer
            # if line_writer is not None:
            #     line_writer.close()
            # if route_writer is not None:
            #     route_writer.close()


def nearest_facilities_from_point_worker(shared_custom, lst, travelCost, bRoutes=True,
                                         bDistances=True, ipos=0):
    G, target_df = shared_custom.task()

    nearest_facilities = {}

    with lock:
        bar = mTqdm(lst, desc="worker-{}".format(ipos), position=ipos, leave=False)
    # with mTqdm(lst, desc=current_process().name, position=ipos, leave=False) as bar:
    for t in lst:
        start_node = t[0]
        fid = t[1]
        # try:
        distances, routes = nx.single_source_dijkstra(G, start_node, weight='length',
                                                      cutoff=travelCost)

        match_routes = {}
        match_distances = {}

        if len(routes) > 1:  # 只有一个是本身，不算入搜索结果
            match_df = target_df[target_df['nodeID'].apply(lambda x: x in routes)]

            # for row in match_df.itertuples():
            #     match_node = row.nodeID
            #     target_fid = row.fid
            for match_node, target_fid in zip(match_df["nodeID"], match_df["fid"]):

                if bRoutes:
                    route = routes[match_node]
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
        # except:
        #     log.error("发生错误节点:{}".format(str(start_node)))
        #     continue

    with lock:
        bar.close()

    return nearest_facilities
    # if connection is not None:
    #     connection.send(nearest_facilities)
    # else:
    #     return nearest_facilities
class routesTransfer:
    def __init__(self, G, res, start_points_dict, target_points_dict):
        self.G = G
        self.res = res
        self.start_points_dict = start_points_dict
        self.target_points_dict = target_points_dict
        # self.nearest_facilities = nearest_facilities
        # self.start_dict = start_dict
        # self.target_dict = target_dict
        # self.target_weight_dict = target_weight_dict
        # self.layer_target = layer_target

    def task(self):
        return self.G, self.res, self.start_points_dict, self.target_points_dict


if __name__ == '__main__':
    freeze_support()
    gdal.SetConfigOption('CPL_LOG', 'NUL')
    # QgsApplication.setPrefixPath('', True)
    # app = QgsApplication([], True)
    # app.initQgis()

    # nearest_facilities_from_layer(
    #     r"D:\空间模拟\PublicSupplyDemand\Data\sz_road_cgcs2000_test.shp",
    #     r"D:\空间模拟\PublicSupplyDemand\Data\building_test.shp",
    #     r"D:\空间模拟\PublicSupplyDemand\Data\2022年现状幼儿园.shp",
    #     travelCost=1000,
    #     out_type=0,
    #     direction_field="")

    # sys.exit(app.exec_())
    nearest()
