import os.path
import sys
import traceback
from multiprocessing import freeze_support
from time import strftime

import click

from Core.core import DataType
from Core.graph import Direction
from Core.log4p import Log

log = Log(__name__)


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
@click.option("--include-bounds", '-b', type=bool, required=False, default=False,
              help="是否输出可达性边界每条边的起点和终点. 可选, 默认值为否.")
@click.option("--concave-hull-ratio", '-r', type=click.FloatRange(0, 1, clamp=False), required=False, default=0.3,
              help="可达性凹包的阈值, 取值范围0-1, 越接近1则凹包越平滑. 可选, 默认值为0.3.")
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
def accessibility(network, network_layer, direction_field, forward_value, backward_value, both_value,
                  default_direction, spath, spath_layer, out_fields, cost, include_bounds, concave_hull_ratio,
                  distance_tolerance, out_type, out_graph_type, out_path, cpu_count):
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
        include_bounds=include_bounds,
        concave_hull_ratio=concave_hull_ratio,
        panMap=panMap,
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


def accessible_from_layer(
        network_path,
        network_layer,
        start_path,
        start_layer_name,
        travelCosts=[sys.float_info.max],
        include_bounds=False,
        concave_hull_ratio=0.3,
        direction_field="",
        forwardValue="",
        backwardValue="",
        bothValue="",
        panMap=None,
        distance_tolerance=500,  # 从原始点到网络最近snap点的距离容差，如果超过说明该点无法到达网络，不进行计算
        defaultDirection=Direction.DirectionBoth,
        out_type=0,
        out_graph_type='gpickle',
        out_path="res",
        cpu_core=1):
    pass


def export_to_file(geoms, fields, crs, source_attributes, out_path, out_type):
    pass


if __name__ == '__main__':
    freeze_support()
    # gdal.SetConfigOption('CPL_LOG', 'NUL')
    accessibility()
