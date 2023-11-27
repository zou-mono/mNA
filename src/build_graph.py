import os
import traceback
from time import time, strftime

import click
from osgeo import gdal

from Core.DataFactory import get_suffix
from Core.graph import import_graph_to_network, create_graph_from_file, export_network_to_graph
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
@click.option("--simplify", type=bool, required=False, default=True,
              help="是否进行图的拓扑简化, 可选. ")
@click.option("--retain-all", type=bool, required=False, default=False,
              help="是否只保留最大弱连通图, 可选. ")
@click.option("--out-type", type=click.Choice(['gpickle', 'graphml', 'gml', 'gexf'], case_sensitive=False),
              required=False, default='gpickle',
              help="如果原始网络数据是空间数据(shp, geojson, gdb等), 则需要设置存储的图文件格式, "
                   "默认值gpicke. 支持gpickle, graphml, gml, gexf.")
@click.option("--out-path", "-o", type=str, required=False, default=None,
              help="输出路径, 可选, 默认值为当前目录下的res目录, 文件名默认为network.")
def build_graph(network, network_layer, direction_field, forward_value, backward_value, both_value,
            default_direction, simplify, retain_all, out_type, out_path):
    """从网络数据构建图结构."""

    if out_path is None:
        out_dir = os.path.join(os.getcwd(), "res")
        out_file = "network"
    else:
        out_file, file_ext = os.path.splitext(os.path.basename(out_path))
        out_dir = os.path.dirname(out_path)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    start_time = time()

    log.info("读取网络数据...")

    try:
        log.info("读取网络数据, 路径为{}...".format(network))

        input_type = get_suffix(network)

        if input_type == 'gml' or input_type == 'gexf' or input_type == 'graphml' or input_type == 'gpickle':
            net = import_graph_to_network(network, input_type)
            if net is not None:
                log.info(f"从输入文件读取后构建的图共{len(net)}个节点, {len(net.edges)}条边")
        else:
            net = create_graph_from_file(network,
                                         network_layer,
                                         direction_field=direction_field,
                                         bothValue=both_value,
                                         simplify=simplify,
                                         retain_all=retain_all,
                                         backwardValue=backward_value,
                                         forwardValue=forward_value)

            if net is not None:
                export_network_to_graph(out_type, net, out_dir, out_file)

        if net is None:
            log.error("网络数据存在问题, 无法创建图结构.")
            return

        end_time = time()
        log.info("计算完成, 共耗时{}秒".format(str(end_time - start_time)))
    except:
        log.error(traceback.format_exc())
        log.error("计算未完成, 请查看日志: {}.".format(os.path.abspath(log.logName)))


if __name__ == '__main__':
    gdal.SetConfigOption('CPL_LOG', 'NUL')
    build_graph()


