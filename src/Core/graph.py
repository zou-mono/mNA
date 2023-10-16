import os
import traceback
from time import strftime

from networkx import Graph
from osgeo import ogr, osr
from osgeo.ogr import DataSource, GeometryTypeToName, Geometry, Layer
from osgeo.osr import SpatialReference
from shapely import Point, LineString, distance, STRtree
from shapely.ops import nearest_points, split, snap
from shapely.wkt import loads

from Core.common import check_line_type
from Core.simplification import simplify_graph
from Core.utils_graph import get_largest_component, add_edge_lengths, graph_to_gdfs, _is_endpoint_of_edge, \
    split_line_by_point
from Core.DataFactory import workspaceFactory
from Core.log4p import Log, mTqdm
import networkx as nx
import numpy as np

log = Log(__name__)


class Direction(int):
    DirectionForward = 1
    DirectionBackward = 2
    DirectionBoth = 0


def create_graph_from_file(path,
                           input_layer_name="",
                           direction_field="",
                           forwardValue="",
                           backwardValue="",
                           bothValue="",
                           retain_all=False,
                           simplify=True,
                           defaultDirection=Direction.DirectionBoth
                           ):

    wks = workspaceFactory()
    ds: DataSource = wks.get_ds(path)
    input_layer = None

    try:
        input_layer, input_layer_name = wks.get_layer(ds, path, input_layer_name)

        if not check_line_type(input_layer):
            log.error("网络数据不满足几何类型要求，只允许Polyline或multiPolyline类型.")
            return False

        crs: SpatialReference = input_layer.GetSpatialRef()

        # create the MultiDiGraph and set its graph-level attributes
        metadata = {
            "created_date": strftime('%Y-%m-%d-%H-%M-%S'),
            "created_with": "mNA",
            "crs": crs.ExportToWkt()
        }

        o_nodes = _nodes_from_network(input_layer)
        nodes, paths = _parse_nodes_paths(o_nodes, input_layer, direction_field,
                                          forwardValue, backwardValue, bothValue, defaultDirection)

        G = nx.MultiDiGraph(**metadata)
        G.add_nodes_from(nodes.items())
        _add_paths(G, paths.values())

        if not retain_all:
            log.info("构建最大连通图...")
            G = get_largest_component(G)

        log.info("创建的图G共有{}条边，{}个节点".format(len(G.edges), len(G)))

        if len(G.edges) > 0:
            G = add_edge_lengths(G)

        if simplify:
            G = simplify_graph(G)

        # strDriverName = "ESRI Shapefile"
        # oDriver = ogr.GetDriverByName(strDriverName)
        # if oDriver == None:
        #     return "驱动不可用："+strDriverName
        # # 创建数据源
        # oDS = oDriver.CreateDataSource("points.shp")
        # if oDS == None:
        #     return "创建文件失败：points.shp"
        # # 创建一个多边形图层，指定坐标系为WGS84
        # papszLCO = []
        # # geosrs = osr.SpatialReference()
        # # geosrs.SetWellKnownGeogCS("WGS84")
        # # 线：ogr_type = ogr.wkbLineString
        # # 点：ogr_type = ogr.wkbPoint
        # ogr_type = ogr.wkbPoint
        # # 面的类型为Polygon，线的类型为Polyline，点的类型为Point
        # oLayer = oDS.CreateLayer("points", crs, ogr_type, papszLCO)
        # if oLayer == None:
        #     return "图层创建失败！"
        #
        # # 创建id字段
        # oId = ogr.FieldDefn("id", ogr.OFTInteger)
        # oLayer.CreateField(oId, 1)
        # # 创建name字段
        # oName = ogr.FieldDefn("name", ogr.OFTString)
        # oLayer.CreateField(oName, 1)
        # oDefn = oLayer.GetLayerDefn()
        #
        # for node in G.nodes.data():
        #     oFeaturePolygon = ogr.Feature(oDefn)
        #     v = node[1]
        #     point = ogr.Geometry(ogr.wkbPoint)
        #     point.AddPoint(v['x'], v['y'])
        #     oFeaturePolygon.SetGeometry(point)
        #     oLayer.CreateFeature(oFeaturePolygon)
        #
        # # 创建完成后，关闭进程
        # oDS.Destroy()

        return G
    except Exception:
        log.error(traceback.format_exc())
        return None
    finally:
        if ds is not None:
            ds.Destroy()
        del wks
        del input_layer
        del ds


def _add_paths(G, paths):
    oneway_values = {Direction.DirectionForward, Direction.DirectionBackward}
    reversed_values = {Direction.DirectionBackward}

    for path in paths:
        # extract/remove the ordered list of nodes from this path element so
        # we don't add it as a superfluous attribute to the edge later
        nodes = path.pop("nodes")

        # reverse the order of nodes in the path if this path is both one-way
        # and only allows travel in the opposite direction of nodes' order
        is_one_way = _is_path_one_way(path, oneway_values)
        if is_one_way and _is_path_reversed(path, reversed_values):
            nodes.reverse()

        # zip path nodes to get (u, v) tuples like [(0,1), (1,2), (2,3)].
        edges = list(zip(nodes[:-1], nodes[1:]))

        # add all the edge tuples and give them the path's tag:value attrs
        path["reversed"] = False
        G.add_edges_from(edges, **path)

        # if the path is NOT one-way, reverse direction of each edge and add
        # this path going the opposite direction too
        if not is_one_way:
            path["reversed"] = True
            G.add_edges_from([(v, u) for u, v in edges], **path)


def _is_path_one_way(path, oneway_values):
    """
    判断是否是单向路
    """
    return True if path['direction'] in oneway_values else False


def _is_path_reversed(path, reversed_values):
    """
    判断是否是反向路
    """
    return True if path['direction'] in reversed_values else False


def _nodes_from_network(layer: Layer):
    nodes = {}
    inode = 0

    for feature in layer:
        geom: Geometry = feature.GetGeometryRef()
        fid = feature.GetFID()

        if geom.GetGeometryName() == "LINESTRING":
            points = geom.GetPoints()
            for point in points:
                node = {"id": inode, "x": point[0], "y": point[1], "feature_id": fid, "bOrigin": True}
                # node = {"x": point[0], "y": point[1], "feature_id": fid, "bOrigin": True}
                nodes.update({(point[0], point[1]): node})
                inode += 1
        elif geom.GetGeometryName() == "MULTILINESTRING":
            for part in geom:
                points = part.GetPoints()
                for point in points:
                    node = {"id": inode, "x": point[0], "y": point[1], "feature_id": fid, "bOrigin": True}
                    # node = {"x": point[0], "y": point[1], "feature_id": fid, "bOrigin": True}
                    nodes.update({(point[0], point[1]): node})
                    inode += 1

    return nodes


def _parse_nodes_paths(o_nodes, layer, direction_field='oneway',
                       forwardValue='F', backwardValue='T', bothValue='B',
                       defaultDirection=Direction.DirectionBoth):
    nodes = {}
    paths = {}

    direction_idx = layer.FindFieldIndex(direction_field, False)

    ipath = 0
    for feature in layer:
        geom: Geometry = feature.GetGeometryRef()
        fid = feature.GetFID()

        if direction_idx == -1:
            direction = defaultDirection
        else:
            f_v = feature.GetField(direction_idx)
            if f_v == forwardValue:
                direction = Direction.DirectionForward
            elif f_v == backwardValue:
                direction = Direction.DirectionBackward
            # elif v == bothValue:
            #     direction = Direction.DirectionBoth
            else:
                direction = Direction.DirectionBoth

        if geom.GetGeometryName() == "LINESTRING":
            points = geom.GetPoints()
            node_ids = []
            for point in points:
                v = o_nodes[(point[0], point[1])]
                nodes[v['id']] = v
                node_ids.append(v['id'])

            paths[ipath] = {
                # 'id': ipath,
                'feature_id': fid,
                'nodes': node_ids,
                'direction': direction,
                'geometry': loads(geom.ExportToWkt())
            }

            ipath += 1

        elif geom.GetGeometryName() == "MULTILINESTRING":
            for part in geom:
                points = part.GetPoints()
                node_ids = []
                for point in points:
                    v = o_nodes[(point[0], point[1])]
                    nodes[v['id']] = v
                    node_ids.append(v['id'])

                paths[ipath] = {
                    # 'id': ipath,
                    'feature_id': fid,
                    'nodes': node_ids,
                    'direction': direction,
                    'geometry': loads(part.ExportToWkt())
                }

                ipath += 1

    return nodes, paths


def _split_edges(G, points_along_edge):
    # nodes, edges = graph_to_gdfs(G)
    # s = []
    for eid, node_pts in points_along_edge.items():  # 遍历需要split的边
        line = G.edges[eid]['geometry']
        edge_attrs = G.edges[eid]

        along_disances = [line.project(node_pt[1]) for node_pt in node_pts]

        #  根据长度，过近的两点只保留一个

        # 对线上的点按照线上距离进行排序，确定split点的序号
        pt_orders = sorted(range(len(along_disances)), key=along_disances.__getitem__)

        res_line = line
        for i in range(len(pt_orders)):
            cur_order = pt_orders[i]
            if i == 0:  # 第一个点要加上首端点
                cur_node = eid[0]
                next_order = pt_orders[0]
                next_node = node_pts[next_order][0]
                split_point = node_pts[next_order][1]
                geomColl = split_line_by_point(split_point, res_line).geoms
                if len(geomColl) == 1:
                    print("error")
                res_line = geomColl[1]
                geom = geomColl[0]
            elif i == len(pt_orders) - 1:  # 最后一个点要加上末端点
                cur_order = pt_orders[-1]
                cur_node = node_pts[cur_order][0]
                next_node = eid[1]
                geom = res_line
            else:  # 中间点是新增node
                cur_node = node_pts[cur_order][0]
                next_order = pt_orders[i + 1]
                next_node = node_pts[next_order][0]
                split_point = node_pts[next_order][1]
                geomColl = split_line_by_point(split_point, res_line).geoms
                res_line = geomColl[1]
                geom = geomColl[0]

            G.add_node(node_pts[cur_order][0], **{
                'id': node_pts[cur_order][0],
                'y': node_pts[cur_order][1].y,
                'x': node_pts[cur_order][1].x,
                'bOrigin': False})

            G.add_edge(cur_node, next_node,
                       **{**edge_attrs,
                          'geometry': geom,
                          'length': geom.length})

            if len(pt_orders) == 1:  # 只有一个点的时候，要再加上一条edge，从split点到末端点
                cur_order = pt_orders[-1]
                cur_node = node_pts[cur_order][0]
                next_node = eid[1]
                G.add_edge(cur_node, next_node,
                           **{**edge_attrs,
                              'geometry': res_line,
                              'length': res_line.length})

        G.remove_edge(eid[0], eid[1], eid[2])  # MultiDiGraph删除边要注意加上key，否则有可能把所有边都删除

    # nodes, edges = graph_to_gdfs(G)
    return G


def makeGraph(G: Graph, additionalPoints, o_max=-1, distance_tolerance=500, rtree=None):
    lst = [-1] * len(additionalPoints)
    target_node_ids = np.array(lst)  # 将additionalPoints转换为目标node的label id，用于后续路径计算

    G = G.copy()

    # edge_geoms = edges['geometry']
    edge_geoms = list(nx.get_edge_attributes(G, "geometry").values())
    if rtree is None:
        log.debug("对edges构建rtree, 计算设施的最近邻edge...")
        rtree = STRtree(edge_geoms)

    pos = rtree.query_nearest(additionalPoints, all_matches=False, return_distance=False,
                              max_distance=distance_tolerance)

    # nodes, edges = graph_to_gdfs(G)
    if o_max == -1:
        # o_max = max(nodes['id'])  # 从最大序号开始为新增split点编号
        o_max = max(G.nodes)
    points_along_edge = {}  # 存储需要打断的edge上附着的split点

    edge_ids = list(G.edges)

    log.debug("计算edge上所有的附着点...")
    for i in mTqdm(range(pos.shape[1])):
        line = edge_geoms[pos[1][i]]

        ne = nearest_points(line, additionalPoints[pos[0][i]])[0]

        eid = edge_ids[pos[1][i]]
        # eid = edges.index[pos[1][i]]

        # if eid[0] == 2074 and eid[1] == 2074:
        #     print("debug")
        edge_attrs = G.edges[eid]

        # is_endpoint, flag = _is_endpoint_of_edge(ne, edges.loc[eid]['geometry'])
        is_endpoint, flag = _is_endpoint_of_edge(ne, edge_attrs['geometry'])

        if not is_endpoint:
            # 加入split点，更新图

            # edge_attrs = edges.loc[eid].to_dict()
            o_max += 1

            if eid in points_along_edge:
                d = points_along_edge[eid]
                d.append((o_max, ne))
                points_along_edge[eid] = d
            else:
                points_along_edge[eid] = [(o_max, ne)]

            target_node_ids[pos[0][i]] = o_max
            # target_node_ids.append(o_max)

            if eid[0] != eid[1]:  # MultiDiGraph有可能出现loop，这种排除， 例如(2074,2074,0)
                # 判断双向边
                r_eid = (eid[1], eid[0], eid[2])
                if r_eid in G.edges:
                    if G.edges[r_eid]["feature_id"] == edge_attrs["feature_id"]:
                        if r_eid in points_along_edge:
                            d = points_along_edge[r_eid]
                            d.append((o_max, ne))
                            points_along_edge[r_eid] = d
                        else:
                            points_along_edge[r_eid] = [(o_max, ne)]
        else:
            node_id = eid[1] if flag == "s" else eid[0]
            # target_node_ids.append(int(node_id))
            target_node_ids[pos[0][i]] = int(node_id)

            # print(net.nodes(data=True)[o_max])
            # G.edges([151609,39627,0]) # 可以筛选

    log.debug("按照附着点顺序打断edge,构建新node和edge...")
    _split_edges(G, points_along_edge)

    return G, list(target_node_ids)


if __name__ == '__main__':
    create_graph_from_file(r"D:\空间模拟\PublicSupplyDemand\Data\mini_graph.shp",
                           direction_field="oneway",
                           forwardValue='F',
                           backwardValue='T',
                           bothValue='B')
