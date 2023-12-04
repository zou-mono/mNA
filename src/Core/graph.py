import ast
import contextlib
import math
import os
import pickle
import traceback
from time import strftime

from networkx import Graph
from osgeo import ogr, osr
from osgeo.ogr import DataSource, GeometryTypeToName, Geometry, Layer
from osgeo.osr import SpatialReference
from shapely import Point, LineString, distance, STRtree, get_num_points, wkt
from shapely.ops import nearest_points, split, snap
from shapely.wkt import loads

from Core.check import check_line_type
from Core.simplification import simplify_graph
from Core.utils_graph import get_largest_component, add_edge_lengths, _is_endpoint_of_edge, \
    split_line_by_point, is_projectPoint_in_segment
from Core.DataFactory import workspaceFactory, get_suffix
from Core.log4p import Log, mTqdm
import networkx as nx
import numpy as np
from rtree import index

log = Log(__name__)


class Direction(int):
    DirectionForward = 1
    DirectionBackward = 2
    DirectionBoth = 0


def import_graph_to_network(input_path, input_type):
    net = None
    if input_type == 'gpickle':
        with open(input_path, 'rb') as f:
            net = pickle.load(f)
    else:
        net = _load_graph(input_path, input_type)

    return net


def export_network_to_graph(out_graph_type, net, out_dir, out_file='network'):
    try:
        out_graph_type = out_graph_type.lower()
        out_net_path = os.path.abspath(os.path.join(out_dir, "{}.{}".format(out_file, out_graph_type)))

        if out_graph_type == 'gpickle':
            out_net_path = os.path.abspath(os.path.join(out_dir, "{}.gpickle".format(out_file)))
            with open(out_net_path, 'wb') as f:
                pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
        else:
            _save_graph(net, out_net_path, out_graph_type)

        log.info("网络数据保存至{}".format(out_net_path))
        return True
    except:
        log.error("保存网络数据至图文件时发生错误. {}".format(traceback.format_exc()))
        return False


def _load_graph(
        filepath=None, input_type='graphml', graph_str=None, node_dtypes=None, edge_dtypes=None, graph_dtypes=None
):
    """
    Load an OSMnx-saved GraphML file from disk or GraphML string.

    This function converts node, edge, and graph-level attributes (serialized
    as strings) to their appropriate data types. These can be customized as
    needed by passing in dtypes arguments providing types or custom converter
    functions. For example, if you want to convert some attribute's values to
    `bool`, consider using the built-in `ox.io._convert_bool_string` function
    to properly handle "True"/"False" string literals as True/False booleans:
    `ox.load_graphml(fp, node_dtypes={my_attr: ox.io._convert_bool_string})`.

    If you manually configured the `all_oneway=True` setting, you may need to
    manually specify here that edge `oneway` attributes should be type `str`.

    Note that you must pass one and only one of `filepath` or `graphml_str`.
    If passing `graphml_str`, you may need to decode the bytes read from your
    file before converting to string to pass to this function.

    Parameters
    ----------
    filepath : string or pathlib.Path
        path to the GraphML file
    graphml_str : string
        a valid and decoded string representation of a GraphML file's contents
    node_dtypes : dict
        dict of node attribute names:types to convert values' data types. the
        type can be a python type or a custom string converter function.
    edge_dtypes : dict
        dict of edge attribute names:types to convert values' data types. the
        type can be a python type or a custom string converter function.
    graph_dtypes : dict
        dict of graph-level attribute names:types to convert values' data
        types. the type can be a python type or a custom string converter
        function.

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    # specify default graph/node/edge attribute values' data types
    default_graph_dtypes = {"simplified": _convert_bool_string}
    default_node_dtypes = {
        "elevation": float,
        "elevation_res": float,
        "lat": float,
        "lon": float,
        "osmid": int,
        "street_count": int,
        "x": float,
        "y": float,
    }
    default_edge_dtypes = {
        "bearing": float,
        "grade": float,
        "grade_abs": float,
        "length": float,
        "oneway": _convert_bool_string,
        "osmid": int,
        "reversed": _convert_bool_string,
        "speed_kph": float,
        "travel_time": float,
    }

    # override default graph/node/edge attr types with user-passed types, if any
    if graph_dtypes is not None:
        default_graph_dtypes.update(graph_dtypes)
    if node_dtypes is not None:
        default_node_dtypes.update(node_dtypes)
    if edge_dtypes is not None:
        default_edge_dtypes.update(edge_dtypes)

    if filepath is not None:
        # read the graph file from disk
        source = filepath
        if input_type == 'graphml':
            G = nx.read_graphml( filepath, node_type=default_node_dtypes["osmid"], force_multigraph=True)
        elif input_type == 'gml':
            G = nx.read_gml(filepath, destringizer=int)
        elif input_type == 'gexf':
            G = nx.read_gexf(filepath, node_type=default_node_dtypes["osmid"])

    # convert graph/node/edge attribute data types
    G = _convert_graph_attr_types(G, default_graph_dtypes)
    G = _convert_node_attr_types(G, default_node_dtypes)
    G = _convert_edge_attr_types(G, default_edge_dtypes)

    return G


def _convert_graph_attr_types(G, dtypes=None):
    """
    Convert graph-level attributes using a dict of data types.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    dtypes : dict
        dict of graph-level attribute names:types

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    # remove node_default and edge_default metadata keys if they exist
    G.graph.pop("node_default", None)
    G.graph.pop("edge_default", None)

    for attr in G.graph.keys() & dtypes.keys():
        G.graph[attr] = dtypes[attr](G.graph[attr])

    return G


def _convert_node_attr_types(G, dtypes=None):
    """
    Convert graph nodes' attributes using a dict of data types.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    dtypes : dict
        dict of node attribute names:types

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    for _, data in G.nodes(data=True):
        for attr in data.keys() & dtypes.keys():
            data[attr] = dtypes[attr](data[attr])
    return G


def _convert_edge_attr_types(G, dtypes=None):
    """
    Convert graph edges' attributes using a dict of data types.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    dtypes : dict
        dict of edge attribute names:types

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    # for each edge in the graph, eval attribute value lists and convert types
    for _, _, data in G.edges(data=True, keys=False):
        # remove extraneous "id" attribute added by graphml saving
        data.pop("id", None)

        # first, eval stringified lists to convert them to list objects
        # edge attributes might have a single value, or a list if simplified
        for attr, value in data.items():
            if isinstance(value, (int, float)):
                data[attr] = value
            else:
                if value.startswith("[") and value.endswith("]"):
                    with contextlib.suppress(SyntaxError, ValueError):
                        data[attr] = ast.literal_eval(value)

        # next, convert attribute value types if attribute appears in dtypes
        for attr in data.keys() & dtypes.keys():
            if isinstance(data[attr], list):
                # if it's a list, eval it then convert each item
                data[attr] = [dtypes[attr](item) for item in data[attr]]
            else:
                # otherwise, just convert the single value
                data[attr] = dtypes[attr](data[attr])

        # if "geometry" attr exists, convert its well-known text to LineString
        if "geometry" in data:
            data["geometry"] = wkt.loads(data["geometry"])

    return G


def _convert_bool_string(value):
    """
    Convert a "True" or "False" string literal to corresponding boolean type.

    This is necessary because Python will otherwise parse the string "False"
    to the boolean value True, that is, `bool("False") == True`. This function
    raises a ValueError if a value other than "True" or "False" is passed.

    If the value is already a boolean, this function just returns it, to
    accommodate usage when the value was originally inside a stringified list.

    Parameters
    ----------
    value : string {"True", "False"}
        the value to convert

    Returns
    -------
    bool
    """
    if value in {"True", "False"}:
        return value == "True"

    if isinstance(value, bool):
        return value

    # otherwise the value is not a valid boolean
    msg = f"布尔值错误: {value!r}"
    raise ValueError(msg)


def _save_graph(G, filepath=None, out_type='graphml', gephi=False, encoding="utf-8"):
    """
    Save graph to disk as GraphML file.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    filepath : string or pathlib.Path
        path to the GraphML file including extension. if None, use default
        data folder + graph.graphml
    gephi : bool
        if True, give each edge a unique key/id to work around Gephi's
        interpretation of the GraphML specification
    encoding : string
        the character encoding for the saved file

    Returns
    -------
    None
    """
    if gephi:
        # for gephi compatibility, each edge's key must be unique as an id
        # uvkd = ((u, v, k, d) for k, (u, v, d) in enumerate(G.edges(keys=False, data=True)))
        # G = nx.MultiDiGraph(uvkd)
        net = nx.MultiDiGraph()
        net.add_nodes_from(G.nodes().items())
        for k, (u, v, d) in enumerate(G.edges(keys=False, data=True)):
            net.add_edge(u, v, key=k, attr=d)
        G = net.copy()
    else:
        # make a copy to not mutate original graph object caller passed in
        G = G.copy()

    # stringify all the graph attribute values
    for attr, value in G.graph.items():
        G.graph[attr] = str(value)

    # stringify all the node attribute values
    for _, data in G.nodes(data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    # stringify all the edge attribute values
    for _, _, data in G.edges(keys=False, data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    if out_type == 'graphml':
        nx.write_graphml(G, path=filepath, encoding=encoding)
    elif out_type == "gml":
        nx.write_gml(G, path=filepath)
    elif out_type == "gexf":
        nx.write_gexf(G, path=filepath, encoding=encoding)


def create_graph_from_file(path,
                           input_layer_name="",
                           direction_field="",
                           forwardValue="",
                           backwardValue="",
                           bothValue="",
                           retain_all=False,
                           simplify=True,
                           bNodes=False,
                           bPaths=False,
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

        log.debug("根据网络数据构建原始图结构...")
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

        if bNodes and bPaths:
            return G, nodes, paths

        if bNodes:
            return G, nodes

        if bPaths:
            return G, paths

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

    for path in mTqdm(paths, total=len(paths)):
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

        # for e in edges:
        # add all the edge tuples and give them the path's tag:value attrs
        path["reversed"] = False

        for edge in edges:
            # if edge[0] == 1903 and edge[1] == 6688:
            #     print("debug")

            path["geometry"] = LineString(
                [Point((G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"])),
                 Point((G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"]))]
            )

            # G.add_edge(edge[0], edge[1], **path)
            bdup = False
            if edge[0] in G:
                if edge[1] in G[edge[0]]:
                    dup_edges = G[edge[0]][edge[1]]
                    for key, v in dup_edges.items():
                        if v['geometry'].equals(path["geometry"]):
                            bdup = True
                            break
            if not bdup:
                G.add_edge(edge[0], edge[1], **path)

            if not is_one_way:
                path["reversed"] = True
                # G.add_edges_from([(v, u) for u, v in edges], **path)
                path["geometry"] = LineString(
                    [Point((G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"])),
                     Point((G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]))]
                )

                bdup = False
                if edge[1] in G:
                    if edge[0] in G[edge[1]]:
                        dup_edges = G[edge[1]][edge[0]]
                        for key, v in dup_edges.items():
                            if v['geometry'].equals(path["geometry"]):
                                bdup = True
                                break

                if not bdup:
                    G.add_edge(edge[1], edge[0], **path)
                # G.add_edge(edge[1], edge[0], **path)

        # G.add_edges_from(edges, **path)

        # # if the path is NOT one-way, reverse direction of each edge and add
        # # this path going the opposite direction too
        # if not is_one_way:
        #     path["reversed"] = True
        #     G.add_edges_from([(v, u) for u, v in edges], **path


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
                'direction': direction
                # 'geometry': loads(geom.ExportToWkt())
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
                    'direction': direction
                    # 'geometry': loads(part.ExportToWkt())
                }

                ipath += 1

    return nodes, paths


def makeGraph(G: Graph, additionalPoints, o_max=-1, distance_tolerance=500, duplication_tolerance=5, rtree=None):
    # duplication_tolerance = 5  # 判断两个点是否重复的容差，越大则最后的图节点越少，运行速度越快

    lst = [-1] * len(additionalPoints)
    node_ids = np.array(lst)  # 将additionalPoints转换为node的label id，用于后续路径计算

    G = G.copy()

    # edge_geoms = edges['geometry']
    edge_geoms = list(nx.get_edge_attributes(G, "geometry").values())
    if rtree is None:
        log.debug("对edges构建rtree, 计算设施的最近邻edge...")
        rtree = STRtree(edge_geoms)

    pos = rtree.query_nearest(additionalPoints, all_matches=False, return_distance=False,
                              max_distance=distance_tolerance)

    if o_max == -1:
        # o_max = max(nodes['id'])  # 从最大序号开始为新增split点编号
        o_max = max(G.nodes)
    points_along_edge = {}  # 存储需要打断的edge上附着的split点

    edge_ids = list(G.edges)

    # log.debug("计算edge上所有的附着点...")
    for i in mTqdm(range(pos.shape[1]), total=pos.shape[1]):
        # if pos[1][i] == 533:
        #     print("debug")

        line = edge_geoms[pos[1][i]]

        ne = nearest_points(line, additionalPoints[pos[0][i]])[0]

        eid = edge_ids[pos[1][i]]
        # eid = edges.index[pos[1][i]]
        # r_eid = (eid[1], eid[0], eid[2])

        # if eid[0] == 3023 and eid[1] == 9437:
        #     print("debug")
        # if eid == (269515, 269514, 1):
        #     print("debug")

        breverse = True
        if eid in points_along_edge:
            # 如果当前边nodes为空，则判断对向边是否为空；如果对向边nodes不为空，则使用对向边作为当前边
            if points_along_edge[eid]['nodes'] is None:
                if points_along_edge[eid]['reverse_eid'] is not None:
                    r_eid = points_along_edge[eid]['reverse_eid']
                    if points_along_edge[r_eid]['nodes'] is not None:
                        eid = r_eid

                    breverse = False

        edge_attrs = G.edges[eid]

        # is_endpoint, flag = _is_endpoint_of_edge(ne, edges.loc[eid]['geometry'])
        is_endpoint, flag = _is_endpoint_of_edge(ne, edge_attrs['geometry'], tolerance=duplication_tolerance)

        if not is_endpoint:
            # 加入split点，更新图

            # edge_attrs = edges.loc[eid].to_dict()
            if eid in points_along_edge:
                # edges上的点要先判断是否与已有点很近，如果很近，那就不加入split
                node_pts = points_along_edge[eid]['nodes']
                bflag = True
                exist_node_id = -1
                for node_pt in node_pts:
                    if distance(node_pt[1], ne) <= duplication_tolerance:
                        bflag = False
                        exist_node_id = node_pt[0]
                        break

                if bflag:
                    o_max += 1
                    exist_node_id = o_max
                    node_pts.append((o_max, ne))
                    points_along_edge[eid]['nodes'] = node_pts

                node_ids[pos[0][i]] = exist_node_id
            else:
                #  edge上的第一个点肯定要加入的
                o_max += 1
                points_along_edge[eid] = {
                    'reverse_eid': None,   # 对向边的eid
                    'nodes': [(o_max, ne)]
                }
                # if o_max == 786237:
                #     print("debug")
                node_ids[pos[0][i]] = o_max

            # target_node_ids.append(o_max)

            # if eid[0] != eid[1]:  # MultiDiGraph有可能出现loop，这种排除， 例如(2074,2074,0)
            if not breverse:
                continue

            # 判断对向边
            if eid[1] in G:
                if eid[0] in G[eid[1]]:
                    r_eids = G[eid[1]][eid[0]]
                    # 这里需要注意: 找对向边有可能遇到多重边的情况，所以只有图形相同的才能视为对向边
                    for key, v in r_eids.items():
                        # MultiDiGraph有可能出现loop，这种要判断key， 例如(2074,2074,0)和(2074,2074,1)
                        if eid[0] == eid[1] and key == eid[2]:
                            continue

                        if v['geometry'].equals(line):
                            r_eid = (eid[1], eid[0], key)
                            if r_eid in G.edges:
                                points_along_edge[eid]['reverse_eid'] = r_eid
                                points_along_edge[r_eid] = {
                                    'reverse_eid': eid,
                                    'nodes': None
                                }
                                break
                                # if r_eid in points_along_edge:
                                #     d = points_along_edge[r_eid]
                                #     d.append((o_max, ne))
                                #     points_along_edge[r_eid] = d
                                # else:
                                #     points_along_edge[r_eid] = [(o_max, ne)]
                                # break
            # r_eid = (eid[1], eid[0], eid[2])
            # if r_eid in G.edges:
            #     if G.edges[r_eid]["feature_id"] == edge_attrs["feature_id"]:
            #         if r_eid in points_along_edge:
            #             d = points_along_edge[r_eid]
            #             d.append((o_max, ne))
            #             points_along_edge[r_eid] = d
            #         else:
            #             points_along_edge[r_eid] = [(o_max, ne)]
        else:
            node_id = eid[1] if flag == "s" else eid[0]
            # target_node_ids.append(int(node_id))
            node_ids[pos[0][i]] = int(node_id)

            # print(net.nodes(data=True)[o_max])
            # G.edges([151609,39627,0]) # 可以筛选

    log.debug("按照附着点顺序打断edge,构建新node和edge...")
    _split_edges(G, points_along_edge)

    return G, list(node_ids)


def _split_edges(G, points_along_edge):
    # nodes, edges = graph_to_gdfs(G)
    # s = []

    for eid, data in mTqdm(points_along_edge.items(), total=len(points_along_edge)):  # 遍历需要split的边
        node_pts = data['nodes']

        if node_pts is None:  # 只处理一侧的边，对向边为空不处理
            continue

        has_reverse = False if data['reverse_eid'] is None else True

        if eid not in G.edges:
            log.error("eid: {}, eid[0]: ({}, {}), eid[1]: ({}, {})".format(eid, G.nodes[eid[0]]['x'], G.nodes[eid[0]]['y'],
                                                     G.nodes[eid[1]]['x'], G.nodes[eid[1]]['y']))

        line = G.edges[eid]['geometry']
        edge_attrs = G.edges[eid]

        # if len(node_pts) > 1:
        along_disances = [line.project(node_pt[1]) for node_pt in node_pts]
        pt_orders = sorted(range(len(along_disances)), key=along_disances.__getitem__)
            # along_disances = list(sorted(along_disances))
        # else:
        #     pt_orders = [0]

        res_nodes = [(eid[0], Point(G.nodes[eid[0]]['x'], G.nodes[eid[0]]['y']))]
        res_nodes.extend([node_pts[n] for n in pt_orders])
        res_nodes.extend([(eid[1], Point(G.nodes[eid[1]]['x'], G.nodes[eid[1]]['y']))])

        i = 0
        res_line = line
        split_geoms = []
        for u, v in zip(res_nodes[:-1], res_nodes[1:]):
            if i < len(res_nodes) - 2:
                geomColl = split_line_by_point(v[1], res_line).geoms
                geom = geomColl[0]
                if len(geomColl) == 1:
                    print("debug")
                res_line = geomColl[1]
            elif i == len(res_nodes) - 2:
                geom = res_line

            G.add_edge(u[0], v[0],
                       **{**edge_attrs,
                          'geometry': geom,
                          'length': geom.length})

            split_geoms.append(geom)  # 存储为反向边使用

            if i > 0:
                G.add_node(u[0], **{
                    'id': u[0],
                    'x': u[1].x,
                    'y': u[1].y,
                    'bOrigin': False})

            i += 1

        G.remove_edge(eid[0], eid[1], eid[2])  # MultiDiGraph删除边要注意加上key，否则有可能把所有边都删除

        if has_reverse:
            r_eid = data['reverse_eid']

            res_nodes.reverse()

            i = 0
            for u, v in zip(res_nodes[:-1], res_nodes[1:]):
                # if u[0] == 9923 and v[0] == 951:
                #     print("debug")

                geom = split_geoms[len(split_geoms) - 1 - i]
                G.add_edge(u[0], v[0],
                           **{**edge_attrs,
                              'geometry': geom.reverse(),  # 这里geometry也要反向
                              'length': geom.length})
                i += 1

            G.remove_edge(r_eid[0], r_eid[1], r_eid[2])
    # for eid, node_pts in points_along_edge.items():  # 遍历需要split的边
    #     line = G.edges[eid]['geometry']
    #     edge_attrs = G.edges[eid]
    #
    #     # 如果超过1个点，对线上的点按照线上距离进行排序，确定split点的序号
    #     if len(node_pts) > 1:
    #         along_disances = [line.project(node_pt[1]) for node_pt in node_pts]
    #         pt_orders = sorted(range(len(along_disances)), key=along_disances.__getitem__)
    #         along_disances = list(sorted(along_disances))
    #     else:
    #         pt_orders = [0]
    #
    #     res_line = line
    #     left_order = 0  # 用来存储保留下来的点序号
    #     for i in range(len(pt_orders)):
    #         cur_order = pt_orders[i]
    #         if i == 0:  # 第一个点要加上首端点
    #             cur_node = eid[0]
    #             next_order = pt_orders[0]
    #             next_node = node_pts[next_order][0]
    #             split_point = node_pts[next_order][1]
    #             geomColl = split_line_by_point(split_point, res_line).geoms
    #             # if len(geomColl) == 1:
    #             #     print("error")
    #             res_line = geomColl[1]
    #             geom = geomColl[0]
    #         elif i == len(pt_orders) - 1:  # 最后一个点要加上末端点
    #             cur_order = pt_orders[-1]
    #             cur_node = node_pts[cur_order][0]
    #             next_node = eid[1]
    #             geom = res_line
    #         else:  # 中间点是新增node
    #             cur_node = node_pts[cur_order][0]
    #             # pre_order = pt_orders[i - 1]
    #             # pre_node = node_pts[pre_order][0]
    #             # pre_order = pt_orders[i - 1]
    #             # pre_pt = node_pts[pre_order][1]
    #
    #             if along_disances[i] - along_disances[left_order] <= tolerance:
    #                 continue
    #
    #             # if distance(sole_pt, cur_pt) <= tolerance:
    #             #     sole_pt = cur_pt
    #             #     continue
    #
    #             split_point = node_pts[cur_order][1]
    #             geomColl = split_line_by_point(split_point, res_line).geoms
    #             if len(geomColl) == 1:
    #                 print("debug")
    #             res_line = geomColl[1]
    #             geom = geomColl[0]
    #             left_order = i
    #
    #         if geom.length < tolerance:
    #             print("debug")
    #
    #         G.add_node(node_pts[cur_order][0], **{
    #             'id': node_pts[cur_order][0],
    #             'y': node_pts[cur_order][1].y,
    #             'x': node_pts[cur_order][1].x,
    #             'bOrigin': False})
    #
    #         G.add_edge(cur_node, next_node,
    #                    **{**edge_attrs,
    #                       'geometry': geom,
    #                       'length': geom.length})
    #
    #         if len(pt_orders) == 1:  # 只有一个点的时候，要再加上一条edge，从split点到末端点
    #             cur_order = pt_orders[-1]
    #             cur_node = node_pts[cur_order][0]
    #             next_node = eid[1]
    #             G.add_edge(cur_node, next_node,
    #                        **{**edge_attrs,
    #                           'geometry': res_line,
    #                           'length': res_line.length})
    #
    #     G.remove_edge(eid[0], eid[1], eid[2])  # MultiDiGraph删除边要注意加上key，否则有可能把所有边都删除

    # nodes, edges = graph_to_gdfs(G)
    return G


def _split_edges_by_point(G, eid, o_max, split_point):
    line = G.edges[eid]['geometry']
    edge_attrs = G.edges[eid]

    geomColl = split_line_by_point(split_point, line).geoms

    if len(geomColl) == 1:
        print("debug")

    node_id = o_max + 1
    G.add_node(node_id, **{
        'id': node_id,
        'y': split_point.y,
        'x': split_point.x,
        'bOrigin': False})

    G.add_edge(eid[0], node_id,
               **{**edge_attrs,
                  'geometry': geomColl[0],
                  'length': geomColl[0].length})

    G.add_edge(node_id, eid[1],
               **{**edge_attrs,
                  'geometry': geomColl[1],
                  'length': geomColl[1].length})

    G.remove_edge(eid[0], eid[1], eid[2])

    if eid[0] != eid[1]:  # MultiDiGraph有可能出现loop，这种排除， 例如(2074,2074,0)
        # 判断对向边
        if eid[1] in G:
            if eid[0] in G[eid[1]]:
                r_eids = G[eid[1]][eid[0]]

                # 这里需要注意: 找对向边有可能遇到多重边的情况，所以只有图形相同的才能视为对向边
                for key, v in r_eids.items():
                    if v['geometry'].equals(line):
                        r_eid = (eid[1], eid[0], key)
                        if r_eid in G.edges:
                            G.add_edge(eid[1], node_id,
                                       **{**edge_attrs,
                                          'geometry': geomColl[1],
                                          'length': geomColl[1].length})

                            G.add_edge(node_id, eid[0],
                                       **{**edge_attrs,
                                          'geometry': geomColl[0],
                                          'length': geomColl[0].length})

                            G.remove_edge(eid[1], eid[0], key)
                            break

    return G, node_id


def makeGraph2(G: Graph, additionalPoints, o_max=-1, rtree=None, distance_tolerance=500):
    lst = [-1] * len(additionalPoints)
    node_ids = np.array(lst)  # 将additionalPoints转换为node的label id，用于后续路径计算

    G = G.copy()

    edge_geoms = list(nx.get_edge_attributes(G, "geometry").values())
    edge_ids = list(G.edges)

    segments = []
    if rtree is None:
        log.debug("对edges构建rtree, 计算设施的最近邻edge...")
        #  把所有segments都放入rtree
        rtree = index.Index()
        i = 0
        for iedge, edge in enumerate(edge_geoms):
            iseg_index = 0
            for u, v in zip(edge.coords[:-1], edge.coords[1:]):
                box = (min(u[0], v[0]), min(u[1], v[1]), max(u[0], v[0]), max(u[1], v[1]))
                rtree.insert(i, box, obj=(iedge, iseg_index))
                segments.append(LineString([u, v]))
                iseg_index += 1
                i += 1

    # pos = rtree.query_nearest(additionalPoints, all_matches=False, return_distance=False,
    #                           max_distance=distance_tolerance)

    pos = [list(rtree.nearest((pt.x, pt.y), 1, objects=True)) for pt in additionalPoints]

    if o_max == -1:
        # o_max = max(nodes['id'])  # 从最大序号开始为新增split点编号
        o_max = max(G.nodes)

    log.debug("计算edge上所有的附着点...")
    points_along_edge = {}  # 存储需要打断的edge上附着的vertex点

    for i in mTqdm(range(len(additionalPoints))):
        ne_segments = pos[i]
        additionalPoint = additionalPoints[i]

        if len(ne_segments) != 2:
            print("!=2")

        min_distance = math.inf
        min_n = -1
        for n, seg in enumerate(ne_segments):
            iseg = seg.id  # segments总序号
            segement = segments[iseg]
            project_dis = distance(segement, additionalPoint)

            if project_dis < min_distance:
                min_distance = project_dis
                min_n = n

        if min_distance > distance_tolerance:
            continue

        seg = ne_segments[min_n]
        iedge = seg.object[0]  # edge序号
        iseg_index = seg.object[1]  # segment在所属edge的序号
        iseg = seg.id  # segments总序号
        eid = edge_ids[iedge]
        segement = segments[iseg]
        line = edge_geoms[iedge]

        num_vertexes = get_num_points(line)
        if iseg_index == 0 or iseg_index + 1 == num_vertexes:
            node_id = -1
            if not is_projectPoint_in_segment(additionalPoint, segement):
                # 如果当前点不在线段上，则肯定是附着到edge的端点，不需要记录vertex点
                if iseg_index == 0:
                    node_id = eid[0]
                elif iseg_index + 1 == num_vertexes:
                    node_id = eid[1]

                node_ids[i] = node_id
                continue

        # eid = edge_ids[edge_order]
        # line = edge_geoms[edge_order]

        # if eid[0] == 1870 and eid[1] == 7350:
        #     print("debug")

        # edge_attrs = G.edges[eid]
        # line = edge_attrs['geometry']

        # nearest_vertex(additionalPoints[point_order], line)

        # ne = nearest_points(line, additionalPoints[pos[0][i]])[0]

        # is_endpoint, flag = _is_endpoint_of_edge(ne, edges.loc[eid]['geometry'])
        # is_endpoint, flag = _is_endpoint_of_edge(ne, edge_attrs['geometry'])

        # if not is_endpoint:
        #     # 加入split点，更新图
        #     new_G, o_max = _split_edges_by_point(G, eid, o_max, ne)
        #     node_ids[pos[0][i]] = o_max
        # else:
        #     node_id = eid[1] if flag == "s" else eid[0]
        #     node_ids[pos[0][i]] = int(node_id)

    return G, list(node_ids)


class GraphTransfer:
    def __init__(self, G, df):
        self.G = G
        self.df = df

    def task(self):
        return self.G, self.df


if __name__ == '__main__':
    create_graph_from_file(r"D:\空间模拟\PublicSupplyDemand\Data\mini_graph.shp",
                           direction_field="oneway",
                           forwardValue='F',
                           backwardValue='T',
                           bothValue='B')
