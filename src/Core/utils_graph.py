import networkx as nx
import numpy as np
from rtree import index
from shapely import Point, LineString, distance, STRtree
from shapely.ops import nearest_points, split, snap

from Core.log4p import Log

log = Log(__name__)


def get_largest_component(G, strongly=False):
    """
    Get subgraph of G's largest weakly/strongly connected component.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strongly : bool
        if True, return the largest strongly instead of weakly connected
        component

    Returns
    -------
    G : networkx.MultiDiGraph
        the largest connected component subgraph of the original graph
    """
    if strongly:
        kind = "强"
        is_connected = nx.is_strongly_connected
        connected_components = nx.strongly_connected_components
    else:
        kind = "弱"
        is_connected = nx.is_weakly_connected
        connected_components = nx.weakly_connected_components

    if not is_connected(G):
        # get all the connected components in graph then identify the largest
        largest_cc = max(connected_components(G), key=len)
        n = len(G)

        # induce (frozen) subgraph then unfreeze it by making new MultiDiGraph
        G = nx.MultiDiGraph(G.subgraph(largest_cc))
        log.debug("生成最大{}连通图(节点:{}/{})".format(kind, len(G), n))

    return G


def add_edge_lengths(G, edges=None):
    """
    Add `length` attribute (in meters) to each edge.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        unprojected, unsimplified input graph
    edges : tuple
        tuple of (u, v, k) tuples representing subset of edges to add length
        attributes to. if None, add lengths to all edges.

    Returns
    -------
    G : networkx.MultiDiGraph
        graph with edge length attributes
    """
    if edges is None:
        uvk = tuple(G.edges)
    else:
        uvk = edges

    # extract edge IDs and corresponding coordinates from their nodes
    x = G.nodes(data="x")
    y = G.nodes(data="y")
    try:
        # two-dimensional array of coordinates: y0, x0, y1, x1
        c = np.array([(y[u], x[u], y[v], x[v]) for u, v, k in uvk])
        # ensure all coordinates can be converted to float and are non-null
        assert not np.isnan(c.astype(float)).any()
    except (AssertionError, KeyError) as e:  # pragma: no cover
        msg = "some edges missing nodes, possibly due to input data clipping issue"
        raise ValueError(msg) from e

    # calculate great circle distances, round, and fill nulls with zeros
    dists = euclidean_dist_vec(c[:, 0], c[:, 1], c[:, 2], c[:, 3]).round(4)
    dists[np.isnan(dists)] = 0
    nx.set_edge_attributes(G, values=dict(zip(uvk, dists)), name="length")

    log.debug("图G增加长度属性length")
    return G


def split_line_by_point(split_point, line):
    minimum_distance = nearest_points(split_point, line)[1]
    res = split(snap(line, minimum_distance, 0.0001), minimum_distance)

    return res


#  判断一个点是否edge的两端
def _is_endpoint_of_edge(point: Point, edge: LineString, tolerance=5):
    # *_, last_new_1, last_1 = linestring1.coords

    if distance(Point(edge.coords[0]), point) <= tolerance:
        return True, "s"
    else:
        if distance(Point(edge.coords[-1]), point) <= tolerance:
            return True, "e"
        else:
            return False, ""


# 判断当前点在线段的垂足是否在线段上
def is_projectPoint_in_segment(point: Point, segment: LineString):
    a = [point.x, point.y]
    b, c = segment.coords
    t = (a[0]-b[0])*(c[0]-b[0]) + (a[1]-b[1])*(c[1]-b[1])
    t = t / ((c[0]-b[0])**2 + (c[1]-b[1])**2)

    if 0 < t < 1:
        return True
    else:
        return False

def euclidean_dist_vec(y1, x1, y2, x2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
#     """
#     Convert a MultiDiGraph to node and/or edge GeoDataFrames.
#
#     This function is the inverse of `graph_from_gdfs`.
#
#     Parameters
#     ----------
#     G : networkx.MultiDiGraph
#         input graph
#     nodes : bool
#         if True, convert graph nodes to a GeoDataFrame and return it
#     edges : bool
#         if True, convert graph edges to a GeoDataFrame and return it
#     node_geometry : bool
#         if True, create a geometry column from node x and y attributes
#     fill_edge_geometry : bool
#         if True, fill in missing edge geometry fields using nodes u and v
#
#     Returns
#     -------
#     geopandas.GeoDataFrame or tuple
#         gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes
#         is indexed by osmid and gdf_edges is multi-indexed by u, v, key
#         following normal MultiDiGraph structure.
#     """
#     crs = G.graph["crs"]
#
#     if nodes:
#         if not G.nodes:  # pragma: no cover
#             msg = "graph contains no nodes"
#             raise ValueError(msg)
#
#         nodes, data = zip(*G.nodes(data=True))
#
#         if node_geometry:
#             # convert node x/y attributes to Points for geometry column
#             for d in data:
#                 if "x" not in d:
#                     print("debug")
#                 else:
#                     geom = (Point(d["x"], d["y"]) for d in data)
#
#             # geom = (Point(d["x"], d["y"]) for d in data)
#             gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
#         else:
#             gdf_nodes = gpd.GeoDataFrame(data, index=nodes)
#
#         gdf_nodes.index.rename("osmid", inplace=True)
#         log.debug("将图G的节点转换为GeoDataFrame对象")
#
#     if edges:
#         if not G.edges:  # pragma: no cover
#             msg = "graph contains no edges"
#             raise ValueError(msg)
#
#         u, v, k, data = zip(*G.edges(keys=True, data=True))
#
#         if fill_edge_geometry:
#             # subroutine to get geometry for every edge: if edge already has
#             # geometry return it, otherwise create it using the incident nodes
#             x_lookup = nx.get_node_attributes(G, "x")
#             y_lookup = nx.get_node_attributes(G, "y")
#
#             def _make_geom(u, v, data, x=x_lookup, y=y_lookup):
#                 if "geometry" in data:
#                     return data["geometry"]
#
#                 # otherwise
#                 return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))
#
#             geom = map(_make_geom, u, v, data)
#             gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))
#
#         else:
#             gdf_edges = gpd.GeoDataFrame(data)
#             if "geometry" not in gdf_edges.columns:
#                 # if no edges have a geometry attribute, create null column
#                 gdf_edges = gdf_edges.set_geometry([None] * len(gdf_edges))
#             gdf_edges = gdf_edges.set_crs(crs)
#
#         # add u, v, key attributes as index
#         gdf_edges["u"] = u
#         gdf_edges["v"] = v
#         gdf_edges["key"] = k
#         gdf_edges.set_index(["u", "v", "key"], inplace=True)
#
#         log.debug("将图G的边转换为GeoDataFrame对象")
#
#     if nodes and edges:
#         return gdf_nodes, gdf_edges
#
#     if nodes:
#         return gdf_nodes
#
#     if edges:
#         return gdf_edges
#
#     # otherwise
#     msg = "you must request nodes or edges or both"
#     raise ValueError(msg)
