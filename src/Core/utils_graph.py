import networkx as nx
import numpy as np
from numba import njit
from shapely import Point, LineString, distance, STRtree
from shapely.ops import nearest_points, split, snap

from warnings import warn
from Core.log4p import Log
import geopandas as gpd
import pandas as pd

log = Log(__name__)

ninety_degrees_rad = 90.0 * np.pi / 180.0


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


def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
    Convert a MultiDiGraph to node and/or edge GeoDataFrames.

    This function is the inverse of `graph_from_gdfs`.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y attributes
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using nodes u and v

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes
        is indexed by osmid and gdf_edges is multi-indexed by u, v, key
        following normal MultiDiGraph structure.
    """
    crs = G.graph["crs"]

    if nodes:
        if not G.nodes:  # pragma: no cover
            msg = "graph contains no nodes"
            raise ValueError(msg)

        nodes, data = zip(*G.nodes(data=True))

        if node_geometry:
            # convert node x/y attributes to Points for geometry column
            geom = (Point(d["x"], d["y"]) for d in data)
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
        else:
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes)

        gdf_nodes.index.rename("osmid", inplace=True)
        log.info("Created nodes GeoDataFrame from graph")

    if edges:
        if not G.edges:  # pragma: no cover
            msg = "graph contains no edges"
            raise ValueError(msg)

        u, v, k, data = zip(*G.edges(keys=True, data=True))

        if fill_edge_geometry:
            # subroutine to get geometry for every edge: if edge already has
            # geometry return it, otherwise create it using the incident nodes
            x_lookup = nx.get_node_attributes(G, "x")
            y_lookup = nx.get_node_attributes(G, "y")

            def _make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if "geometry" in data:
                    return data["geometry"]

                # otherwise
                return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))

            geom = map(_make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))

        else:
            gdf_edges = gpd.GeoDataFrame(data)
            if "geometry" not in gdf_edges.columns:
                # if no edges have a geometry attribute, create null column
                gdf_edges = gdf_edges.set_geometry([None] * len(gdf_edges))
            gdf_edges = gdf_edges.set_crs(crs)

        # add u, v, key attributes as index
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        gdf_edges["key"] = k
        gdf_edges.set_index(["u", "v", "key"], inplace=True)

        log.info("Created edges GeoDataFrame from graph")

    if nodes and edges:
        return gdf_nodes, gdf_edges

    if nodes:
        return gdf_nodes

    if edges:
        return gdf_edges

    # otherwise
    msg = "you must request nodes or edges or both"
    raise ValueError(msg)


def graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None):
    """
    Convert node and edge GeoDataFrames to a MultiDiGraph.

    This function is the inverse of `graph_to_gdfs` and is designed to work in
    conjunction with it.

    However, you can convert arbitrary node and edge GeoDataFrames as long as
    1) `gdf_nodes` is uniquely indexed by `osmid`, 2) `gdf_nodes` contains `x`
    and `y` coordinate columns representing node geometries, 3) `gdf_edges` is
    uniquely multi-indexed by `u`, `v`, `key` (following normal MultiDiGraph
    structure). This allows you to load any node/edge shapefiles or GeoPackage
    layers as GeoDataFrames then convert them to a MultiDiGraph for graph
    analysis. Note that any `geometry` attribute on `gdf_nodes` is discarded
    since `x` and `y` provide the necessary node geometry information instead.

    Parameters
    ----------
    gdf_nodes : geopandas.GeoDataFrame
        GeoDataFrame of graph nodes uniquely indexed by osmid
    gdf_edges : geopandas.GeoDataFrame
        GeoDataFrame of graph edges uniquely multi-indexed by u, v, key
    graph_attrs : dict
        the new G.graph attribute dict. if None, use crs from gdf_edges as the
        only graph-level attribute (gdf_edges must have crs attribute set)

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    if not ("x" in gdf_nodes.columns and "y" in gdf_nodes.columns):  # pragma: no cover
        msg = "gdf_nodes must contain x and y columns"
        raise ValueError(msg)

    # if gdf_nodes has a geometry attribute set, drop that column (as we use x
    # and y for geometry information) and warn the user if the geometry values
    # differ from the coordinates in the x and y columns
    if hasattr(gdf_nodes, "geometry"):
        try:
            all_x_match = (gdf_nodes.geometry.x == gdf_nodes["x"]).all()
            all_y_match = (gdf_nodes.geometry.y == gdf_nodes["y"]).all()
            assert all_x_match
            assert all_y_match
        except (AssertionError, ValueError):  # pragma: no cover
            # AssertionError if x/y coords don't match geometry column
            # ValueError if geometry column contains non-point geometry types
            warn(
                "discarding the gdf_nodes geometry column, though its "
                "values differ from the coordinates in the x and y columns",
                stacklevel=2,
            )
        gdf_nodes = gdf_nodes.drop(columns=gdf_nodes.geometry.name)

    # create graph and add graph-level attribute dict
    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}
    G = nx.MultiDiGraph(**graph_attrs)

    # add edges and their attributes to graph, but filter out null attribute
    # values so that edges only get attributes with non-null values
    attr_names = gdf_edges.columns.to_list()
    for (u, v, k), attr_vals in zip(gdf_edges.index, gdf_edges.values):
        data_all = zip(attr_names, attr_vals)
        data = {name: val for name, val in data_all if isinstance(val, list) or pd.notnull(val)}
        G.add_edge(u, v, key=k, **data)

    # add any nodes with no incident edges, since they wouldn't be added above
    G.add_nodes_from(set(gdf_nodes.index) - set(G.nodes))

    # now all nodes are added, so set nodes' attributes
    for col in gdf_nodes.columns:
        nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())

    log.info("Created graph from node/edge GeoDataFrames")
    return G


def angle_between_lines(line1, line2, deg=True):
    coords_1 = line1.coords
    coords_2 = line2.coords

    line1_vertical = (coords_1[1][0] - coords_1[0][0]) == 0.0
    line2_vertical = (coords_2[1][0] - coords_2[0][0]) == 0.0

    # Vertical lines have undefined slope, but we know their angle in rads is = 90° * π/180
    if line1_vertical and line2_vertical:
        # Perpendicular vertical lines
        return 0.0
    if line1_vertical or line2_vertical:
        # 90° - angle of non-vertical line
        non_vertical_line = line2 if line1_vertical else line1

        _arc = abs((90.0 * np.pi / 180.0) - np.arctan(slope(non_vertical_line)))
        if deg:
            return np.degrees(_arc)
        else:
            return _arc

    m1 = slope(line1)
    m2 = slope(line2)

    _arc = np.arctan((m1 - m2)/(1 + m1*m2))

    if deg:
        return np.degrees(_arc)
    else:
        return _arc


def slope(line):
    # Assignments made purely for readability. One could opt to just one-line return them
    x0 = line.coords[0][0]
    y0 = line.coords[0][1]
    x1 = line.coords[1][0]
    y1 = line.coords[1][1]
    return (y1 - y0) / (x1 - x0)


@njit(cache=True, nogil=True)
def angle_between_vectors(vector1, vector2, deg=True, orientation=False):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)

    if orientation:
        minor = np.linalg.det(
            np.stack((v1_u[-2:], v2_u[-2:]))
        )
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
    else:
        sign = 1

    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)

    sign_arc = sign * np.arccos(dot_p)
    if deg:
        return np.degrees(sign_arc)
    else:
        return sign_arc


@njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


@njit(cache=True, nogil=True)
def angle_between_vecters2(v1, v2, orientation=False):
    # print(np.linalg.det(np.stack((v1[-2:], v2[-2:]))))
    # print(np.dot(v1, v2))
    angle = np.math.atan2(np.linalg.det(np.stack((v1[-2:], v2[-2:]))), np.dot(v1, v2))
    if not orientation:
        angle = abs(angle)

    return np.degrees(angle)
