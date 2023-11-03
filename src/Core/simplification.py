"""Simplify, correct, and consolidate network topology."""

import logging as lg

import networkx as nx
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

# from . import stats
# from . import utils
# from . import utils_graph
from ._errors import GraphSimplificationError
from .log4p import Log

log = Log(__name__)


def _is_endpoint(G, node, strict=True):
    """
    Is node a true endpoint of an edge.

    Return True if the node is a "real" endpoint of an edge in the network,
    otherwise False. OSM data includes lots of nodes that exist only as points
    to help streets bend around curves. An end point is a node that either:
    1) is its own neighbor, ie, it self-loops.
    2) or, has no incoming edges or no outgoing edges, ie, all its incident
    edges point inward or all its incident edges point outward.
    3) or, it does not have exactly two neighbors and degree of 2 or 4.
    4) or, if strict mode is false, if its edges have different OSM IDs.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    node : int
        the node to examine
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs

    Returns
    -------
    bool
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    if G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    if not ((n == 2) and (d in {2, 4})):  # noqa: PLR2004
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    # rule 4
    if not strict:
        # non-strict mode: do its incident edges have different OSM IDs?
        # first collect all the OSM way IDs for incoming edges
        # then collect all the OSM way IDs for outgoing edges
        # if there is more than 1 OSM ID then it is an endpoint, otherwise not
        incoming = [G.edges[u, node, k]["osmid"] for u in G.predecessors(node) for k in G[u][node]]
        outgoing = [G.edges[node, v, k]["osmid"] for v in G.successors(node) for k in G[node][v]]
        return len(set(incoming + outgoing)) > 1

    # if none of the preceding rules passed, then it is not an endpoint
    return False


def _build_path(G, endpoint, endpoint_successor, endpoints):
    """
    Build a path of nodes from one endpoint node to next endpoint node.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    endpoint : int
        the endpoint node from which to start the path
    endpoint_successor : int
        the successor of endpoint through which the path to the next endpoint
        will be built
    endpoints : set
        the set of all nodes in the graph that are endpoints

    Returns
    -------
    path : list
        the first and last items in the resulting path list are endpoint
        nodes, and all other items are interstitial nodes that can be removed
        subsequently
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for this_successor in G.successors(endpoint_successor):
        successor = this_successor
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]

                    # otherwise, this can happen due to OSM digitization error
                    # where a one-way street turns into a two-way here, but
                    # duplicate incoming one-way edges are present
                    msg = f"Unexpected simplify pattern handled near {successor}"
                    log.warning(msg)
                    return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    msg = f"Unexpected simplify pattern failed near {successor}"
                    raise GraphSimplificationError(msg)

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path


def _get_paths_to_simplify(G, strict=True):
    """
    Generate all the paths to be simplified between endpoint nodes.

    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs

    Yields
    ------
    path_to_simplify : list
        a generator of paths to simplify
    """
    # first identify all the nodes that are endpoints
    endpoints = {n for n in G.nodes if _is_endpoint(G, n, strict=strict)}
    log.debug("识别出{}个端点".format(len(endpoints)))

    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints)


def _remove_rings(G):
    """
    Remove self-contained rings from graph.

    This identifies any connected components that form a self-contained ring
    without any endpoints, and removes them from the graph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph

    Returns
    -------
    G : networkx.MultiDiGraph
        graph with self-contained rings removed
    """
    nodes_in_rings = set()
    for wcc in nx.weakly_connected_components(G):
        if not any(_is_endpoint(G, n) for n in wcc):
            nodes_in_rings.update(wcc)
    G.remove_nodes_from(nodes_in_rings)
    return G


def simplify_graph(G, strict=True, remove_rings=True, track_merged=False):
    """
    Simplify a graph's topology by removing interstitial nodes.

    Simplifies graph topology by removing all nodes that are not intersections
    or dead-ends. Create an edge directly between the end points that
    encapsulate them, but retain the geometry of the original edges, saved as
    a new `geometry` attribute on the new edge. Note that only simplified
    edges receive a `geometry` attribute. Some of the resulting consolidated
    edges may comprise multiple OSM ways, and if so, their multiple attribute
    values are stored as a list. Optionally, the simplified edges can receive
    a `merged_edges` attribute that contains a list of all the (u, v) node
    pairs that were merged together.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have incident edges with different OSM IDs. Lets you keep
        nodes at elbow two-way intersections, but sometimes individual blocks
        have multiple OSM IDs within them too.
    remove_rings : bool
        if True, remove isolated self-contained rings that have no endpoints
    track_merged : bool
        if True, add `merged_edges` attribute on simplified edges, containing
        a list of all the (u, v) node pairs that were merged together

    Returns
    -------
    G : networkx.MultiDiGraph
        topologically simplified graph, with a new `geometry` attribute on
        each simplified edge
    """
    if "simplified" in G.graph and G.graph["simplified"]:  # pragma: no cover
        msg = "This graph has already been simplified, cannot simplify it again."
        raise GraphSimplificationError(msg)

    log.info("开始拓扑简化图G...")

    # define edge segment attributes to sum upon edge simplification
    attrs_to_sum = {"length", "travel_time"}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, strict=strict):
        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        merged_edges = []
        path_attributes = {}
        for u, v in zip(path[:-1], path[1:]):
            if track_merged:
                # keep track of the edges that were merged
                merged_edges.append((u, v))

            # there should rarely be multiple edges between interstitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)
            edge_count = G.number_of_edges(u, v)
            if edge_count != 1:
                log.debug("简化过程中识别出节点{}到节点{}之间存在{}条边".format(edge_count, u, v))

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            # We can't assume that there exists an edge from u to v
            # with key=0, so we get a list of all edges from u to v
            # and just take the first one.
            edge_data = list(G.get_edge_data(u, v).values())[0]
            for attr in edge_data:
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr in path_attributes:
            if attr in attrs_to_sum:
                # if this attribute must be summed, sum it now
                path_attributes[attr] = sum(path_attributes[attr])
            elif len(set(path_attributes[attr])) == 1:
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th):
                path_attributes[attr] = path_attributes[attr][0]
            else:
                # otherwise, if there are multiple values, keep one of each
                path_attributes[attr] = list(set(path_attributes[attr]))

        # construct the new consolidated edge's geometry for this path
        path_attributes["geometry"] = LineString(
            [Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in path]
        )

        if track_merged:
            # add the merged edges as a new attribute of the simplified edge
            path_attributes["merged_edges"] = merged_edges

        # add the nodes and edge to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes}
        )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination

    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        G = _remove_rings(G)

    # mark graph as having been simplified
    G.graph["simplified"] = True

    # msg = (
    #     f"Simplified graph: {initial_node_count:,} to {len(G):,} nodes, "
    #     f"{initial_edge_count:,} to {len(G.edges):,} edges"
    # )
    log.info("简化后的图G: 节点从{}个简化为{}个，边从{}条简化为{}条".format(initial_node_count, len(G),
                                                             initial_edge_count, len(G.edges)))
    return G
