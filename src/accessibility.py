import os.path
import traceback

from PyQt5.QtCore import QVariant
from qgis._analysis import QgsVectorLayerDirector, QgsNetworkDistanceStrategy, QgsGraphBuilder, QgsGraphAnalyzer
from qgis._core import QgsPointXY, QgsVectorLayer, QgsGeometry, QgsProject, QgsVectorFileWriter, QgsWkbTypes, QgsFields, \
    QgsFeature, QgsGeometryUtils, QgsField, QgsSpatialIndex, QgsApplication

from Core import mSave_Option, create_writer, DataType, export_db, remove_temp_db
from common import resource_path, set_main_path
from log4p import Log, mTqdm
from progressbar import printProgressBar

log = Log(__name__)


# 计算从点出发的可达范围，输出包括可达点(service_nodes)、可达边(service_lines)、可达路径(service_routes)、可达范围(cover_area)
def accessible_from_layer(input_network_data,
                         input_facility_data,
                         out_path="",
                         travelCost=0.0,
                         direction_field="",
                         forwardValue="F",
                         backwardValue="T",
                         bothValue="B",
                         include_bounds=False,
                         concave_hull_threshold=0.3,
                         defaultDirection=QgsVectorLayerDirector.Direction.DirectionBoth,
                         out_type=0):

    log.info("读取网络数据...")
    mNetwork = QgsVectorLayer(input_network_data, "road_network")
    # network_index = QgsSpatialIndex(mNetwork.getFeatures())

    log.info("读取设施点数据...")
    mPoints = QgsVectorLayer(input_facility_data, "facilities")
    startPoints = []
    source_attributes = {}
    i = 0

    features = mPoints.getFeatures()
    for current, f in enumerate(features):
        if not f.hasGeometry():
            continue

        for p in f.geometry().vertices():
            startPoints.append(QgsPointXY(p))
            source_attributes[i] = f.attributes()
            i += 1

    #  测试用单点
    startPoints = [QgsPointXY(520096, 2506194)]

    log.info("构建图结构...")

    directionField = mNetwork.fields().lookupField(direction_field)

    strategy = QgsNetworkDistanceStrategy()
    mDirector = QgsVectorLayerDirector(
        mNetwork, directionField, forwardValue, backwardValue, bothValue, defaultDirection)

    mDirector.addStrategy(strategy)

    mBuilder = QgsGraphBuilder(mNetwork.crs())
    snappedPoints = mDirector.makeGraph(mBuilder, startPoints)

    #  清理掉距离节点太远的点
    startPoints_clear = []
    for idx, startPt in enumerate(startPoints):
        if snappedPoints[idx].distance(startPt) <= travelCost:
            startPoints_clear.append(startPoints[idx])

    mBuilder = QgsGraphBuilder(mNetwork.crs())
    snappedPoints_clear = mDirector.makeGraph(mBuilder, startPoints_clear)

    graph = mBuilder.takeGraph()
    geoms = accessibility(graph, startPoints_clear, snappedPoints_clear, travelCost, include_bounds, concave_hull_threshold)

    main_path = resource_path("")
    set_main_path(main_path)
    out_path = os.path.join(main_path, "res")

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    log.info("导出结果数据...")

    fields = mPoints.fields()
    bFlag = export_to_file(geoms, fields, mPoints.crs(), source_attributes, out_path, out_type)

    if bFlag:
        log.info("所有步骤完成，结果保存至{}".format(out_path))
    else:
        log.error("计算发生错误，请检查日志文件.")


def accessibility(graph,
                  startPoints,
                  snappedPoints,
                  travelCost=0.0,
                  include_bounds=False,
                  concave_hull_threshold=0.3):
    result = {}

    total = len(snappedPoints)
    # printProgressBar(0, total)

    for i, p in enumerate(mTqdm(snappedPoints)):
        idxStart = graph.findVertex(snappedPoints[i])
        tree, costs = QgsGraphAnalyzer.dijkstra(graph, idxStart, 0)

        vertices = set()
        area_points = []  # 存储可达范围的节点(concave hull算法)
        lines = []
        routes = []  # 存储从出发点到可达边缘点的路径
        interpolated_end = {}

        for vertex, start_vertex_cost in enumerate(costs):
            inbound_edge_index = tree[vertex]
            if inbound_edge_index == -1 and vertex != idxStart:
                # 无法到达的vertex
                continue

            if start_vertex_cost > travelCost:
                # cost太大，放弃
                continue

            vertices.add(vertex)
            start_point = graph.vertex(vertex).point()

            # 遍历从这个vertex出发的所有edge
            for edge_id in graph.vertex(vertex).outgoingEdges():
                edge = graph.edge(edge_id)
                end_vertex_cost = start_vertex_cost + edge.cost(0)
                end_point = graph.vertex(edge.toVertex()).point()
                if end_vertex_cost <= travelCost:
                    # end vertex满足cost条件，纳入结果
                    vertices.add(edge.toVertex())
                    lines.append([start_point, end_point])
                else:
                    # travelCost在edge的中间位置, 进行位置插补计算
                    interpolated_end_point = QgsGeometryUtils.interpolatePointOnLineByValue(
                        start_point.x(),
                        start_point.y(), start_vertex_cost, end_point.x(),
                        end_point.y(), end_vertex_cost, travelCost)
                    area_points.append(interpolated_end_point)

                    # 把中间点所在的edge记录下来，以便后续回溯route使用
                    interpolated_end[(interpolated_end_point.x(), interpolated_end_point.y())] = edge_id
                    lines.append([start_point, interpolated_end_point])

        for v in vertices:
            area_points.append(graph.vertex(v).point())
            # area_within_points.append(graph.vertex(v).point())

        geom_within = QgsGeometry.fromMultiPointXY(area_points)
        cover_area = geom_within.concaveHull(concave_hull_threshold, False)

        # geom_interior = QgsGeometry.fromMultiPointXY(area_within_points)  # 内部vertex点，不考虑插补出来的中间点
        # cover_interior = geom_interior.concaveHull(concave_hull_threshold, False)

        if len(area_points) < 3:
            geom_within = None
            cover_area = None
        else:
            for pt in cover_area.vertices():
                pt = QgsPointXY(pt.x(), pt.y())
                eIdx = graph.findVertex(pt)

                if eIdx == -1:  # 走到了最远edge的中间就达到了travelCost
                    route = [pt]
                    end_edge = interpolated_end[(pt.x(), pt.y())]
                    eIdx = graph.edge(end_edge).fromVertex()
                    route.append(graph.vertex(eIdx).point())
                else:
                    route = [graph.vertex(eIdx).point()]

                # cost = costs[eIdx]
                current = eIdx
                while current != idxStart:  # 回溯最短路径树到出发点
                    current = graph.edge(tree[current]).fromVertex()
                    route.append(graph.vertex(current).point())

                route.reverse()

                if len(route) > 1:
                    routes.append(route)

        geom_upper = None
        geom_lower = None
        if include_bounds:
            upperBoundary = []
            lowerBoundary = []

            vertices = []
            for vertex, c in enumerate(costs):
                if c > travelCost and tree[vertex] != -1:
                    vertexId = graph.edge(tree[vertex]).fromVertex()
                    if costs[vertexId] <= travelCost:
                        vertices.append(vertex)

            for v in vertices:
                upperBoundary.append(
                    graph.vertex(graph.edge(
                        tree[v]).toVertex()).point())
                lowerBoundary.append(
                    graph.vertex(graph.edge(
                        tree[v]).fromVertex()).point())

            geom_upper = QgsGeometry.fromMultiPointXY(upperBoundary)
            geom_lower = QgsGeometry.fromMultiPointXY(lowerBoundary)

        geom_lines = QgsGeometry.fromMultiPolylineXY(lines)
        geom_routes = QgsGeometry.fromMultiPolylineXY(routes)

        if geom_lines.isEmpty():
            geom_lines = None
        if geom_routes.isEmpty():
            geom_routes = None

        result[i] = {
            "org_pt": startPoints[i],
            "within": geom_within,
            "upper": geom_upper,
            "lower": geom_lower,
            "lines": geom_lines,
            "routes": geom_routes,
            "c_area": cover_area
        }

        # printProgressBar(i + 1, total)

    # print("OK")
    return result


def export_to_file(geoms, fields, crs, source_attributes, out_path, out_type):
    fields.append(QgsField('type', QVariant.String, '', 254, 0))
    fields.append(QgsField('startID', QVariant.String, '', 254, 0))
    fields.append(QgsField('startPos', QVariant.String, '', 254, 0))

    writer_area = None
    writer_nodes = None
    writer_lines = None
    writer_routes = None

    out_file_area = None
    out_file_nodes = None
    out_file_lines = None
    out_file_routes = None

    transform_context = QgsProject.instance().transformContext()

    save_options_area = mSave_Option(out_type)
    save_options_area.layerName = "service_area"

    save_options_nodes = mSave_Option(out_type)
    save_options_nodes.layerName = "service_nodes"

    save_options_lines = mSave_Option(out_type)
    save_options_lines.layerName = "service_lines"

    save_options_routes = mSave_Option(out_type)
    save_options_routes.layerName = "service_routes"

    try:
        if out_type == DataType.shapefile.value:
            out_file_area = os.path.join(out_path, "service_area.shp")
            out_file_nodes = os.path.join(out_path, "service_nodes.shp")
            out_file_lines = os.path.join(out_path, "service_lines.shp")
            out_file_routes = os.path.join(out_path, "service_routes.shp")
        elif out_type == DataType.sqlite.value:
            out_file_area = os.path.join(out_path, "service_area.sqlite")
            out_file_nodes = os.path.join(out_path, "service_nodes.sqlite")
            out_file_lines = os.path.join(out_path, "service_lines.sqlite")
            out_file_routes = os.path.join(out_path, "service_routes.sqlite")

        Save_Options = mSave_Option(out_type)
        Save_Options.layerName = "service_area"
        save_options_area = Save_Options

        writer_area = create_writer(out_file_area, "service_area", fields, QgsWkbTypes.Type.Polygon, crs,
                                      transform_context, save_options_area)

        Save_Options = mSave_Option(out_type)
        Save_Options.layerName = "service_nodes"
        save_options_nodes = Save_Options

        writer_nodes = create_writer(out_file_nodes, "service_nodes", fields, QgsWkbTypes.Type.MultiPoint, crs,
                                    transform_context, save_options_nodes)


        Save_Options = mSave_Option(out_type)
        Save_Options.layerName = "service_lines"
        save_options_lines = Save_Options

        writer_lines = create_writer(out_file_lines, "service_lines", fields, QgsWkbTypes.Type.MultiLineString, crs,
                                     transform_context, save_options_lines)

        Save_Options = mSave_Option(out_type)
        Save_Options.layerName = "service_routes"
        save_options_routes = Save_Options

        writer_routes = create_writer(out_file_routes, "service_routes", fields, QgsWkbTypes.Type.MultiLineString, crs,
                                     transform_context, save_options_routes)

        # if out_file_area is not None:
        #     writer_area = create_writer(out_file_area, "service_area", fields, QgsWkbTypes.Type.Polygon, crs,
        #                                 transform_context, save_options_area)
        #
        # if out_file_nodes is not None:
        #     writer_nodes = create_writer(out_file_nodes, "service_nodes", fields, QgsWkbTypes.Type.MultiPoint, crs,
        #                                  transform_context, save_options_nodes)
        #
        # if out_file_lines is not None:
        #     writer_lines = create_writer(out_file_lines, "service_lines", fields, QgsWkbTypes.Type.MultiLineString, crs,
        #                                  transform_context, save_options_lines)
        #
        # if out_file_routes is not None:
        #     writer_routes = create_writer(out_file_routes, "service_routes", fields, QgsWkbTypes.Type.MultiLineString,
        #                                   crs,
        #                                   transform_context, save_options_routes)

        for key, geom in geoms.items():
            origPoint = geom["org_pt"].toString()

            #  导出cover_area
            if geom["c_area"] is not None and writer_area is not None:
                pFeature = QgsFeature(fields)
                pFeature.setGeometry(geom['c_area'])
                attrs = source_attributes[key]
                attrs = attrs + ['service_area', key, origPoint]
                pFeature.setAttributes(attrs)
                writer_area.addFeature(pFeature)

            # 导出service_nodes
            if writer_nodes is not None:
                pFeature = QgsFeature(fields)
                attrs = source_attributes[key]
                if geom["within"] is not None:
                    pFeature.setGeometry(geom['within'])
                    attrs = attrs + ['within', key, origPoint]
                if geom["upper"] is not None:
                    pFeature.setGeometry(geom['upper'])
                    attrs = attrs + ['upper', key, origPoint]
                if geom["lower"] is not None:
                    pFeature.setGeometry(geom['lower'])
                    attrs = attrs + ['lower', key, origPoint]

                pFeature.setAttributes(attrs)
                writer_nodes.addFeature(pFeature)

            # 导出service_lines
            if writer_lines is not None:
                if geom['lines'] is not None:
                    pFeature = QgsFeature(fields)
                    pFeature.setGeometry(geom['lines'])
                    attrs = source_attributes[key]
                    attrs = attrs + ['lines', key, origPoint]
                    # attrs.extend(['lines', key, origPoint])
                    pFeature.setAttributes(attrs)
                    writer_lines.addFeature(pFeature)

            # 导出service_routes
            if writer_routes is not None:
                if geom['routes'] is not None:
                    pFeature = QgsFeature(fields)
                    pFeature.setGeometry(geom['routes'])
                    attrs = source_attributes[key]
                    attrs = attrs + ['routes', key, origPoint]
                    # attrs.extend(['routes', key, origPoint])
                    pFeature.setAttributes(attrs)
                    writer_routes.addFeature(pFeature)

        del writer_area
        del writer_nodes
        del writer_lines
        del writer_routes

        if out_type == DataType.sqlite.value:
            out_file = os.path.join(out_path, "service_db.sqlite")

            url = "{}|layername={}".format(out_file_area, "service_area")
            lyr1 = QgsVectorLayer(url, "service_area", 'ogr')
            lyr1.setProviderEncoding('utf-8')
            if lyr1.isValid():
                export_db(lyr1.clone(), out_file, transform_context, save_options_area)

            url = "{}|layername={}".format(out_file_nodes, "service_nodes")
            lyr2 = QgsVectorLayer(url, "service_nodes", 'ogr')
            lyr2.setProviderEncoding('utf-8')
            if lyr2.isValid():
                export_db(lyr2.clone(), out_file, transform_context, save_options_nodes)

            url = "{}|layername={}".format(out_file_lines, "service_lines")
            lyr3 = QgsVectorLayer(url, "service_lines", 'ogr')
            lyr3.setProviderEncoding('utf-8')
            if lyr3.isValid():
                export_db(lyr3.clone(), out_file, transform_context, save_options_lines)

            url = "{}|layername={}".format(out_file_routes, "service_routes")
            lyr4 = QgsVectorLayer(url, "service_routes", 'ogr')
            lyr4.setProviderEncoding('utf-8')
            if lyr4.isValid():
                export_db(lyr4.clone(), out_file, transform_context, save_options_routes)

            del lyr1
            del lyr2
            del lyr3
            del lyr4

            remove_temp_db(out_file_area)
            remove_temp_db(out_file_nodes)
            remove_temp_db(out_file_lines)
            remove_temp_db(out_file_routes)

        return True
    except:
        log.error(traceback.format_exc())
        return False
    finally:
        if 'writer_area' in locals():
            del writer_area
        if 'writer_nodes' in locals():
            del writer_nodes
        if 'writer_lines' in locals():
            del writer_lines
        if 'writer_routes' in locals():
            del writer_routes


if __name__ == '__main__':
    # startPt = QgsPointXY(520096, 2506194)
    # startPoints = []
    # startPoints.append(startPt)

    QgsApplication.setPrefixPath('', True)
    app = QgsApplication([], True)
    app.initQgis()

    accessible_from_layer(r"D:\空间模拟\PublicSupplyDemand\Data\sz_road_cgcs2000_test.shp",
                             r"D:\空间模拟\PublicSupplyDemand\Data\2022年现状幼儿园.shp",
                             travelCost=1000,
                             out_type=0,
                             direction_field="oneway")

    # mPoints = QgsVectorLayer(r"D:\空间模拟\PublicSupplyDemand\Data\2022年现状幼儿园.shp", "facilities")
    # points = []
    # source_attributes = {}
    # i = 0
    #
    # features = mPoints.getFeatures()
    # for current, f in enumerate(features):
    #     if not f.hasGeometry():
    #         continue
    #
    #     for p in f.geometry().vertices():
    #         points.append(QgsPointXY(p))
    #         source_attributes[i] = f.attributes()
    #         i += 1
    #
    # allocate(r"D:\空间模拟\PublicSupplyDemand\Data\sz_road_cgcs2000_test.shp",
    #          points, direction_field="oneway")
