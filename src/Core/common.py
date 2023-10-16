import os

from future.moves import sys
from osgeo.ogr import Layer, GeometryTypeToName, FieldDefn, Feature, Geometry
from osgeo import ogr
from qgis._analysis import QgsGraphBuilder
from qgis._core import QgsWkbTypes, QgsPointXY, QgsVectorLayer, QgsField, QgsGeometry, QgsFeatureIterator, \
    QgsSpatialIndex
from shapely import Point
import pandas as pd


def set_main_path(path):
    global g_main_path
    g_main_path = path


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def center_points(layer: QgsVectorLayer):
    features = layer.getFeatures()

    center_pts = []

    for current, f in enumerate(features):
        if not f.hasGeometry():
            continue

        geom = f.geometry()
        center_pt = geom.pointOnSurface()
        center_pts.append(center_pt)

    return center_pts


def clear_beyond_point(mNetwork, mDirector, layer, points, point_lst, distance_tolerance):
    mBuilder = QgsGraphBuilder(mNetwork.crs())
    snappedPoints = mDirector.makeGraph(mBuilder, points)

    Points_clear = []
    point_clear_lst = []
    Features = {}  # FID: order
    i = 0
    for idx, pt in enumerate(points):
        if snappedPoints[idx].distance(pt) <= distance_tolerance:
            Points_clear.append(points[idx])
            point_clear_lst.append(point_lst[idx])
            Features[point_lst[idx]] = {
                'feature': layer.getFeature(point_lst[idx]),
                'id': i
            }
            i += 1

    return Points_clear, point_clear_lst, Features


def clear_beyond_point2(layer, index: QgsSpatialIndex, points, point_lst, distance_tolerance):
    Points_clear = []
    point_clear_lst = []
    Features = {}  # FID: order
    i = 0
    for idx, pt in enumerate(points):
        lst = index.nearestNeighbor(pt, neighbors=1, maxDistance=distance_tolerance)
        if len(lst) > 0:
        # if snappedPoints[idx].distance(pt) <= distance_tolerance:
            Points_clear.append(points[idx])
            point_clear_lst.append(point_lst[idx])
            Features[point_lst[idx]] = {
                'feature': layer.getFeature(point_lst[idx]),
                'id': i
            }
            i += 1

    return Points_clear, point_clear_lst, Features


def get_centerPoints(features: QgsFeatureIterator, feature_type):
    Points = []
    points_lst = []  # 用于记录原始feature id和筛选后 序号的对应关系
    # source_attributes = {}
    i = 0

    # features = layer.getFeatures()
    for current, f in enumerate(features):
        if not f.hasGeometry():
            continue

        geom: QgsGeometry = f.geometry()

        if feature_type == QgsWkbTypes.Type.Polygon or feature_type == QgsWkbTypes.Type.MultiPolygon:
            center_pt = geom.pointOnSurface()
            if not center_pt.isNull():
                Points.append(center_pt.asPoint())
                points_lst.append(f.id())
            # source_attributes[i] = f.attributes()
            # org_ids[i] = len(Points) - 1
        elif feature_type == QgsWkbTypes.Type.Point:
            Points.append(geom.asPoint())
            points_lst.append(f.id())

        elif feature_type == QgsWkbTypes.Type.MultiPoint:
            x = 0
            y = 0
            c = 0
            for p in geom.vertices():
                x = x + QgsPointXY(p).x()
                y = y + QgsPointXY(p).y()
                c += 1

            Points.append(QgsPointXY(x / c, y / c))
            points_lst.append(f.id())

        # points_dict[f.id()] = i
        i += 1

    return Points, points_lst


def check_line_type(layer: Layer):
    layer_type = layer.GetGeomType()

    if layer_type == 2 or layer_type == 5:
        return True
    else:
        return False


def check_geom_type(layer: Layer):
    layer_type = layer.GetGeomType()

    if layer_type == 3 or layer_type == 6 or layer_type == 1 or layer_type == 4:
        return True
    else:
        return False


def check_field_type(field: FieldDefn):
    field_type = field.GetType()
    if field_type == ogr.OFTReal or field_type == ogr.OFTInteger or field_type == ogr.OFTInteger64:
        return True
    else:
        return False


def launderName(name):
    dir = os.path.dirname(name)
    basename, suffix = os.path.splitext(name)
    if os.path.exists(name):
        basename = basename + "_1"
        name = os.path.join(dir, basename + suffix)

    if not (os.path.exists(name)):
        return name
    else:
        return launderName(name)


def get_centerPoints(layer):
    Points = []
    points_lst = []  # 用于记录原始feature id和筛选后 序号的对应关系
    i = 0

    layer.ResetReading()
    for feature in layer:
        geom: Geometry = feature.GetGeometryRef()
        feature_type = geom.GetGeometryType()

        if feature_type == ogr.wkbPolygon or feature_type == ogr.wkbMultiPolygon:
            center_pt = geom.PointOnSurface()
            if not center_pt.IsEmpty():
                Points.append(Point([center_pt.GetX(), center_pt.GetY()]))
                points_lst.append(feature.GetFID())
        elif feature_type == ogr.wkbPoint:
            Points.append(Point([geom.GetX(), geom.GetY()]))
            points_lst.append(feature.GetFID())
        elif feature_type == ogr.wkbMultiPoint:
            x = 0
            y = 0
            c = 0
            for part in geom:
                pt = part.GetPoint()
                x = x + pt.GetX()
                y = y + pt.GetY()
                c += 1

            Points.append(Point([x / c, y / c]))
            points_lst.append(feature.GetFID())

        # points_dict[f.id()] = i
        i += 1

    # res = zip(Points, points_lst)
    return pd.DataFrame({'fid': points_lst, 'geom': Points})


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton
