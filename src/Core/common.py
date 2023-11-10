import os

import sys
from osgeo.ogr import Layer, GeometryTypeToName, FieldDefn, Feature, Geometry
from osgeo import ogr
from shapely import Point
import pandas as pd


def set_main_path(path):
    global g_main_path
    g_main_path = path


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


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
    res = pd.DataFrame({'fid': points_lst, 'geom': Points})
    res.set_index('fid')
    return res


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton
