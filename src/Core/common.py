import os

import sys
from pathlib import Path
from typing import Union, List

import colorama
from colorama import Fore
from osgeo.ogr import Layer, GeometryTypeToName, FieldDefn, Feature, Geometry
from osgeo import ogr, gdal
from shapely import Point
import pandas as pd
from tqdm import tqdm

PathLikeOrStr = Union[str, os.PathLike]

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


def get_extension(filename: PathLikeOrStr) -> str:
    """
    returns the suffix without the leading dot.
    special case for shp.zip
    """
    if os.fspath(filename).lower().endswith(".shp.zip"):
        return "shp.zip"
    ext = _get_suffix(filename)
    if ext.startswith("."):
        ext = ext[1:]
    return ext


def _get_suffix(filename: PathLikeOrStr) -> str:
    return Path(filename).suffix  # same as os.path.splitext(filename)[1]


def GetOutputDriverFor(
        filename: PathLikeOrStr,
        is_raster=True,
        default_raster_format="GTiff",
        default_vector_format="ESRI Shapefile",
) -> str:
    if not filename:
        return "MEM"
    drv_list = GetOutputDriversFor(filename, is_raster)
    ext = get_extension(filename)
    if not drv_list:
        if not ext:
            return default_raster_format if is_raster else default_vector_format
        else:
            raise Exception("Cannot guess driver for %s" % filename)
    elif len(drv_list) > 1 and not (drv_list[0] == "GTiff" and drv_list[1] == "COG"):
        print(
            "Several drivers matching %s extension. Using %s"
            % (ext if ext else "", drv_list[0])
        )
    return drv_list[0]

    # GMT is registered before netCDF for opening reasons, but we want
    # netCDF to be used by default for output.
    if (
            ext.lower() == "nc"
            and len(drv_list) >= 2
            and drv_list[0].upper() == "GMT"
            and drv_list[1].upper() == "NETCDF"
    ):
        drv_list = ["NETCDF", "GMT"]

    return drv_list


def GetOutputDriversFor(filename: PathLikeOrStr, is_raster=True) -> List[str]:
    filename = os.fspath(filename)
    drv_list = []
    ext = get_extension(filename)
    if ext.lower() == "vrt":
        return ["VRT"]
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if (
                drv.GetMetadataItem(gdal.DCAP_CREATE) is not None
                or drv.GetMetadataItem(gdal.DCAP_CREATECOPY) is not None
        ) and drv.GetMetadataItem(
            gdal.DCAP_RASTER if is_raster else gdal.DCAP_VECTOR
        ) is not None:
            if ext and DoesDriverHandleExtension(drv, ext):
                drv_list.append(drv.ShortName)
            else:
                prefix = drv.GetMetadataItem(gdal.DMD_CONNECTION_PREFIX)
                if prefix is not None and filename.lower().startswith(prefix.lower()):
                    drv_list.append(drv.ShortName)

    # GMT is registered before netCDF for opening reasons, but we want
    # netCDF to be used by default for output.
    if (
            ext.lower() == "nc"
            and len(drv_list) >= 2
            and drv_list[0].upper() == "GMT"
            and drv_list[1].upper() == "NETCDF"
    ):
        drv_list = ["NETCDF", "GMT"]

    return drv_list


def DoesDriverHandleExtension(drv: gdal.Driver, ext: str) -> bool:
    exts = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
    return exts is not None and exts.lower().find(ext.lower()) >= 0


def stdout_moveto(n):
    sys.stdout.write('\n' * n + _term_move_up() * -n)
    # getattr(sys.stdout, 'flush', lambda: None)()
    sys.stdout.flush()


def _term_move_up():  # pragma: no cover
    return '' if (os.name == 'nt') and (colorama is None) else '\x1b[A'


def stdout_clear(pos):
    COLOUR_RESET = '\x1b[0m'
    stdout_moveto(pos)
    # sys.stdout.write(' ' * 100)
    sys.stdout.write(f'\r{COLOUR_RESET}')  # place cursor back at the beginning of line
    stdout_moveto(-pos)
    sys.stdout.write(' ' * 100)
    sys.stdout.write('\r')


def progress_callback(complete, message, cb_data):
    # '''Emit progress report in numbers for 10% intervals and dots for 3%'''
    # block = u'\u2588\u2588'
    # ncol = 80
    # # if int(complete*100) % 10 == 0:
    # #     print(f'{complete*100:.0f}', end='', flush=True)
    # if int(complete*100) % 3 == 0:
    #     n = int(ncol * (complete*100 / 3) / 100)
    #     # sys.stdout.write('\n')
    #     # sys.stdout.write('')
    #     # print(f'\r {complete*100:.0f}%{block}', end='', flush=True)
    #     stdout_moveto(cb_data)
    #     sys.stdout.write(f'\r {Fore.BLUE}{complete*100:.0f}% {block * n}')
    #     sys.stdout.write('')
    #     stdout_moveto(-cb_data)
    # if int(complete*100) == 100:
    #     # print('\r', end='', flush=True)
    #     # sys.stdout.write(f'\r {COLOUR_RESET}')
    #     # sys.stdout.write('')
    #     stdout_clear(cb_data)
    bar = cb_data[0]
    total = cb_data[1]
    if int(complete*100) < 100:
        bar.update(int(total * 0.01))
    else:
        bar.close()


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton
