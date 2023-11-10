import os
import re
import traceback
from enum import Enum
from multiprocessing.managers import BaseManager

from Core.log4p import Log

log = Log(__name__)


class DataType(Enum):
    shapefile = 0
    geojson = 1
    fileGDB = 2
    sqlite = 3
    csv = 4
    xlsx = 5
    dbf = 6
    memory = 7
    openFileGDB = 8
    FGDBAPI = 9
    cad_dwg = 10


DataType_suffix = {
    DataType.shapefile: 'shp',
    DataType.geojson: 'geojson',
    DataType.fileGDB: 'gdb',
    DataType.FGDBAPI: 'gdb',
    DataType.csv: 'csv',
    DataType.sqlite: 'sqlite'
}

DataType_dict = {
    DataType.shapefile: "ESRI Shapefile",
    DataType.geojson: "geojson",
    DataType.fileGDB: "FileGDB",
    DataType.cad_dwg: "CAD",
    DataType.sqlite: "SQLite"
}


def remove_temp_db(db_path):
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            log.debug("删除临时文件数据库文件{}成功！".format(db_path))
    except:
        log.warning("删除临时文件数据库文件{}出错，可能是数据库文件被占用，请手动删除!".format(db_path))


def check_layer_name(name):
    p1 = r'[-!&<>"\'?@=$~^`#%*()/\\:;{}\[\]|+.]'  # 不允许出现非法字符
    res = re.sub(p1, '_', name)
    p2 = r'( +)'  # 去除空格
    res = re.sub(p2, '', res)
    p3 = r'^[0-9]'  # 不允许以数字开头
    bExist = re.match(p3, name)
    if bExist is not None:
        res = "_" + res
    return res


class QueueManager(BaseManager):
    pass