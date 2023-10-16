import os
import traceback
from enum import Enum

from Core.log4p import Log

log = Log(__name__)


class DataType(Enum):
    shapefile = 0
    geojson = 1
    cad_dwg = 2
    fileGDB = 3
    csv = 4
    xlsx = 5
    dbf = 6
    memory = 7
    openFileGDB = 8
    sqlite = 9
    FGDBAPI = 10


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