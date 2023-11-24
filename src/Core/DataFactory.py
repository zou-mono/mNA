import os
import traceback

from osgeo import ogr, osr, gdal

# from UICore import fgdb, filegdbapi
from osgeo.ogr import DataSource

from Core import fgdb
from Core.common import launderName
from Core.log4p import Log
from Core.core import DataType

log = Log(__name__)
# gdal.SetConfigOption('CPL_LOG', 'NUL')

class workspaceFactory(object):
    def __init__(self):
        self.datasource = None
        self.driver = None
        self.driverName = ""

    def get_factory(self, factory):
        wks = None
        if factory == DataType.shapefile or factory == DataType.dbf:
            wks = shapefileWorkspaceFactory()
        elif factory == DataType.geojson:
            wks = geojsonWorkspaceFactory()
        elif factory == DataType.fileGDB:
            wks = filegdbWorkspaceFactory()
        elif factory == DataType.cad_dwg:
            wks = dwgWorkspaceFactory()
        elif factory == DataType.openFileGDB:
            wks = openfilegdbWorkspaceFactory()
        elif factory == DataType.sqlite:
            wks = sqliteWorkspaceFactory()
        elif factory == DataType.FGDBAPI:
            wks = fgdbapiWorkspaceFactory()
        elif factory == DataType.geojsonl:
            wks = geojonslWorkspaceFactory()
        elif factory == DataType.memory:
            wks = memoryWorkspaceFactory()
        elif factory == DataType.jsonfg:
            wks = jsonfgWorkspackeFactory()
        if wks is None:
            log.error("不支持的空间数据格式!")

        return wks

    def createDataSource(self, path, options=None):
        if options is None:
            options = []
        if self.driver is None:
            log.error("缺失图形文件引擎{}!".format(self.driverName))
            return None
        else:
            self.datasource = self.driver.CreateDataSource(path, options)
        return self.datasource

    def createFromExistingDataSource(self, in_layer, out_path, layer_name, srs,
                                     datasetCreationOptions, layerCreationOptions, new_fields=None,
                                     geom_type=ogr.wkbMultiLineString, panMap=None, open=True):

        out_DS = self.createDataSource(out_path, options=datasetCreationOptions)
        out_layer = out_DS.CreateLayer(layer_name, srs=srs, geom_type=geom_type, options=layerCreationOptions)

        in_defn = in_layer.GetLayerDefn()
        # for i in range(in_defn.GetFieldCount()):
        #     fieldName = in_defn.GetFieldDefn(i).GetName()
        #     fieldTypeCode = in_defn.GetFieldDefn(i).GetType()
        #     new_field = ogr.FieldDefn(fieldName, fieldTypeCode)
        #     out_layer.CreateField(new_field)
        #     del new_field
        if panMap is None:
            for i in range(in_defn.GetFieldCount()):
                fieldName = in_defn.GetFieldDefn(i).GetName()
                fieldTypeCode = in_defn.GetFieldDefn(i).GetType()
                new_field = ogr.FieldDefn(fieldName, fieldTypeCode)
                out_layer.CreateField(new_field)
                del new_field
        else:
            for i in panMap:
                if i > -1:
                    fieldName = in_defn.GetFieldDefn(i).GetName()
                    fieldTypeCode = in_defn.GetFieldDefn(i).GetType()
                    new_field = ogr.FieldDefn(fieldName, fieldTypeCode)
                    out_layer.CreateField(new_field)
                    del new_field

        if new_fields is not None:
            for new_field in new_fields:
                out_layer.CreateField(new_field)
                del new_field

        if open:
            return out_DS, out_layer
        else:
            if out_DS is not None:
                out_DS.Destroy()
            out_DS = None
            del out_layer
            return None, None

    def openFromFile(self, file, access=0):
        if self.driver is None:
            log.error("缺失图形文件引擎{}!".format(self.driverName))
            return None
        else:
            try:
                if self.driver == "fgdbapi":
                    gdb = fgdb.GeoDatabase()
                    bflag, err_msg = gdb.Open(file)
                    if not bflag:
                        raise Exception(err_msg)
                    else:
                        self.datasource = gdb
                else:
                    self.datasource = self.driver.Open(file, access)
                return self.datasource
            except Exception as e:
                if self.driver == "fgdbapi":
                    log.error("打开文件{}发生错误!\n{}".format(file, e))
                else:
                    log.error("打开文件{}发生错误!\n{}".format(file, traceback.format_exc()))
                return None

    def get_ds(self, path, access=0):
        fileType = get_suffix(path)

        if fileType == DataType.shapefile:
            wks = self.get_factory(DataType.shapefile)
        elif fileType == DataType.geojson:
            wks = self.get_factory(DataType.geojson)
        elif fileType == DataType.fileGDB:
            wks = self.get_factory(DataType.fileGDB)
        elif fileType == DataType.sqlite:
            wks = self.get_factory(DataType.sqlite)
        elif fileType == DataType.geojsonl:
            wks = self.get_factory(DataType.geojsonl)
        else:
            raise TypeError("不识别的图形文件格式!")

        datasource = wks.openFromFile(path, access)
        return datasource

    def get_layer(self, ds: DataSource, path, layer_name):
        if ds.GetDriver().name == "ESRI Shapefile" or ds.GetDriver().name == "GeoJSON":
            file_name, file_ext = os.path.splitext(os.path.basename(path))
            layer_name = file_name
            layer = ds.GetLayer(layer_name)
        else:
            layer = ds.GetLayer(layer_name)

        return layer, layer_name

    def openLayer(self, name):
        if self.datasource is not None:
            try:
                layer = self.datasource.GetLayer(name)
                if layer is None:
                    log.warning("图层{}不存在!".format(name))
                    return None
                else:
                    return layer
            except:
                log.error("读取图层{}发生错误!\n{}".format(name, traceback.format_exc()))
                return None

    def getLayers(self):
        layers = []
        if self.datasource is not None:
            for layer_idx in range(self.datasource.GetLayerCount()):
                layer = self.datasource.GetLayerByIndex(layer_idx)
                layers.append(layer)
            return layers
        else:
            return []

    def getLayerNames(self):
        layer_names = []
        if self.datasource is not None:
            for layer_idx in range(self.datasource.GetLayerCount()):
                layer = self.datasource.GetLayerByIndex(layer_idx)
                layer_names.append(layer.GetName())
            return layer_names
        else:
            return []

    def cloneLayer(self, in_layer, output_path, out_layer_name, out_srs, out_format):
        if self.driver is None:
            return None

        out_DS = None
        out_layer = None
        try:
            in_defn = in_layer.GetLayerDefn()

            if os.path.exists(output_path):
                # log.info("datasource已存在，在已有datasource基础上创建图层.")

                if out_format == DataType.shapefile or out_format == DataType.geojson:
                    self.driver.DeleteDataSource(output_path)

                    # 如果无法删除则再修改图层名新建一个
                    if os.path.exists(output_path):
                        output_path = launderName(output_path)
                        out_layer_name, suffix = os.path.splitext(os.path.basename(output_path))
                    out_DS = self.driver.CreateDataSource(output_path)
                elif out_format == DataType.fileGDB:
                    out_DS = self.openFromFile(output_path)
                    # out_layer = out_DS.GetLayer(out_layer_name)
                    # print(out_DS.TestCapability(ogr.ODsCDeleteLayer))
                    # if out_layer is not None:
                    #     out_DS.DeleteLayer(out_layer_name)
            else:
                out_DS = self.driver.CreateDataSource(output_path)

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(out_srs)

            if out_format == DataType.shapefile:
                encode_name = in_layer.GetMetadataItem("SOURCE_ENCODING", "SHAPEFILE")
                if encode_name is None:
                    encode_name = in_layer.GetMetadataItem("ENCODING_FROM_LDID", "SHAPEFILE")
                    if encode_name is None:
                        encode_name = in_layer.GetMetadataItem("ENCODING_FROM_CPG", "SHAPEFILE")

                if encode_name is not None:
                    out_layer = out_DS.CreateLayer(out_layer_name, srs=srs, geom_type=in_layer.GetGeomType(),
                                                  options=['ENCODING={}'.format(encode_name)])
            elif out_format == DataType.fileGDB or out_format == DataType.geojson:
                out_layer = out_DS.CreateLayer(out_layer_name, srs=srs, geom_type=in_layer.GetGeomType())

            if out_layer is None:
                raise Exception("创建图层失败.")

            # out_layer.CreateField(in_defn)
            for i in range(in_defn.GetFieldCount()):
                fieldName = in_defn.GetFieldDefn(i).GetName()
                fieldTypeCode = in_defn.GetFieldDefn(i).GetType()
                new_field = ogr.FieldDefn(fieldName, fieldTypeCode)
                out_layer.CreateField(new_field)

            return output_path, out_layer.GetName()
        except Exception as e:
            log.error("{}\n{}".format(e, traceback.format_exc()))
            return None, None
        finally:
            out_DS = None
            out_layer = None


def addFeature(in_feature, fid, geometry, out_layer, panMap, new_values=None, bjson=False):
    out_Feature = None
    try:
        # if isinstance(out_layer, ogr.Layer):
        out_Feature = ogr.Feature(out_layer.GetLayerDefn())
        # elif isinstance(out_layer, fgdb.Table):
        #     out_Feature = fgdb.Row()
        #     out_layer.CreateRowObject(out_Feature)
        #     out_Feature.SetGeometry2(geometry)
        #     out_layer.Insert(out_Feature)

        if fid > 0:
            out_Feature.SetFID(fid)
        # out_Feature.SetFromWithMap(in_feature, 1, panMap)
        out_Feature.SetFrom(in_feature)
        out_Feature.SetGeometry(geometry)
        # poDstGeometry = poDstFeature.GetGeometryRef()

        if new_values is not None:
            for k, v in new_values.items():
                out_Feature.SetField(k, v)

        if not bjson:
            out_layer.CreateFeature(out_Feature)
            return 1
        else:
            ret_str = out_Feature.ExportToJson()
            return ret_str
    except UnicodeEncodeError:
        log.error("错误发生在第{}个要素.\n{}".format(fid, "字符编码无法转换，请检查输入文件的字段!"))
        return -1
    except RuntimeError:
        log.error("错误发生在第{}个要素.\n{}".format(fid, "无法拷贝属性值"))
        return -2
    except:
        log.error("错误发生在第{}个要素.\n{}".format(fid, traceback.format_exc()))
        return -10000
    finally:
        del out_Feature


class jsonfgWorkspackeFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "JSONFG"
        self.driver = ogr.GetDriverByName(self.driverName)


class memoryWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "Memory"
        self.driver = ogr.GetDriverByName(self.driverName)


class geojonslWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "GeoJSONSeq"
        self.driver = ogr.GetDriverByName(self.driverName)


class fgdbapiWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driver = "fgdbapi"


class shapefileWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "ESRI Shapefile"
        self.driver = ogr.GetDriverByName(self.driverName)


class geojsonWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "GeoJSON"
        self.driver = ogr.GetDriverByName(self.driverName)


class filegdbWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "FileGDB"
        self.driver = ogr.GetDriverByName(self.driverName)


class openfilegdbWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "OpenFileGDB"
        self.driver = ogr.GetDriverByName(self.driverName)


class dwgWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "CAD"
        self.driver = ogr.GetDriverByName(self.driverName)


class sqliteWorkspaceFactory(workspaceFactory):
    def __init__(self):
        super().__init__()
        self.driverName = "SQLite"
        self.driver = ogr.GetDriverByName(self.driverName)


def get_suffix(path):
    suffix = None
    basename = os.path.basename(path)
    filename, suffix = os.path.splitext(path)
    # if basename.find('.') > 0:
    #     suffix = basename.split('.')[1]

    if suffix is None:
        return None

    if suffix.lower() == '.shp':
        return DataType.shapefile
    elif suffix.lower() == '.geojson':
        return DataType.geojson
    elif suffix.lower() == '.gdb':
        return DataType.fileGDB
    elif suffix.lower() == ".sqlite":
        return DataType.sqlite
    elif suffix.lower() == ".gml":
        return 'gml'
    elif suffix.lower() == ".gexf":
        return 'gexf'
    elif suffix.lower() == '.gpickle':
        return 'gpickle'
    elif suffix.lower() == '.graphml':
        return 'graphml'
    else:
        return None
