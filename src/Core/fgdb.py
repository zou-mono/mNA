import re

from Core import filegdbapi
from Core.filegdbapi import FieldDef, EnumRows, FieldInfo
from Core.log4p import Log
import xml.etree.ElementTree as ET
from enum import Enum
from datetime import datetime
from osgeo import ogr

log = Log(__name__)


class FieldType(Enum):
    fieldTypeSmallInteger = 0
    fieldTypeInteger = 1
    fieldTypeSingle = 2
    fieldTypeDouble = 3
    fieldTypeString = 4
    fieldTypeDate = 5
    fieldTypeOID = 6
    fieldTypeGeometry = 7
    fieldTypeBlob = 8
    fieldTypeRaster = 9
    fieldTypeGUID = 10
    fieldTypeGlobalID = 11
    fieldTypeXML = 12

class ShapeType(Enum):
    shapeNull               =  0
    shapePoint              =  1
    shapePointM             = 21
    shapePointZM            = 11
    shapePointZ             =  9
    shapeMultipoint         =  8
    shapeMultipointM        = 28
    shapeMultipointZM       = 18
    shapeMultipointZ        = 20
    shapePolyline           =  3
    shapePolylineM          = 23
    shapePolylineZM         = 13
    shapePolylineZ          = 10
    shapePolygon            =  5
    shapePolygonM           = 25
    shapePolygonZM          = 15
    shapePolygonZ           = 19
    shapeMultiPatchM        = 31
    shapeMultiPatch         = 32
    shapeGeneralPolyline    = 50
    shapeGeneralPolygon     = 51
    shapeGeneralPoint       = 52
    shapeGeneralMultipoint  = 53
    shapeGeneralMultiPatch  = 54


class GeoDatabase(filegdbapi.Geodatabase):
    m_layers = []

    def __init__(self):
        super(GeoDatabase, self).__init__()
        self.table = filegdbapi.Table()

    def Open(self, file):
        hr = 0
        try:
            gdb = filegdbapi.Geodatabase()
            hr = filegdbapi.OpenGeodatabase(file, gdb)
            if hr < 0:
                raise Exception("打开GDB发生错误，代码{}".format(str(hr)))
            else:
                self.datasource = gdb
                bflag, err_msg = self.__LoadLayers("\\")
                if not bflag:
                    raise Exception(err_msg)
                return True, gdb
        except Exception as e:
            return False, e

    # 按层次打开所有table和feature class
    def __LoadLayers(self, root):
        tables = filegdbapi.wstringsp()
        featureclasses = filegdbapi.wstringsp()
        featuredatasets = filegdbapi.wstringsp()

        if self.datasource.GetChildDatasets(root, "Table", tables) < 0:
            del tables
            return False, "读取路径{}下的Table时发生错误".format(root)
        tables = tables.value()
        self.__OpenFGDBTables(tables)

        if self.datasource.GetChildDatasets(root, "Feature Class", featureclasses) < 0:
            del featureclasses
            return False, "读取路径{}下的Feature Class时发生错误".format(root)
        featureclasses = featureclasses.value()
        self.__OpenFGDBTables(featureclasses)

        if self.datasource.GetChildDatasets(root, "Feature Dataset", featuredatasets) < 0:
            del featuredatasets
            return False, "读取路径{}下的Feature Dataset时发生错误".format(root)
        featuredatasets = featuredatasets.value()
        for i in range(len(featuredatasets)):
            featuredataset = featuredatasets[i]
            featureclasses = filegdbapi.wstringsp()
            if self.datasource.GetChildDatasets(featuredataset, "Feature Class", featureclasses) < 0:
                del featureclasses
                return False, "读取Feature Dataset{}下的Feature Class时发生错误".format(featuredataset)
            featureclasses = featureclasses.value()
            self.__OpenFGDBTables(featureclasses)
        return True, ""

    def __OpenFGDBTables(self, layers):
        for layer in layers:
            tbl = Table()
            hr = self.datasource.OpenTable(layer, tbl)
            if hr < 0:
                del tbl
                log.error("打开Table{}时发生错误.".format(layer))
                continue

            self.m_layers.append(tbl)

    def GetLayerByName(self, name):
        tbl = Table()
        hr = self.datasource.OpenTable(name, tbl)
        if hr < 0:
            del tbl
            log.error("打开Table{}时发生错误.".format(name))
            return None
        else:
            tbl.ResetReading()
            return tbl

    # def CreateLayer(self, name):
    #     self.CreateTable()

    def StartTransaction(self, force=True):
        pass

    def CommitTransaction(self):
        pass

    def FlushCache(self):
        pass

class Table(filegdbapi.Table):
    def __init__(self):
        super(Table, self).__init__()
        self.m_featureDef = FeatureDefn()
        self.m_enumRows = EnumRows()
        # hr = self.Search("*", "1=1", True, rows)
        # self.m_enumRows = rows

    def ResetReading(self):
        hr = self.Search("*", "1=1", True, self.m_enumRows)

    def GetLayerDefn(self):
        tableDef = filegdbapi.stringp()
        hr = self.GetDefinition(tableDef)
        # if hr < 0:
        #     del tableDef
        #     log.error("获取GDB的元数据错误")
        #     return None
        tableDef = tableDef.value()

        root = ET.fromstring(tableDef)
        if root is None:
            log.error("获取GDB的元数据错误")
            return None

        if root.tag != "DataElement":
            log.error("获取GDB的元数据错误")
            return None

        Fields = root.find("Fields")
        FieldArray = Fields.find("FieldArray")
        for field in FieldArray:
            fieldDef = FieldDefn()

            name = field.find("Name")
            if name is not None:
                fieldDef.SetName(name.text)

            alias_name = field.find("AliasName")
            if alias_name is not None:
                fieldDef.SetAlias(alias_name.text)

            type = field.find("Type")
            if type is not None:
                fieldDef.SetType(type.text)

            self.m_featureDef.append(fieldDef)

        return self.m_featureDef

    def CreateField(self, field_def, approx_ok=True):
        hr = self.AddField(field_def)
        if hr < 0:
            log.error("创建字段错误. 代码: {}".format(str(hr)))
            return False
        else:
            self.ResetReading()
            return True

    def FindFieldIndex(self, get_field_name, approx_ok=True):
        fieldInfo = FieldInfo()
        hr = self.GetFieldInformation(fieldInfo)

        if hr < 0:
            raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))

        field_count = filegdbapi.intp()
        hr = fieldInfo.GetFieldCount(field_count)
        if hr < 0:
            raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))
        field_count = field_count.value()

        for i in range(field_count):
            field_name = filegdbapi.wstringp()
            hr = fieldInfo.GetFieldName(i, field_name)
            if hr < 0:
                raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))

            field_name = field_name.value()
            if not approx_ok:
                ret = re.search(field_name, get_field_name, re.IGNORECASE)
                if ret is not None:
                    return i
            else:
                if field_name == get_field_name:
                    return i

        return -1

    def GetFeatureCount(self):
        count = filegdbapi.intp()
        hr = self.GetRowCount(count)
        if hr < 0:
            log.error("获取要素数量错误. 代码: {}".format(str(hr)))
            return -1
        else:
            return count.value()

    def GetNextFeature(self):
        row = Row()
        hr = self.m_enumRows.Next(row)
        if hr < 0:
            log.error("获取要素错误. 代码:{}".format(str(hr)))
            return None
        elif hr == 0:
            return row
        elif hr == 1:
            return None

    def SetFeature(self, row):
        hr = self.Update(row)
        if hr < 0:
            log.error("更新字段错误. 代码:{}".format(str(hr)))

    def CreateFeature(self):
        row = Row()
        hr = self.CreateRowObject(row)
        if hr < 0:
            log.error("创建要素失败. 代码:{}".format(str(hr)))
            return None
        return row


class Row(filegdbapi.Row):
    def __init__(self):
        super(Row, self).__init__()

    def GetFID(self):
        objectID = filegdbapi.intp()
        hr = self.GetOID(objectID)
        if hr < 0:
            return -1
        else:
            return objectID.value()

    # 输入ogr的Geometry
    def SetGeometry(self, ogrGeometry):
        nOGRType = ogr.GT_Flatten(ogrGeometry.getGeometryType())
        print(nOGRType)

    def GetFieldIndex(self, get_field_name):
        try:
            fieldInfo = FieldInfo()
            hr = self.GetFieldInformation(fieldInfo)

            if hr < 0:
                raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))

            field_count = filegdbapi.intp()
            hr = fieldInfo.GetFieldCount(field_count)
            if hr < 0:
                raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))
            field_count = field_count.value()

            for i in range(field_count):
                field_name = filegdbapi.wstringp()
                hr = fieldInfo.GetFieldName(i, field_name)
                if hr < 0:
                    raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))

                field_name = field_name.value()
                if field_name == get_field_name:
                    return i

            return -1
        except Exception as e:
            log.error(e)
            return -1

    def GetField(self, fld_index):
        if isinstance(fld_index, str):
            fld_index = self.GetFieldIndex(fld_index)
        if (fld_index < 0) or (fld_index > self.GetFieldCount()):
            raise KeyError("获取字段函数GetField()错误.")
        fld_type = self.GetFieldType(fld_index)

        if fld_type == FieldType.fieldTypeInteger.value:
            value = filegdbapi.intp()
            self.GetInteger(fld_index, value)
            value = value.value()
            return value

        if fld_type == FieldType.fieldTypeSmallInteger.value:
            value = filegdbapi.shortp()
            self.GetShort(fld_index, value)
            value = value.value()
            return value

        if fld_type == FieldType.fieldTypeOID.value:
            value = filegdbapi.intp()
            self.GetOID(value)
            value = value.value()
            return value

        if fld_type == FieldType.fieldTypeDate.value:
            value = filegdbapi.tmp()
            # self.GetDate(fld_index, value)
            value = filegdbapi.GetDate_string(self, fld_index, value)
            value2 = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            del value
            return value2

        if fld_type == FieldType.fieldTypeString.value:
            value = filegdbapi.wstringp()
            self.GetString(fld_index, value)
            value = value.value()
            return value

        if fld_type == FieldType.fieldTypeDouble.value:
            value = filegdbapi.doublep()
            self.GetDouble(fld_index, value)
            value = value.value()
            return value

        if fld_type == FieldType.fieldTypeSingle.value:
            value = filegdbapi.floatp()
            self.GetFloat(fld_index, value)
            value = value.value()
            return value

    def SetField(self, *args):
        """
        SetField(self, int id, char value)
        SetField(self, char name, char value)
        SetField(self, int id, int value)
        SetField(self, char name, int value)
        SetField(self, int id, double value)
        SetField(self, char name, double value)
        SetField(self, int id, int year, int month, int day, int hour, int minute,
            int second, int tzflag)
        SetField(self, char name, int year, int month, int day, int hour,
            int minute, int second, int tzflag)
        """
        hr = 0

        fld_index = args[0]
        if isinstance(fld_index, str):
            fld_index = self.GetFieldIndex(fld_index)
        if (fld_index < 0) or (fld_index > self.GetFieldCount()):
            raise KeyError("获取字段函数GetField()错误.")
        fld_type = self.GetFieldType(fld_index)

        if len(args) == 2 and args[1] is None:
            hr = self.SetNull(fld_index)

        if len(args) == 2:
            if fld_type == FieldType.fieldTypeInteger.value and isinstance(args[1], int):
                hr = self.SetInteger(fld_index, args[1])

            if fld_type == FieldType.fieldTypeSmallInteger.value and isinstance(args[1], int):
                hr = self.SetShort(fld_index, args[1])

            if fld_type == FieldType.fieldTypeString.value and isinstance(args[1], str):
                hr = self.SetString(fld_index, args[1])

            if fld_type == FieldType.fieldTypeDouble.value and isinstance(args[1], float):
                hr = self.SetDouble(fld_index, args[1])

            if fld_type == FieldType.fieldTypeSingle.value and isinstance(args[1], float):
                hr = self.SetFloat(fld_index, args[1])
        else:
            vtime = "{}-{}-{} {}:{}:{}".format(str(args[1]), str(args[2]), str(args[3]), str(args[4]), str(args[5]), str(args[6]))
            hr = filegdbapi.SetDate_string(self, fld_index, vtime)

        if hr < 0:
            raise Exception("第{}个字段赋值错误.".format(str(fld_index)))

    def GetFieldType(self, fld_index):
        fieldInfo = FieldInfo()
        hr = self.GetFieldInformation(fieldInfo)
        if hr < 0:
            raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))

        fieldType = filegdbapi.new_fieldtypep()
        hr = fieldInfo.GetFieldType(fld_index, fieldType)
        if hr < 0:
            raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))
        fieldType = filegdbapi.fieldtypep_value(fieldType)

        return fieldType

    def GetFieldCount(self):
        fieldInfo = FieldInfo()
        hr = self.GetFieldInformation(fieldInfo)

        if hr < 0:
            raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))

        field_count = filegdbapi.intp()
        hr = fieldInfo.GetFieldCount(field_count)
        if hr < 0:
            raise Exception("获取要素字段元数据错误. 代码:{}".format(str(hr)))
        field_count = field_count.value()

        return field_count

    def SetGeometry(self, wkt):
        hr = -10000
        ogr_geometry = ogr.CreateGeometryFromWkt(wkt)
        geomType = ogr_geometry.GetGeometryType()

        pGeom = None

        if geomType == ogr.wkbPoint or geomType == ogr.wkbPointZM or geomType == ogr.wkbPointM:
            point = filegdbapi.Point()
            point.x = ogr_geometry.GetX()
            point.y = ogr_geometry.GetY()
            pGeom = filegdbapi.PointShapeBuffer()

            if geomType == ogr.wkbPoint:
                hr = filegdbapi.SetPoint(pGeom, point)
            elif geomType == ogr.wkbPointZM:
                z_value = ogr_geometry.GetZ()
                m_value = ogr_geometry.GetM()
                hr = filegdbapi.SetPoint(pGeom, point, z_value, m_value)
            elif geomType == ogr.wkbPointM:
                m_value = ogr_geometry.GetM()
                hr = filegdbapi.SetPoint(pGeom, point, None, m_value)

        if geomType == ogr.wkbMultiPoint or geomType == ogr.wkbMultiPointZM or geomType == ogr.wkbMultiPointM:
            ogr_points = ogr_geometry.GetPoints()
            numPts = len(ogr_points)
            pointArray = filegdbapi.pointArray(numPts)
            mArray = filegdbapi.doubleArray(numPts)
            zArray = filegdbapi.doubleArray(numPts)
            i = 0
            for ogr_point in ogr_points:
                pointArray[i].x = ogr_point[0]
                pointArray[i].y = ogr_point[1]
                if len(ogr_point) >= 3:
                    zArray[i] = ogr_point[2]
                if len(ogr_point) == 4:
                    mArray[i] = ogr_point[3]
                i += 1
            pGeom = filegdbapi.MultiPointShapeBuffer()

            if geomType == ogr.wkbMultiPoint:
                hr = filegdbapi.SetMultiPoint(pGeom, pointArray, numPts)
            elif geomType == ogr.wkbMultiPointM:
                hr = filegdbapi.SetMultiPoint(pGeom, pointArray, numPts, None, mArray)
            elif geomType == ogr.wkbMultiPointZM:
                hr = filegdbapi.SetMultiPoint(pGeom, pointArray, numPts, zArray, mArray)

        if geomType == ogr.wkbLineString or geomType == ogr.wkbLineStringZM or geomType == ogr.wkbLineStringM:
            ogr_points = ogr_geometry.GetPoints()
            numPts = len(ogr_points)
            pointArray = filegdbapi.pointArray(numPts)
            mArray = filegdbapi.doubleArray(numPts)
            zArray = filegdbapi.doubleArray(numPts)
            i = 0
            for ogr_point in ogr_points:
                pointArray[i].x = ogr_point[0]
                pointArray[i].y = ogr_point[1]
                if len(ogr_point) >= 3:
                    zArray[i] = ogr_point[2]
                if len(ogr_point) == 4:
                    mArray[i] = ogr_point[3]
                i += 1
            pGeom = filegdbapi.MultiPartShapeBuffer()

            if geomType == ogr.wkbLineString:
                hr = filegdbapi.SetPolyline(pGeom, pointArray, numPts, [0], 1)
            elif geomType == ogr.wkbLineStringM:
                hr = filegdbapi.SetPolyline(pGeom, pointArray, numPts, [0], 1, None, mArray)
            elif geomType == ogr.wkbLineStringZM:
                hr = filegdbapi.SetPolyline(pGeom, pointArray, numPts, [0], 1, zArray, mArray)

        if geomType == ogr.wkbMultiLineString or geomType == ogr.wkbMultiLineStringZM or geomType == ogr.wkbMultiLineStringM or \
            geomType == ogr.wkbPolygon or geomType == ogr.wkbPolygonZM or geomType == ogr.wkbPolygonM:
            ogr_points = []
            ogr_parts = [0]
            numPts = 0
            numParts = 0
            for part in ogr_geometry:
                points = part.GetPoints()
                ogr_numPts = part.GetPointCount()
                ogr_points.extend(points)
                numPts = numPts + ogr_numPts
                numParts += 1
                ogr_parts.append(numPts)
            ogr_parts = ogr_parts[:-1]

            pointArray, partArray, mArray, zArray = self.ogrWriteToArray(geomType, ogr_points, numPts, ogr_parts, numParts)

            pGeom = filegdbapi.MultiPartShapeBuffer()

            if geomType == ogr.wkbMultiLineString:
                hr = filegdbapi.SetPolyline(pGeom, pointArray, numPts, partArray, numParts)
            elif geomType == ogr.wkbMultiLineStringM:
                hr = filegdbapi.SetPolyline(pGeom, pointArray, numPts, partArray, numParts, None, mArray)
            elif geomType == ogr.wkbMultiLineStringZM:
                hr = filegdbapi.SetPolyline(pGeom, pointArray, numPts, partArray, numParts, zArray, mArray)
            elif geomType == ogr.wkbPolygon:
                hr = filegdbapi.SetPolygon(pGeom, pointArray, numPts, partArray, numParts)
            elif geomType == ogr.wkbPolygonM:
                hr = filegdbapi.SetPolygon(pGeom, pointArray, numPts, partArray, numParts, None, mArray)
            elif geomType == ogr.wkbPolygonZM:
                hr = filegdbapi.SetPolygon(pGeom, pointArray, numPts, partArray, numParts, zArray, mArray)

        if geomType == ogr.wkbMultiPolygon or geomType == ogr.wkbMultiPolygonZM or geomType == ogr.wkbMultiPolygonM:
            ogr_points = []
            ogr_parts = [0]
            numPts = 0
            numParts = 0
            for part in ogr_geometry:
                for ring in part:
                    points = ring.GetPoints()
                    ogr_numPts = ring.GetPointCount()
                    ogr_points.extend(points)
                    numPts = numPts + ogr_numPts
                numParts += 1
                ogr_parts.append(numPts)
            ogr_parts = ogr_parts[:-1]

            pointArray, partArray, mArray, zArray = self.ogrWriteToArray(geomType, ogr_points, numPts, ogr_parts, numParts)

            pGeom = filegdbapi.MultiPartShapeBuffer()
            if geomType == ogr.wkbMultiPolygon:
                hr = filegdbapi.SetPolygon(pGeom, pointArray, numPts, partArray, numParts)
            elif geomType == ogr.wkbMultiPolygonM:
                hr = filegdbapi.SetPolygon(pGeom, pointArray, numPts, partArray, numParts, None, mArray)
            elif geomType == ogr.wkbMultiPolygonZM:
                hr = filegdbapi.SetPolygon(pGeom, pointArray, numPts, partArray, numParts, zArray, mArray)

        if hr == 0 and pGeom is not None:
            hr = self.SetGeometry(pGeom)

        return hr

    def ogrWriteToArray(self, geomType, ogr_points, numPts, ogr_parts, numParts):
        pointArray = filegdbapi.pointArray(numPts)
        partArray = filegdbapi.intArray(numParts)
        mArray = filegdbapi.doubleArray(numPts)
        zArray = filegdbapi.doubleArray(numPts)

        i = 0
        for ogr_point in ogr_points:
            pointArray[i].x = ogr_point[0]
            pointArray[i].y = ogr_point[1]
            if ogr.GT_HasM(geomType):
                mArray[i] = ogr_point[3]
            if ogr.GT_HasZ(geomType):
                zArray[i] = ogr_point[2]
            i += 1

        i = 0
        for ogr_part in ogr_parts:
            partArray[i] = ogr_part
            i += 1

        return pointArray, partArray, mArray, zArray


class FeatureDefn(list):
    def __init__(self):
        super(FeatureDefn, self).__init__()

    def GetFieldDefn(self, i):
        return self[i]

    def GetFieldCount(self):
        return len(self)


class FieldDefn(filegdbapi.FieldDef):
    def __init__(self, name="t", type=4):
        self.SetType(type)
        self.SetName(name)

        super(FieldDefn, self).__init__()

    def SetType(self, type):
        self.type = type

    def GetType(self):
        return self.type

    def SetName(self, name):
        self.name = name

    def GetName(self):
        return self.name

    def SetAlias(self, alias):
        self.alias = alias

    def GetAlias(self):
        return self.alias

    def SetWidth(self, length):
        self.SetLength(length)
        self.length = length

    def GetWidth(self):
        return self.length
