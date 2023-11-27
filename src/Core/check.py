from osgeo import ogr
from osgeo.ogr import FieldDefn, Layer

from Core.log4p import Log

log = Log(__name__)

def check_line_type(layer: Layer):
    layer_type = layer.GetGeomType()

    if layer_type == 2 or layer_type == 5:
        return True
    else:
        return False


def check_geom_type(layer: Layer):
    layer_type = layer.GetGeomType()

    if layer_type == ogr.wkbPoint or layer_type == ogr.wkbMultiPoint or layer_type == ogr.wkbPointM or \
            layer_type == ogr.wkbPointZM or layer_type == ogr.wkbPoint25D or layer_type == ogr.wkbPolygon or \
            layer_type == ogr.wkbPolygonM or layer_type == ogr.wkbPolygonZM or layer_type == ogr.wkbPolygon25D:
        return True
    else:
        return False


def check_field_type(field: FieldDefn):
    field_type = field.GetType()
    if field_type == ogr.OFTReal or field_type == ogr.OFTInteger or field_type == ogr.OFTInteger64:
        return True
    else:
        return False


def init_check(layer, capacity_field=None, suffix="", target_weight_idx=-1, panMap=None):
    layer_name = layer.GetName()

    poSrcFDefn = layer.GetLayerDefn()
    nSrcFieldCount = poSrcFDefn.GetFieldCount()
    res_panMap = [-1] * nSrcFieldCount
    if panMap is not None:
        for iField in panMap:
            poSrcFieldDefn = poSrcFDefn.GetFieldDefn(iField)
            # iDstField = poSrcFDefn.GetFieldIndex(poSrcFieldDefn.GetNameRef())
            if poSrcFieldDefn is not None:
                res_panMap[iField] = iField
    else:
        for iField in range(poSrcFDefn.GetFieldCount()):
            poSrcFieldDefn = poSrcFDefn.GetFieldDefn(iField)
            iDstField = poSrcFDefn.GetFieldIndex(poSrcFieldDefn.GetNameRef())
            if iDstField >= 0:
                res_panMap[iField] = iDstField

    # res_panMap = []
    # in_defn = layer.GetLayerDefn()
    # if panMap is not None:
    #     for i in range(in_defn.GetFieldCount()):
    #         if i not in panMap:
    #             res_panMap.append(-1)
    #         else:
    #             res_panMap.append(i)
    # else:
    #     for i in range(in_defn.GetFieldCount()):
    #         res_panMap.append(i)

    if not check_geom_type(layer):
        log.error("{}设施数据不满足几何类型要求,只允许Polygon,multiPolygon,Point,multiPoint类型".format(suffix, layer_name))
        return False, res_panMap, None, None, None, None
    else:
        if capacity_field is None:
            return True, res_panMap, None, None, None, None

    capacity_idx = layer.FindFieldIndex(capacity_field, False)
    if capacity_idx == -1:
        log.error("{}设施数据'{}'缺少容量字段{},无法进行后续计算".format(suffix, layer_name, capacity_field))
        return False, res_panMap, None, None, None, None

    if not check_field_type(layer.GetLayerDefn().GetFieldDefn(capacity_idx)):
        log.error("设施数据'{}'的字段{}不满足类型要求,只允许int, double类型".format(suffix, layer_name, capacity_field))
        return False, res_panMap, None, None, None, None

    query_str = '''"{}" > 0'''.format(capacity_field)
    layer.SetAttributeFilter(query_str)

    capacity = 0
    capacity_dict = {}
    weight_dict = {}
    layer.ResetReading()
    for feature in layer:
        fid = feature.GetFID()
        v = feature.GetField(capacity_field)
        capacity_dict[fid] = v
        capacity = capacity + v

        if target_weight_idx > 0:
            weight = feature.GetField(target_weight_idx)
            weight_dict[fid] = weight

    log.info("{}设施总容量为{}".format(suffix, capacity))

    return True, res_panMap, capacity, capacity_dict, capacity_idx, weight_dict