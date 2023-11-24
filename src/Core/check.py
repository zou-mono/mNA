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


def init_check(layer, capacity_field=None, suffix="", panMap=None, target_weight_idx=-1):
    layer_name = layer.GetName()

    if panMap is not None:
        in_defn = layer.GetLayerDefn()
        for i in panMap:
            fieldDefn = in_defn.GetFieldDefn(i)
            if fieldDefn is None:
                panMap.remove(i)

    if not check_geom_type(layer):
        log.error("{}设施数据不满足几何类型要求,只允许Polygon,multiPolygon,Point,multiPoint类型".format(suffix, layer_name))
        return False
    else:
        if capacity_field is None:
            return True, panMap

    capacity_idx = layer.FindFieldIndex(capacity_field, False)
    if capacity_idx == -1:
        log.error("{}设施数据'{}'缺少容量字段{},无法进行后续计算".format(suffix, layer_name, capacity_field))
        return False

    if not check_field_type(layer.GetLayerDefn().GetFieldDefn(capacity_idx)):
        log.error("设施数据'{}'的字段{}不满足类型要求,只允许int, double类型".format(suffix, layer_name, capacity_field))
        return False

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

    return True, panMap, capacity, capacity_dict, capacity_idx, weight_dict