import os
from collections import deque

import numpy as np
from osgeo import ogr, gdal
from osgeo.ogr import DataSource, Layer, CreateGeometryFromWkt
from osgeo.osr import SpatialReference
from shapely import STRtree, from_wkt, Point, geometry, buffer

from Core.DataFactory import workspaceFactory, creation_options, addFeature
from Core.check import check_line_type
from Core.core import DataType_num, DataType, DataType_suffix
from Core.graph import _nodes_from_network, _parse_nodes_paths, Direction
from Core.log4p import Log
from Core.utils_graph import angle_between_lines, angle_between_vectors, angle_between_vecters2

log = Log(__name__)

# remainedWayLevel = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
remainedWayLevel = ["motorway", "trunk", "primary", "secondary", "tertiary"]
m_parallelTolerance = 30

class streetHander:
    def __init__(self):
        self.m_allSubWayId2SubWayObjs = None
        self.o_nodes = None
        self._nodes = None
        self._ways = None
        self.input_layer = None
        self.m_search_dis = 0
        self.m_max_wayid = -1
        self.out_type = DataType.shapefile.value
        self.m_bufferDistance = 100 / 111000  # 缓冲区距离100米, 1度约等于111公里，换算成度是100 / 111000

    def build_street(self,
                     path,
                     input_layer_name,
                     direction_field="",
                     waylevel_field="highway",
                     forwardValue="F",
                     backwardValue="T",
                     bothValue="B",
                     defaultDirection=0,
                     out_type='shp',
                     out_path="res"):
        wks = workspaceFactory()
        ds: DataSource = wks.get_ds(path, access=1)

        input_layer, input_layer_name = wks.get_layer(ds, path, input_layer_name)
        self.input_layer = input_layer
        # log.info("创建空间索引...")
        # exec_str = r"Create Spatial Index ON {}".format(input_layer_name)
        # ds.ExecuteSQL(exec_str)

        self.out_type = DataType_num[out_type.lower()]
        self.out_path = out_path
        self.waylevel_field = waylevel_field

        if not check_line_type(input_layer):
            log.error("网络数据不满足几何类型要求，只允许Polyline或multiPolyline类型.")
            return False

        crs: SpatialReference = input_layer.GetSpatialRef()

        if crs.IsProjected():
            self.m_search_dis = 1  # 如果是投影距离，则搜索距离为1米
        else:
            self.m_search_dis = 1 / 111000  # 如果是经纬度距离，则搜索距离为1/111公里

        self.o_nodes = _nodes_from_network(input_layer)
        self._nodes, self._ways = _parse_nodes_paths(self.o_nodes, input_layer, direction_field, waylevel_field,
                                                     forwardValue, backwardValue, bothValue, defaultDirection,
                                                     key='fid')

        self.m_max_wayid = max(self._ways.keys())

        self.m_simplifiedNetwork = self.merge_ways_to_lines()
        self.generateSimplifiedLines()
        # self._export_lines("01_link_ways_allLayer", waylevel_field, self.m_simplifiedNetwork)

        print("OK")

    def merge_ways_to_lines(self):
        aggregatedWayInfoMap = {}
        for iway, way in self._ways.items():
            if way['waylevel'] in remainedWayLevel:
                if way['waylevel'] in aggregatedWayInfoMap:
                    aggregatedWayInfoMap[way['waylevel']].update({
                        iway: way
                    })
                else:
                    aggregatedWayInfoMap[way['waylevel']] = {
                        iway: way
                    }

        out_res = {}
        for way_level, way_dic in aggregatedWayInfoMap.items():
            rtree, pos, parent = self._generate_STRtree(self.waylevel_field, way_level)

            if way_level == 'primary':
                print("debug")

            while len(way_dic) > 0:
                completeWayIds = deque()
                way1Id = -1

                for k, v in way_dic.items():
                    way1Id = k
                    break

                completeWayIds.append(way1Id)

                way1FromNode = self._nodes[way_dic[way1Id]['nodes'][0]]
                way1ToNode = self._nodes[way_dic[way1Id]['nodes'][-1]]

                adj_ways = self._query_adj_way(rtree, way1FromNode)
                way2Id, share_type = self._query_way2(adj_ways, parent, pos, way1Id)

                if way2Id > 0:
                    if share_type == 1:
                        completeWayIds.appendleft(way2Id)
                    elif share_type == 2:
                        completeWayIds.append(way2Id)

                adj_ways = self._query_adj_way(rtree, way1ToNode)
                way2Id, share_type = self._query_way2(adj_ways, parent, pos, way1Id)

                if way2Id > 0:
                    if share_type == 1:
                        completeWayIds.appendleft(way2Id)
                    elif share_type == 2:
                        completeWayIds.append(way2Id)

                if len(completeWayIds) > 1:
                    new_way = self._create_new_way(completeWayIds)
                    self._ways[self.m_max_wayid] = new_way
                    way_dic[self.m_max_wayid] = new_way
                    parent[self.m_max_wayid] = self.m_max_wayid

                    for visited in completeWayIds:
                        parentId = self._find_parent(visited, parent)
                        parent[visited] = self.m_max_wayid
                        parent[parentId] = self.m_max_wayid

                        if parentId in way_dic:
                            del way_dic[parentId]

                elif len(completeWayIds) == 1:
                    parentId = self._find_parent(completeWayIds[0], parent)
                    if parentId in way_dic:
                        del way_dic[parentId]

            lines = set()
            for wayId in parent:
                # parentId = self._find_parent(wayId, parent)
                if parent[wayId] == wayId:
                    # lines.add(wayId)
                    lines.add(wayId)

            lines_lst = []
            for wayId in lines:
                geom_line = self._from_wayid_to_geometry(wayId)
                lines_lst.append({
                    'id': wayId,
                    'geom': geom_line
                })

            out_res[way_level] = lines_lst

        return out_res

    def generateSimplifiedLines(self):
        for waylevel, lines in self.m_simplifiedNetwork.items():
            lineId2lengthDic = {}

            for line in lines:
                _id = line['id']
                lineId2lengthDic[_id] = line['geom']

            while len(lineId2lengthDic) > 0:
                longest_key = self._get_longest(lineId2lengthDic)
                geom_line = lineId2lengthDic[longest_key]
                geom_buff = buffer(geom_line, self.m_bufferDistance)

                for line_id, line_geom in lineId2lengthDic.items():
                    pass

        print("generateSimplifiedLines")

    def _export_lines(self, out_layer_name, waylevel_field, out_res):
        datasetCreationOptions = []
        layerCreationOptions = []

        srs = self.input_layer.GetSpatialRef()

        datasetCreationOptions, layerCreationOptions, out_type_f, out_format = creation_options(self.out_type)

        out_suffix = DataType_suffix[out_type_f]

        # poSrcFDefn = self.input_layer.GetLayerDefn()
        # nSrcFieldCount = poSrcFDefn.GetFieldCount()
        # panMap = [-1] * nSrcFieldCount
        #
        # for iField in range(poSrcFDefn.GetFieldCount()):
        #     poSrcFieldDefn = poSrcFDefn.GetFieldDefn(iField)
        #     iDstField = poSrcFDefn.GetFieldIndex(poSrcFieldDefn.GetNameRef())
        #     if iDstField >= 0:
        #         panMap[iField] = iDstField

        line_out_path = os.path.join(self.out_path, "{}.{}".format(out_layer_name, out_suffix))

        wks = workspaceFactory().get_factory(out_type_f)
        out_line_ds, out_line_layer = wks.createFromExistingDataSource(self.input_layer, line_out_path, out_layer_name, srs,
                                                                       datasetCreationOptions, layerCreationOptions,
                                                                       new_fields=[],
                                                                       geom_type=ogr.wkbLineString, open=True)

        for waylevel, lines in out_res.items():
            for line in lines:
                # line = self._ways[wayId]
                # a = [[self._nodes[nodeId]['x'], self._nodes[nodeId]['y']] for nodeId in line['nodes']]
                # geom_line = geometry.LineString(a)
                # geom_line = self._from_wayid_to_geometry(wayId)
                geom_line = line['geom']
                wayId = line['id']

                out_Feature = ogr.Feature(out_line_layer.GetLayerDefn())
                out_Feature.SetGeometry(CreateGeometryFromWkt(geom_line.wkt))
                out_Feature.SetField(waylevel_field, waylevel)
                out_Feature.SetField("ogc_fid", wayId)

                out_line_layer.CreateFeature(out_Feature)

    def _from_wayid_to_geometry(self, wayId):
        line = self._ways[wayId]
        a = [[self._nodes[nodeId]['x'], self._nodes[nodeId]['y']] for nodeId in line['nodes']]
        geom_line = geometry.LineString(a)
        return geom_line

    def _find_parent(self, _id, dic):
        while _id in dic:
            if dic[_id] == _id:
                return _id
            _id = dic[_id]

        return _id

    def _create_new_way(self, completeWayIds):
        _fids = []
        _node_ids = []
        _direction = Direction.DirectionBoth
        _waylevel = ""

        ifirst = True
        for _id in completeWayIds:
            _fids.append(_id)

            if ifirst:
                ifirst = False
                _node_ids = self._ways[_id]['nodes']
            else:
                _node_ids = _node_ids + self._ways[_id]['nodes'][1:]

            _direction = self._ways[_id]['direction']
            _waylevel = self._ways[_id]['waylevel']

        self.m_max_wayid = self.m_max_wayid + 1

        if _direction == Direction.DirectionBackward:
            _direction = Direction.DirectionForward
            _node_ids.reverse()

        return {
            'feature_id': _fids,
            'nodes': _node_ids,
            'direction': _direction,
            'waylevel': _waylevel
        }

    def _generate_STRtree(self, waylevel_field, way_level):
        self.input_layer.SetAttributeFilter("{} = '{}'".format(waylevel_field, way_level))
        self.input_layer.ResetReading()

        geoms = []
        pos = []
        parent = {}  # 用于记录原始way和合并后way的关系
        for feat in self.input_layer:
            geom = feat.GetGeometryRef()
            fid = feat.GetFID()
            geoms.append(from_wkt(geom.ExportToWkt()))
            pos.append(fid)
            parent[fid] = fid

        rtree = STRtree(geoms)
        return rtree, pos, parent

    def _query_adj_way(self, rtree, node):
        search_pt = Point(node['x'], node['y'])
        adj_ways = rtree.query(search_pt, predicate='touches', distance=self.m_search_dis)
        return adj_ways

    # 0表示无连接点，1表示way2在way1的前面，2表示way2在way1的后面
    def _isTwoWayShareSameNode(self, way1Id, way2Id):
        way1_direction = self._ways[way1Id]['direction']
        way2_direction = self._ways[way2Id]['direction']

        way1SubwayFrontId = self._ways[way1Id]['nodes'][0]
        way1SubwayBackId = self._ways[way1Id]['nodes'][-1]
        way2SubwayFrontId = self._ways[way2Id]['nodes'][0]
        way2SubwayBackId = self._ways[way2Id]['nodes'][-1]

        if way1SubwayFrontId == way2SubwayFrontId:
            if (way1_direction == Direction.DirectionBoth and way2_direction == Direction.DirectionBoth) or (
                    way1_direction == Direction.DirectionForward and way2_direction == Direction.DirectionBackward):
                return 1
            elif way1_direction == Direction.DirectionBackward and way2_direction == Direction.DirectionForward:
                return 2
        elif way1SubwayFrontId == way2SubwayBackId:
            if (way1_direction == Direction.DirectionBoth and way2_direction == Direction.DirectionBoth) or (
                    way1_direction == Direction.DirectionForward and way2_direction == Direction.DirectionForward):
                return 1
            elif way1_direction == Direction.DirectionBackward and way2_direction == Direction.DirectionBackward:
                return 2
        elif way1SubwayBackId == way2SubwayFrontId:
            if way1_direction == Direction.DirectionBackward and way2_direction == Direction.DirectionBackward:
                return 1
            elif (way1_direction == Direction.DirectionBoth and way2_direction == Direction.DirectionBoth) or (
                    way1_direction == Direction.DirectionForward and way2_direction == Direction.DirectionForward):
                return 2
        elif way1SubwayBackId == way2SubwayBackId:
            if way1_direction == Direction.DirectionBackward and way2_direction == Direction.DirectionForward:
                return 1
            elif (way1_direction == Direction.DirectionBoth and way2_direction == Direction.DirectionBoth) or (
                    way1_direction == Direction.DirectionForward and way2_direction == Direction.DirectionBackward):
                return 2

        return 0

    def _query_way2(self, adj_ways, parent, pos, way1Id):
        min_angle = m_parallelTolerance
        share_type = 0
        way2Id = -1
        for adj_way in adj_ways:
            parentId = self._find_parent(pos[adj_way], parent)
            if parentId != way1Id:
                _way2Id = parentId
                _share_type = self._isTwoWayShareSameNode(way1Id, _way2Id)

                if _share_type > 0:
                    connect_type, angleMap = self._calAngleOfConnectedSubWay(way1Id, _way2Id)
                    if angleMap < min_angle:  # 特殊情况下，adj_way有多条，则选择夹角最小的
                        min_angle = angleMap
                        share_type = _share_type
                        way2Id = _way2Id

        return way2Id, share_type

    def _get_longest(self, lineId2lengthDic):
        longest_key = max(lineId2lengthDic, key=lambda x: lineId2lengthDic[x].length)
        # longest = lineId2lengthDic[longest_key]['geom'].length
        return longest_key

    def _calAngleOfConnectedSubWay(self, way1Id, way2Id):
        way1SubwayFrontFromNode = self._ways[way1Id]['nodes'][0]
        way1SubwayBackToNode = self._ways[way1Id]['nodes'][-1]
        way2SubwayFrontFromNode = self._ways[way2Id]['nodes'][0]
        way2SubwayBackToNode = self._ways[way2Id]['nodes'][-1]

        if way1SubwayFrontFromNode == way2SubwayFrontFromNode:
            vector1 = (
                self._nodes[self._ways[way1Id]['nodes'][1]]['x'] - self._nodes[self._ways[way1Id]['nodes'][0]]['x'],
                self._nodes[self._ways[way1Id]['nodes'][1]]['y'] - self._nodes[self._ways[way1Id]['nodes'][0]]['y'])
            vector2 = (
                self._nodes[self._ways[way2Id]['nodes'][1]]['x'] - self._nodes[self._ways[way2Id]['nodes'][0]]['x'],
                self._nodes[self._ways[way2Id]['nodes'][1]]['y'] - self._nodes[self._ways[way2Id]['nodes'][0]]['y'])
            # angleMap = angle_between_vectors(self._npf(vector1), self._npf(vector2))
            angleMap = angle_between_vecters2(self._npf(vector1), self._npf(vector2))
            return 1, abs(180 - angleMap)

        elif way1SubwayFrontFromNode == way2SubwayBackToNode:
            vector1 = (
                self._nodes[self._ways[way1Id]['nodes'][1]]['x'] - self._nodes[self._ways[way1Id]['nodes'][0]]['x'],
                self._nodes[self._ways[way1Id]['nodes'][1]]['y'] - self._nodes[self._ways[way1Id]['nodes'][0]]['y'])
            vector2 = (
                self._nodes[self._ways[way2Id]['nodes'][-1]]['x'] - self._nodes[self._ways[way2Id]['nodes'][-2]]['x'],
                self._nodes[self._ways[way2Id]['nodes'][-1]]['y'] - self._nodes[self._ways[way2Id]['nodes'][-2]]['y'])
            angleMap = angle_between_vecters2(self._npf(vector1), self._npf(vector2))
            return 2, angleMap

        elif way1SubwayBackToNode == way2SubwayFrontFromNode:
            vector1 = (
                self._nodes[self._ways[way1Id]['nodes'][-1]]['x'] - self._nodes[self._ways[way1Id]['nodes'][-2]]['x'],
                self._nodes[self._ways[way1Id]['nodes'][-1]]['y'] - self._nodes[self._ways[way1Id]['nodes'][-2]]['y'])
            vector2 = (
                self._nodes[self._ways[way2Id]['nodes'][1]]['x'] - self._nodes[self._ways[way2Id]['nodes'][0]]['x'],
                self._nodes[self._ways[way2Id]['nodes'][1]]['y'] - self._nodes[self._ways[way2Id]['nodes'][0]]['y'])
            angleMap = angle_between_vecters2(self._npf(vector1), self._npf(vector2))
            return 3, angleMap

        elif way1SubwayBackToNode == way2SubwayBackToNode:
            vector1 = (
                self._nodes[self._ways[way1Id]['nodes'][-1]]['x'] - self._nodes[self._ways[way1Id]['nodes'][-2]]['x'],
                self._nodes[self._ways[way1Id]['nodes'][-1]]['y'] - self._nodes[self._ways[way1Id]['nodes'][-2]]['y'])
            vector2 = (
                self._nodes[self._ways[way2Id]['nodes'][-1]]['x'] - self._nodes[self._ways[way2Id]['nodes'][-2]]['x'],
                self._nodes[self._ways[way2Id]['nodes'][-1]]['y'] - self._nodes[self._ways[way2Id]['nodes'][-2]]['y'])
            angleMap = angle_between_vecters2(self._npf(vector1), self._npf(vector2))
            return 4, abs(180 - angleMap)

    def _npf(self, x):
        return np.array(x, dtype=float)


if __name__ == '__main__':
    sh = streetHander()
    sh.build_street(r"D:\Codes\Traffic-model\TSMM\data\sz_highway.shp", "sz_highway", waylevel_field='highway')
