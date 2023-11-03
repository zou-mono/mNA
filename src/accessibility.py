import os.path
import traceback

from log4p import Log, mTqdm

from Core.graph import Direction

log = Log(__name__)


# 计算从点出发的可达范围，输出包括可达点(service_nodes)、可达边(service_lines)、可达路径(service_routes)、可达范围(cover_area)
def accessible_from_layer(input_network_data,
                         input_facility_data,
                         out_path="",
                         travelCost=0.0,
                         direction_field="",
                         forwardValue="F",
                         backwardValue="T",
                         bothValue="B",
                         include_bounds=False,
                         concave_hull_threshold=0.3,
                         defaultDirection=Direction.DirectionBoth,
                         out_type=0):

    pass


def accessibility(graph,
                  startPoints,
                  snappedPoints,
                  travelCost=0.0,
                  include_bounds=False,
                  concave_hull_threshold=0.3):
    pass


def export_to_file(geoms, fields, crs, source_attributes, out_path, out_type):
    pass


if __name__ == '__main__':
    pass
