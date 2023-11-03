import os, sys


# os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0]) + r"\Library\share\proj"
# os.environ['GDAL_DRIVER_PATH'] = os.path.dirname(sys.argv[0]) + r"\Library\lib\gdalplugins"
# os.environ['GDAL_DATA'] = os.path.dirname(sys.argv[0]) + r"\Library\share\gdal"
# path = os.path.dirname(sys.argv[0])

# def resource_path(relative):
#     return os.path.join(
#         os.environ.get(
#             "_MEIPASS2",
#             os.path.abspath(".")
#         ),
#         relative
#     )

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

root_path = resource_path("")

path = os.getcwd()
os.environ['PATH'] = r"%WINDIR%\system32;%WINDIR%;%WINDIR%\system32\WBem;"
# os.environ['PATH'] = os.environ['PATH'] + r"{}\scip\bin".format(os.path.realpath(path))
# os.environ['PROJ_LIB'] = r".\Library\share\proj"
# os.environ['GDAL_DRIVER_PATH'] = r".\Library\lib\gdalplugins"
# os.environ['GDAL_DATA'] = r".\Library\share\gdal"

os.environ['PROJ_LIB'] = os.path.join(root_path, "Library\share\proj")
os.environ['GDAL_DRIVER_PATH'] = os.path.join(root_path, "Library\lib\gdalplugins")
os.environ['GDAL_DATA'] = os.path.join(root_path, "Library\share\gdal")

# os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0]) + r"\Library\share\proj"
# os.environ['GDAL_DRIVER_PATH'] = os.path.dirname(sys.argv[0]) + r"\Library\lib\gdalplugins"
# os.environ['GDAL_DATA'] = os.path.dirname(sys.argv[0]) + r"\Library\share\gdal"

# print("PROJ_LIB的路径:{}".format(os.path.abspath(os.environ['PROJ_LIB'])))
# print("PATH的路径:{}".format(os.environ['PATH']))





