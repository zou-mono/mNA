from multiprocessing import freeze_support

import click
import sys

from osgeo import gdal

from allocate import allocate
from build_graph import build_graph
from nearest import nearest

@click.group()
def cli():
    pass

if __name__ == '__main__':
    freeze_support()
    gdal.SetConfigOption('CPL_LOG', 'NUL')

    cli.add_command(allocate)
    cli.add_command(nearest)
    cli.add_command(build_graph)
    cli()