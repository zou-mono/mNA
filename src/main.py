import click
import sys

from allocate import allocate
from nearest import nearest

@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(allocate)
    cli.add_command(nearest)
    cli()