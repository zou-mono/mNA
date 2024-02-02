import geopandas as gpd

def is_projected(crs):
    """
    Determine if a coordinate reference system is projected or not.

    Parameters
    ----------
    crs : string or pyproj.CRS
        the identifier of the coordinate reference system, which can be
        anything accepted by `pyproj.CRS.from_user_input()` such as an
        authority string or a WKT string

    Returns
    -------
    projected : bool
        True if crs is projected, otherwise False
    """
    return gpd.GeoSeries(crs=crs).crs.is_projected
