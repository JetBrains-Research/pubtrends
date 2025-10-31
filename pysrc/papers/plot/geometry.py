import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPoint
from shapely.geometry import Polygon
from shapely.affinity import scale

def compute_component_boundaries(
        df: pd.DataFrame,
        madk: int = 2,
        min_filter_size:int = 5,
):
    boundaries = {}
    for comp, g in df.groupby("comp"):
        xs, ys = g['x'], g['y']
        # MAD filtering
        cx, cy = np.median(xs), np.median(ys)
        rs = np.hypot(xs - cx, ys - cy)  # radial distances
        med = np.median(rs)
        mad = np.median(np.abs(rs - med))
        pts = [Point(x, y) for x, y, r in zip(xs, ys, rs) if r <= med + mad * madk]
        if len(pts) < min_filter_size:
            pts = [Point(x, y) for x, y in zip(xs, ys)]
        # Compute convex hull
        multipoint = MultiPoint(pts)
        hull = multipoint.convex_hull.buffer(0)
        scaled_hull = scale(hull, xfact=1.05, yfact=1.05, origin='center')
        boundaries[comp] = scaled_hull

    return boundaries




def shapely_to_bokeh_multipolygons(poly: Polygon):
    xs, ys = [], []
    ex_x, ex_y = poly.exterior.xy
    xs.append(list(ex_x)); ys.append(list(ex_y))
    for hole in poly.interiors:
        hx, hy = hole.xy
        xs.append(list(hx)); ys.append(list(hy))
    return [[xs]], [[ys]]


def comp_boundaries(df):
    return compute_component_boundaries(rescaled_comp_corrds(df))


def rescaled_comp_corrds(df) -> pd.DataFrame:
    dfxyc = df[['x', 'y', 'comp']].copy()
    xs, ys = dfxyc['x'], dfxyc['y']
    xmin, xrange = np.min(xs), np.max(xs) - np.min(xs)
    ymin, yrange = np.min(ys), np.max(ys) - np.min(ys)
    dfxyc['x'] = np.array((dfxyc['x'] - xmin) / xrange * 100)
    dfxyc['y'] = np.array((dfxyc['y'] - ymin) / yrange * 100)
    return dfxyc
