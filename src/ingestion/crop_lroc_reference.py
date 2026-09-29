import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pathlib import Path


INPUT = Path(
    r"data/reference/lroc/WAC_GLOBAL_O000N0000_100M.TIF"
)

OUTPUT = Path(
    r"data/processed/reference/tycho_wac_100m.tif"
)

# Tycho / TMC overlap region
WEST = 347.8
EAST = 350.0
SOUTH = -45.0
NORTH = -41.5


def lon_360_to_180(lon):
    """Convert 0..360 longitude to -180..180."""
    return lon - 360 if lon > 180 else lon


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    west = lon_360_to_180(WEST)
    east = lon_360_to_180(EAST)

    moon_geographic = (
        '+proj=longlat +R=1737400 +no_defs'
    )

    with rasterio.open(INPUT) as src:

        print("Source:", INPUT)
        print("CRS:", src.crs)

        # Convert lunar lon/lat to the mosaic's orthographic coordinates
        projected_bounds = transform_bounds(
            moon_geographic,
            src.crs,
            west,
            SOUTH,
            east,
            NORTH,
            densify_pts=21,
        )

        print("Geographic bounds:")
        print(" west :", WEST)
        print(" east :", EAST)
        print(" south:", SOUTH)
        print(" north:", NORTH)

        print("Projected bounds:", projected_bounds)

        window = from_bounds(
            *projected_bounds,
            transform=src.transform
        )

        window = window.round_offsets().round_lengths()

        data = src.read(1, window=window)

        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            compress="deflate",
        )

        print("Crop size:", data.shape)
        print("Valid pixels:", (data != src.nodata).sum())

        with rasterio.open(OUTPUT, "w", **profile) as dst:
            dst.write(data, 1)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()