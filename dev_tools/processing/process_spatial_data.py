import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
import os
from glob import glob

import geopandas as gpd

from shapely.geometry import shape, MultiPolygon
from rasterio.plot import show

import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import numpy as np
import os

# import elevation
from tqdm import tqdm


def get_elevation_bands(
    rasterio_path, basin_polygon, crs, code, band_ranges, band_range=500
):
    """Function to get elevation bands from a raster within a basin polygon."""

    with rasterio.open(rasterio_path) as src:
        out_image, out_transform = mask(src, [basin_polygon], crop=True)
        out_meta = src.meta.copy()
        nan_value = src.nodata

    # set the nodata value to nan
    out_image[out_image == nan_value] = np.nan

    if band_ranges is None:
        min_elevation = np.nanmin(out_image[0])
        max_elevation = np.nanmax(out_image[0])
        this_start = min_elevation
        band_ranges_new = []

        while this_start <= max_elevation:
            band_ranges_new.append((this_start, this_start + band_range))
            this_start += band_range

        band_ranges = band_ranges_new

    mean_elevation_per_range = [(lower + upper) / 2 for lower, upper in band_ranges]

    # Initialize an array to store elevation band IDs
    elevation_bands = np.zeros_like(out_image[0])

    # Iterate over each band range
    for band_id, (lower, upper) in enumerate(band_ranges, start=1):
        # Mask pixels within the current band range
        mask_band = np.logical_and(out_image[0] > lower, out_image[0] <= upper)
        # Assign band ID to masked pixels
        elevation_bands[mask_band] = band_id

    # Convert elevation bands to polygons
    shapes_gen = shapes(elevation_bands.astype("uint8"), transform=out_transform)

    # loop through the shapes and create a GeoDataFrame
    elev_dict = {}
    for poly, value in shapes_gen:
        if value not in elev_dict:
            elev_dict[value] = [shape(poly)]
        else:
            elev_dict[value].append(shape(poly))

    elev_values = []
    elev_polygons = []
    mean_elevation = []
    for value, polys in elev_dict.items():
        elev_values.append(value)
        elev_polygons.append(MultiPolygon(polys))
        mean_elevation.append(mean_elevation_per_range[int(value) - 1])

    elev_polygons = gpd.GeoDataFrame(
        {"geometry": elev_polygons, "CODE": code, "elevation_band": elev_values},
        crs=crs,
    )

    return elev_polygons


def create_shapefiles_with_elevation_bands(
    path_to_shp, path_to_dems, elevation_bands=None, band_range=500
):
    # Create an empty list to store elevation band GeoDataFrames
    all_elev_polygons = []
    basins_outline = gpd.read_file(path_to_shp)

    # Load elevation raster
    crs = basins_outline.crs

    for index, basin in basins_outline.iterrows():
        code_basin = basin["CODE"]

        rasterio_path = path_to_dems + "/" + code_basin + ".tif"

        total_area = basin.geometry.area

        elev_polygons = get_elevation_bands(
            rasterio_path,
            basin["geometry"],
            crs=crs,
            code=code_basin,
            band_ranges=elevation_bands,
            band_range=band_range,
        )

        # elev_polygons_projected = elev_polygons.to_crs(epsg=32642)
        relative_area = elev_polygons["geometry"].area / total_area

        # Assign back to the original GeoDataFrame (which remains in EPSG:4326)
        elev_polygons["relative_area"] = relative_area

        # Add the elevation band GeoDataFrame to the list
        all_elev_polygons.append(elev_polygons)

    combined_elev_polygons = pd.concat(all_elev_polygons)
    combined_elev_polygons["id"] = (
        combined_elev_polygons.CODE
        + "_"
        + combined_elev_polygons.elevation_band.astype(int).astype(str)
    )

    # drop elevation band 0
    combined_elev_polygons = combined_elev_polygons[
        combined_elev_polygons["elevation_band"] != 0
    ]

    return combined_elev_polygons


# Function to extract mean value within each polygon
def extract_sca(raster, polygons):
    from exactextract import exact_extract

    include_cols = [
        col for col in polygons.columns if col in ["CODE", "elevation_band"]
    ]

    sca = exact_extract(
        raster, polygons, ["frac", "unique"], include_cols=include_cols, output="pandas"
    )

    def get_sca(row):
        unique_vals = np.array(row["unique"])
        fracs = np.array(row["frac"])
        return fracs[unique_vals == 8.0][0] if 8.0 in unique_vals else 0.0

    # Create transformed dataframe using only the columns that were included
    result_cols = {col: sca[col] for col in include_cols}
    result_cols["sca"] = sca.apply(get_sca, axis=1)

    transformed_df = pd.DataFrame(result_cols)

    return transformed_df


def extract_sca_rasterio(raster_path, polygons):
    include_cols = [
        col for col in polygons.columns if col in ["CODE", "elevation_band"]
    ]
    result_cols = {col: [] for col in include_cols}
    result_cols["sca"] = []

    with rasterio.open(raster_path) as src:
        # Process each polygon separately
        for idx, polygon in polygons.iterrows():
            # Get polygon geometry
            geom = [polygon.geometry.__geo_interface__]

            # Mask raster with polygon
            out_image, out_transform = mask(
                src, geom, crop=True, nodata=0, all_touched=True
            )

            # Get valid data (excluding nodata)
            valid_data = out_image[0][out_image[0] != 0]

            if len(valid_data) > 0:
                # Calculate snow percentage (pixels with value 8.0)
                snow_pixels = np.sum(valid_data == 8.0)
                total_pixels = len(valid_data)
                snow_percentage = snow_pixels / total_pixels
            else:
                snow_percentage = 0.0

            # Store the results
            for col in include_cols:
                result_cols[col].append(polygon[col])
            result_cols["sca"].append(snow_percentage)

    return pd.DataFrame(result_cols)


def iter_over_shp(path_to_sca, shapefiles, year_limit=None, day_limit=None):
    """
    Function to iterate over all the shapefiles and extract the SCA values for each shapefile
    year_limit: tuple of two integers, the lower and upper limit of years to process
    """

    # Get and sort year folders
    year_folders = [
        f
        for f in glob(os.path.join(path_to_sca, "*"))
        if os.path.isdir(f) and f.split(os.path.sep)[-1].isdigit()
    ]
    year_folders.sort()

    # sort the shapefile by 'elevation_band' and 'CODE'
    shapefiles = shapefiles.sort_values(by=["elevation_band", "CODE"])

    # Count total number of files for overall progress
    total_files = sum(len(glob(os.path.join(year, "*.asc"))) for year in year_folders)

    pbar = tqdm(total=total_files, desc="Processing files")

    results_df = pd.DataFrame()

    for year in year_folders:
        year_ = os.path.basename(year)

        if year_limit:
            year_int = int(year_)
            lower, upper = year_limit
            if year_int < lower or year_int > upper:
                continue

        # Get all ASC files in the folder
        asc_files = glob(os.path.join(year, "*.asc"))
        # Sort the files by the day of year
        asc_files.sort(
            key=lambda x: int(os.path.basename(x).split("_")[0][4:]), reverse=True
        )

        # Create year-specific progress bar
        months_passed = []

        for file_path in asc_files:
            file = os.path.basename(file_path)
            day_of_year = file.split("_")[0][4:]

            if day_limit:
                day_int = int(day_of_year)
                lower, upper = day_limit
                if day_int < lower or day_int > upper:
                    continue

            date = pd.to_datetime(year_ + day_of_year, format="%Y%j")
            day_of_month = date.day
            month = date.month

            try:
                sca_values = extract_sca(file_path, shapefiles)

                sca_values["date"] = date

                results_df = pd.concat([results_df, sca_values], ignore_index=True)
                months_passed.append(month)
            except Exception as e:
                print(f"Error processing file {file_path}")
                print(e)
                continue

            # Update progress bar with current year and date
            pbar.set_description(f"Year {year_} - {date.strftime('%Y-%m-%d')}")
            pbar.update(1)

    pbar.close()

    return results_df


def extract_sca_data(
    path_to_dem,
    path_to_shp,
    path_to_sca,
    elevation_bands=None,
    band_range=500,
    year_limit=None,
    day_limit=None,
    selected_basin=None,
):
    # Create shapefiles with elevation bands
    shapefiles = create_shapefiles_with_elevation_bands(
        path_to_shp, path_to_dem, elevation_bands=elevation_bands, band_range=band_range
    )

    if selected_basin:
        shapefiles = shapefiles[shapefiles["CODE"].isin(selected_basin)]

    # Extract SCA data
    sca_data = iter_over_shp(
        path_to_sca, shapefiles, year_limit=year_limit, day_limit=day_limit
    )

    return sca_data, shapefiles
