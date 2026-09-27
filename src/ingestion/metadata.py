import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PDS_NS = "http://pds.nasa.gov/pds4/pds/v1"
ISDA_NS = "https://isda.issdc.gov.in/pds4/isda/v1"

NS = {
    "pds": PDS_NS,
    "isda": ISDA_NS,
}


def get_text(parent: ET.Element, path: str, default: Any = None) -> Any:
    """Return the text of an XML element, or default if it doesn't exist."""
    element = parent.find(path, NS)

    if element is None:
        return default

    return element.text.strip() if element.text else default


def get_float(parent: ET.Element, path: str, default: Any = None) -> Any:
    """Return an XML value as float."""
    value = get_text(parent, path, default)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def get_int(parent: ET.Element, path: str, default: Any = None) -> Any:
    """Return an XML value as integer."""
    value = get_text(parent, path, default)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def read_metadata(xml_path: str | Path) -> dict[str, Any]:
    """
    Read metadata from a Chandrayaan-2 TMC PDS4 XML label.
    """

    xml_path = Path(xml_path)

    if not xml_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {xml_path}")
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed metadata XML in {xml_path}: {exc}") from exc
    root = tree.getroot()
    if root.tag != f"{{{PDS_NS}}}Product_Observational":
        raise ValueError(f"Malformed metadata: expected PDS4 Product_Observational in {xml_path}")

    metadata = {}

    # -----------------------------
    # Identification
    # -----------------------------

    metadata["logical_identifier"] = get_text(
        root,
        "pds:Identification_Area/pds:logical_identifier"
    )

    metadata["version_id"] = get_text(
        root,
        "pds:Identification_Area/pds:version_id"
    )

    metadata["title"] = get_text(
        root,
        "pds:Identification_Area/pds:title"
    )

    # -----------------------------
    # Observation
    # -----------------------------

    metadata["start_date_time"] = get_text(
        root,
        "pds:Observation_Area/pds:Time_Coordinates/pds:start_date_time"
    )

    metadata["stop_date_time"] = get_text(
        root,
        "pds:Observation_Area/pds:Time_Coordinates/pds:stop_date_time"
    )

    metadata["purpose"] = get_text(
        root,
        "pds:Observation_Area/pds:Primary_Result_Summary/pds:purpose"
    )

    metadata["processing_level"] = get_text(
        root,
        "pds:Observation_Area/pds:Primary_Result_Summary/pds:processing_level"
    )

    metadata["target"] = get_text(
        root,
        "pds:Observation_Area/pds:Target_Identification/pds:name"
    )

    # -----------------------------
    # Instrument
    # -----------------------------

    metadata["instrument"] = get_text(
        root,
        "pds:Observation_Area/pds:Observing_System/"
        "pds:Observing_System_Component["
        "pds:type='Instrument']/pds:name"
    )

    # -----------------------------
    # ISDA product parameters
    # -----------------------------

    product_params = root.find(
        "pds:Observation_Area/pds:Mission_Area/"
        "isda:Product_Parameters",
        NS
    )

    if product_params is not None:

        metadata["job_id"] = get_text(
            product_params,
            "isda:job_id"
        )

        metadata["imaging_orbit_number"] = get_int(
            product_params,
            "isda:imaging_orbit_number"
        )

        metadata["dumping_orbit_number"] = get_int(
            product_params,
            "isda:dumping_orbit_number"
        )

        metadata["spacecraft_altitude_km"] = get_float(
            product_params,
            "isda:spacecraft_altitude"
        )

        metadata["pixel_resolution_m_per_pixel"] = get_float(
            product_params,
            "isda:pixel_resolution"
        )

        metadata["roll_deg"] = get_float(
            product_params,
            "isda:roll"
        )

        metadata["pitch_deg"] = get_float(
            product_params,
            "isda:pitch"
        )

        metadata["yaw_deg"] = get_float(
            product_params,
            "isda:yaw"
        )

        metadata["sun_azimuth_deg"] = get_float(
            product_params,
            "isda:sun_azimuth"
        )

        metadata["sun_elevation_deg"] = get_float(
            product_params,
            "isda:sun_elevation"
        )

        metadata["solar_incidence_deg"] = get_float(
            product_params,
            "isda:solar_incidence"
        )

        metadata["projection"] = get_text(
            product_params,
            "isda:projection"
        )

        metadata["area"] = get_text(
            product_params,
            "isda:area"
        )

    # -----------------------------
    # Geometry
    # -----------------------------

    geometry = root.find(
        "pds:Observation_Area/pds:Mission_Area/"
        "isda:Geometry_Parameters/"
        "isda:System_Level_Coordinates",
        NS
    )

    if geometry is not None:

        metadata["upper_left_latitude"] = get_float(
            geometry,
            "isda:upper_left_latitude"
        )

        metadata["upper_left_longitude"] = get_float(
            geometry,
            "isda:upper_left_longitude"
        )

        metadata["upper_right_latitude"] = get_float(
            geometry,
            "isda:upper_right_latitude"
        )

        metadata["upper_right_longitude"] = get_float(
            geometry,
            "isda:upper_right_longitude"
        )

        metadata["lower_left_latitude"] = get_float(
            geometry,
            "isda:lower_left_latitude"
        )

        metadata["lower_left_longitude"] = get_float(
            geometry,
            "isda:lower_left_longitude"
        )

        metadata["lower_right_latitude"] = get_float(
            geometry,
            "isda:lower_right_latitude"
        )

        metadata["lower_right_longitude"] = get_float(
            geometry,
            "isda:lower_right_longitude"
        )

    # -----------------------------
    # Image product information
    # -----------------------------

    image_file = root.find(
        "pds:File_Area_Observational/pds:File",
        NS
    )

    if image_file is not None:

        metadata["file_name"] = get_text(
            image_file,
            "pds:file_name"
        )

        metadata["file_size_bytes"] = get_int(
            image_file,
            "pds:file_size"
        )

        metadata["md5_checksum"] = get_text(
            image_file,
            "pds:md5_checksum"
        )

    image_array = root.find(
        "pds:File_Area_Observational/"
        "pds:Array_2D_Image",
        NS
    )

    if image_array is not None:

        metadata["offset_bytes"] = get_int(image_array, "pds:offset")
        metadata["axes"] = get_int(image_array, "pds:axes")
        metadata["axis_index_order"] = get_text(image_array, "pds:axis_index_order")
        metadata["axis_sequence"] = {}

        metadata["data_type"] = get_text(
            image_array,
            "pds:Element_Array/pds:data_type"
        )

        axes = image_array.findall(
            "pds:Axis_Array",
            NS
        )

        for axis in axes:

            axis_name = get_text(
                axis,
                "pds:axis_name"
            )

            elements = get_int(
                axis,
                "pds:elements"
            )

            if axis_name in metadata["axis_sequence"]:
                raise ValueError(f"Malformed metadata: duplicate axis {axis_name}")
            metadata["axis_sequence"][axis_name] = get_int(axis, "pds:sequence_number")

            if axis_name == "Line":
                metadata["image_height"] = elements

            elif axis_name == "Sample":
                metadata["image_width"] = elements

    for field in ("file_name", "data_type"):
        if not metadata.get(field):
            raise ValueError(f"Malformed metadata: missing {field} in {xml_path}")
    for field in ("file_size_bytes", "image_height", "image_width"):
        value = metadata.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Malformed metadata: {field} must be a positive integer in {xml_path}")
    if metadata.get("offset_bytes") is None or metadata["offset_bytes"] < 0:
        raise ValueError(f"Malformed metadata: offset must be a nonnegative integer in {xml_path}")
    if metadata.get("axes") != 2 or set(metadata["axis_sequence"]) != {"Line", "Sample"}:
        raise ValueError("Malformed metadata: expected exactly two axes, Line and Sample")
    if sorted(v for v in metadata["axis_sequence"].values() if v is not None) != [1, 2]:
        raise ValueError("Malformed metadata: axis sequence numbers must be 1 and 2")
    if metadata["axis_index_order"] not in {"Last Index Fastest", "First Index Fastest"}:
        raise ValueError("Malformed metadata: invalid axis_index_order")
    return metadata


if __name__ == "__main__":

    xml_path = r"data\test\borrow_k\data\raw\20260629\ch2_tmc_nrf_20260629T2059373111_d_img_d18.xml"

    metadata = read_metadata(xml_path)

    print("\nBorrow K Metadata")
    print("=" * 50)

    for key, value in metadata.items():
        print(f"{key}: {value}")
