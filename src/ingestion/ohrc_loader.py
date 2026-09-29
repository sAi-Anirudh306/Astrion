"""Label-driven OHRC scientific image loading, never writable source mappings."""
from pathlib import Path
from numbers import Integral
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .metadata import read_metadata, NS, get_text
from .image_loader import ImageData


def read_ohrc_metadata(label: str | Path) -> dict:
    """Preserve generic PDS4 metadata plus complete raw XML and normalized fields.

    The observed calibrated Array_2D_Image is one band of UnsignedByte, with
    contiguous samples. Byte order is not applicable to one-byte samples.
    Other types/layouts fail explicitly until supported by inspected products.
    """
    path = Path(label)
    metadata = read_metadata(path)
    if str(metadata.get('instrument', '')).lower() not in {'ohrc', 'orbiter high resolution camera'}:
        raise ValueError('Unsupported sensor: XML instrument is not OHRC')
    if metadata['data_type'] != 'UnsignedByte':
        raise ValueError('Unsupported OHRC type: expected inspected UnsignedByte layout')
    if metadata['axis_sequence'] != {'Line': 1, 'Sample': 2} or metadata['axis_index_order'] != 'Last Index Fastest':
        raise ValueError('Unsupported OHRC axis layout')
    if metadata.get('processing_level') != 'Calibrated':
        raise ValueError('Only label-declared calibrated OHRC products supported')
    raw_xml = path.read_text(encoding='utf-8')
    root = ET.fromstring(raw_xml)
    params = root.find('.//isda:Product_Parameters', NS)
    def optional(name):
        value = get_text(params, 'isda:'+name) if params is not None else None
        if value is None:
            return None
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f'Nonfinite {name}')
        return number
    normalized = dict(instrument='OHRC', product_type='calibrated browse' if metadata['file_name'].lower().endswith('.png') else 'calibrated image', calibrated=True,
        observation_start=metadata.get('start_date_time'), observation_stop=metadata.get('stop_date_time'),
        orbit=metadata.get('imaging_orbit_number'), width=metadata['image_width'], height=metadata['image_height'],
        bands=1, dtype='uint8', sample_bits=8, byte_order='not applicable (one-byte samples)',
        native_gsd_m=metadata.get('pixel_resolution_m_per_pixel'), altitude_km=metadata.get('spacecraft_altitude_km'),
        focal_length_mm=optional('focal_length'), detector_pixel_width_micrometres=optional('detector_pixel_width'),
        sun_azimuth_deg=metadata.get('sun_azimuth_deg'), sun_elevation_deg=metadata.get('sun_elevation_deg'),
        solar_incidence_deg=metadata.get('solar_incidence_deg'),
        attitude_degrees={axis: metadata.get(axis+'_deg') for axis in ('roll', 'pitch', 'yaw')},
        footprint={corner: [metadata.get(corner+'_longitude'), metadata.get(corner+'_latitude')]
                   for corner in ('upper_left', 'upper_right', 'lower_left', 'lower_right')})
    return dict(metadata, normalized=normalized, raw_xml=raw_xml)


def validate_ohrc_file(path: str | Path, metadata: dict) -> int:
    path = Path(path)
    if path.name != metadata['file_name']:
        raise ValueError('Image filename differs from XML label')
    actual = path.stat().st_size
    expected = metadata['offset_bytes'] + metadata['image_height']*metadata['image_width']
    if path.suffix.lower() == '.img' and expected != metadata['file_size_bytes']:
        raise ValueError('OHRC dimensions/offset disagree with declared file size')
    if actual != metadata['file_size_bytes']:
        raise ValueError(f'OHRC file-size mismatch: expected {metadata["file_size_bytes"]}, got {actual}')
    return actual


def _window(window: tuple | None, height: int, width: int) -> tuple[int, int, int, int]:
    if window is None:
        return 0, height, 0, width
    if (not isinstance(window, tuple) or len(window) != 4 or
            any(isinstance(v, bool) or not isinstance(v, Integral) for v in window)):
        raise ValueError('Window must be integer (row_start,row_stop,column_start,column_stop)')
    r0, r1, c0, c1 = window
    if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
        raise ValueError('Window outside image or empty')
    return tuple(int(v) for v in window)


def load_ohrc_image(image_path: str | Path, metadata_path: str | Path | None = None,
                    *, window: tuple[int, int, int, int] | None = None,
                    max_bytes: int = 256*1024*1024) -> ImageData:
    """Read full small images or efficient windows via read-only memmap.

    Calibrated scientific uint8 values are preserved, not display-normalized.
    Full loads above max_bytes require an explicitly raised limit or a window.
    ImageData.product_kind='raw' means scientific samples, not uncalibrated.
    """
    path = Path(image_path).resolve()
    label = Path(metadata_path).resolve() if metadata_path else path.with_suffix('.xml')
    if path.suffix.lower() not in {'.img', '.png'}:
        raise ValueError('OHRC supports calibrated IMG and labelled browse PNG')
    metadata = read_ohrc_metadata(label)
    actual = validate_ohrc_file(path, metadata)
    h, w = metadata['image_height'], metadata['image_width']
    r0, r1, c0, c1 = _window(window, h, w)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral) or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    if (r1-r0)*(c1-c0) > max_bytes:
        raise ValueError('Requested OHRC load exceeds max_bytes; use a smaller window')
    if path.suffix.lower() == '.img':
        mapped = np.memmap(path, dtype='u1', mode='r', offset=metadata['offset_bytes'], shape=(h, w))
        data = np.array(mapped[r0:r1, c0:c1], copy=True)
        del mapped
        kind = 'raw'
    else:
        if window is not None or metadata['offset_bytes'] != 0:
            raise ValueError('Browse PNG requires full image and zero offset')
        data = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if data is None or data.shape != (h, w) or data.dtype != np.uint8:
            raise ValueError('OHRC browse dimensions/type do not match label')
        kind = 'browse'
    metadata.update(actual_file_size=actual, subset=dict(row_start=r0, row_stop=r1,
        column_start=c0, column_stop=c1, shape=list(data.shape),
        first_sample_byte_offset=metadata['offset_bytes']+r0*w+c0,
        row_stride_bytes=w, coordinate_mapping='full_xy = crop_xy + [column_start,row_start]'))
    return ImageData(data, metadata, path, label, 'OHRC', kind)


def crop_to_full(points: np.ndarray, subset: dict) -> np.ndarray:
    """Translate valid crop (x,y) pixel centers to full-product coordinates."""
    points = np.asarray(points)
    shape = subset['shape']
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'fiu' or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx2 points')
    if np.any(points < 0) or np.any(points > np.array(shape[::-1])-1):
        raise ValueError('Point outside crop')
    return points.astype(float)+[subset['column_start'], subset['row_start']]


def read_geometry_label(path: str | Path) -> dict:
    """Read the actual labelled four-column table; retain unspecified semantics."""
    root = ET.parse(path).getroot()
    table = root.find('.//pds:Table_Delimited', NS)
    if table is None:
        raise ValueError('Missing geometry Table_Delimited')
    fields = [dict(name=get_text(f, 'pds:name'), number=int(get_text(f, 'pds:field_number')),
                   dtype=get_text(f, 'pds:data_type'), unit=get_text(f, 'pds:unit'),
                   description=get_text(f, 'pds:description'))
              for f in table.findall('pds:Record_Delimited/pds:Field_Delimited', NS)]
    fields.sort(key=lambda f: f['number'])
    if [f['name'] for f in fields] != ['Longitude', 'Latitude', 'Pixel', 'Scan']:
        raise ValueError('Unsupported geometry field names/order')
    if [f['dtype'] for f in fields] != ['ASCII_Real', 'ASCII_Real', 'ASCII_Integer', 'ASCII_Integer']:
        raise ValueError('Unsupported geometry field types')
    if get_text(table, 'pds:field_delimiter') != 'Comma':
        raise ValueError('Unsupported geometry delimiter')
    file = root.find('.//pds:File_Area_Observational/pds:File', NS)
    header = root.find('.//pds:Header', NS)
    return dict(fields=fields, records=int(get_text(table, 'pds:records')),
        table_offset=int(get_text(table, 'pds:offset')), header_length=int(get_text(header, 'pds:object_length')),
        record_delimiter=get_text(table, 'pds:record_delimiter'),
        file_name=get_text(file, 'pds:file_name'), file_size=int(get_text(file, 'pds:file_size')),
        units_specified=False, origin_specified=False, longitude_convention_specified=False,
        missing_value_convention=None, raw_xml=Path(path).read_text())


def load_ohrc_geometry(csv_path: str | Path, image_shape: tuple[int, int], *,
                       allow_header_length_discrepancy: bool = False,
                       label_path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Validate schema/counts and filter explicitly reported invalid numeric rows.

    The inspected product proves zero-based pixel/scan via 0..width-1 and
    0..height-1 extrema; origin/degree units are inferred, not XML declarations.
    Only the known header-length versus table-offset discrepancy can be waived.
    """
    import io
    if not isinstance(allow_header_length_discrepancy, bool):
        raise ValueError('Header exception flag must be boolean')
    if len(image_shape) != 2 or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in image_shape):
        raise ValueError('Image shape must contain two positive integers')
    path = Path(csv_path)
    schema = read_geometry_label(label_path or path.with_suffix('.xml'))
    content = path.read_bytes()
    if len(content) != schema['file_size'] or path.name != schema['file_name']:
        raise ValueError('Geometry filename/file-size mismatch')
    header = content.splitlines(keepends=True)[0]
    discrepancies = []
    if len(header) != schema['table_offset']:
        raise ValueError('Actual header does not match XML table offset')
    if len(header) != schema['header_length']:
        if not allow_header_length_discrepancy or (schema['header_length'], schema['table_offset'], len(header)) != (31, 30, 30):
            raise ValueError('XML header length conflicts with table offset/actual header')
        discrepancies.append('Explicit exception: XML header length differs; using verified XML table offset')
    if header.decode('ascii').strip() != 'Longitude,Latitude,Pixel,Scan':
        raise ValueError('CSV header differs from XML fields')
    body = content[schema['table_offset']:]
    if schema['record_delimiter'] != 'Carriage-Return Line-Feed' or body.count(b'\r\n') != schema['records']:
        raise ValueError('CSV record delimiter/count mismatch')
    values = np.loadtxt(io.BytesIO(body), delimiter=',', ndmin=2)
    if values.shape != (schema['records'], 4):
        raise ValueError('CSV shape differs from XML')
    h, w = image_shape
    valid = (np.isfinite(values).all(axis=1) & (values[:, 0] >= -180) & (values[:, 0] <= 360)
        & (np.abs(values[:, 1]) <= 90) & (values[:, 2] >= 0) & (values[:, 2] <= w-1)
        & (values[:, 3] >= 0) & (values[:, 3] <= h-1)
        & (values[:, 2] == np.floor(values[:, 2])) & (values[:, 3] == np.floor(values[:, 3])))
    data = values[valid].copy()
    if not len(data):
        raise ValueError('No valid geometry rows')
    pixels, scans = np.unique(data[:, 2]), np.unique(data[:, 3])
    duplicates = len(data)-len(np.unique(data, axis=0))
    summary = dict(schema=schema, total_data_rows=len(values), total_lines_including_header=len(values)+1,
        valid_rows=len(data), invalid_rows=int((~valid).sum()), invalid_original_indices=np.flatnonzero(~valid).tolist(),
        duplicate_records=duplicates, unique_pixels=len(pixels), unique_scans=len(scans),
        ranges={key: [float(data[:, i].min()), float(data[:, i].max())]
                for i, key in enumerate(['longitude', 'latitude', 'pixel', 'scan'])},
        pixel_spacing=np.unique(np.diff(pixels)).tolist(), scan_spacing=np.unique(np.diff(scans)).tolist(),
        complete_rectilinear_grid=len(np.unique(data[:, 2:], axis=0)) == len(pixels)*len(scans),
        scan_major_pixel_minor=bool(np.array_equal(np.lexsort((data[:, 2], data[:, 3])), np.arange(len(data)))),
        coordinate_interpretation='Pixel=x, Scan=y; zero-based inferred from extrema matching image bounds; angular degrees inferred from matching labelled degree-valued corners.',
        longitude_interpretation='Local east-longitude degrees consistent with image label; positive range alone cannot distinguish signed versus 0..360 convention.',
        discrepancies=discrepancies)
    return data, summary
