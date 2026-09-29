"""PDS4 IIRS BSQ cube access: scientific samples, never display imagery.

XML and ENVI HDR must agree. The opt-in 20210628 signedness exception follows
the XML only after checking this exact product, layout, size, MD5 and EVERY
sample's sign bit. It is not a general XML-over-HDR precedence rule. Source
labels are preserved verbatim. No radiometric calibration is performed.
"""
from __future__ import annotations

import hashlib
from numbers import Integral, Real
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .image_loader import ImageData
from .metadata import NS, get_text


EXCEPTION_STEM = 'ch2_iir_nri_20210628T1759527938_d_img_d32'
EXCEPTION_MD5 = 'f0695b144502fc771c05a1731d24915b'
EXCEPTION_BYTES = 3283840000
TYPES = {'UnsignedLSB2': ('<u2', 12), 'SignedLSB2': ('<i2', 2)}


def discover_iirs(directory: str | Path) -> tuple[Path, ...]:
    """Discover labelled QUB/HDR triplets in stable path order; no downloads."""
    return tuple(p for p in sorted(Path(directory).rglob('*.qub'))
                 if p.with_suffix('.xml').is_file() and p.with_suffix('.hdr').is_file())


def read_envi_header(path: str | Path) -> dict[str, str]:
    """Parse the inspected scalar ENVI header; reject ambiguous/duplicate keys."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if not lines or lines[0].strip() != 'ENVI':
        raise ValueError('Expected ENVI header')
    result = {}
    for line in lines[1:]:
        if not line.strip() or line.lstrip().startswith(';'):
            continue
        if '=' not in line:
            raise ValueError('Unsupported ENVI header syntax')
        key, value = (s.strip() for s in line.split('=', 1))
        key = key.lower()
        if key in result or not value:
            raise ValueError('Duplicate or empty ENVI field')
        result[key] = value
    return result


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def read_iirs_metadata(label: str | Path, header: str | Path | None = None) -> dict:
    """Read declarations without resolving a signedness conflict or reading QUB.

    Only the inspected contiguous BSQ, little-endian 16-bit layout is supported.
    Unsupported organizations fail explicitly. Missing optional scientific fields
    stay None. Band widths are reported as supplied, not relabelled as sampling.
    """
    label = Path(label)
    raw_xml = label.read_text(encoding='utf-8')
    root = ET.fromstring(raw_xml)
    if root.tag != '{'+NS['pds']+'}Product_Observational':
        raise ValueError('Expected PDS4 Product_Observational')
    instrument = [get_text(c, 'pds:name') for c in root.findall('.//pds:Observing_System_Component', NS)
                  if get_text(c, 'pds:type') == 'Instrument']
    if len(instrument) != 1 or instrument[0].lower() not in {'iirs', 'imaging infrared spectrometer'}:
        raise ValueError('Unsupported sensor: XML instrument is not IIRS')
    array = root.find('.//pds:File_Area_Observational/pds:Array_3D_Spectrum', NS)
    if array is None:
        raise ValueError('Expected IIRS Array_3D_Spectrum')
    def required(parent, field):
        if parent is None:
            raise ValueError(f'Missing IIRS parent for {field}')
        text = get_text(parent, 'pds:'+field)
        if text is None:
            raise ValueError(f'Missing IIRS field {field}')
        return text
    axes = array.findall('pds:Axis_Array', NS)
    sequence = [(required(a, 'axis_name'), int(required(a, 'sequence_number'))) for a in axes]
    if (int(required(array, 'axes')) != 3 or sorted(sequence, key=lambda a:a[1]) !=
            [('BAND', 1), ('LINE', 2), ('SAMPLE', 3)] or
            required(array, 'axis_index_order') != 'Last Index Fastest'):
        raise ValueError('Unsupported IIRS layout: expected contiguous BSQ')
    dims = {required(a, 'axis_name'): _positive(int(required(a, 'elements')), 'dimension') for a in axes}
    dtype = required(array.find('pds:Element_Array', NS), 'data_type')
    if dtype not in TYPES:
        raise ValueError(f'Unsupported IIRS datatype {dtype}')
    offset = int(required(array, 'offset'))
    if offset != 0:
        raise ValueError('Only inspected zero-offset IIRS cubes supported')
    file = root.find('pds:File_Area_Observational/pds:File', NS)
    name = required(file, 'file_name')
    if Path(name).name != name or not name.lower().endswith('.qub'):
        raise ValueError('Expected plain QUB filename')
    size = _positive(int(required(file, 'file_size')), 'file size')
    expected = dims['BAND']*dims['LINE']*dims['SAMPLE']*2
    if size != expected:
        raise ValueError('XML file size disagrees with contiguous cube dimensions')
    header_path = Path(header) if header else label.with_suffix('.hdr')
    hdr = read_envi_header(header_path)
    expected_hdr = {'samples': str(dims['SAMPLE']), 'lines': str(dims['LINE']),
                    'bands': str(dims['BAND']), 'header offset': '0',
                    'file type': 'ENVI Standard', 'interleave': 'bsq', 'byte order': '0'}
    for key, value in expected_hdr.items():
        if hdr.get(key) != value:
            raise ValueError(f'XML/HDR conflict or unsupported field: {key}')
    hdr_type = int(hdr.get('data type', '-1'))
    if hdr_type not in {2, 12}:
        raise ValueError('Unsupported ENVI datatype')
    status = get_text(root, './/pds:Primary_Result_Summary/pds:processing_level')
    parts = Path(name).stem.split('_')
    if len(parts) != 7 or parts[:2] != ['ch2', 'iir'] or len(parts[2]) != 3:
        raise ValueError('Unrecognized IIRS product identifier')
    id_status = {'r': 'Raw', 'c': 'Calibrated', 'd': 'Derived'}.get(parts[2][1])
    if id_status is None or status != id_status:
        raise ValueError('Product ID and XML processing level disagree')
    lid = get_text(root, 'pds:Identification_Area/pds:logical_identifier')
    if not lid or lid.rsplit(':', 1)[-1] != Path(name).stem.lower():
        raise ValueError('Logical identifier and filename disagree')
    def optional(name, numeric=False):
        text = get_text(root, './/isda:'+name)
        if text is None:
            return None
        if not numeric:
            return text
        number = float(text)
        if not np.isfinite(number):
            raise ValueError(f'Nonfinite metadata {name}')
        return number
    bins = array.findall('.//pds:Band_Bin', NS)
    wavelengths, widths, units = [None]*dims['BAND'], [None]*dims['BAND'], set()
    seen = set()
    for bin in bins:
        number = int(required(bin, 'band_number'))
        if not 1 <= number <= dims['BAND'] or number in seen:
            raise ValueError('Invalid/duplicate official band number')
        seen.add(number)
        element = bin.find('pds:center_wavelength', NS)
        value = float(required(bin, 'center_wavelength'))
        if not np.isfinite(value) or value <= 0 or not element.get('unit'):
            raise ValueError('Invalid wavelength or missing units')
        units.add(element.get('unit'))
        wavelengths[number-1] = value
        width = bin.find('pds:band_width', NS)
        if width is not None:
            v = float(width.text)
            if not np.isfinite(v) or v <= 0 or width.get('unit') != element.get('unit'):
                raise ValueError('Invalid band width')
            widths[number-1] = v
    if len(units) > 1:
        raise ValueError('Mixed wavelength units unsupported')
    # Actual product contains no bad-band or special-value declarations. Never
    # silently ignore a future declaration whose semantics have not been added.
    if any(any(k in e.tag.lower() for k in ('special_constant', 'bad_band', 'invalid_constant')) for e in array.iter()):
        raise ValueError('Unsupported IIRS validity metadata; explicit interpretation required')
    known = [v for v in wavelengths if v is not None]
    footprint = {c: {'longitude': optional(c+'_longitude', True), 'latitude': optional(c+'_latitude', True)}
                 for c in ('upper_left', 'upper_right', 'lower_left', 'lower_right')}
    return dict(instrument='IIRS', mission=get_text(root, './/pds:Investigation_Area/pds:name'),
        product_id=Path(name).stem, logical_identifier=lid, product_type=status,
        processing_level=status, product_id_status=id_status, product_id_codes=dict(phase=parts[2][0], type=parts[2][1], instrument=parts[2][2]),
        observation_start=get_text(root, './/pds:start_date_time'), observation_stop=get_text(root, './/pds:stop_date_time'),
        orbit=optional('imaging_orbit_number'), height=dims['LINE'], width=dims['SAMPLE'], bands=dims['BAND'],
        xml_data_type=dtype, hdr_data_type=hdr_type, dtype=str(np.dtype(TYPES[dtype][0])),
        signedness='unsigned' if dtype.startswith('Unsigned') else 'signed', byte_order='little-endian', interleave='BSQ',
        header_bytes=offset, expected_image_bytes=expected, expected_total_bytes=size, file_name=name,
        md5_checksum=get_text(file, 'pds:md5_checksum'), native_gsd_m=optional('pixel_resolution', True),
        wavelengths=wavelengths, wavelength_units=next(iter(units), None),
        wavelength_min=min(known) if known else None, wavelength_max=max(known) if known else None,
        spectral_sampling=np.diff(wavelengths).tolist() if len(known)==dims['BAND'] else None,
        band_widths=widths, spectral_resolution=None, invalid_bands=[], validity_declared=False,
        nodata=None, saturation=None, radiometric_units=None, stored_quantity='DN' if status=='Raw' else None,
        footprint=footprint, sun_azimuth=optional('sun_azimuth', True), sun_elevation=optional('sun_elevation', True),
        solar_incidence=optional('solar_incidence', True), raw_xml=raw_xml,
        raw_hdr=header_path.read_text(encoding='utf-8'), hdr_fields=hdr)


def _scan_words(path: Path) -> dict:
    """Bounded-memory full binary evidence scan, not a sample of the cube."""
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    count = sign_bits = 0
    minimum, maximum = 65535, 0
    with path.open('rb') as stream:
        while chunk := stream.read(16*1024*1024):
            md5.update(chunk)
            sha256.update(chunk)
            if len(chunk) % 2:
                raise ValueError('Incomplete 16-bit sample')
            values = np.frombuffer(chunk, dtype='<u2')
            count += len(values)
            sign_bits += int(np.count_nonzero(values >= 32768))
            minimum, maximum = min(minimum, int(values.min())), max(maximum, int(values.max()))
    return dict(md5=md5.hexdigest(), sha256=sha256.hexdigest(), samples=count,
                sign_bit_set=sign_bits, minimum=minimum, maximum=maximum)


def _compatibility(path: Path, metadata: dict, enabled: bool, actual: int) -> dict:
    conflict = metadata['hdr_data_type'] != TYPES[metadata['xml_data_type']][1]
    report = dict(xml=metadata['xml_data_type'], hdr_envi_data_type=metadata['hdr_data_type'],
                  conflict=conflict, exception_applied=False, evidence=None)
    if not conflict:
        return report
    if not enabled:
        raise ValueError('XML/HDR signedness conflict; strict consistency required')
    if not (metadata['product_id']==EXCEPTION_STEM and path.name==EXCEPTION_STEM+'.qub'
            and metadata['xml_data_type']=='UnsignedLSB2' and metadata['hdr_data_type']==2
            and (metadata['height'],metadata['width'],metadata['bands'])==(25655,250,256)
            and metadata['interleave']=='BSQ' and metadata['byte_order']=='little-endian'
            and metadata['header_bytes']==0 and metadata['expected_total_bytes']==actual==EXCEPTION_BYTES
            and metadata['md5_checksum']==EXCEPTION_MD5):
        raise ValueError('Conflict is outside the approved product-specific exception')
    evidence = _scan_words(path)
    if (evidence['md5'] != EXCEPTION_MD5 or evidence['sign_bit_set'] != 0
            or evidence['samples']*2 != actual):
        raise ValueError('Product-specific signedness exception evidence failed')
    report.update(exception_applied=True, evidence=evidence,
        decision='Explicitly authorized XML uint16 interpretation for this validated product only; HDR int16 retained unchanged.')
    return report


class IIRSCube:
    """Read-only BSQ memmap with independent outputs in (band,row,column) order.

    An integer band yields a 2D image. Sequences/None yield a 3D selected/full
    cube. Python band indices are zero-based; XML band numbers are one-based.
    Construction validates metadata and file once; reuse the instance for reads.
    """
    def __init__(self, path: str | Path, metadata_path: str | Path | None = None,
                 *, allow_validated_signedness_exception: bool = False):
        if not isinstance(allow_validated_signedness_exception, bool):
            raise ValueError('Signedness exception requires an explicit boolean opt-in')
        if Path(path).suffix.lower() not in {'.qub','.xml','.hdr'}:
            raise ValueError('Expected IIRS QUB, XML or HDR path')
        self.path = Path(path).resolve().with_suffix('.qub')
        self.label = Path(metadata_path).resolve() if metadata_path else self.path.with_suffix('.xml')
        self.metadata = read_iirs_metadata(self.label)
        m = self.metadata
        actual = self.path.stat().st_size
        if self.path.name != m['file_name'] or actual != m['expected_total_bytes']:
            raise ValueError('IIRS filename or exact file-size mismatch')
        m['actual_bytes'] = actual
        m['compatibility'] = _compatibility(self.path, m, allow_validated_signedness_exception, actual)
        self._array = np.memmap(self.path, mode='r', dtype=TYPES[m['xml_data_type']][0],
                                shape=(m['bands'], m['height'], m['width']))

    def close(self) -> None:
        """Release the mapping (including Windows file handles); outputs are copies."""
        if self._array is not None:
            self._array._mmap.close()
            self._array = None

    def __enter__(self) -> IIRSCube:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def band_indices(self, bands=None) -> list[int]:
        if bands is None:
            return list(range(self.metadata['bands']))
        values = [bands] if isinstance(bands, Integral) else bands
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError('Band indices must be one-dimensional')
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values)==0:
            raise ValueError('Expected nonempty band indices')
        if any(isinstance(b, bool) or not isinstance(b, Integral) or not 0<=b<self.metadata['bands'] for b in values):
            raise ValueError('Band index out of range or noninteger')
        if len(set(values)) != len(values):
            raise ValueError('Duplicate band indices')
        return [int(b) for b in values]

    def read(self, bands=None, *, window: tuple[int,int,int,int] | None = None,
             max_bytes: int = 256*1024*1024) -> ImageData:
        """Half-open window (row_start,row_stop,col_start,col_stop); no clamping."""
        m = self.metadata
        if self._array is None:
            raise ValueError('Cube is closed')
        r0,r1,c0,c1 = validate_window(window, m['height'], m['width'])
        indices = self.band_indices(bands)
        if len(indices)*(r1-r0)*(c1-c0)*2 > _positive(max_bytes, 'max_bytes'):
            raise ValueError('Requested cube exceeds max_bytes; select bands/window')
        data = np.stack([np.array(self._array[b,r0:r1,c0:c1], copy=True) for b in indices])
        if isinstance(bands, Integral):
            data = data[0]
        metadata = dict(m, selected_bands=indices, official_band_numbers=[b+1 for b in indices],
                        subset=dict(row_start=r0,row_stop=r1,column_start=c0,column_stop=c1),
                        array_order='row,column' if data.ndim==2 else 'band,row,column')
        return ImageData(data, metadata, self.path, self.label, 'IIRS', 'raw')

    def wavelength(self, band: int) -> float | None:
        return self.metadata['wavelengths'][self.band_indices(band)[0]]

    def nearest_band(self, wavelength: float) -> int:
        if isinstance(wavelength,bool) or not isinstance(wavelength,Real) or not np.isfinite(wavelength) or wavelength<=0:
            raise ValueError('Wavelength must be finite and positive in metadata units')
        indices = [b for b,w in enumerate(self.metadata['wavelengths']) if w is not None and b not in self.metadata['invalid_bands']]
        if not indices:
            raise ValueError('No valid wavelength metadata')
        return min(indices, key=lambda b:(abs(self.metadata['wavelengths'][b]-wavelength),b))

    def mean(self, bands, *, window=None) -> tuple[np.ndarray, np.ndarray]:
        """Streaming float64 accumulation; skip declared bad bands/invalid samples.

        Output is float32 mean DN, not radiance. Zero valid contributions -> NaN
        and false mask. Actual product declares no invalid bands or nodata;
        zero is therefore retained, not guessed to be missing terrain.
        """
        if self._array is None:
            raise ValueError('Cube is closed')
        indices = [b for b in self.band_indices(bands) if b not in self.metadata['invalid_bands']]
        if not indices:
            raise ValueError('No usable selected bands')
        r0,r1,c0,c1 = validate_window(window,self.metadata['height'],self.metadata['width'])
        total = np.zeros((r1-r0,c1-c0),np.float64)
        count = np.zeros(total.shape,np.int32)
        for b in indices:
            values = self._array[b,r0:r1,c0:c1]
            valid = valid_samples(values,self.metadata.get('nodata'))
            total += np.where(valid,values,0)
            count += valid
        output = np.full(total.shape,np.nan,np.float32)
        np.divide(total,count,out=output,where=count>0)
        return output,count>0


def validate_window(window, height: int, width: int) -> tuple[int,int,int,int]:
    if window is None:
        return 0,height,0,width
    if not isinstance(window,tuple) or len(window)!=4 or any(isinstance(v,bool) or not isinstance(v,Integral) for v in window):
        raise ValueError('Window must be four integers')
    r0,r1,c0,c1 = map(int,window)
    if not (0<=r0<r1<=height and 0<=c0<c1<=width):
        raise ValueError('Window outside cube or empty')
    return r0,r1,c0,c1


def valid_samples(values: np.ndarray, nodata=None) -> np.ndarray:
    mask = np.isfinite(values)
    if nodata is not None:
        mask &= values != nodata
    return mask


def representation_bands(cube: IIRSCube) -> dict:
    """Predetermined near-visible diagnostic: nearest 1000 nm, mean 900..1100 nm.

    No match outcomes influence this choice. If nm wavelengths are unavailable,
    use the middle usable band and at most nine central usable bands, explicitly
    reporting the fallback. 'Usable' means not declared invalid, not validated
    detector quality. No spectral interpolation or calibration is performed.
    """
    m = cube.metadata
    usable = [b for b in range(m['bands']) if b not in m['invalid_bands']]
    if not usable:
        raise ValueError('No usable bands')
    interval = [b for b in usable if m['wavelengths'][b] is not None and 900<=m['wavelengths'][b]<=1100]
    if m['wavelength_units']=='nm' and interval:
        single, mean, rule = cube.nearest_band(1000), interval, 'Nearest 1000 nm; mean over 900..1100 nm, inclusive'
    else:
        center = len(usable)//2
        single, mean = usable[center], usable[max(0,center-4):center+5]
        rule = 'Wavelength interval unavailable: middle usable band and up to nine central usable bands'
    return dict(single_band=single, mean_bands=mean, rule=rule)


def load_iirs_image(path: str | Path, metadata_path=None, *, bands=None, window=None,
                    allow_validated_signedness_exception=False) -> ImageData:
    """Common interface; callers must explicitly reduce a returned 3D cube."""
    with IIRSCube(path, metadata_path,
        allow_validated_signedness_exception=allow_validated_signedness_exception) as cube:
        return cube.read(bands,window=window)
