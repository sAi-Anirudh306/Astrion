"""Milestone 1 unit and real Borrow K integration tests.

Run from the project root: .venv/Scripts/python.exe -m unittest discover -s tests -v
All synthetic files are created in a temporary directory, never under data/.
"""

from pathlib import Path
import hashlib
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from src.ingestion.image_loader import ImageData, load_image
from src.ingestion.metadata import PDS_NS, read_metadata
from src.ingestion.tmc_loader import load_tmc_image


ROOT = Path(__file__).resolve().parents[1]
BORROW = ROOT / "data/test/borrow_k"
RAW = BORROW / "data/raw/20260629/ch2_tmc_nrf_20260629T2059373111_d_img_d18.img"
BROWSE = BORROW / "browse/raw/20260629/ch2_tmc_nrf_20260629T2059373111_b_brw_d18.png"


class TMCUnitTests(unittest.TestCase):
    """Use small asymmetric images and known bytes to detect decoding errors."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "sample.img"
        self.path.write_bytes(b"\x01\x00\x00\x01\xff\xff\x02\x00\x34\x12\x00\x00")
        self.label = self.path.with_suffix(".xml")
        self.label.write_text(f'''<Product_Observational xmlns="{PDS_NS}">
          <Observation_Area><Observing_System><Observing_System_Component>
            <type>Instrument</type><name>terrain mapping camera</name>
          </Observing_System_Component></Observing_System></Observation_Area>
          <File_Area_Observational><File><file_name>sample.img</file_name>
            <file_size unit="byte">12</file_size></File>
            <Array_2D_Image><offset unit="byte">0</offset><axes>2</axes>
              <axis_index_order>Last Index Fastest</axis_index_order>
              <Element_Array><data_type>UnsignedLSB2</data_type></Element_Array>
              <Axis_Array><axis_name>Line</axis_name><elements>2</elements>
                <sequence_number>1</sequence_number></Axis_Array>
              <Axis_Array><axis_name>Sample</axis_name><elements>3</elements>
                <sequence_number>2</sequence_number></Axis_Array>
            </Array_2D_Image></File_Area_Observational></Product_Observational>''', encoding="utf-8")

    def change(self, tag: str, value: str) -> None:
        """Change a synthetic label field only."""
        tree = ET.parse(self.label)
        element = tree.find(f".//{{{PDS_NS}}}{tag}")
        assert element is not None
        element.text = value
        tree.write(self.label, encoding="utf-8")

    def make_png(self) -> None:
        self.path = self.path.with_suffix(".png")
        pixels = np.array([[0, 128, 255], [3, 4, 5]], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", pixels)
        self.assertTrue(ok)
        self.path.write_bytes(encoded.tobytes())
        self.change("file_name", self.path.name)
        self.change("file_size", str(self.path.stat().st_size))
        self.change("data_type", "UnsignedByte")

    def test_metadata(self) -> None:
        metadata = read_metadata(self.label)
        self.assertEqual((metadata["image_height"], metadata["image_width"]), (2, 3))
        self.assertEqual(metadata["file_size_bytes"], 12)
        self.assertEqual(metadata["data_type"], "UnsignedLSB2")

    def test_raw_values_endianness_and_dispatch(self) -> None:
        result = load_image(self.path, sensor="TMC-2")
        self.assertIsInstance(result, ImageData)
        self.assertEqual(result.data.dtype.str, "<u2")
        np.testing.assert_array_equal(result.data, [[1, 256, 65535], [2, 4660, 0]])
        self.assertEqual(result.product_kind, "raw")
        before = self.path.read_bytes()
        result.data[0, 0] = 42
        self.assertEqual(self.path.read_bytes(), before)

    def test_explicit_label(self) -> None:
        other = self.label.with_name("explicit.xml")
        self.label.rename(other)
        self.assertEqual(load_tmc_image(self.path, other).data.shape, (2, 3))

    def test_read_only_mapping_matches_regular_ingestion(self) -> None:
        mapped = load_tmc_image(self.path, mmap=True).data
        self.assertIsInstance(mapped, np.memmap)
        np.testing.assert_array_equal(mapped, load_tmc_image(self.path).data)
        with self.assertRaises(ValueError):
            mapped[0, 0] = 10
        mapped._mmap.close()

    def test_mapping_preserves_offset_layout_and_row_window(self) -> None:
        self.path.write_bytes(b'head' + self.path.read_bytes())
        self.change('offset', '4')
        self.change('file_size', '16')
        mapped = load_tmc_image(self.path, row_range=(1, 2), mmap=True).data
        np.testing.assert_array_equal(mapped, [[2, 4660, 0]])
        mapped._mmap.close()
        self.change('axis_index_order', 'First Index Fastest')
        mapped = load_tmc_image(self.path, mmap=True).data
        np.testing.assert_array_equal(mapped, load_tmc_image(self.path).data)
        mapped._mmap.close()

    def test_mapping_validates_file_size_and_format(self) -> None:
        self.change('file_size', '11')
        with self.assertRaisesRegex(ValueError, 'File-size mismatch'):
            load_tmc_image(self.path, mmap=True)
        self.make_png()
        with self.assertRaisesRegex(ValueError, 'only for raw'):
            load_tmc_image(self.path, mmap=True)

    def test_offset(self) -> None:
        self.path.write_bytes(b"head" + self.path.read_bytes())
        self.change("offset", "4")
        self.change("file_size", "16")
        self.assertEqual(load_image(self.path).data[0, 1], 256)

    def test_first_index_fastest(self) -> None:
        self.change("axis_index_order", "First Index Fastest")
        np.testing.assert_array_equal(load_image(self.path).data, [[1, 65535, 4660], [256, 2, 0]])

    def test_unsigned_byte_raw(self) -> None:
        self.change("data_type", "UnsignedByte")
        self.change("file_size", "6")
        self.path.write_bytes(bytes(range(6)))
        result = load_image(self.path)
        self.assertEqual(result.data.dtype, np.dtype("u1"))
        np.testing.assert_array_equal(result.data, [[0, 1, 2], [3, 4, 5]])

    def test_file_size_mismatch(self) -> None:
        for size in (0, 10, 14):
            with self.subTest(size=size):
                self.path.write_bytes(bytes(size))
                with self.assertRaisesRegex(ValueError, "File-size mismatch"):
                    load_image(self.path)

    def test_inconsistent_dimensions(self) -> None:
        self.change("elements", "3")
        with self.assertRaisesRegex(ValueError, "Dimension/file-size mismatch"):
            load_image(self.path)

    def test_bad_metadata_fields(self) -> None:
        for tag, value in (("elements", "0"), ("elements", "abc"), ("elements", "-1"),
                           ("file_size", "bad"), ("offset", "-1"), ("axes", "3"),
                           ("sequence_number", "2"), ("axis_index_order", "unknown")):
            original = self.label.read_bytes()
            with self.subTest(tag=tag, value=value):
                self.change(tag, value)
                with self.assertRaisesRegex(ValueError, "Malformed metadata"):
                    load_image(self.path)
                self.label.write_bytes(original)

    def test_missing_dimensions(self) -> None:
        tree = ET.parse(self.label)
        array = tree.find(f".//{{{PDS_NS}}}Array_2D_Image")
        assert array is not None
        array.remove(array.find(f"{{{PDS_NS}}}Axis_Array"))
        tree.write(self.label)
        with self.assertRaisesRegex(ValueError, "image_height"):
            load_image(self.path)

    def test_malformed_xml(self) -> None:
        self.label.write_text("<broken>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed metadata XML"):
            read_metadata(self.label)

    def test_missing_files(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_image(self.path.with_name("absent.img"))
        self.label.unlink()
        with self.assertRaises(FileNotFoundError):
            load_image(self.path)

    def test_unsupported_type(self) -> None:
        self.change("data_type", "UnsupportedType")
        with self.assertRaisesRegex(ValueError, "Unsupported data type"):
            load_image(self.path)

    def test_unsupported_sensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported sensor"):
            load_image(self.path, sensor="OHRC")
        self.change("name", "IIRS")
        with self.assertRaisesRegex(ValueError, "Unsupported sensor"):
            load_tmc_image(self.path)

    def test_wrong_label_filename(self) -> None:
        self.change("file_name", "other.img")
        with self.assertRaisesRegex(ValueError, "file_name mismatch"):
            load_image(self.path)

    def test_browse(self) -> None:
        self.make_png()
        result = load_image(self.path)
        self.assertEqual(result.product_kind, "browse")
        self.assertEqual(result.data.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(result.data, [[0, 128, 255], [3, 4, 5]])

    def test_browse_dimension_mismatch(self) -> None:
        self.make_png()
        self.change("elements", "3")
        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            load_image(self.path)

    def test_browse_dtype_mismatch(self) -> None:
        self.make_png()
        self.change("data_type", "UnsignedLSB2")
        with self.assertRaisesRegex(ValueError, "Data type mismatch"):
            load_image(self.path)

    def test_corrupt_png(self) -> None:
        self.make_png()
        self.path.write_bytes(bytes(self.path.stat().st_size))
        with self.assertRaisesRegex(ValueError, "Invalid PNG"):
            load_image(self.path)


class BorrowKIntegrationTests(unittest.TestCase):
    """Require the actual dataset; missing fixtures are failures, not skips."""

    def test_raw_metadata(self) -> None:
        metadata = read_metadata(RAW.with_suffix(".xml"))
        self.assertEqual(metadata["image_height"], 21513)
        self.assertEqual(metadata["image_width"], 4000)
        self.assertEqual(metadata["data_type"], "UnsignedLSB2")
        self.assertEqual(metadata["pixel_resolution_m_per_pixel"], 6.27)
        self.assertEqual(metadata["projection"], "Selenographic")
        self.assertEqual(metadata["target"], "Moon")
        self.assertEqual(metadata["imaging_orbit_number"], 30533)
        self.assertAlmostEqual(metadata["upper_left_latitude"], 69.769058)

    def test_raw_image_size_dtype_and_values(self) -> None:
        result = load_image(RAW)
        self.assertEqual(RAW.stat().st_size, 172104000)
        self.assertEqual(result.data.shape, (21513, 4000))
        self.assertEqual(result.data.dtype.str, "<u2")
        self.assertEqual(result.data.nbytes, result.metadata["file_size_bytes"])
        # Independently decode sampled bytes, including the last pixel.
        with RAW.open("rb") as stream:
            for row, column in ((0, 0), (100, 123), (21512, 3999)):
                stream.seek((row * 4000 + column) * 2)
                self.assertEqual(result.data[row, column], int.from_bytes(stream.read(2), "little"))
        self.assertEqual(hashlib.md5(result.data.tobytes()).hexdigest(), result.metadata["md5_checksum"])
        self.assertGreater(int(result.data.max()), 255)

    def test_browse_image(self) -> None:
        result = load_image(BROWSE)
        self.assertEqual(result.data.shape, (2151, 400))
        self.assertEqual(result.data.dtype, np.dtype("uint8"))
        self.assertEqual(result.product_kind, "browse")
        self.assertEqual(BROWSE.stat().st_size, result.metadata["file_size_bytes"])
        self.assertEqual(hashlib.md5(BROWSE.read_bytes()).hexdigest(), result.metadata["md5_checksum"])


if __name__ == "__main__":
    unittest.main()
