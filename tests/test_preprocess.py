import io

import pytest
from PIL import Image

from app.preprocess import ImageError, prepare_image
from fakes import png_bytes


def test_small_png_kept_as_png():
    p = prepare_image(png_bytes((64, 48)))
    assert (p.width, p.height) == (64, 48)
    assert p.media_type == "image/png"
    assert not p.resized


def test_large_image_downscaled_preserving_aspect():
    p = prepare_image(png_bytes((3000, 2000)), max_edge=1500)
    assert max(p.width, p.height) == 1500
    assert abs(p.width / p.height - 1.5) < 0.01
    assert (p.original_width, p.original_height) == (3000, 2000)
    assert p.resized


def test_rgba_flattened_onto_white():
    p = prepare_image(png_bytes((10, 10), (0, 0, 0, 0), mode="RGBA"))
    img = Image.open(io.BytesIO(p.data)).convert("RGB")
    assert img.getpixel((5, 5)) == (255, 255, 255)


def test_exif_orientation_applied():
    img = Image.new("RGB", (40, 20), (10, 120, 200))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 degrees
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    p = prepare_image(buf.getvalue())
    assert (p.width, p.height) == (20, 40)
    assert p.media_type == "image/jpeg"


def _jpeg(size, orientation=None) -> bytes:
    img = Image.new("RGB", size, (30, 160, 90))
    buf = io.BytesIO()
    if orientation:
        exif = img.getexif()
        exif[0x0112] = orientation
        img.save(buf, "JPEG", exif=exif)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


def test_large_jpeg_reports_true_original_size():
    p = prepare_image(_jpeg((4000, 3000)), max_edge=1000)   # decoder downscales, original size is still reported
    assert (p.original_width, p.original_height) == (4000, 3000)
    assert (p.width, p.height) == (1000, 750)
    assert p.resized


def test_large_rotated_jpeg_swaps_original_size():
    p = prepare_image(_jpeg((4000, 3000), orientation=6), max_edge=1000)
    assert (p.original_width, p.original_height) == (3000, 4000)
    assert (p.width, p.height) == (750, 1000)


def test_too_many_pixels_rejected(monkeypatch):
    monkeypatch.setattr("app.preprocess.MAX_PIXELS", 1000)
    with pytest.raises(ImageError, match="megapixels"):
        prepare_image(png_bytes((64, 48)))


def test_content_block_shape():
    block = prepare_image(png_bytes()).to_content_block()
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"]


@pytest.mark.parametrize("raw", [b"", b"not an image at all"])
def test_invalid_input_rejected(raw):
    with pytest.raises(ImageError):
        prepare_image(raw)
