"""Image validation, normalisation and encoding for the Claude Messages API."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

MAX_UPLOAD_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = config.MAX_IMAGE_MEGAPIXELS * 1_000_000  # reject decompression bombs / memory hogs
MAX_ENCODED_BYTES = 5 * 1024 * 1024  # API limit is 10 MB base64; stay well below


class ImageError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    media_type: str
    width: int
    height: int
    original_width: int
    original_height: int

    @property
    def resized(self) -> bool:
        return (self.width, self.height) != (self.original_width, self.original_height)

    @property
    def b64(self) -> str:
        return base64.standard_b64encode(self.data).decode("ascii")

    def to_content_block(self) -> dict:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": self.media_type, "data": self.b64},
        }


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def prepare_image(raw: bytes, max_edge: int | None = None) -> PreparedImage:
    """Validate, orient, flatten, downscale and encode an image for the API.

    - honours EXIF orientation
    - flattens transparency onto white and converts to RGB
    - downscales so the longest side is <= max_edge
    - keeps PNG for PNG inputs (sharper text) when small enough, otherwise JPEG
    - animated images use their first frame
    """
    max_edge = max_edge or config.MAX_IMAGE_EDGE
    if not raw:
        raise ImageError("Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageError(f"Image is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    try:
        img = Image.open(io.BytesIO(raw))
        src_w, src_h = img.size
        if src_w * src_h > MAX_PIXELS:
            raise ImageError(f"Image has more than {config.MAX_IMAGE_MEGAPIXELS} megapixels.", 413)
        src_format = (img.format or "").upper()
        if src_format == "JPEG":
            # decode at reduced size (still >= max_edge): a 48 MP photo then needs a fraction of the memory
            img.draft("RGB", (max_edge, max_edge))
        img.load()
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("The file is not a readable image (JPEG, PNG, WebP, GIF, BMP, TIFF).") from exc

    orientation = img.getexif().get(0x0112, 1)
    img = ImageOps.exif_transpose(img)
    # report the real (rotated) size of the original, even if the decoder scaled it down
    original_w, original_h = (src_h, src_w) if orientation in (5, 6, 7, 8) else (src_w, src_h)

    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")

    scale = max_edge / max(img.size)
    if scale < 1:
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    data: bytes | None = None
    media_type = "image/jpeg"
    if src_format == "PNG":
        candidate = _encode_png(img)
        if len(candidate) <= MAX_ENCODED_BYTES:
            data, media_type = candidate, "image/png"
    if data is None:
        for quality in (90, 80, 70, 55):
            candidate = _encode_jpeg(img, quality)
            if len(candidate) <= MAX_ENCODED_BYTES:
                data, media_type = candidate, "image/jpeg"
                break
        else:
            raise ImageError("Could not compress the image below the size limit.", 413)

    return PreparedImage(
        data=data,
        media_type=media_type,
        width=img.width,
        height=img.height,
        original_width=original_w,
        original_height=original_h,
    )
