import logging
from pathlib import Path
from typing import Iterable

from PIL import Image
from rembg import new_session, remove

from .remove_background import remove_background
from .center_and_resize import center_and_resize


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def iter_image_files(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def warmup_rembg(session) -> None:
    warmup = remove(Image.new("RGBA", (64, 64), (0, 0, 0, 0)), session=session)


def process_image_file(
    input_path: Path,
    output_path: Path,
    session,
    target_size: int = 224,
    target_area_fraction: float = 0.65,
    max_occupancy: float = 0.9,
    bbox_margin_ratio: float = 0.08,
) -> bool:
    try:
        with Image.open(input_path) as img:
            img_rgba = img.convert("RGBA")

        img_no_bg = remove_background(img_rgba, session=session)
        final_img = center_and_resize(
            img_no_bg,
            target_size=target_size,
            target_area_fraction=target_area_fraction,
            max_occupancy=max_occupancy,
            bbox_margin_ratio=bbox_margin_ratio,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(output_path, format="PNG")
        return True

    except Exception as e:
        logging.error(f"Failed to process '{input_path}': {e}")
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 224,
    target_area_fraction: float = 0.65,
    max_occupancy: float = 0.9,
    bbox_margin_ratio: float = 0.08,
) -> None:
    session = new_session()
    warmup_rembg(session)

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = list(iter_image_files(input_dir))
    if not files:
        logging.warning(f"No images found in {input_dir}")
        return

    logging.info(f"Found {len(files)} images in {input_dir}")

    for idx, in_path in enumerate(files, start=1):
        rel = in_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".png")

        logging.info(f"[{idx}/{len(files)}] {in_path} -> {out_path}")
        _ = process_image_file(
            input_path=in_path,
            output_path=out_path,
            session=session,
            target_size=target_size,
            target_area_fraction=target_area_fraction,
            max_occupancy=max_occupancy,
            bbox_margin_ratio=bbox_margin_ratio,
        )