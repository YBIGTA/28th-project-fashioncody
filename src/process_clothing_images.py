import argparse
import logging
from pathlib import Path

from rembg import remove, new_session
from PIL import Image


def remove_background(image: Image.Image, session) -> Image.Image:
    """
    Use rembg to remove background and return an RGBA image.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    result = remove(image, session=session)
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    return result


def center_and_resize(
    image_rgba: Image.Image,
    target_size: int = 224,
    target_area_fraction: float = 0.65,
    max_occupancy: float = 0.9,
    bbox_margin_ratio: float = 0.08,
) -> Image.Image:
    """
    Center the clothing item based on the alpha channel and resize to target_size x target_size.

    - Uses the alpha channel to find the bounding box of the clothing.
    - Scales so the clothing occupies roughly target_area_fraction of the final image area.
    - Preserves aspect ratio.
    - Places the item centered on a square transparent canvas.
    """
    if image_rgba.mode != "RGBA":
        image_rgba = image_rgba.convert("RGBA")
    alpha = image_rgba.split()[3]

    bbox = alpha.getbbox()
    if bbox is None:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    left, upper, right, lower = bbox
    box_w = right - left
    box_h = lower - upper

    if box_w == 0 or box_h == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    
    img_w, img_h = image_rgba.size 
    margin_x = int(round(box_w * bbox_margin_ratio)) 
    margin_y = int(round(box_h * bbox_margin_ratio))

    left2 = max(0, left - margin_x)   
    upper2 = max(0, upper - margin_y)
    right2 = min(img_w, right + margin_x) 
    lower2 = min(img_h, lower + margin_y)

    box_w2 = right2 - left2  
    box_h2 = lower2 - upper2
    if box_w2 == 0 or box_h2 == 0: 
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    obj = image_rgba.crop((left2, upper2, right2, lower2)) 

    obj_area = box_w2 * box_h2  
    desired_area = target_area_fraction * (target_size * target_size)
    scale_area = (desired_area / obj_area) ** 0.5

    max_side = max(box_w2, box_h2) 
    scale_fit = (target_size * max_occupancy) / max_side
    scale = min(scale_area, scale_fit)

    new_w = max(1, int(round(box_w2 * scale))) 
    new_h = max(1, int(round(box_h2 * scale))) 

    obj_resized = obj.resize((new_w, new_h), resample=Image.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2

    canvas.paste(obj_resized, (left, top), obj_resized)

    return canvas


def process_image_file(
    input_path: Path,
    output_path: Path,
    session,
    target_size: int = 224,
    target_area_fraction: float = 0.65,
    max_occupancy: float = 0.9,
) -> bool:
    """
    Process a single image:
    - Load
    - Remove background
    - Center and resize
    - Save as PNG with alpha

    Returns True on success, False on failure.
    """
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGBA")

        img_no_bg = remove_background(img, session=session)
        final_img = center_and_resize(
            img_no_bg,
            target_size=target_size,
            target_area_fraction=target_area_fraction,
            max_occupancy=max_occupancy,
            bbox_margin_ratio=0.08,
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
) -> None:
    """
    Batch-process all JPG/PNG images in input_dir and write PNG results to output_dir.
    """
    session = new_session()
    warmup_result = remove(Image.new("RGBA", (64, 64), (0, 0, 0, 0)), session=session)
    supported_exts = {".jpg", ".jpeg", ".png"}
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_exts
    ]

    if not files:
        logging.warning(f"No images found in {input_dir}")
        return

    logging.info(f"Found {len(files)} images in {input_dir}")

    for idx, in_path in enumerate(files, start=1):
        rel = in_path.relative_to(input_dir)
        out_filename = rel.with_suffix(".png")
        out_path = output_dir / out_filename

        logging.info(f"[{idx}/{len(files)}] Processing {in_path} -> {out_path}")
        success = process_image_file(
            in_path,
            out_path,
            target_size=target_size,
            target_area_fraction=target_area_fraction,
            max_occupancy=max_occupancy,
            session=session
        )
        if not success:
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch remove backgrounds, center, and resize clothing images to 224x224 PNGs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing input JPG/PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where processed PNG images will be saved.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output image size (size x size). Default: 224",
    )
    parser.add_argument(
        "--area_fraction",
        type=float,
        default=0.65,
        help="Target area fraction (0–1) that the clothing bounding box should occupy. Default: 0.65",
    )
    parser.add_argument(
        "--max_occupancy",
        type=float,
        default=0.9,
        help="Maximum fraction of side length that the clothing can occupy. Default: 0.9",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Target size: {args.size}x{args.size}")
    logging.info(f"Target area fraction: {args.area_fraction}")
    logging.info(f"Max side occupancy: {args.max_occupancy}")

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=args.size,
        target_area_fraction=args.area_fraction,
        max_occupancy=args.max_occupancy,
    )

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()