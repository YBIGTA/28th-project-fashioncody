from PIL import Image

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