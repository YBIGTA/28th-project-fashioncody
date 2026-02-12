from rembg import remove
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