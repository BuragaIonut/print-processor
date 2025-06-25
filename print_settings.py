from PIL import Image

# Common printing formats (width, height in mm)
PRINT_FORMATS = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A5": (148, 210),
    "Letter": (215.9, 279.4),  # 8.5 × 11 inches
    "Legal": (215.9, 355.6),   # 8.5 × 14 inches
    "Business Card": (89, 51),  # 3.5 × 2 inches
    "Postcard": (102, 152),     # 4 × 6 inches
    "Flyer": (215.9, 279.4),   # Same as Letter
    "Small Poster": (457, 610), # 18 × 24 inches
    "Large Poster": (610, 914), # 24 × 36 inches
    "Custom": None
}

def mm_to_inches(mm):
    """Convert millimeters to inches"""
    return mm / 25.4

def inches_to_mm(inches):
    """Convert inches to millimeters"""
    return inches * 25.4

def mm_to_pixels(mm, dpi):
    """Convert millimeters to pixels at given DPI"""
    inches = mm_to_inches(mm)
    return int(inches * dpi)

def resize_image_for_print(img, target_width_mm, target_height_mm, dpi):
    """Resize image to exact print dimensions"""
    target_width_px = mm_to_pixels(target_width_mm, dpi)
    target_height_px = mm_to_pixels(target_height_mm, dpi)
    
    # Resize with high quality
    resized_img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
    return resized_img 