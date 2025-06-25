from PIL import ImageDraw
from print_settings import mm_to_pixels

def add_cut_lines(img, bleed_mm, dpi, line_color='black', line_width=1):
    """Add cut lines to show where to cut"""
    if bleed_mm <= 0:
        return img
    
    # Make a copy to draw on
    cut_img = img.copy()
    draw = ImageDraw.Draw(cut_img)
    
    # Calculate bleed in pixels
    bleed_px = mm_to_pixels(bleed_mm, dpi)
    
    # Get image dimensions
    width, height = img.size
    
    # Draw cut lines at the edges of the original content (not including bleed)
    # Vertical lines
    draw.line([(bleed_px, 0), (bleed_px, height)], fill=line_color, width=line_width)
    draw.line([(width - bleed_px, 0), (width - bleed_px, height)], fill=line_color, width=line_width)
    
    # Horizontal lines  
    draw.line([(0, bleed_px), (width, bleed_px)], fill=line_color, width=line_width)
    draw.line([(0, height - bleed_px), (width, height - bleed_px)], fill=line_color, width=line_width)
    
    return cut_img 