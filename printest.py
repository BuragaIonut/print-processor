def calculate_print_requirements(paper_format, dpi, padding_mm, bleeding_mm):
    """
    Calculate all print-related dimensions
    
    Args:
        paper_format: dict with 'width_mm' and 'height_mm'
        dpi: target DPI (e.g., 300)
        padding_mm: padding in millimeters
        bleeding_mm: bleeding in millimeters
    """
    
    # Convert mm to pixels
    mm_to_px = dpi / 25.4
    
    # Paper dimensions in pixels
    paper_width_px = int(paper_format['width_mm'] * mm_to_px)
    paper_height_px = int(paper_format['height_mm'] * mm_to_px)
    
    # Bleed area (paper + bleeding on all sides)
    bleed_width_px = int((paper_format['width_mm'] + 2 * bleeding_mm) * mm_to_px)
    bleed_height_px = int((paper_format['height_mm'] + 2 * bleeding_mm) * mm_to_px)
    
    # Content area (paper - padding on all sides)
    content_width_px = int((paper_format['width_mm'] - 2 * padding_mm) * mm_to_px)
    content_height_px = int((paper_format['height_mm'] - 2 * padding_mm) * mm_to_px)
    
    # Calculate percentages for image processing
    padding_h_percent = padding_mm / paper_format['width_mm']
    padding_v_percent = padding_mm / paper_format['height_mm']
    bleeding_h_percent = bleeding_mm / paper_format['width_mm']
    bleeding_v_percent = bleeding_mm / paper_format['height_mm']
    
    return {
        'paper_size_px': (paper_width_px, paper_height_px),
        'bleed_size_px': (bleed_width_px, bleed_height_px),
        'content_size_px': (content_width_px, content_height_px),
        'padding_percent': (padding_h_percent, padding_v_percent),
        'bleeding_percent': (bleeding_h_percent, bleeding_v_percent),
        'total_extension_needed': (
            padding_h_percent + bleeding_h_percent,
            padding_v_percent + bleeding_v_percent
        )
    }
    
a4_requirements = calculate_print_requirements(
    paper_format={'width_mm': 210, 'height_mm': 297},
    dpi=300,
    padding_mm=100,
    bleeding_mm=3
)

def determine_print_extension_strategy(original_image, print_requirements):
    """
    Determine how much extension is needed for print requirements
    """
    orig_h, orig_w = original_image.shape[:2]
    orig_aspect = orig_w / orig_h
    
    content_w, content_h = print_requirements['content_size_px']
    content_aspect = content_w / content_h
    
    bleed_w, bleed_h = print_requirements['bleed_size_px']
    
    # Determine how the original image fits in the content area
    if orig_aspect > content_aspect:
        # Image is wider - fit by width
        scale_factor = content_w / orig_w
        fitted_w = content_w
        fitted_h = int(orig_h * scale_factor)
        # Center vertically in content area
        v_margin = (content_h - fitted_h) // 2
        h_margin = 0
    else:
        # Image is taller - fit by height
        scale_factor = content_h / orig_h
        fitted_h = content_h
        fitted_w = int(orig_w * scale_factor)
        # Center horizontally in content area
        h_margin = (content_w - fitted_w) // 2
        v_margin = 0
    
    # Calculate total extension needed
    padding_h, padding_v = print_requirements['padding_percent']
    bleeding_h, bleeding_v = print_requirements['bleeding_percent']
    
    # Total extension needed as percentage of final bleed size
    total_h_extension = (padding_h + bleeding_h) * 2  # both sides
    total_v_extension = (padding_v + bleeding_v) * 2  # both sides
    
    return {
        'fitted_size': (fitted_w, fitted_h),
        'scale_factor': scale_factor,
        'margins': (h_margin, v_margin),
        'extension_needed': (total_h_extension, total_v_extension),
        'final_bleed_size': (bleed_w, bleed_h)
    }

print_requirements = calculate_print_requirements(
    paper_format={'width_mm': 210, 'height_mm': 297},
    dpi=300,
    padding_mm=100,
    bleeding_mm=3
)
print_extension = determine_print_extension_strategy(a4_requirements, print_requirements)
print(print_extension)