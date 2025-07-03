# Print-Oriented Padding and Bleeding Implementation Guide

## Core Concept
When preparing images for print (like A4 at 300 DPI), you need to:
1. Calculate the final print dimensions in pixels
2. Determine padding and bleeding percentages based on physical measurements
3. Extend the original image to create enough content
4. Crop to the exact print format with proper positioning

## Print Format Calculations

### A4 Format at 300 DPI
```
A4 dimensions: 210mm × 297mm
At 300 DPI: 2480px × 3508px
```

### Example: A4 with 10mm padding and 3mm bleeding
```
Total bleed area: 210mm + 2×3mm = 216mm × 297mm + 2×3mm = 303mm
Bleed area in pixels: 2551px × 3579px

Content area (image + padding): 210mm - 2×10mm = 190mm × 297mm - 2×10mm = 277mm  
Content area in pixels: 2244px × 3272px

Padding percentages:
- Horizontal padding = 10mm / 210mm = 4.76% of A4 width
- Vertical padding = 10mm / 297mm = 3.37% of A4 height

Bleeding percentages:
- Horizontal bleeding = 3mm / 210mm = 1.43% of A4 width  
- Vertical bleeding = 3mm / 297mm = 1.01% of A4 height
```

## Implementation Strategy

### Step 1: Calculate Print Requirements
```python
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

# Example usage
a4_requirements = calculate_print_requirements(
    paper_format={'width_mm': 210, 'height_mm': 297},
    dpi=300,
    padding_mm=10,
    bleeding_mm=3
)
```

### Step 2: Determine Extension Strategy
```python
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
```

### Step 3: Execute Print-Oriented Extension
```python
def execute_print_extension(original_image, extension_strategy, print_requirements):
    """
    Execute the extension sequence for print requirements
    """
    
    # First, resize original to fit content area properly
    fitted_w, fitted_h = extension_strategy['fitted_size']
    resized_image = resize(original_image, (fitted_h, fitted_w))
    
    # Determine API extension sequence based on how much extension we need
    h_ext_needed, v_ext_needed = extension_strategy['extension_needed']
    
    # Choose extension path based on requirements
    if h_ext_needed > 0.3 and v_ext_needed > 0.3:
        # Need significant extension in both directions
        return execute_dual_direction_extension(resized_image, extension_strategy)
    elif h_ext_needed > v_ext_needed:
        # Need more horizontal extension
        return execute_horizontal_focused_extension(resized_image, extension_strategy)
    else:
        # Need more vertical extension
        return execute_vertical_focused_extension(resized_image, extension_strategy)

def execute_dual_direction_extension(image, strategy):
    """
    For cases needing significant extension in both directions
    """
    # Step 1: Extend to square (adds content in the narrower dimension)
    if image.shape[1] > image.shape[0]:  # Landscape
        extended_1 = extend_to_api_format(image, (1024, 1024))  # adds top/bottom
        extended_2 = extend_to_api_format(extended_1, (1280, 720))  # adds left/right
    else:  # Portrait
        extended_1 = extend_to_api_format(image, (1024, 1024))  # adds left/right
        extended_2 = extend_to_api_format(extended_1, (720, 1280))  # adds top/bottom
    
    return extended_2

def extend_to_api_format(image, target_size):
    """
    Placeholder for actual API call
    This would call your diffusion model API
    """
    # Resize image to target_size for API call
    api_input = resize(image, target_size)
    
    # Call your extension API
    # extended_result = your_extension_api(api_input, target_size)
    
    # For demo purposes, return resized input
    return api_input
```

### Step 4: Final Print Crop and Positioning
```python
def create_final_print_output(extended_image, print_requirements, extension_strategy):
    """
    Create the final print-ready output with proper positioning
    """
    bleed_w, bleed_h = print_requirements['bleed_size_px']
    content_w, content_h = print_requirements['content_size_px']
    fitted_w, fitted_h = extension_strategy['fitted_size']
    h_margin, v_margin = extension_strategy['margins']
    
    # Create final canvas at bleed size
    final_canvas = np.zeros((bleed_h, bleed_w, 3), dtype=np.uint8)
    
    # Calculate padding and bleeding in pixels
    padding_h_px = int(print_requirements['padding_percent'][0] * bleed_w)  
    padding_v_px = int(print_requirements['padding_percent'][1] * bleed_h)
    bleeding_h_px = int(print_requirements['bleeding_percent'][0] * bleed_w)
    bleeding_v_px = int(print_requirements['bleeding_percent'][1] * bleed_h)
    
    # Position the extended image content
    # The original image should be centered in the content area
    # Content area starts at bleeding offset
    content_start_x = bleeding_h_px
    content_start_y = bleeding_v_px
    
    # Original image position within content area
    image_start_x = content_start_x + padding_h_px + h_margin
    image_start_y = content_start_y + padding_v_px + v_margin
    
    # Extract and place the core image content
    # This requires careful mapping from the extended image back to final position
    extracted_region = extract_centered_region(extended_image, (fitted_w, fitted_h))
    
    # Place in final canvas
    final_canvas[
        image_start_y:image_start_y + fitted_h,
        image_start_x:image_start_x + fitted_w
    ] = extracted_region
    
    # Fill padding and bleeding areas with extended content
    fill_padding_and_bleeding_areas(final_canvas, extended_image, print_requirements)
    
    return final_canvas

def extract_centered_region(extended_image, target_size):
    """
    Extract the central region that corresponds to our original image
    """
    ext_h, ext_w = extended_image.shape[:2]
    target_w, target_h = target_size
    
    start_x = (ext_w - target_w) // 2
    start_y = (ext_h - target_h) // 2
    
    return extended_image[start_y:start_y + target_h, start_x:start_x + target_w]

def fill_padding_and_bleeding_areas(canvas, extended_image, print_requirements):
    """
    Fill the padding and bleeding areas with appropriate extended content
    """
    # This is complex and depends on how much extended content you have
    # You'll need to map regions from the extended image to the padding/bleeding areas
    # while ensuring smooth transitions
    pass
```

## Complete Example Workflows

### Example 1: Portrait Photo for A4 Print
```python
# Original photo: 1500x2000 (portrait)
original_photo = load_image("portrait_1500x2000.jpg")

# A4 print requirements
print_req = calculate_print_requirements(
    paper_format={'width_mm': 210, 'height_mm': 297},
    dpi=300,
    padding_mm=10,
    bleeding_mm=3
)

# Results:
# Paper: 2480x3508px
# Bleed area: 2551x3579px  
# Content area: 2244x3272px
# Image will fit as: 2244x2992px (fitted by width)
# Vertical margin: 140px top/bottom in content area

# Extension strategy:
extension_strategy = determine_print_extension_strategy(original_photo, print_req)

# Execute extensions:
# 1. Resize 1500x2000 → 2244x2992 (fit content area)
# 2. Extend via API to add ~307px horizontal, ~287px vertical content
# 3. Final crop and positioning to 2551x3579px bleed size

final_print_ready = process_image_for_print(original_photo, print_req)
# Output: 2551x3579px image ready for A4 printing with proper bleeding
```

### Example 2: Landscape Photo for A4 Print  
```python
# Original photo: 3000x2000 (landscape, 3:2 ratio)
original_landscape = load_image("landscape_3000x2000.jpg")

print_req = calculate_print_requirements(
    paper_format={'width_mm': 210, 'height_mm': 297},
    dpi=300,
    padding_mm=10,
    bleeding_mm=3
)

# Image will fit as: 2244x1496px (fitted by width)
# Vertical margin: 888px top/bottom in content area
# Needs significant vertical extension for bleeding

extension_strategy = determine_print_extension_strategy(original_landscape, print_req)

final_print_ready = process_image_for_print(original_landscape, print_req)
# Output: 2551x3579px image ready for A4 printing
```

### Example 3: Square Photo for A4 Print
```python  
# Original photo: 2000x2000 (square)
original_square = load_image("square_2000x2000.jpg")

# Same A4 requirements
print_req = calculate_print_requirements(
    paper_format={'width_mm': 210, 'height_mm': 297},
    dpi=300,
    padding_mm=10, 
    bleeding_mm=3
)

# Image will fit as: 2244x2244px (fitted by width)
# Vertical margin: 514px top/bottom in content area
# Needs moderate vertical extension

final_print_ready = process_image_for_print(original_square, print_req)
# Output: 2551x3579px image ready for A4 printing
```

## Key Implications

### 1. **Fixed Output Dimensions**
Unlike the previous approach, the output is always the bleed size (e.g., 2551x3579px for A4), regardless of input image dimensions.

### 2. **Content Preservation**
The original image content is preserved and positioned correctly within the content area, with padding and bleeding filled by extended content.

### 3. **Physical Accuracy**
All measurements are based on actual physical dimensions, ensuring accurate printing results.

### 4. **Extension Requirements Vary**
- Portrait images in portrait format need less extension
- Landscape images in portrait format need significant vertical extension
- Square images need moderate extension in the longer dimension

### 5. **Quality Considerations**
- Images smaller than the content area will be upscaled
- Very small images may need significant upscaling, affecting quality
- Extension API calls should use appropriate intermediate sizes

### 6. **Complex Positioning Logic**
The final positioning requires careful calculation to ensure:
- Original image is properly centered in content area
- Padding areas are filled with appropriate extended content
- Bleeding areas extend naturally from padding areas

This approach ensures your images are print-ready with exact physical measurements while maintaining the original image's integrity within the designated content area.