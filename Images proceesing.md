# Content-Aware Padding and Bleeding Implementation Guide

## Core Concept
To add padding/bleeding to an image:
1. **Extend** the image to create more content around the edges
2. **Crop** back to original dimensions (centered) to get the bleeding effect
3. **Scale** if needed to maintain original size

## API Constraints
- Best aspect ratios: 16:9, 9:16, 1:1
- Target resolutions: 1024x1024, 1280x720, 720x1280
- Extension directions: left+right OR top+bottom per call

## Case Analysis by Original Format

### Case 1: Portrait Input (9:16 ratio)

#### Scenario 1A: Portrait → Add Horizontal Bleeding
**Goal**: Extend left/right, then crop back to portrait

**Example with API-preferred dimensions:**
```
Original: 720x1280 (portrait)
Step 1: Extend to 1024x1024 (square) - adds left+right content
Step 2: Crop center 720x1280 from the 1024x1024 result
Result: Original size portrait with horizontal bleeding
```

**Example with arbitrary dimensions:**
```
Original: 400x700 (portrait, aspect ratio 0.57)
Step 0: Resize to 731x1280, then crop to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 2: Crop center 360x630 region (10% bleeding) from extended image
Step 3: Scale 360x630 → 400x700 (back to original size)
Result: Original 400x700 portrait with horizontal bleeding

Another example:
Original: 600x1200 (portrait, aspect ratio 0.5)
Step 0: Resize to 640x1280, then pad to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (square) - adds left+right  
Step 2: Crop center 540x1080 region (10% bleeding)
Step 3: Scale 540x1080 → 600x1200 (back to original size)
Result: Original 600x1200 portrait with horizontal bleeding
```

#### Scenario 1B: Portrait → Add Vertical Bleeding
**Goal**: Need vertical extension, requires 2-step process

**Example with API-preferred dimensions:**
```
Original: 720x1280 (portrait)
Step 1: Extend to 1024x1024 (square) - adds left+right content
Step 2: Extend to 720x1280 (portrait) - adds top+bottom content
Step 3: Crop center 720x1280 from final result
Result: Original size portrait with vertical bleeding
```

**Examples with arbitrary dimensions:**
```
Original: 500x900 (portrait, aspect ratio 0.56)
Step 0: Resize to 711x1280, then crop to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 2: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 3: Crop center 450x810 region (10% bleeding)
Step 4: Scale 450x810 → 500x900 (back to original size)
Result: Original 500x900 portrait with vertical bleeding

Another example:
Original: 300x800 (portrait, aspect ratio 0.375)
Step 0: Resize to 480x1280, then pad to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 2: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 3: Crop center 270x720 region (10% bleeding)
Step 4: Scale 270x720 → 300x800 (back to original size)
Result: Original 300x800 portrait with vertical bleeding
```

#### Scenario 1C: Portrait → Add Both Horizontal & Vertical Bleeding
**Goal**: Maximum bleeding in both directions

**Example with API-preferred dimensions:**
```
Original: 720x1280 (portrait)
Step 1: Extend to 1024x1024 (square) - adds left+right
Step 2: Extend to 720x1280 (portrait) - adds top+bottom
Step 3: Crop center region smaller than 720x1280, then scale up
Result: Original size portrait with bleeding in all directions
```

**Example with arbitrary dimensions:**
```
Original: 600x900 (portrait, aspect ratio 2:3)
Step 0: Resize to 853x1280, then crop to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 2: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom  
Step 3: Crop center 540x810 region (10% bleeding)
Step 4: Scale 540x810 → 600x900 (back to original size)
Result: Original 600x900 portrait with bleeding in all directions

Another example:
Original: 450x1000 (portrait, aspect ratio 0.45)
Step 0: Resize to 576x1280, then pad to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 2: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 3: Crop center 405x900 region (10% bleeding)
Step 4: Scale 405x900 → 450x1000 (back to original size)
Result: Original 450x1000 portrait with bleeding in all directions
```

### Case 2: Square Input (1:1 ratio)

#### Scenario 2A: Square → Add Horizontal Bleeding
**Example with API-preferred dimensions:**
```
Original: 1024x1024 (square)
Step 1: Extend to 1280x720 (landscape) - adds left+right content
Step 2: Crop center 1024x1024 from the 1280x720 result
Result: Original size square with horizontal bleeding
```

**Examples with arbitrary dimensions:**
```
Original: 800x800 (square)
Step 0: Resize to 1024x1024 for API (perfect fit)
Step 1: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 2: Crop center 720x720 region (10% bleeding)
Step 3: Scale 720x720 → 800x800 (back to original size)
Result: Original 800x800 square with horizontal bleeding

Another example:
Original: 512x512 (square)
Step 0: Resize to 1024x1024 for API (2x upscale)
Step 1: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 2: Crop center 461x461 region (10% bleeding)
Step 3: Scale 461x461 → 512x512 (back to original size)
Result: Original 512x512 square with horizontal bleeding
```

#### Scenario 2B: Square → Add Vertical Bleeding
**Example with API-preferred dimensions:**
```
Original: 1024x1024 (square)
Step 1: Extend to 720x1280 (portrait) - adds top+bottom content
Step 2: Crop center 1024x1024 from the 720x1280 result
Result: Original size square with vertical bleeding
```

**Examples with arbitrary dimensions:**
```
Original: 600x600 (square)
Step 0: Resize to 1024x1024 for API
Step 1: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 2: Crop center 540x540 region (10% bleeding)
Step 3: Scale 540x540 → 600x600 (back to original size)
Result: Original 600x600 square with vertical bleeding

Another example:
Original: 900x900 (square)
Step 0: Resize to 1024x1024 for API
Step 1: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 2: Crop center 810x810 region (10% bleeding)
Step 3: Scale 810x810 → 900x900 (back to original size)
Result: Original 900x900 square with vertical bleeding
```

#### Scenario 2C: Square → Add Both Bleeding (Option 1)
**Example with API-preferred dimensions:**
```
Original: 1024x1024 (square)
Step 1: Extend to 1280x720 (landscape) - adds left+right
Step 2: Extend to 1024x1024 (square) - adds top+bottom
Step 3: Crop center region smaller than 1024x1024, then scale up
Result: Original size square with bleeding in all directions
```

**Examples with arbitrary dimensions:**
```
Original: 640x640 (square)
Step 0: Resize to 1024x1024 for API
Step 1: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 2: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 3: Crop center 576x576 region (10% bleeding)
Step 4: Scale 576x576 → 640x640 (back to original size)
Result: Original 640x640 square with bleeding in all directions

Another example:
Original: 750x750 (square)
Step 0: Resize to 1024x1024 for API
Step 1: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 2: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 3: Crop center 675x675 region (10% bleeding)
Step 4: Scale 675x675 → 750x750 (back to original size)
Result: Original 750x750 square with bleeding in all directions
```

#### Scenario 2D: Square → Add Both Bleeding (Option 2)
**Example with API-preferred dimensions:**
```
Original: 1024x1024 (square)
Step 1: Extend to 720x1280 (portrait) - adds top+bottom
Step 2: Extend to 1024x1024 (square) - adds left+right
Step 3: Crop center region smaller than 1024x1024, then scale up
Result: Original size square with bleeding in all directions
```

**Examples with arbitrary dimensions:**
```
Original: 500x500 (square)
Step 0: Resize to 1024x1024 for API
Step 1: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 2: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 3: Crop center 450x450 region (10% bleeding)
Step 4: Scale 450x450 → 500x500 (back to original size)
Result: Original 500x500 square with bleeding in all directions

Another example:
Original: 1200x1200 (square)
Step 0: Resize to 1024x1024 for API (slight downscale)
Step 1: Extend 1024x1024 → 720x1280 (portrait) - adds top+bottom
Step 2: Extend 720x1280 → 1024x1024 (square) - adds left+right
Step 3: Crop center 1080x1080 region (10% bleeding)
Step 4: Scale 1080x1080 → 1200x1200 (back to original size)
Result: Original 1200x1200 square with bleeding in all directions
```

### Case 3: Landscape Input (16:9 ratio)

#### Scenario 3A: Landscape → Add Vertical Bleeding
**Example with API-preferred dimensions:**
```
Original: 1280x720 (landscape)
Step 1: Extend to 1024x1024 (square) - adds top+bottom content
Step 2: Crop center 1280x720 from the 1024x1024 result
Result: Original size landscape with vertical bleeding
```

**Examples with arbitrary dimensions:**
```
Original: 1600x900 (landscape, aspect ratio 1.78)
Step 0: Resize to 1280x720 for API (perfect aspect match)
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Crop center 1440x810 region (10% bleeding)
Step 3: Scale 1440x810 → 1600x900 (back to original size)
Result: Original 1600x900 landscape with vertical bleeding

Another example:
Original: 1000x600 (landscape, aspect ratio 1.67)
Step 0: Resize to 1280x768, then crop to 1280x720 for API
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Crop center 900x540 region (10% bleeding)
Step 3: Scale 900x540 → 1000x600 (back to original size)
Result: Original 1000x600 landscape with vertical bleeding
```

#### Scenario 3B: Landscape → Add Horizontal Bleeding
**Example with API-preferred dimensions:**
```
Original: 1280x720 (landscape)
Step 1: Extend to 1024x1024 (square) - adds top+bottom
Step 2: Extend to 1280x720 (landscape) - adds left+right
Step 3: Crop center 1280x720 from final result
Result: Original size landscape with horizontal bleeding
```

**Examples with arbitrary dimensions:**
```
Original: 1920x1080 (landscape, aspect ratio 1.78)
Step 0: Resize to 1280x720 for API (perfect aspect match)
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 3: Crop center 1728x972 region (10% bleeding)
Step 4: Scale 1728x972 → 1920x1080 (back to original size)
Result: Original 1920x1080 landscape with horizontal bleeding

Another example:
Original: 800x450 (landscape, aspect ratio 1.78)
Step 0: Resize to 1280x720 for API (perfect aspect match)
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 3: Crop center 720x405 region (10% bleeding)
Step 4: Scale 720x405 → 800x450 (back to original size)
Result: Original 800x450 landscape with horizontal bleeding
```

#### Scenario 3C: Landscape → Add Both Bleeding
**Example with API-preferred dimensions:**
```
Original: 1280x720 (landscape)
Step 1: Extend to 1024x1024 (square) - adds top+bottom
Step 2: Extend to 1280x720 (landscape) - adds left+right
Step 3: Crop center region smaller than 1280x720, then scale up
Result: Original size landscape with bleeding in all directions
```

**Examples with arbitrary dimensions:**
```
Original: 1440x810 (landscape, aspect ratio 1.78)
Step 0: Resize to 1280x720 for API (perfect aspect match)
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 3: Crop center 1296x729 region (10% bleeding)
Step 4: Scale 1296x729 → 1440x810 (back to original size)
Result: Original 1440x810 landscape with bleeding in all directions

Another example:
Original: 1200x800 (landscape, aspect ratio 1.5)
Step 0: Resize to 1080x720, then pad to 1280x720 for API
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 3: Crop center 1080x720 region (10% bleeding)
Step 4: Scale 1080x720 → 1200x800 (back to original size)
Result: Original 1200x800 landscape with bleeding in all directions

Wide landscape example:
Original: 2000x800 (landscape, aspect ratio 2.5)
Step 0: Resize to 1800x720, then crop to 1280x720 for API
Step 1: Extend 1280x720 → 1024x1024 (square) - adds top+bottom
Step 2: Extend 1024x1024 → 1280x720 (landscape) - adds left+right
Step 3: Crop center 1800x720 region (10% bleeding)
Step 4: Scale 1800x720 → 2000x800 (back to original size)
Result: Original 2000x800 landscape with bleeding in all directions
```

## Complete Workflow for Arbitrary Input Sizes

### Universal Algorithm

```python
def process_arbitrary_image_for_bleeding(image, bleeding_direction, bleeding_amount=0.1):
    """
    Process any image size for bleeding effect
    
    Args:
        image: Input image of any dimensions
        bleeding_direction: 'horizontal', 'vertical', or 'both'
        bleeding_amount: Float 0-1, amount of bleeding (0.1 = 10%)
    """
    
    # Step 0: Prepare for API
    original_h, original_w = image.shape[:2]
    original_aspect = original_w / original_h
    
    # Determine image category and best API format
    if original_aspect > 1.5:  # Landscape-ish
        api_format = "landscape"
        api_size = (1280, 720)
    elif original_aspect < 0.7:  # Portrait-ish  
        api_format = "portrait"
        api_size = (720, 1280)
    else:  # Square-ish
        api_format = "square"
        api_size = (1024, 1024)
    
    # Resize to API format (may change aspect ratio slightly)
    api_ready_image = resize_and_pad_to_api_format(image, api_size)
    
    # Step 1-N: Execute extension sequence
    extended_image = execute_extension_sequence(
        api_ready_image, 
        api_format, 
        bleeding_direction
    )
    
    # Final step: Crop with bleeding and scale back to original size
    result = apply_bleeding_crop_and_scale(
        extended_image, 
        original_size=(original_h, original_w),
        bleeding_amount=bleeding_amount
    )
    
    return result

def resize_and_pad_to_api_format(image, target_size):
    """
    Resize image to fit API requirements while preserving as much content as possible
    """
    target_w, target_h = target_size
    current_h, current_w = image.shape[:2]
    
    # Calculate scaling to fit within target while preserving aspect ratio
    scale_w = target_w / current_w
    scale_h = target_h / current_h
    scale = min(scale_w, scale_h)
    
    # Resize maintaining aspect ratio
    new_w = int(current_w * scale)
    new_h = int(current_h * scale)
    resized = resize(image, (new_h, new_w))
    
    # Pad/crop to exact API dimensions
    if new_w < target_w or new_h < target_h:
        # Pad with edge pixels or use content-aware fill
        padded = pad_to_size(resized, target_size)
        return padded
    else:
        # Center crop if needed
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        return resized[start_y:start_y+target_h, start_x:start_x+target_w]

def execute_extension_sequence(image, original_format, bleeding_direction):
    """
    Execute the appropriate extension sequence based on format and desired bleeding
    """
    if original_format == "portrait":
        if bleeding_direction == "horizontal":
            return extend_api(image, (1024, 1024))  # portrait -> square
        elif bleeding_direction == "vertical":
            step1 = extend_api(image, (1024, 1024))  # portrait -> square  
            return extend_api(step1, (720, 1280))    # square -> portrait
        else:  # both
            step1 = extend_api(image, (1024, 1024))  # portrait -> square
            return extend_api(step1, (720, 1280))    # square -> portrait
    
    elif original_format == "landscape":
        if bleeding_direction == "vertical":
            return extend_api(image, (1024, 1024))   # landscape -> square
        elif bleeding_direction == "horizontal":
            step1 = extend_api(image, (1024, 1024))  # landscape -> square
            return extend_api(step1, (1280, 720))    # square -> landscape  
        else:  # both
            step1 = extend_api(image, (1024, 1024))  # landscape -> square
            return extend_api(step1, (1280, 720))    # square -> landscape
    
    else:  # square
        if bleeding_direction == "horizontal":
            return extend_api(image, (1280, 720))    # square -> landscape
        elif bleeding_direction == "vertical":
            return extend_api(image, (720, 1280))    # square -> portrait
        else:  # both - choose one path
            step1 = extend_api(image, (1280, 720))   # square -> landscape
            return extend_api(step1, (1024, 1024))   # landscape -> square

def apply_bleeding_crop_and_scale(extended_image, original_size, bleeding_amount):
    """
    Apply final crop with bleeding effect and scale back to original dimensions
    """
    original_h, original_w = original_size
    ext_h, ext_w = extended_image.shape[:2]
    
    # Calculate crop dimensions (smaller than original for bleeding effect)
    crop_w = int(original_w * (1 - bleeding_amount))
    crop_h = int(original_h * (1 - bleeding_amount))
    
    # We need to find the best crop region in the extended image
    # that corresponds to where our original content would be
    
    # Find center region of extended image
    center_x = ext_w // 2
    center_y = ext_h // 2
    
    # Calculate crop region
    start_x = center_x - crop_w // 2
    start_y = center_y - crop_h // 2
    
    # Ensure we don't go out of bounds
    start_x = max(0, min(start_x, ext_w - crop_w))
    start_y = max(0, min(start_y, ext_h - crop_h))
    
    # Crop the region
    cropped = extended_image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Scale back to original dimensions
    final_image = resize(cropped, (original_h, original_w))
    
    return final_image
```

### Example Workflows

**Example 1: Random Portrait Image**
```python
# Input: 500x800 portrait image
original_image = load_image("portrait_500x800.jpg")

# Process for 15% bleeding in all directions
result = process_arbitrary_image_for_bleeding(
    original_image, 
    bleeding_direction="both", 
    bleeding_amount=0.15
)

# Output: 500x800 image with bleeding effect
```

**Detailed steps for 500x800 portrait:**
```
Original: 500x800 (portrait, aspect ratio 0.625)
Step 0: Resize to 800x1280, then crop to 720x1280 for API
Step 1: Extend 720x1280 → 1024x1024 (adds left+right content)  
Step 2: Extend 1024x1024 → 720x1280 (adds top+bottom content)
Step 3: Crop center 425x680 region (15% bleeding)
Step 4: Scale 425x680 → 500x800 (back to original size)
Result: 500x800 portrait with bleeding in all directions
```

**Example 2: Wide Landscape Image**
```
Original: 1920x1080 (landscape, aspect ratio 1.78)
Step 0: Resize to 1280x720 for API
Step 1: Extend 1280x720 → 1024x1024 (adds top+bottom content)
Step 2: Extend 1024x1024 → 1280x720 (adds left+right content)  
Step 3: Crop center 1632x918 region (15% bleeding)
Step 4: Scale 1632x918 → 1920x1080 (back to original size)
Result: 1920x1080 landscape with bleeding in all directions
```

### Step 1: Determine Bleeding Requirements
```python
def determine_bleeding_strategy(original_aspect, bleeding_direction):
    """
    bleeding_direction options:
    - 'horizontal': left/right bleeding
    - 'vertical': top/bottom bleeding  
    - 'both': all directions bleeding
    """
    
    if original_aspect > 1:  # Landscape
        if bleeding_direction == 'vertical':
            return ['landscape_to_square']
        elif bleeding_direction == 'horizontal':
            return ['landscape_to_square', 'square_to_landscape']
        else:  # both
            return ['landscape_to_square', 'square_to_landscape']
    
    elif original_aspect < 1:  # Portrait
        if bleeding_direction == 'horizontal':
            return ['portrait_to_square']
        elif bleeding_direction == 'vertical':
            return ['portrait_to_square', 'square_to_portrait']
        else:  # both
            return ['portrait_to_square', 'square_to_portrait']
    
    else:  # Square
        if bleeding_direction == 'horizontal':
            return ['square_to_landscape']
        elif bleeding_direction == 'vertical':
            return ['square_to_portrait']
        else:  # both
            return ['square_to_landscape', 'landscape_to_square']  # or portrait path
```

### Step 2: Execute Extension Sequence
```python
def execute_extension_sequence(image, steps, target_final_size):
    current_image = image
    
    for step in steps:
        if step == 'portrait_to_square':
            current_image = extend_api(current_image, target_size=(1024, 1024))
        elif step == 'square_to_landscape':
            current_image = extend_api(current_image, target_size=(1280, 720))
        elif step == 'square_to_portrait':
            current_image = extend_api(current_image, target_size=(720, 1280))
        elif step == 'landscape_to_square':
            current_image = extend_api(current_image, target_size=(1024, 1024))
    
    return current_image
```

### Step 3: Crop and Scale
```python
def apply_bleeding_crop(extended_image, original_size, bleeding_amount):
    """
    bleeding_amount: percentage of padding/bleeding (e.g., 0.1 for 10% bleeding)
    """
    ext_h, ext_w = extended_image.shape[:2]
    orig_h, orig_w = original_size
    
    # Calculate crop size (smaller than original to create bleeding effect)
    crop_w = int(orig_w * (1 - bleeding_amount))
    crop_h = int(orig_h * (1 - bleeding_amount))
    
    # Center crop
    start_x = (ext_w - crop_w) // 2
    start_y = (ext_h - crop_h) // 2
    
    cropped = extended_image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Scale back to original dimensions
    final_image = resize(cropped, (orig_h, orig_w))
    
    return final_image
```

## Key Considerations

1. **Bleeding Amount**: The smaller you crop relative to original size, the more bleeding effect you get
2. **Quality Loss**: Multiple API calls and scaling may reduce quality
3. **Computational Cost**: More complex bleeding (both directions) requires 2 API calls
4. **Content Consistency**: The AI model should maintain style consistency across extensions

## Optimization Tips

- For maximum efficiency, choose single-direction bleeding when possible
- Test different bleeding percentages (5%, 10%, 15%) to find optimal visual results  
- Consider caching intermediate results for similar images
- Pre-resize input images to exact target resolutions for best API performance