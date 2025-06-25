from PIL import Image, ImageFilter
import numpy as np
import random
from collections import Counter
from print_settings import mm_to_pixels
import base64
import io
import os
import cv2
from skimage import segmentation, morphology, measure
from scipy import ndimage

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.strip():
        client = OpenAI(api_key=api_key)
        OPENAI_CONFIGURED = True
    else:
        client = None
        OPENAI_CONFIGURED = False
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_CONFIGURED = False
    client = None

def validate_openai_setup():
    """Validate OpenAI setup and return status message"""
    if not OPENAI_AVAILABLE:
        return False, "OpenAI library not installed. Run: pip install openai"
    
    if not OPENAI_CONFIGURED:
        return False, "OpenAI API key not found. Set OPENAI_API_KEY environment variable"
    
    # Test API connection
    try:
        # Simple test call to verify API key
        client.models.list()
        return True, "OpenAI ready"
    except Exception as e:
        return False, f"OpenAI API error: {str(e)}"

def test_ai_padding_simple():
    """Simple test function for AI padding with a small test image"""
    if not OPENAI_CONFIGURED:
        return None, "OpenAI not configured"
    
    try:
        # Create a simple test image
        test_img = Image.new('RGB', (100, 100), (255, 0, 0))  # Red square
        result = create_ai_padding(test_img, 20, "natural", "Add a simple blue background")
        return result, "Success"
    except Exception as e:
        return None, str(e)

def get_dominant_colors(img, num_colors=5):
    """Get dominant colors from image"""
    # Convert to numpy array and reshape
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    
    # Sample pixels for performance
    sample_size = min(10000, len(pixels))
    sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
    
    # Count color occurrences (with some clustering)
    color_counts = Counter()
    for pixel in sampled_pixels:
        # Round colors to reduce variation
        rounded = tuple((pixel // 16) * 16)
        color_counts[rounded] += 1
    
    # Get most common colors
    dominant = [color for color, count in color_counts.most_common(num_colors)]
    return dominant if dominant else [(128, 128, 128)]

def get_average_edge_color(img, edge):
    """Get average color of an edge"""
    width, height = img.size
    
    if edge == 'left':
        edge_pixels = [img.getpixel((0, y)) for y in range(height)]
    elif edge == 'right':
        edge_pixels = [img.getpixel((width-1, y)) for y in range(height)]
    elif edge == 'top':
        edge_pixels = [img.getpixel((x, 0)) for x in range(width)]
    elif edge == 'bottom':
        edge_pixels = [img.getpixel((x, height-1)) for x in range(width)]
    else:
        return (128, 128, 128)  # Default gray
    
    # Calculate average color
    avg_r = sum(p[0] for p in edge_pixels) // len(edge_pixels)
    avg_g = sum(p[1] for p in edge_pixels) // len(edge_pixels)
    avg_b = sum(p[2] for p in edge_pixels) // len(edge_pixels)
    
    return (avg_r, avg_g, avg_b)

def interpolate_color(color1, color2, ratio=0.5):
    """Interpolate between two colors"""
    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
    return (r, g, b)

def get_average_image_color(img):
    """Get average color of entire image"""
    # Sample the image for performance
    img_array = np.array(img)
    return tuple(np.mean(img_array, axis=(0, 1)).astype(int))

def darken_color(color, factor):
    """Darken a color by a factor (0-1)"""
    return tuple(max(0, int(c * (1 - factor))) for c in color)

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def analyze_image_edges(img):
    """Analyze the edges of the image to understand what needs to be extended"""
    width, height = img.size
    
    # Sample edge pixels more thoroughly
    edge_samples = {
        'top': [img.getpixel((x, 0)) for x in range(0, width, max(1, width//20))],
        'bottom': [img.getpixel((x, height-1)) for x in range(0, width, max(1, width//20))],
        'left': [img.getpixel((0, y)) for y in range(0, height, max(1, height//20))],
        'right': [img.getpixel((width-1, y)) for y in range(0, height, max(1, height//20))]
    }
    
    # Analyze edge characteristics
    edge_analysis = {}
    for edge, pixels in edge_samples.items():
        avg_color = tuple(sum(p[i] for p in pixels) // len(pixels) for i in range(3))
        
        # Determine if edge is uniform or varied
        color_variance = sum(abs(p[i] - avg_color[i]) for p in pixels for i in range(3)) / (len(pixels) * 3)
        
        edge_analysis[edge] = {
            'avg_color': avg_color,
            'variance': color_variance,
            'is_uniform': color_variance < 30,  # Low variance = uniform
            'brightness': sum(avg_color) / 3
        }
    
    return edge_analysis

def detect_image_content_type(img):
    """Detect the type of content in the image to guide padding generation"""
    width, height = img.size
    
    # Sample center and corners
    center_color = img.getpixel((width//2, height//2))
    corner_colors = [
        img.getpixel((0, 0)),
        img.getpixel((width-1, 0)),
        img.getpixel((0, height-1)),
        img.getpixel((width-1, height-1))
    ]
    
    # Analyze overall image characteristics
    dominant_colors = get_dominant_colors(img, 5)
    avg_color = get_average_image_color(img)
    
    # Determine content type based on color distribution and edge analysis
    edge_analysis = analyze_image_edges(img)
    
    # Check if it's likely a product photo (uniform background)
    uniform_edges = sum(1 for edge in edge_analysis.values() if edge['is_uniform'])
    
    if uniform_edges >= 3:
        content_type = "product_photo"  # Likely has uniform background
    elif avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
        content_type = "light_background"  # Light/white background
    elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
        content_type = "dark_background"  # Dark background
    else:
        content_type = "complex_scene"  # Complex scene or portrait
    
    return {
        'type': content_type,
        'dominant_colors': dominant_colors,
        'avg_color': avg_color,
        'edge_analysis': edge_analysis
    }

def generate_padding_prompt(img, padding_style="natural"):
    """Generate highly specific prompts for seamless AI padding generation"""
    
    # Analyze the image content
    content_info = detect_image_content_type(img)
    content_type = content_info['type']
    dominant_colors = content_info['dominant_colors']
    avg_color = content_info['avg_color']
    edge_analysis = content_info['edge_analysis']
    
    # Create color descriptions
    primary_color = f"RGB({dominant_colors[0][0]}, {dominant_colors[0][1]}, {dominant_colors[0][2]})"
    secondary_color = f"RGB({dominant_colors[1][0]}, {dominant_colors[1][1]}, {dominant_colors[1][2]})" if len(dominant_colors) > 1 else primary_color
    
    # Build edge-specific instructions
    edge_instructions = []
    for edge, analysis in edge_analysis.items():
        color = analysis['avg_color']
        if analysis['is_uniform']:
            edge_instructions.append(f"{edge} edge: continue the uniform {color} color")
        else:
            edge_instructions.append(f"{edge} edge: extend the varied colors and patterns")
    
    # Base prompt templates focused on seamless extension
    base_prompts = {
        "natural": f"""Seamlessly extend the image edges outward to create natural padding. 
CRITICAL: Only extend what already exists at the edges - do not add new objects, text, or unrelated elements.
Edge extension rules: {'; '.join(edge_instructions)}.
Use colors {primary_color} and {secondary_color} for smooth transitions.
The padding must look like a natural continuation of the existing image edges.""",
        
        "artistic": f"""Create artistic padding by extending the existing edge content with enhanced visual appeal.
CRITICAL: Base all padding on the actual edge content - no new objects or text.
Edge extension: {'; '.join(edge_instructions)}.
Enhance with subtle artistic effects using {primary_color} and {secondary_color}.
Keep the artistic elements minimal and focused on edge extension only.""",
        
        "minimalist": f"""Generate clean, minimal padding by extending edges with simplified, elegant transitions.
CRITICAL: Only extend existing edge colors and patterns - no additional elements.
Edge rules: {'; '.join(edge_instructions)}.
Use {primary_color} and {secondary_color} for smooth, minimal gradients.
Keep the padding extremely simple and unobtrusive.""",
        
        "textured": f"""Extend the image edges with subtle texture that matches the existing content.
CRITICAL: Base textures on what's already at the edges - no new objects or designs.
Edge extension: {'; '.join(edge_instructions)}.
Add gentle texture using {primary_color} and {secondary_color} tones.
Texture should feel like a natural continuation of the image surface.""",
        
        "blurred": f"""Create soft padding by extending and blurring the existing edge content outward.
CRITICAL: Only blur and extend what's already at the edges - no new content.
Edge blur rules: {'; '.join(edge_instructions)}.
Use {primary_color} and {secondary_color} for smooth color transitions.
Blur should fade naturally from the existing edges."""
    }
    
    # Content-type specific adjustments
    if content_type == "product_photo":
        suffix = " IMPORTANT: This appears to be a product photo - extend the background uniformly without adding new objects or text."
    elif content_type == "light_background":
        suffix = " IMPORTANT: Maintain the light background by extending it smoothly - no dark elements or text."
    elif content_type == "dark_background":
        suffix = " IMPORTANT: Maintain the dark background by extending it smoothly - no bright elements or text."
    else:
        suffix = " IMPORTANT: This is a complex scene - only extend the existing edge content, never add new objects, people, or text."
    
    # Get base prompt and add content-specific instructions
    prompt = base_prompts.get(padding_style, base_prompts["natural"])
    prompt += suffix
    
    # Add final technical requirements
    prompt += " The result must be seamless with no visible boundaries between original and extended content."
    
    return prompt

def optimize_prompt_for_seamless_results(prompt, content_info):
    """Optimize the prompt to ensure seamless results and avoid common issues"""
    
    # Common problematic phrases that lead to irrelevant content
    avoid_phrases = [
        "add objects", "add text", "add elements", "add details", "add features",
        "create new", "generate new", "include", "place", "insert", "put",
        "logo", "watermark", "signature", "writing", "letters", "words"
    ]
    
    # Ensure we're not accidentally including problematic language
    prompt_lower = prompt.lower()
    for phrase in avoid_phrases:
        if phrase in prompt_lower:
            print(f"Warning: Potentially problematic phrase detected: '{phrase}'")
    
    # Add specific negative instructions to prevent common issues
    seamless_reinforcement = [
        "NO text, letters, words, or writing of any kind",
        "NO logos, watermarks, or signatures", 
        "NO new objects that weren't at the edges",
        "NO people, faces, or body parts unless extending existing ones",
        "NO geometric shapes unless extending existing patterns",
        "ONLY extend and continue what already exists at the image boundaries"
    ]
    
    # Add the reinforcement to the prompt
    reinforcement_text = " STRICT RULES: " + "; ".join(seamless_reinforcement)
    
    # Ensure prompt isn't too long (DALL-E 2 limit is 1000 chars)
    total_length = len(prompt) + len(reinforcement_text)
    if total_length > 950:  # Leave some buffer
        # Truncate the original prompt to make room for reinforcement
        max_prompt_length = 950 - len(reinforcement_text)
        prompt = prompt[:max_prompt_length].rsplit(' ', 1)[0]  # Cut at word boundary
    
    final_prompt = prompt + reinforcement_text
    
    return final_prompt

def create_ai_padding(img, padding_px, padding_style="natural", custom_prompt=None, remove_objects=False, object_removal_sensitivity=0.3):
    """
    Create AI-generated padding following the specified workflow:
    1. Remove objects (mandatory if enabled)
    2. Create RGBA image with original + padding dimensions
    3. Scale to OpenAI-compatible size
    4. Process with OpenAI
    5. Scale back to original + padding size
    
    Args:
        remove_objects: Whether to remove distracting objects before AI processing
        object_removal_sensitivity: 0.0-1.0, higher = more aggressive object removal
    
    Returns:
        tuple: (final_result, intermediate_images_dict)
    """
    
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not available. Please install with: pip install openai")
    
    if not client:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    # Dictionary to store intermediate images for display
    intermediate_images = {}
    
    # Step 0: MANDATORY object removal preprocessing if enabled
    # This MUST happen before any AI processing as requested by user
    processed_img = img
    object_mask = None
    if remove_objects:
        print(f"0. MANDATORY: Removing distracting objects (sensitivity: {object_removal_sensitivity})...")
        try:
            processed_img, object_mask = detect_and_remove_objects(img, method="auto", sensitivity=object_removal_sensitivity)
            print(f"   Objects removed from {np.sum(object_mask)} pixels")
            intermediate_images['object_mask'] = Image.fromarray((object_mask * 255).astype(np.uint8))
            intermediate_images['after_object_removal'] = processed_img
            print(f"   ✅ Object removal completed BEFORE AI processing")
        except Exception as e:
            print(f"   ❌ Object removal failed: {e}, using original image")
            processed_img = img
    
    # This is the final processed image that will be sent to AI
    intermediate_images['processed_for_ai'] = processed_img
    
    # Step 1: Create new RGBA image using DOUBLE the padding dimensions (buffer zone)
    original_width, original_height = processed_img.size
    target_width = original_width + (2 * padding_px)  # Final target size
    target_height = original_height + (2 * padding_px)
    
    # DOUBLE the padding for OpenAI processing (buffer zone)
    buffer_padding_px = padding_px * 2
    buffer_width = original_width + (2 * buffer_padding_px)
    buffer_height = original_height + (2 * buffer_padding_px)
    
    print(f"AI Padding Workflow with Buffer Zone:")
    print(f"1. Original size: {original_width}×{original_height}")
    print(f"2. Target size with padding: {target_width}×{target_height}")
    print(f"3. Buffer size with DOUBLE padding: {buffer_width}×{buffer_height}")
    
    # Convert to RGB if needed
    if processed_img.mode != 'RGB':
        processed_img = processed_img.convert('RGB')
    
    # Create the BUFFER RGBA image with DOUBLE padding
    padded_img = Image.new('RGBA', (buffer_width, buffer_height), (255, 255, 255, 0))  # Transparent
    
    # Center the processed image in the buffer
    paste_x = buffer_padding_px
    paste_y = buffer_padding_px
    
    # Convert processed to RGBA and paste
    if processed_img.mode != 'RGBA':
        img_rgba = processed_img.convert('RGBA')
    else:
        img_rgba = processed_img
    
    padded_img.paste(img_rgba, (paste_x, paste_y))
    intermediate_images['padded_rgba'] = padded_img
    
    # Step 2: Scale the BUFFER RGBA image to OpenAI model accepted dimensions
    # OpenAI dall-e-2 supports: 256x256, 512x512, 1024x1024 (square only)
    valid_sizes = [256, 512, 1024]
    
    # Find the best size that accommodates our BUFFER image
    max_dimension = max(buffer_width, buffer_height)
    openai_size = min([s for s in valid_sizes if s >= max_dimension], default=1024)
    
    print(f"4. Creating square canvas with equal padding for OpenAI: {openai_size}×{openai_size}")
    
    # Instead of stretching, maintain aspect ratio and center with equal padding
    # Calculate scale to fit the buffer image in the square while maintaining aspect ratio
    scale_factor = openai_size / max_dimension
    scaled_buffer_width = int(buffer_width * scale_factor)
    scaled_buffer_height = int(buffer_height * scale_factor)
    
    # Scale the buffer image maintaining aspect ratio
    scaled_padded = padded_img.resize((scaled_buffer_width, scaled_buffer_height), Image.Resampling.LANCZOS)
    
    # Create square canvas and center the scaled image (this maintains equal padding)
    canvas_padded = Image.new('RGBA', (openai_size, openai_size), (255, 255, 255, 0))
    paste_x = (openai_size - scaled_buffer_width) // 2
    paste_y = (openai_size - scaled_buffer_height) // 2
    canvas_padded.paste(scaled_padded, (paste_x, paste_y))
    
    # Store positioning info for later use
    canvas_info = {
        'paste_x': paste_x,
        'paste_y': paste_y,
        'scaled_width': scaled_buffer_width,
        'scaled_height': scaled_buffer_height,
        'scale_factor': scale_factor
    }
    
    # Convert to RGB for OpenAI - fill transparent padding areas with edge colors instead of white
    openai_rgb = Image.new('RGB', (openai_size, openai_size), (255, 255, 255))
    
    # Get the RGB version without alpha for analysis
    canvas_rgb = Image.new('RGB', (openai_size, openai_size), (255, 255, 255))
    canvas_rgb.paste(canvas_padded, (0, 0), canvas_padded)
    
    # Analyze the canvas image to get edge colors for better padding fill
    edge_colors = {
        'top': get_average_edge_color(canvas_rgb, 'top'),
        'bottom': get_average_edge_color(canvas_rgb, 'bottom'),
        'left': get_average_edge_color(canvas_rgb, 'left'),
        'right': get_average_edge_color(canvas_rgb, 'right')
    }
    
    # Fill transparent areas with appropriate edge colors instead of white
    for y in range(openai_size):
        for x in range(openai_size):
            # Check if this pixel is transparent in the original
            alpha = canvas_padded.getpixel((x, y))[3] if canvas_padded.mode == 'RGBA' else 255
            if alpha == 0:  # Transparent pixel (padding area)
                # Determine which edge color to use based on position
                center_x, center_y = openai_size // 2, openai_size // 2
                
                if y < center_y and x < center_x:
                    # Top-left quadrant
                    fill_color = interpolate_color(edge_colors['top'], edge_colors['left'], 0.5)
                elif y < center_y and x >= center_x:
                    # Top-right quadrant  
                    fill_color = interpolate_color(edge_colors['top'], edge_colors['right'], 0.5)
                elif y >= center_y and x < center_x:
                    # Bottom-left quadrant
                    fill_color = interpolate_color(edge_colors['bottom'], edge_colors['left'], 0.5)
                else:
                    # Bottom-right quadrant
                    fill_color = interpolate_color(edge_colors['bottom'], edge_colors['right'], 0.5)
                
                openai_rgb.putpixel((x, y), fill_color)
            else:
                # Non-transparent pixel, use the original
                rgb_pixel = canvas_rgb.getpixel((x, y))
                openai_rgb.putpixel((x, y), rgb_pixel)
    
    intermediate_images['openai_input'] = openai_rgb
    

    
    # Generate and optimize prompt
    if custom_prompt:
        # Still optimize custom prompts
        content_info = detect_image_content_type(processed_img)
        prompt = optimize_prompt_for_seamless_results(custom_prompt, content_info)
    else:
        # Generate optimized prompt based on processed image
        content_info = detect_image_content_type(processed_img)
        base_prompt = generate_padding_prompt(processed_img, padding_style)
        prompt = optimize_prompt_for_seamless_results(base_prompt, content_info)
    
    print(f"4. Optimized prompt ({len(prompt)} chars): {prompt[:150]}...")
    
    try:
        # Step 3: Send to OpenAI for processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as base_file:
            openai_rgb.save(base_file.name, format='PNG')
            base_path = base_file.name
            
        # Create enhanced mask for better blending
        mask = Image.new('RGBA', (openai_size, openai_size), (0, 0, 0, 0))  # Fully transparent
        
        # Calculate original image area in the canvas context
        orig_canvas_width = int(original_width * canvas_info['scale_factor'])
        orig_canvas_height = int(original_height * canvas_info['scale_factor'])
        orig_x = canvas_info['paste_x'] + int(buffer_padding_px * canvas_info['scale_factor'])
        orig_y = canvas_info['paste_y'] + int(buffer_padding_px * canvas_info['scale_factor'])
        
        # Make the entire original image area opaque (preserve it completely)
        # Note: We preserve the ORIGINAL image area, not the processed one
        if orig_canvas_width > 0 and orig_canvas_height > 0:
            opaque_area = Image.new('RGBA', (orig_canvas_width, orig_canvas_height), (255, 255, 255, 255))
            mask.paste(opaque_area, (orig_x, orig_y))
            
        print(f"4.5. Created preservation mask for area: {orig_x},{orig_y} to {orig_x + orig_canvas_width},{orig_y + orig_canvas_height}")
        intermediate_images['openai_mask'] = mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as mask_file:
            mask.save(mask_file.name, format='PNG')
            mask_path = mask_file.name
        
        # Check file sizes
        base_size = os.path.getsize(base_path) / (1024 * 1024)  # MB
        if base_size > 4.0:
            raise Exception(f"Image too large: {base_size:.2f} MB (max 4MB for DALL-E 2)")
        
        # Make API call to OpenAI
        with open(base_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            response = client.images.edit(
                model="dall-e-2",
                image=image_file,
                mask=mask_file,
                prompt=prompt,
                size=f"{openai_size}x{openai_size}",
                response_format="b64_json",
                n=1
            )
        
        # Clean up temporary files
        try:
            os.unlink(base_path)
            os.unlink(mask_path)
        except:
            pass
        
        # Get the generated image from OpenAI
        ai_generated = base64_to_image(response.data[0].b64_json)
        intermediate_images['openai_output'] = ai_generated
        
        print(f"5. Received from OpenAI: {ai_generated.size}")
        
        # Step 4: Extract the relevant area and scale back to buffer size
        print(f"5. Extracting from canvas and scaling back to buffer size: {buffer_width}×{buffer_height}")
        
        # Extract the relevant area from the OpenAI result (remove the canvas padding)
        extract_x = canvas_info['paste_x']
        extract_y = canvas_info['paste_y']
        extract_width = canvas_info['scaled_width']
        extract_height = canvas_info['scaled_height']
        
        ai_cropped = ai_generated.crop((extract_x, extract_y, extract_x + extract_width, extract_y + extract_height))
        
        # Scale back to buffer dimensions
        buffer_result = ai_cropped.resize((buffer_width, buffer_height), Image.Resampling.LANCZOS)
        intermediate_images['scaled_to_buffer'] = buffer_result
        
        # Step 5: Trim the buffer to target size (remove half the padding on each side)
        print(f"6. Trimming buffer to target size: {target_width}×{target_height}")
        
        # Calculate trim coordinates (remove half the extra padding)
        trim_left = padding_px  # Remove half of the double padding
        trim_top = padding_px
        trim_right = buffer_width - padding_px
        trim_bottom = buffer_height - padding_px
        
        final_result = buffer_result.crop((trim_left, trim_top, trim_right, trim_bottom))
        intermediate_images['scaled_back'] = final_result
        
        # Step 5: Replace the center with the ORIGINAL image (not the processed one)
        # This ensures we keep the original content while using the AI-generated padding
        final_result.paste(img, (padding_px, padding_px))
        intermediate_images['final_result'] = final_result
        
        print(f"7. Final result with original content restored: {final_result.size}")
        print(f"   Buffer zone strategy: Used {buffer_padding_px}px padding for AI, trimmed to {padding_px}px")
        
        # Return both the final result and intermediate images
        return final_result, intermediate_images
        
    except Exception as e:
        # Clean up temporary files in case of error
        try:
            if 'base_path' in locals():
                os.unlink(base_path)
            if 'mask_path' in locals():
                os.unlink(mask_path)
        except:
            pass
        
        # More detailed error information
        error_msg = str(e)
        if "image_generation_user_error" in error_msg:
            error_msg += " - This may be due to image format, size, or prompt issues."
        
        raise Exception(f"OpenAI API error: {error_msg}")

def create_content_aware_padding(img, padding_px):
    """Create content-aware padding similar to content-aware bleed"""
    width, height = img.size
    new_width = width + (2 * padding_px)
    new_height = height + (2 * padding_px)
    
    # Analyze image content
    dominant_colors = get_dominant_colors(img, 3)
    
    # Create base with dominant color
    padded_img = Image.new('RGB', (new_width, new_height), dominant_colors[0])
    
    # Add subtle texture using edge extension with fade effect
    if padding_px > 0:
        # Extend edges with fade effect - same as content aware bleed
        for i in range(padding_px):
            # Gradient starts with dominant color (margin) and fades to edge color (near image)
            fade_ratio = i / padding_px  # 0 at margin, 1 near image
            
            # Top edge
            top_line = img.crop((0, 0, width, 1))
            top_faded = Image.blend(Image.new('RGB', (width, 1), dominant_colors[0]), top_line, fade_ratio)
            padded_img.paste(top_faded, (padding_px, i))
            
            # Bottom edge
            bottom_line = img.crop((0, height-1, width, height))
            bottom_faded = Image.blend(Image.new('RGB', (width, 1), dominant_colors[0]), bottom_line, fade_ratio)
            padded_img.paste(bottom_faded, (padding_px, new_height - 1 - i))
            
            # Left edge  
            left_line = img.crop((0, 0, 1, height))
            left_faded = Image.blend(Image.new('RGB', (1, height), dominant_colors[0]), left_line, fade_ratio)
            padded_img.paste(left_faded, (i, padding_px))
            
            # Right edge
            right_line = img.crop((width-1, 0, width, height))
            right_faded = Image.blend(Image.new('RGB', (1, height), dominant_colors[0]), right_line, fade_ratio)
            padded_img.paste(right_faded, (new_width - 1 - i, padding_px))
    
    # Paste original image
    padded_img.paste(img, (padding_px, padding_px))
    return padded_img

def create_soft_shadow_padding(img, padding_px):
    """Create soft shadow padding effect"""
    width, height = img.size
    new_width = width + (2 * padding_px)
    new_height = height + (2 * padding_px)
    
    # Create base with slightly darker background
    avg_color = get_average_image_color(img)
    shadow_color = darken_color(avg_color, 0.15)
    
    padded_img = Image.new('RGB', (new_width, new_height), shadow_color)
    
    # Create soft gradient around the edges
    for i in range(padding_px):
        # Create gradient effect from edge to center
        fade_ratio = (padding_px - i) / padding_px
        current_color = interpolate_color(shadow_color, avg_color, fade_ratio * 0.3)
        
        # Create frame
        frame_img = Image.new('RGB', (new_width - 2*i, new_height - 2*i), current_color)
        padded_img.paste(frame_img, (i, i))
    
    # Apply subtle blur for smooth transition
    padded_img = padded_img.filter(ImageFilter.GaussianBlur(radius=padding_px/8))
    
    # Paste original image
    padded_img.paste(img, (padding_px, padding_px))
    
    return padded_img

def create_gradient_fade_padding(img, padding_px):
    """Create gradient fade padding"""
    width, height = img.size
    new_width = width + (2 * padding_px)
    new_height = height + (2 * padding_px)
    
    # Get edge colors
    edge_colors = {
        'top': get_average_edge_color(img, 'top'),
        'bottom': get_average_edge_color(img, 'bottom'),
        'left': get_average_edge_color(img, 'left'),
        'right': get_average_edge_color(img, 'right')
    }
    
    # Create base with average of edge colors
    avg_edge_color = tuple(sum(c[i] for c in edge_colors.values()) // 4 for i in range(3))
    padded_img = Image.new('RGB', (new_width, new_height), avg_edge_color)
    
    # Create smooth gradients from each edge
    for y in range(new_height):
        for x in range(new_width):
            # Skip center area where image will be placed
            if padding_px <= x < width + padding_px and padding_px <= y < height + padding_px:
                continue
            
            # Calculate distance from nearest edge
            dist_to_left = x
            dist_to_right = new_width - 1 - x
            dist_to_top = y
            dist_to_bottom = new_height - 1 - y
            
            min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            if min_dist < padding_px:
                # Create gradient based on distance
                fade_ratio = min_dist / padding_px
                
                # Determine which edge color to use
                if min_dist == dist_to_left:
                    edge_color = edge_colors['left']
                elif min_dist == dist_to_right:
                    edge_color = edge_colors['right']
                elif min_dist == dist_to_top:
                    edge_color = edge_colors['top']
                else:
                    edge_color = edge_colors['bottom']
                
                blended_color = interpolate_color(avg_edge_color, edge_color, fade_ratio)
                padded_img.putpixel((x, y), blended_color)
    
    # Apply blur for smoothness
    padded_img = padded_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Paste original image
    padded_img.paste(img, (padding_px, padding_px))
    
    return padded_img

def create_color_blend_padding(img, padding_px):
    """Create color blend padding using dominant colors"""
    width, height = img.size
    new_width = width + (2 * padding_px)
    new_height = height + (2 * padding_px)
    
    # Get dominant colors
    dominant_colors = get_dominant_colors(img, 3)
    
    # Create gradient using dominant colors
    padded_img = Image.new('RGB', (new_width, new_height))
    
    # Fill with gradient of dominant colors
    for y in range(new_height):
        for x in range(new_width):
            # Skip center area
            if padding_px <= x < width + padding_px and padding_px <= y < height + padding_px:
                continue
            
            # Create pattern based on position
            color_index = ((x // 20) + (y // 20)) % len(dominant_colors)
            base_color = dominant_colors[color_index]
            
            # Add some variation
            noise_factor = 0.1
            varied_color = tuple(
                max(0, min(255, int(base_color[i] + random.randint(-20, 20) * noise_factor)))
                for i in range(3)
            )
            
            padded_img.putpixel((x, y), varied_color)
    
    # Heavy blur for smooth blending
    padded_img = padded_img.filter(ImageFilter.GaussianBlur(radius=padding_px/4))
    
    # Paste original image
    padded_img.paste(img, (padding_px, padding_px))
    
    return padded_img

def create_vintage_vignette_padding(img, padding_px):
    """Create vintage vignette padding effect"""
    width, height = img.size
    new_width = width + (2 * padding_px)
    new_height = height + (2 * padding_px)
    
    # Get average color and darken it
    avg_color = get_average_image_color(img)
    dark_color = darken_color(avg_color, 0.4)
    
    padded_img = Image.new('RGB', (new_width, new_height), dark_color)
    
    # Create vignette effect
    center_x, center_y = new_width // 2, new_height // 2
    max_distance = ((new_width/2)**2 + (new_height/2)**2)**0.5
    
    for y in range(new_height):
        for x in range(new_width):
            # Skip center area
            if padding_px <= x < width + padding_px and padding_px <= y < height + padding_px:
                continue
            
            # Calculate distance from center
            distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
            
            # Create vignette fade
            fade_ratio = min(1.0, distance / (max_distance * 0.8))
            vignette_color = interpolate_color(avg_color, dark_color, fade_ratio)
            
            padded_img.putpixel((x, y), vignette_color)
    
    # Apply blur
    padded_img = padded_img.filter(ImageFilter.GaussianBlur(radius=padding_px/6))
    
    # Paste original image
    padded_img.paste(img, (padding_px, padding_px))
    
    return padded_img

def create_clean_border_padding(img, padding_px):
    """Create clean border padding"""
    width, height = img.size
    new_width = width + (2 * padding_px)
    new_height = height + (2 * padding_px)
    
    # Use white or light color based on image
    avg_color = get_average_image_color(img)
    
    # Choose white or black based on image brightness
    brightness = sum(avg_color) / 3
    if brightness > 128:
        border_color = (250, 250, 250)  # Light gray
    else:
        border_color = (30, 30, 30)    # Dark gray
    
    padded_img = Image.new('RGB', (new_width, new_height), border_color)
    
    # Create subtle inner shadow for depth
    shadow_width = min(padding_px // 3, 10)
    if shadow_width > 0:
        shadow_color = darken_color(border_color, 0.1)
        
        # Create inner shadow
        for i in range(shadow_width):
            fade_ratio = i / shadow_width
            current_color = interpolate_color(shadow_color, border_color, fade_ratio)
            
            # Draw shadow frame
            for x in range(padding_px - shadow_width + i, width + padding_px + shadow_width - i):
                for y in [padding_px - shadow_width + i, height + padding_px + shadow_width - i - 1]:
                    if 0 <= x < new_width and 0 <= y < new_height:
                        padded_img.putpixel((x, y), current_color)
            
            for y in range(padding_px - shadow_width + i, height + padding_px + shadow_width - i):
                for x in [padding_px - shadow_width + i, width + padding_px + shadow_width - i - 1]:
                    if 0 <= x < new_width and 0 <= y < new_height:
                        padded_img.putpixel((x, y), current_color)
    
    # Paste original image
    padded_img.paste(img, (padding_px, padding_px))
    
    return padded_img

def add_padding(img, padding_mm, dpi, padding_style='content_aware', ai_style='natural', custom_prompt=None, remove_objects=False, object_removal_sensitivity=0.3):
    """Add padding that blends smoothly with the image using various methods"""
    if padding_mm <= 0:
        return img, {}
    
    # Calculate padding in pixels
    padding_px = mm_to_pixels(padding_mm, dpi)
    
    # Choose padding method based on style
    if padding_style == 'ai_advanced':
        if OPENAI_AVAILABLE and client:
            return create_ai_padding(img, padding_px, ai_style, custom_prompt, remove_objects, object_removal_sensitivity)
        else:
            # Fallback to content-aware if OpenAI not available
            print("Warning: OpenAI not available, falling back to content-aware padding")
            return create_content_aware_padding(img, padding_px), {}
    elif padding_style == 'content_aware':
        return create_content_aware_padding(img, padding_px), {}
    elif padding_style == 'soft_shadow':
        return create_soft_shadow_padding(img, padding_px), {}
    elif padding_style == 'gradient_fade':
        return create_gradient_fade_padding(img, padding_px), {}
    elif padding_style == 'color_blend':
        return create_color_blend_padding(img, padding_px), {}
    elif padding_style == 'vintage_vignette':
        return create_vintage_vignette_padding(img, padding_px), {}
    elif padding_style == 'clean_border':
        return create_clean_border_padding(img, padding_px), {}
    else:
        # Default to content-aware
        return create_content_aware_padding(img, padding_px), {}

def detect_and_remove_objects(img, method="auto", sensitivity=0.3):
    """
    Detect and remove objects from image, filling with background color or neighborhood colors
    
    Args:
        img: PIL Image
        method: "auto", "edge_detection", "color_clustering", "slic_segmentation"
        sensitivity: 0.0-1.0, higher = more aggressive removal
    """
    
    # Convert to numpy array
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    if method == "auto":
        # Try multiple methods and combine results
        mask1 = detect_objects_edge_based(img_array, sensitivity)
        mask2 = detect_objects_color_clustering(img_array, sensitivity)
        mask3 = detect_objects_slic_segmentation(img_array, sensitivity)
        
        # Combine masks (object is detected if any method detects it)
        combined_mask = mask1 | mask2 | mask3
        
        # Clean up the mask
        combined_mask = clean_object_mask(combined_mask)
        
    elif method == "edge_detection":
        combined_mask = detect_objects_edge_based(img_array, sensitivity)
    elif method == "color_clustering":
        combined_mask = detect_objects_color_clustering(img_array, sensitivity)
    elif method == "slic_segmentation":
        combined_mask = detect_objects_slic_segmentation(img_array, sensitivity)
    else:
        combined_mask = detect_objects_edge_based(img_array, sensitivity)
    
    # Remove detected objects and fill with background
    cleaned_img = remove_objects_and_fill(img_array, combined_mask)
    
    return Image.fromarray(cleaned_img), combined_mask

def detect_objects_edge_based(img_array, sensitivity):
    """Detect objects using edge detection and contour analysis"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with adaptive thresholds based on sensitivity
    low_threshold = int(50 * (1 - sensitivity))
    high_threshold = int(150 * (1 + sensitivity))
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for objects
    mask = np.zeros(gray.shape, dtype=bool)
    
    # Filter contours by size and position
    height, width = gray.shape
    min_area = (height * width) * (0.01 * sensitivity)  # Minimum 1% of image
    max_area = (height * width) * (0.7)  # Maximum 70% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Check if contour is not touching edges (likely an object, not background)
            x, y, w, h = cv2.boundingRect(contour)
            margin = 5
            
            if (x > margin and y > margin and 
                x + w < width - margin and y + h < height - margin):
                cv2.fillPoly(mask, [contour], True)
    
    return mask

def detect_objects_color_clustering(img_array, sensitivity):
    """Detect objects using color clustering and background detection"""
    
    height, width = img_array.shape[:2]
    
    # Reshape for clustering
    pixels = img_array.reshape(-1, 3)
    
    # Use K-means clustering to find dominant colors
    from sklearn.cluster import KMeans
    n_clusters = max(3, int(8 * sensitivity))  # More clusters = more sensitive
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        
        # Reshape labels back to image shape
        label_image = labels.reshape(height, width)
        
        # Detect background clusters (those present at image edges)
        edge_labels = set()
        
        # Sample edge pixels
        edge_labels.update(label_image[0, :])  # Top edge
        edge_labels.update(label_image[-1, :])  # Bottom edge
        edge_labels.update(label_image[:, 0])  # Left edge
        edge_labels.update(label_image[:, -1])  # Right edge
        
        # Create mask for non-background clusters
        object_mask = np.zeros((height, width), dtype=bool)
        
        for cluster_id in range(n_clusters):
            if cluster_id not in edge_labels:
                # This cluster is not at edges, likely an object
                cluster_mask = (label_image == cluster_id)
                
                # Only consider if cluster is not too large (not background)
                cluster_size = np.sum(cluster_mask)
                if cluster_size < (height * width * 0.6):  # Less than 60% of image
                    object_mask |= cluster_mask
        
        return object_mask
        
    except ImportError:
        print("Warning: sklearn not available, using simpler color detection")
        return detect_objects_simple_color(img_array, sensitivity)

def detect_objects_simple_color(img_array, sensitivity):
    """Simple color-based object detection without sklearn"""
    
    height, width = img_array.shape[:2]
    
    # Get edge colors (likely background)
    edge_colors = []
    
    # Sample edge pixels
    edge_colors.extend(img_array[0, :].tolist())  # Top
    edge_colors.extend(img_array[-1, :].tolist())  # Bottom
    edge_colors.extend(img_array[:, 0].tolist())  # Left
    edge_colors.extend(img_array[:, -1].tolist())  # Right
    
    # Calculate average edge color
    edge_colors = np.array(edge_colors)
    avg_edge_color = np.mean(edge_colors, axis=0)
    
    # Create mask for pixels significantly different from edge color
    color_diff = np.sqrt(np.sum((img_array - avg_edge_color) ** 2, axis=2))
    threshold = 50 * sensitivity  # Adjust threshold based on sensitivity
    
    object_mask = color_diff > threshold
    
    return object_mask

def detect_objects_slic_segmentation(img_array, sensitivity):
    """Detect objects using SLIC superpixel segmentation"""
    
    try:
        # SLIC superpixel segmentation
        n_segments = int(100 * (1 + sensitivity))  # More segments = more sensitive
        segments = segmentation.slic(img_array, n_segments=n_segments, compactness=10, sigma=1)
        
        height, width = img_array.shape[:2]
        
        # Find edge segments (likely background)
        edge_segments = set()
        
        # Sample edge regions
        margin = max(5, int(min(height, width) * 0.05))
        edge_segments.update(np.unique(segments[:margin, :]))  # Top
        edge_segments.update(np.unique(segments[-margin:, :]))  # Bottom
        edge_segments.update(np.unique(segments[:, :margin]))  # Left
        edge_segments.update(np.unique(segments[:, -margin:]))  # Right
        
        # Create mask for non-edge segments
        object_mask = np.zeros((height, width), dtype=bool)
        
        for segment_id in np.unique(segments):
            if segment_id not in edge_segments:
                segment_mask = (segments == segment_id)
                segment_size = np.sum(segment_mask)
                
                # Only consider segments that are not too large
                if segment_size < (height * width * 0.4):
                    object_mask |= segment_mask
        
        return object_mask
        
    except ImportError:
        print("Warning: skimage not available for SLIC segmentation")
        return detect_objects_simple_color(img_array, sensitivity)

def clean_object_mask(mask):
    """Clean up the object detection mask"""
    
    # Remove small noise
    mask = morphology.remove_small_objects(mask, min_size=50)
    
    # Fill small holes
    mask = morphology.remove_small_holes(mask, area_threshold=100)
    
    # Smooth the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask.astype(bool)

def remove_objects_and_fill(img_array, object_mask, method="inpaint"):
    """Remove objects and fill with background using various methods"""
    
    if method == "inpaint":
        return inpaint_removed_objects(img_array, object_mask)
    elif method == "background_color":
        return fill_with_background_color(img_array, object_mask)
    elif method == "neighborhood":
        return fill_with_neighborhood_colors(img_array, object_mask)
    else:
        return inpaint_removed_objects(img_array, object_mask)

def inpaint_removed_objects(img_array, object_mask):
    """Use OpenCV inpainting to fill removed objects"""
    
    try:
        # Convert mask to uint8
        mask_uint8 = (object_mask * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply inpainting
        inpainted_bgr = cv2.inpaint(img_bgr, mask_uint8, 3, cv2.INPAINT_TELEA)
        
        # Convert back to RGB
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        
        return inpainted_rgb
        
    except Exception as e:
        print(f"Inpainting failed: {e}, falling back to background color fill")
        return fill_with_background_color(img_array, object_mask)

def fill_with_background_color(img_array, object_mask):
    """Fill removed objects with detected background color"""
    
    height, width = img_array.shape[:2]
    
    # Get background color from edges (non-masked areas)
    edge_pixels = []
    
    # Sample edge pixels that are not masked
    for y in [0, height-1]:
        for x in range(width):
            if not object_mask[y, x]:
                edge_pixels.append(img_array[y, x])
    
    for x in [0, width-1]:
        for y in range(height):
            if not object_mask[y, x]:
                edge_pixels.append(img_array[y, x])
    
    if edge_pixels:
        background_color = np.mean(edge_pixels, axis=0).astype(np.uint8)
    else:
        # Fallback to overall image average
        background_color = np.mean(img_array[~object_mask], axis=0).astype(np.uint8)
    
    # Fill masked areas with background color
    result = img_array.copy()
    result[object_mask] = background_color
    
    return result

def fill_with_neighborhood_colors(img_array, object_mask):
    """Fill removed objects with colors from their neighborhood"""
    
    result = img_array.copy()
    
    # Dilate mask to get neighborhood
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=2)
    neighborhood_mask = dilated_mask & (~object_mask)
    
    # For each masked pixel, find average of nearby non-masked pixels
    masked_coords = np.where(object_mask)
    
    for y, x in zip(masked_coords[0], masked_coords[1]):
        # Define neighborhood window
        window_size = 7
        y_min = max(0, y - window_size)
        y_max = min(img_array.shape[0], y + window_size + 1)
        x_min = max(0, x - window_size)
        x_max = min(img_array.shape[1], x + window_size + 1)
        
        # Get non-masked pixels in neighborhood
        neighborhood = img_array[y_min:y_max, x_min:x_max]
        neighborhood_mask_window = object_mask[y_min:y_max, x_min:x_max]
        
        valid_pixels = neighborhood[~neighborhood_mask_window]
        
        if len(valid_pixels) > 0:
            avg_color = np.mean(valid_pixels, axis=0).astype(np.uint8)
            result[y, x] = avg_color
    
    return result

# Legacy add_padding function removed - only AI padding is used now
# All padding is handled by create_ai_padding function directly 