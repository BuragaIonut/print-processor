from PIL import Image, ImageFilter
import numpy as np
import random
from collections import Counter
from print_settings import mm_to_pixels

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

def create_content_aware_bleed(img, bleed_px):
    """Create intelligent content-aware bleed with proper corner handling"""
    width, height = img.size
    new_width = width + (2 * bleed_px)
    new_height = height + (2 * bleed_px)
    
    # Get image mode and handle different color spaces
    img_mode = img.mode
    
    # Analyze image content - convert to RGB temporarily for analysis if needed
    if img_mode == 'CMYK':
        rgb_img = img.convert('RGB')
        dominant_colors = get_dominant_colors(rgb_img, 3)
    else:
        dominant_colors = get_dominant_colors(img, 3)
    
    # Create base with dominant color in the same mode as input image
    if img_mode == 'CMYK':
        # For CMYK, convert RGB dominant color to CMYK
        rgb_color = Image.new('RGB', (1, 1), dominant_colors[0])
        cmyk_color = rgb_color.convert('CMYK')
        base_color = cmyk_color.getpixel((0, 0))
        bleed_img = Image.new('CMYK', (new_width, new_height), base_color)
    else:
        bleed_img = Image.new(img_mode, (new_width, new_height), dominant_colors[0])
    
    if bleed_px > 0:
        # Get corner pixels for smooth corner transitions
        corner_tl = img.getpixel((0, 0))
        corner_tr = img.getpixel((width-1, 0))
        corner_bl = img.getpixel((0, height-1))
        corner_br = img.getpixel((width-1, height-1))
        
        # Create corner gradients first to avoid overlap issues
        for y in range(bleed_px):
            for x in range(bleed_px):
                # Calculate distances from corner
                dist_from_corner = max(x, y)
                corner_fade = dist_from_corner / bleed_px
                
                # Create corner colors with proper mode handling
                if img_mode == 'CMYK':
                    # For CMYK, manually interpolate
                    def blend_cmyk_colors(c1, c2, ratio):
                        return tuple(int(c1[i] * (1 - ratio) + c2[i] * ratio) for i in range(4))
                    
                    corner_color = blend_cmyk_colors(base_color, corner_tl, corner_fade)
                    bleed_img.putpixel((x, y), corner_color)
                    
                    corner_color = blend_cmyk_colors(base_color, corner_tr, corner_fade)
                    bleed_img.putpixel((new_width - 1 - x, y), corner_color)
                    
                    corner_color = blend_cmyk_colors(base_color, corner_bl, corner_fade)
                    bleed_img.putpixel((x, new_height - 1 - y), corner_color)
                    
                    corner_color = blend_cmyk_colors(base_color, corner_br, corner_fade)
                    bleed_img.putpixel((new_width - 1 - x, new_height - 1 - y), corner_color)
                else:
                    # For RGB and other modes, use Image.blend
                    corner_color = Image.blend(
                        Image.new(img_mode, (1, 1), dominant_colors[0]), 
                        Image.new(img_mode, (1, 1), corner_tl), 
                        corner_fade
                    ).getpixel((0, 0))
                    bleed_img.putpixel((x, y), corner_color)
                    
                    corner_color = Image.blend(
                        Image.new(img_mode, (1, 1), dominant_colors[0]), 
                        Image.new(img_mode, (1, 1), corner_tr), 
                        corner_fade
                    ).getpixel((0, 0))
                    bleed_img.putpixel((new_width - 1 - x, y), corner_color)
                    
                    corner_color = Image.blend(
                        Image.new(img_mode, (1, 1), dominant_colors[0]), 
                        Image.new(img_mode, (1, 1), corner_bl), 
                        corner_fade
                    ).getpixel((0, 0))
                    bleed_img.putpixel((x, new_height - 1 - y), corner_color)
                    
                    corner_color = Image.blend(
                        Image.new(img_mode, (1, 1), dominant_colors[0]), 
                        Image.new(img_mode, (1, 1), corner_br), 
                        corner_fade
                    ).getpixel((0, 0))
                    bleed_img.putpixel((new_width - 1 - x, new_height - 1 - y), corner_color)
        
        # Now extend edges (but skip corners to avoid overlap)
        for i in range(bleed_px):
            fade_ratio = i / bleed_px
            
            # Handle edge blending with proper color mode support
            if img_mode == 'CMYK':
                # For CMYK, use manual blending to avoid mode mismatch
                def blend_image_manual(base_color, line_img, ratio):
                    blended = line_img.copy()
                    width_img, height_img = line_img.size
                    for y in range(height_img):
                        for x in range(width_img):
                            pixel = line_img.getpixel((x, y))
                            new_pixel = tuple(int(base_color[i] * (1 - ratio) + pixel[i] * ratio) for i in range(len(pixel)))
                            blended.putpixel((x, y), new_pixel)
                    return blended
                
                # Top edge (skip corners)
                top_line = img.crop((0, 0, width, 1))
                top_faded = blend_image_manual(base_color, top_line, fade_ratio)
                bleed_img.paste(top_faded, (bleed_px, i))
                
                # Bottom edge (skip corners)
                bottom_line = img.crop((0, height-1, width, height))
                bottom_faded = blend_image_manual(base_color, bottom_line, fade_ratio)
                bleed_img.paste(bottom_faded, (bleed_px, new_height - 1 - i))
                
                # Left edge (skip corners)
                left_line = img.crop((0, 0, 1, height))
                left_faded = blend_image_manual(base_color, left_line, fade_ratio)
                bleed_img.paste(left_faded, (i, bleed_px))
                
                # Right edge (skip corners)
                right_line = img.crop((width-1, 0, width, height))
                right_faded = blend_image_manual(base_color, right_line, fade_ratio)
                bleed_img.paste(right_faded, (new_width - 1 - i, bleed_px))
            else:
                # For RGB and other modes, use Image.blend
                # Top edge (skip corners)
                top_line = img.crop((0, 0, width, 1))
                top_faded = Image.blend(Image.new(img_mode, (width, 1), dominant_colors[0]), top_line, fade_ratio)
                bleed_img.paste(top_faded, (bleed_px, i))
                
                # Bottom edge (skip corners)
                bottom_line = img.crop((0, height-1, width, height))
                bottom_faded = Image.blend(Image.new(img_mode, (width, 1), dominant_colors[0]), bottom_line, fade_ratio)
                bleed_img.paste(bottom_faded, (bleed_px, new_height - 1 - i))
                
                # Left edge (skip corners)
                left_line = img.crop((0, 0, 1, height))
                left_faded = Image.blend(Image.new(img_mode, (1, height), dominant_colors[0]), left_line, fade_ratio)
                bleed_img.paste(left_faded, (i, bleed_px))
                
                # Right edge (skip corners)
                right_line = img.crop((width-1, 0, width, height))
                right_faded = Image.blend(Image.new(img_mode, (1, height), dominant_colors[0]), right_line, fade_ratio)
                bleed_img.paste(right_faded, (new_width - 1 - i, bleed_px))
    
    # Paste original image
    bleed_img.paste(img, (bleed_px, bleed_px))
    return bleed_img

def create_radial_pattern_bleed(img, bleed_px):
    """Create radial pattern bleed from corners"""
    width, height = img.size
    new_width = width + (2 * bleed_px)
    new_height = height + (2 * bleed_px)
    
    # Get corner colors
    corner_colors = [
        img.getpixel((0, 0)),           # Top-left
        img.getpixel((width-1, 0)),     # Top-right
        img.getpixel((0, height-1)),    # Bottom-left
        img.getpixel((width-1, height-1)) # Bottom-right
    ]
    
    # Create image in same mode as input
    bleed_img = Image.new(img.mode, (new_width, new_height))
    
    # Create radial gradients from each corner
    for y in range(new_height):
        for x in range(new_width):
            # Calculate distances to corners
            d_tl = ((x - 0)**2 + (y - 0)**2)**0.5
            d_tr = ((x - new_width)**2 + (y - 0)**2)**0.5
            d_bl = ((x - 0)**2 + (y - new_height)**2)**0.5
            d_br = ((x - new_width)**2 + (y - new_height)**2)**0.5
            
            # Weight by inverse distance
            weights = [1/(d_tl+1), 1/(d_tr+1), 1/(d_bl+1), 1/(d_br+1)]
            total_weight = sum(weights)
            
            # Blend colors based on weights - handle different color modes
            num_channels = len(corner_colors[0])
            blended_color = []
            for channel in range(num_channels):
                channel_value = sum(corner_colors[i][channel] * weights[i] for i in range(4)) / total_weight
                blended_color.append(int(channel_value))
            
            bleed_img.putpixel((x, y), tuple(blended_color))
    
    # Paste original image
    bleed_img.paste(img, (bleed_px, bleed_px))
    return bleed_img

def create_noise_texture_bleed(img, bleed_px):
    """Create textured noise bleed"""
    width, height = img.size
    new_width = width + (2 * bleed_px)
    new_height = height + (2 * bleed_px)
    
    # Get dominant color - convert to RGB for analysis if needed
    if img.mode == 'CMYK':
        rgb_img = img.convert('RGB')
        dominant_colors = get_dominant_colors(rgb_img, 1)
        # Convert base color to CMYK
        rgb_color = Image.new('RGB', (1, 1), dominant_colors[0])
        cmyk_color = rgb_color.convert('CMYK')
        base_color = cmyk_color.getpixel((0, 0))
    else:
        dominant_colors = get_dominant_colors(img, 1)
        base_color = dominant_colors[0]
    
    # Create noise texture
    bleed_img = Image.new(img.mode, (new_width, new_height), base_color)
    
    # Add noise
    for y in range(new_height):
        for x in range(new_width):
            # Skip the center area where original image will be placed
            if bleed_px <= x < width + bleed_px and bleed_px <= y < height + bleed_px:
                continue
                
            # Add random noise to base color - handle different color modes
            num_channels = len(base_color)
            new_color = []
            for channel in range(num_channels):
                noise = random.randint(-30, 30)
                new_channel = max(0, min(255, base_color[channel] + noise))
                new_color.append(new_channel)
            
            bleed_img.putpixel((x, y), tuple(new_color))
    
    # Apply subtle blur to soften the noise
    bleed_img = bleed_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Paste original image
    bleed_img.paste(img, (bleed_px, bleed_px))
    return bleed_img

def create_dominant_color_bleed(img, bleed_px):
    """Create bleed using dominant color palette"""
    width, height = img.size
    new_width = width + (2 * bleed_px)
    new_height = height + (2 * bleed_px)
    
    # Get dominant colors - convert to RGB for analysis if needed
    if img.mode == 'CMYK':
        rgb_img = img.convert('RGB')
        rgb_dominant_colors = get_dominant_colors(rgb_img, 4)
        # Convert dominant colors to CMYK
        dominant_colors = []
        for rgb_color in rgb_dominant_colors:
            rgb_temp = Image.new('RGB', (1, 1), rgb_color)
            cmyk_temp = rgb_temp.convert('CMYK')
            dominant_colors.append(cmyk_temp.getpixel((0, 0)))
    else:
        dominant_colors = get_dominant_colors(img, 4)
    
    # Create sections with different dominant colors
    bleed_img = Image.new(img.mode, (new_width, new_height))
    
    # Fill different areas with different dominant colors
    section_width = new_width // 2
    section_height = new_height // 2
    
    # Top-left
    for y in range(section_height):
        for x in range(section_width):
            bleed_img.putpixel((x, y), dominant_colors[0])
    
    # Top-right
    for y in range(section_height):
        for x in range(section_width, new_width):
            color_idx = 1 if len(dominant_colors) > 1 else 0
            bleed_img.putpixel((x, y), dominant_colors[color_idx])
    
    # Bottom-left
    for y in range(section_height, new_height):
        for x in range(section_width):
            color_idx = 2 if len(dominant_colors) > 2 else 0
            bleed_img.putpixel((x, y), dominant_colors[color_idx])
    
    # Bottom-right
    for y in range(section_height, new_height):
        for x in range(section_width, new_width):
            color_idx = 3 if len(dominant_colors) > 3 else 0
            bleed_img.putpixel((x, y), dominant_colors[color_idx])
    
    # Apply blur for smooth transitions
    bleed_img = bleed_img.filter(ImageFilter.GaussianBlur(radius=bleed_px/4))
    
    # Paste original image
    bleed_img.paste(img, (bleed_px, bleed_px))
    return bleed_img

def add_bleed(img, bleed_mm, dpi, bleed_type='content_aware', bleed_color=(255, 255, 255)):
    """Add bleed area around the image with different bleed types"""
    if bleed_mm <= 0:
        return img
    
    # Calculate bleed in pixels
    bleed_px = mm_to_pixels(bleed_mm, dpi)
    
    # Get current image size
    width, height = img.size
    
    # Create new image with bleed
    new_width = width + (2 * bleed_px)
    new_height = height + (2 * bleed_px)
    
    if bleed_type == 'solid_color':
        # Solid color bleed using color picker - handle CMYK
        if img.mode == 'CMYK' and len(bleed_color) == 3:
            # Convert RGB bleed color to CMYK
            rgb_temp = Image.new('RGB', (1, 1), bleed_color)
            cmyk_temp = rgb_temp.convert('CMYK')
            bleed_color = cmyk_temp.getpixel((0, 0))
        bleed_img = Image.new(img.mode, (new_width, new_height), bleed_color)
        bleed_img.paste(img, (bleed_px, bleed_px))
        
    elif bleed_type == 'edge_extend':
        # Extend edge pixels
        bleed_img = Image.new(img.mode, (new_width, new_height))
        
        # Fill with extended edge colors
        # Top edge
        top_line = img.crop((0, 0, width, 1))
        top_extended = top_line.resize((new_width, bleed_px))
        bleed_img.paste(top_extended, (0, 0))
        
        # Bottom edge
        bottom_line = img.crop((0, height-1, width, height))
        bottom_extended = bottom_line.resize((new_width, bleed_px))
        bleed_img.paste(bottom_extended, (0, new_height - bleed_px))
        
        # Left edge
        left_line = img.crop((0, 0, 1, height))
        left_extended = left_line.resize((bleed_px, height))
        bleed_img.paste(left_extended, (0, bleed_px))
        
        # Right edge
        right_line = img.crop((width-1, 0, width, height))
        right_extended = right_line.resize((bleed_px, height))
        bleed_img.paste(right_extended, (new_width - bleed_px, bleed_px))
        
        # Paste original image in center
        bleed_img.paste(img, (bleed_px, bleed_px))
        
    elif bleed_type == 'mirror':
        # Mirror the edges
        bleed_img = Image.new('RGB', (new_width, new_height))
        
        # Create mirrored sections
        # Top mirror
        top_section = img.crop((0, 0, width, bleed_px))
        top_mirrored = top_section.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        bleed_img.paste(top_mirrored, (bleed_px, 0))
        
        # Bottom mirror
        bottom_section = img.crop((0, height - bleed_px, width, height))
        bottom_mirrored = bottom_section.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        bleed_img.paste(bottom_mirrored, (bleed_px, new_height - bleed_px))
        
        # Left mirror
        left_section = img.crop((0, 0, bleed_px, height))
        left_mirrored = left_section.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        bleed_img.paste(left_mirrored, (0, bleed_px))
        
        # Right mirror
        right_section = img.crop((width - bleed_px, 0, width, height))
        right_mirrored = right_section.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        bleed_img.paste(right_mirrored, (new_width - bleed_px, bleed_px))
        
        # Fill corners with solid color (average of corner pixels)
        corner_color = img.getpixel((0, 0))
        
        # Top-left corner
        corner_tl = Image.new('RGB', (bleed_px, bleed_px), corner_color)
        bleed_img.paste(corner_tl, (0, 0))
        
        # Top-right corner
        corner_tr = Image.new('RGB', (bleed_px, bleed_px), img.getpixel((width-1, 0)))
        bleed_img.paste(corner_tr, (new_width - bleed_px, 0))
        
        # Bottom-left corner
        corner_bl = Image.new('RGB', (bleed_px, bleed_px), img.getpixel((0, height-1)))
        bleed_img.paste(corner_bl, (0, new_height - bleed_px))
        
        # Bottom-right corner
        corner_br = Image.new('RGB', (bleed_px, bleed_px), img.getpixel((width-1, height-1)))
        bleed_img.paste(corner_br, (new_width - bleed_px, new_height - bleed_px))
        
        # Paste original image in center
        bleed_img.paste(img, (bleed_px, bleed_px))
        
    elif bleed_type == 'content_aware':
        # Intelligent content-aware bleed
        bleed_img = create_content_aware_bleed(img, bleed_px)
        
    else:  # Default to content aware
        bleed_img = create_content_aware_bleed(img, bleed_px)
    
    return bleed_img 