import streamlit as st
from PIL import Image
import fitz
import io
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

print("API KEY SET")
print(os.getenv("OPENAI_API_KEY"))
# Import from our modules
from print_settings import PRINT_FORMATS, mm_to_inches, resize_image_for_print
from bleed import add_bleed
from cut_lines import add_cut_lines


# Set page config
st.set_page_config(
    page_title="Print Processor",
    page_icon="🖨️",
    layout="wide"
)

# Remove PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def rotate_image(img, angle):
    """Rotate image by specified angle with intelligent handling"""
    if angle == 0:
        return img
    
    # For 90-degree increments, use simple rotation to preserve quality
    if angle == 90:
        return img.transpose(Image.ROTATE_90)
    elif angle == 180:
        return img.transpose(Image.ROTATE_180)
    elif angle == 270:
        return img.transpose(Image.ROTATE_270)
    else:
        # For custom angles, use rotate with expand=True to fit rotated content
        # Use high-quality resampling
        return img.rotate(-angle, expand=True, resample=Image.LANCZOS, fillcolor='white')

def detect_shape_and_create_smart_padding(img, target_width_mm, target_height_mm, dpi, padding_mm, detection_sensitivity, background_handling, bg_color_hex):
    """
    Detect the main shape/logo in the image and create smart padding around it
    """
    try:
        from padding import detect_and_remove_objects
        import numpy as np
        from print_settings import mm_to_pixels
        
        # Convert target dimensions to pixels
        target_width_px = mm_to_pixels(target_width_mm, dpi)
        target_height_px = mm_to_pixels(target_height_mm, dpi)
        padding_px = mm_to_pixels(padding_mm, dpi)
        
        # Step 1: Detect the main object/shape
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Use inverse logic - detect background to isolate the shape
        try:
            # Create a mask for the background (what we want to remove/replace)
            _, background_mask = detect_and_remove_objects(img, method="auto", sensitivity=detection_sensitivity)
            # Invert to get the shape mask
            shape_mask = ~background_mask
            
            # Clean up the shape mask
            import cv2
            from skimage import morphology
            
            # Remove small noise and fill holes
            shape_mask = morphology.remove_small_objects(shape_mask, min_size=100)
            shape_mask = morphology.remove_small_holes(shape_mask, area_threshold=500)
            
            # Smooth the mask
            kernel = np.ones((3, 3), np.uint8)
            shape_mask = cv2.morphologyEx(shape_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            shape_mask = shape_mask.astype(bool)
            
        except Exception as e:
            st.warning(f"Shape detection failed: {e}. Using simple edge-based detection.")
            # Fallback: simple edge-based shape detection
            shape_mask = detect_shape_simple(img_array, detection_sensitivity)
        
        # Step 2: Find the bounding box of the detected shape
        shape_coords = np.where(shape_mask)
        if len(shape_coords[0]) == 0:
            st.warning("No shape detected. Using full image.")
            return process_traditional_padding(img, target_width_mm, target_height_mm, dpi, padding_mm)
        
        min_y, max_y = np.min(shape_coords[0]), np.max(shape_coords[0])
        min_x, max_x = np.min(shape_coords[1]), np.max(shape_coords[1])
        
        shape_width = max_x - min_x + 1
        shape_height = max_y - min_y + 1
        
        # Step 3: Calculate optimal positioning within target dimensions
        # Account for padding around the shape
        available_width = target_width_px - (2 * padding_px)
        available_height = target_height_px - (2 * padding_px)
        
        # Calculate scale factor to fit shape within available space
        scale_x = available_width / shape_width
        scale_y = available_height / shape_height
        scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        # Step 4: Create the final image
        final_img = Image.new('RGBA', (target_width_px, target_height_px), (255, 255, 255, 0))
        
        # Step 5: Handle background based on user preference
        if background_handling == "transparent":
            # Keep transparent background
            pass
        elif background_handling == "solid_color":
            # Fill with solid color
            bg_rgb = tuple(int(bg_color_hex[i:i+2], 16) for i in (1, 3, 5))
            final_img = Image.new('RGB', (target_width_px, target_height_px), bg_rgb)
        elif background_handling == "extend_background":
            # Extend the original background color
            background_color = get_dominant_background_color(img, shape_mask)
            final_img = Image.new('RGB', (target_width_px, target_height_px), background_color)
        else:  # intelligent_fill
            # Use intelligent background fill
            final_img = create_intelligent_background_fill(img, shape_mask, (target_width_px, target_height_px))
        
        # Step 6: Resize and position the shape
        if scale_factor < 1.0:
            # Resize the original image
            new_img_width = int(width * scale_factor)
            new_img_height = int(height * scale_factor)
            scaled_img = img.resize((new_img_width, new_img_height), Image.Resampling.LANCZOS)
            
            # Scale the mask too
            scaled_mask = Image.fromarray(shape_mask.astype(np.uint8) * 255).resize(
                (new_img_width, new_img_height), Image.Resampling.LANCZOS
            )
            scaled_mask_array = np.array(scaled_mask) > 128
        else:
            scaled_img = img
            scaled_mask_array = shape_mask
            new_img_width, new_img_height = width, height
        
        # Center the scaled image
        paste_x = (target_width_px - new_img_width) // 2
        paste_y = (target_height_px - new_img_height) // 2
        
        # Paste only the shape part of the image
        if background_handling == "transparent":
            # Create RGBA version with transparency outside shape
            scaled_rgba = scaled_img.convert('RGBA')
            scaled_array = np.array(scaled_rgba)
            scaled_array[~scaled_mask_array] = [0, 0, 0, 0]  # Make background transparent
            masked_img = Image.fromarray(scaled_array)
            final_img.paste(masked_img, (paste_x, paste_y), masked_img)
        else:
            # For non-transparent backgrounds, paste the whole scaled image
            final_img.paste(scaled_img, (paste_x, paste_y))
        
        return final_img, {
            'shape_mask': Image.fromarray((shape_mask * 255).astype(np.uint8)),
            'shape_bounds': f"{shape_width}×{shape_height} at ({min_x},{min_y})",
            'scale_factor': scale_factor,
            'final_position': f"Centered at ({paste_x},{paste_y})"
        }
        
    except Exception as e:
        st.error(f"Shape-aware processing failed: {e}")
        return process_traditional_padding(img, target_width_mm, target_height_mm, dpi, padding_mm), {}

def detect_shape_simple(img_array, sensitivity):
    """Simple shape detection using edge detection"""
    import cv2
    import numpy as np
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to separate object from background
    threshold_value = int(127 * (1 - sensitivity + 0.5))
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find the largest contour (main object)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=bool)
        cv2.fillPoly(mask, [largest_contour], True)
        return mask
    else:
        return np.ones(gray.shape, dtype=bool)  # Fallback to whole image

def get_dominant_background_color(img, shape_mask):
    """Get the dominant background color from areas outside the shape"""
    import numpy as np
    
    img_array = np.array(img)
    background_pixels = img_array[~shape_mask]
    
    if len(background_pixels) > 0:
        return tuple(np.mean(background_pixels, axis=0).astype(int))
    else:
        return (255, 255, 255)  # Default to white

def create_intelligent_background_fill(img, shape_mask, target_size):
    """Create an intelligent background fill based on the original image"""
    from padding import get_dominant_colors
    
    # Get background colors
    background_color = get_dominant_background_color(img, shape_mask)
    
    # Create gradient background
    target_width, target_height = target_size
    background = Image.new('RGB', target_size, background_color)
    
    # Add subtle texture/gradient
    try:
        dominant_colors = get_dominant_colors(img, 3)
        # Create subtle gradient using dominant colors
        for y in range(target_height):
            for x in range(target_width):
                # Create subtle color variation
                distance_factor = ((x - target_width//2)**2 + (y - target_height//2)**2) ** 0.5
                distance_factor = distance_factor / (max(target_width, target_height) / 2)
                
                if len(dominant_colors) > 1:
                    color_index = int(distance_factor * (len(dominant_colors) - 1))
                    color_index = min(color_index, len(dominant_colors) - 1)
                    pixel_color = dominant_colors[color_index]
                else:
                    pixel_color = background_color
                
                background.putpixel((x, y), pixel_color)
    except:
        # Fallback to solid color
        pass
    
    return background

def process_traditional_padding(img, target_width_mm, target_height_mm, dpi, padding_mm):
    """Fallback to traditional padding processing"""
    from print_settings import resize_image_for_print, mm_to_pixels
    
    # Calculate image dimensions without padding
    content_width_mm = target_width_mm - (2 * padding_mm)
    content_height_mm = target_height_mm - (2 * padding_mm)
    
    # Resize image to content area
    resized_img = resize_image_for_print(img, content_width_mm, content_height_mm, dpi)
    
    # Add simple padding
    target_width_px = mm_to_pixels(target_width_mm, dpi)
    target_height_px = mm_to_pixels(target_height_mm, dpi)
    padding_px = mm_to_pixels(padding_mm, dpi)
    
    final_img = Image.new('RGB', (target_width_px, target_height_px), (255, 255, 255))
    final_img.paste(resized_img, (padding_px, padding_px))
    
    return final_img

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return PIL Image"""
    try:
        if uploaded_file.type == "application/pdf":
            # Handle PDF
            pdf_bytes = uploaded_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            if len(pdf_document) == 0:
                st.error("PDF has no pages")
                return None
            
            # Get first page as image
            page = pdf_document[0]
            mat = fitz.Matrix(2.0, 2.0)  # High resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pdf_document.close()
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            return img
        else:
            # Handle image files
            img = Image.open(uploaded_file)
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            return img
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def process_for_print(img, format_size, dpi, padding_mm, padding_style, ai_style, custom_prompt, remove_objects, object_removal_sensitivity, bleed_mm, bleed_type, bleed_color, add_cut_lines_flag, cut_line_color):
    """Complete print processing pipeline with support for all padding methods"""
    width_mm, height_mm = format_size
    intermediate_images = {}
    
    # Step 1: Resize to exact print dimensions first (for non-AI padding methods)
    if padding_style != "ai_advanced":
        st.write("🔄 Resizing image to target dimensions...")
        resized_img = resize_image_for_print(img, width_mm, height_mm, dpi)
        
        # Step 2: Add padding using traditional methods
        if padding_mm > 0:
            st.write(f"📦 Adding {padding_mm}mm padding ({padding_style})...")
            try:
                from padding import add_padding
                padded_img, padding_intermediates = add_padding(resized_img, padding_mm, dpi, padding_style, ai_style, custom_prompt, remove_objects, object_removal_sensitivity)
                intermediate_images.update(padding_intermediates)
            except Exception as e:
                st.error(f"Error adding padding: {str(e)}")
                st.warning("Falling back to content-aware padding...")
                from padding import add_padding
                padded_img, padding_intermediates = add_padding(resized_img, padding_mm, dpi, 'content_aware', ai_style, custom_prompt, remove_objects, object_removal_sensitivity)
                intermediate_images.update(padding_intermediates)
        else:
            padded_img = resized_img
    else:
        # Step 1: AI padding BEFORE resizing for print (special workflow for AI)
        if padding_mm > 0:
            if remove_objects:
                st.write(f"🎯 Removing distracting objects (sensitivity: {object_removal_sensitivity})...")
            st.write(f"🤖 Adding {padding_mm}mm AI-powered padding ({ai_style})...")
            
            # Convert padding from mm to pixels at original image resolution
            # Use a base DPI for the AI processing (we'll scale to final DPI later)
            base_dpi = 300  # Standard base resolution for AI processing
            from print_settings import mm_to_pixels
            padding_px = mm_to_pixels(padding_mm, base_dpi)
            
            try:
                from padding import create_ai_padding
                # AI padding happens at original image dimensions
                padded_img, intermediate_images = create_ai_padding(
                    img, 
                    padding_px, 
                    ai_style, 
                    custom_prompt,
                    remove_objects,
                    object_removal_sensitivity
                )
                st.success("✅ AI padding completed successfully!")
            except Exception as e:
                st.error(f"Error adding AI padding: {str(e)}")
                st.error("❌ AI padding failed - please check your OpenAI setup")
                return img, {}  # Return original image if AI fails
        else:
            padded_img = img
        
        # Step 2: Now resize the AI-padded image to exact print dimensions
        st.write("🔄 Resizing to target print dimensions...")
        padded_img = resize_image_for_print(padded_img, width_mm, height_mm, dpi)
    
    # Step 3: Add bleed (outer area for cutting)
    if bleed_mm > 0:
        st.write(f"📏 Adding {bleed_mm}mm bleed ({bleed_type})...")
        bleed_img = add_bleed(padded_img, bleed_mm, dpi, bleed_type, bleed_color)
    else:
        bleed_img = padded_img
    
    # Step 4: Add cut lines
    if add_cut_lines_flag and bleed_mm > 0:
        st.write(f"✂️ Adding {cut_line_color} cut lines...")
        final_img = add_cut_lines(bleed_img, bleed_mm, dpi, cut_line_color)
    else:
        final_img = bleed_img
    
    return final_img, intermediate_images

def main():
    # Centered title
    st.markdown("<h1 style='text-align: center;'>🖨️ Print Processor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><strong>Professional print preparation tool with DPI adjustment, bleed, and cut lines</strong></p>", unsafe_allow_html=True)
    
    # Sidebar for settings
    st.sidebar.header("📋 Print Settings")
    
    # Print format selection
    format_name = st.sidebar.selectbox(
        "Printing Format",
        list(PRINT_FORMATS.keys()),
        index=0
    )
    
    if format_name == "Custom":
        col1, col2 = st.sidebar.columns(2)
        custom_width = col1.number_input("Width (mm)", min_value=10, max_value=2000, value=210)
        custom_height = col2.number_input("Height (mm)", min_value=10, max_value=2000, value=297)
        format_size = (custom_width, custom_height)
    else:
        format_size = PRINT_FORMATS[format_name]
        st.sidebar.write(f"**{format_name}**: {format_size[0]} × {format_size[1]} mm")
    
    # DPI selection
    dpi_options = [72, 150, 300, 600]
    dpi = st.sidebar.selectbox(
        "DPI (Print Quality)",
        dpi_options,
        index=2,  # Default to 300 DPI
        help="Higher DPI = better quality but larger file size"
    )
    
    # DPI info
    dpi_info = {
        72: "Screen display",
        150: "Draft printing", 
        300: "Good quality printing",
        600: "High quality printing"
    }
    st.sidebar.write(f"*{dpi_info[dpi]}*")
    
    # Padding settings
    st.sidebar.subheader("📦 Padding Settings")
    padding_mm = st.sidebar.number_input(
        "Padding (mm)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        help="Inner border that adds depth and visual space around your image"
    )
    
    # Padding style selector
    if padding_mm > 0:
        # Check if OpenAI is available
        from padding import OPENAI_AVAILABLE, validate_openai_setup
        
        padding_styles = {
            "content_aware": "Content Aware",
            "soft_shadow": "Soft Shadow",
            "gradient_fade": "Gradient Fade",
            "color_blend": "Color Blend",
            "vintage_vignette": "Vintage Vignette",
            "clean_border": "Clean Border"
        }
        
        # Check OpenAI setup
        openai_ready, openai_status = validate_openai_setup()
        
        # Add AI option if available
        if OPENAI_AVAILABLE:
            padding_styles["ai_advanced"] = "🤖 AI Advanced (OpenAI)"
        
        padding_style = st.sidebar.selectbox(
            "Padding Style",
            options=list(padding_styles.keys()),
            format_func=lambda x: padding_styles[x],
            index=0,
            help="Choose the padding generation method"
        )
        
        # AI-specific settings
        if padding_style == "ai_advanced":
            if openai_ready:
                st.sidebar.success(f"✅ {openai_status}")
                
                # Object removal preprocessing
                st.sidebar.subheader("🎯 Object Removal (Experimental)")
                remove_objects = st.sidebar.checkbox(
                    "Remove distracting objects",
                    value=False,
                    help="Automatically detect and remove objects/text/logos before AI processing for cleaner padding"
                )
                
                if remove_objects:
                    object_removal_sensitivity = st.sidebar.slider(
                        "Detection Sensitivity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.3,
                        step=0.1,
                        help="Higher values detect more objects (may remove background elements)"
                    )
                    
                    st.sidebar.info("💡 Works best with flags, logos, business cards, and simple objects on uniform backgrounds")
                else:
                    object_removal_sensitivity = 0.3
                
                # AI style selector
                st.sidebar.subheader("🎨 AI Style")
                ai_styles = {
                    "natural": "Natural Extension",
                    "artistic": "Artistic & Creative",
                    "minimalist": "Minimalist & Clean",
                    "textured": "Textured & Detailed",
                    "blurred": "Blurred & Dreamy"
                }
                
                ai_style = st.sidebar.selectbox(
                    "AI Padding Style",
                    options=list(ai_styles.keys()),
                    format_func=lambda x: ai_styles[x],
                    index=0,
                    help="Style of AI-generated padding"
                )
                
                # Custom prompt option
                use_custom_prompt = st.sidebar.checkbox(
                    "Use Custom Prompt",
                    value=False,
                    help="Provide your own prompt for AI padding generation"
                )
                
                if use_custom_prompt:
                    custom_prompt = st.sidebar.text_area(
                        "Custom AI Prompt",
                        value="",
                        height=100,
                        help="Describe how you want the AI to generate the padding (max 1000 chars for DALL-E 2)",
                        max_chars=1000
                    )
                else:
                    custom_prompt = None
                    
                st.sidebar.info("🤖 AI-generated contextual padding using DALL-E 2")
            else:
                st.sidebar.error(f"❌ {openai_status}")
                st.sidebar.warning("AI padding requires OpenAI API key")
                ai_style = "natural"
                custom_prompt = None
                remove_objects = False
                object_removal_sensitivity = 0.3
        else:
            ai_style = "natural"
            custom_prompt = None
            remove_objects = False
            object_removal_sensitivity = 0.3
            
        # Padding descriptions
        padding_descriptions = {
            "content_aware": "Intelligent padding based on image analysis",
            "soft_shadow": "Soft shadow effect around the image",
            "gradient_fade": "Gradient fade from edge colors",
            "color_blend": "Blended colors from image palette",
            "vintage_vignette": "Vintage vignette effect",
            "clean_border": "Clean, minimal border",
            "ai_advanced": "AI-generated contextual padding (requires OpenAI API)"
        }
        st.sidebar.write(f"*{padding_descriptions[padding_style]}*")
    else:
        padding_style = "content_aware"
        ai_style = "natural"
        custom_prompt = None
        remove_objects = False
        object_removal_sensitivity = 0.3
    
    # Bleed settings
    st.sidebar.subheader("📐 Bleed Settings")
    bleed_mm = st.sidebar.number_input(
        "Bleed (mm)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Extra area around your design for safe cutting"
    )
    
    # Bleed type selector
    bleed_types = {
        "content_aware": "Content Aware",
        "mirror": "Mirror Edges",
        "edge_extend": "Edge Extend",
        "solid_color": "Solid Color"
    }
    
    bleed_type = st.sidebar.selectbox(
        "Bleed Type",
        options=list(bleed_types.keys()),
        format_func=lambda x: bleed_types[x],
        index=0,  # Default to content aware
        help="How to fill the bleed area"
    )
    
    # Add color picker for solid color bleed
    if bleed_type == "solid_color":
        bleed_color = st.sidebar.color_picker(
            "Bleed Color",
            value="#FFFFFF",  # Default to white
            help="Choose the solid color for the bleed area"
        )
        # Convert hex to RGB tuple
        bleed_color_rgb = tuple(int(bleed_color[i:i+2], 16) for i in (1, 3, 5))
    else:
        bleed_color_rgb = (255, 255, 255)  # Default white
    
    # Bleed type descriptions
    bleed_descriptions = {
        "content_aware": "Intelligent bleed based on image analysis",
        "mirror": "Mirror edge content to fill bleed area",
        "edge_extend": "Extend edge pixels to fill bleed area",
        "solid_color": "Fill bleed area with chosen solid color"
    }
    st.sidebar.write(f"*{bleed_descriptions[bleed_type]}*")
    
    # Cut lines settings
    st.sidebar.subheader("✂️ Cut Line Settings")
    add_cut_lines_flag = st.sidebar.checkbox(
        "Add Cut Lines",
        value=True,
        help="Add guide lines showing where to cut"
    )
    
    # Cut line color selector
    cut_line_colors = {
        "black": "Black",
        "red": "Red",
        "blue": "Blue", 
        "green": "Green",
        "magenta": "Magenta",
        "cyan": "Cyan",
        "yellow": "Yellow",
        "white": "White"
    }
    
    cut_line_color = st.sidebar.selectbox(
        "Cut Line Color",
        options=list(cut_line_colors.keys()),
        format_func=lambda x: cut_line_colors[x],
        index=0,  # Default to black
        help="Color for the cut guide lines"
    )
    
    # Auto-rotation settings
    st.sidebar.subheader("🔄 Auto-Rotation")
    auto_rotate = st.sidebar.checkbox(
        "Auto-rotate to match format",
        value=True,
        help="Automatically rotate image to match print format orientation (landscape ↔ portrait)"
    )
    
    if auto_rotate:
        st.sidebar.info("📐 Image will be automatically rotated if needed to prevent stretching")
    
    # Shape-aware processing
    st.sidebar.subheader("🎯 Smart Shape Detection")
    shape_aware_mode = st.sidebar.checkbox(
        "Shape-aware processing",
        value=False,
        help="Automatically detect logos/objects and apply padding around the actual shape instead of the full rectangle"
    )
    
    if shape_aware_mode:
        detection_sensitivity = st.sidebar.slider(
            "Shape Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Higher values detect more detailed shapes"
        )
        
        background_handling = st.sidebar.selectbox(
            "Background Handling",
            ["transparent", "extend_background", "solid_color", "intelligent_fill"],
            index=1,
            help="How to handle the area outside the detected shape"
        )
        
        if background_handling == "solid_color":
            shape_bg_color = st.sidebar.color_picker(
                "Background Color",
                value="#FFFFFF",
                help="Color for areas outside the detected shape"
            )
        else:
            shape_bg_color = "#FFFFFF"
        
        st.sidebar.info("🎯 Perfect for logos, emblems, and objects on solid backgrounds")
    else:
        detection_sensitivity = 0.4
        background_handling = "extend_background"
        shape_bg_color = "#FFFFFF"
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose an image or PDF file",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Supported formats: JPG, PNG, PDF"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Show file info
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**File type:** {uploaded_file.type}")
    
    with col2:
        st.header("📊 Print Specifications")
        
        # Calculate dimensions working backwards from target format size
        from print_settings import mm_to_pixels
        
        # Final dimensions should be exactly the print format size
        final_width_mm = format_size[0]
        final_height_mm = format_size[1]
        
        # Content area after removing bleed
        content_width_mm = final_width_mm - (2 * bleed_mm)
        content_height_mm = final_height_mm - (2 * bleed_mm)
        
        # Image area after removing padding from content area
        image_width_mm = content_width_mm - (2 * padding_mm)
        image_height_mm = content_height_mm - (2 * padding_mm)
        
        final_width_px = mm_to_pixels(final_width_mm, dpi)
        final_height_px = mm_to_pixels(final_height_mm, dpi)
        
        st.write(f"**Final Print Size:** {final_width_mm} × {final_height_mm} mm ({format_name})")
        
        # Check if dimensions are valid
        if image_width_mm <= 0 or image_height_mm <= 0:
            st.error(f"❌ **Error**: Padding ({padding_mm}mm) and bleed ({bleed_mm}mm) are too large for {format_name} format!")
            st.write(f"Maximum total padding + bleed: {min(format_size[0], format_size[1]) / 2:.1f} mm")
        else:
            st.write(f"**Image Content Area:** {image_width_mm:.1f} × {image_height_mm:.1f} mm")
            if padding_mm > 0:
                st.write(f"**With Padding:** {content_width_mm:.1f} × {content_height_mm:.1f} mm")
            if bleed_mm > 0:
                st.write(f"**With Bleed:** {final_width_mm} × {final_height_mm} mm")
            
            st.write(f"**Final Pixels:** {final_width_px} × {final_height_px} px")
            st.write(f"**DPI:** {dpi}")
            
            # Physical dimensions in inches
            width_inches = mm_to_inches(final_width_mm)
            height_inches = mm_to_inches(final_height_mm)
            st.write(f"**Physical Size:** {width_inches:.2f}\" × {height_inches:.2f}\"")
    
    # Process button and results
    if uploaded_file is not None:
        if st.button("🚀 Process for Print", type="primary"):
            with st.spinner("Processing image for print..."):
                # Process the uploaded file
                original_img = process_uploaded_file(uploaded_file)
                
                if original_img is not None:
                    # Show original image info - centered single column
                    st.markdown("<h3 style='text-align: center;'>📸 Original Image</h3>", unsafe_allow_html=True)
                    
                    # Center the image
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(original_img, caption="Original Image", use_container_width=True)
                    
                    # Image info below - centered
                    orig_width, orig_height = original_img.size
                    estimated_dpi = 72
                    orig_width_inches = orig_width / estimated_dpi
                    orig_height_inches = orig_height / estimated_dpi
                    
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <strong>Original Dimensions:</strong> {orig_width} × {orig_height} px<br>
                        <strong>Estimated Print Size:</strong> {orig_width_inches:.2f}" × {orig_height_inches:.2f}" (at 72 DPI)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-rotation logic - check orientation mismatch BEFORE processing
                    rotated_img = original_img
                    rotation_applied = False
                    
                    if auto_rotate:
                        # Determine orientations
                        img_width, img_height = original_img.size
                        format_width, format_height = format_size
                        
                        # Check if image is square (no rotation needed for square images)
                        img_is_square = img_width == img_height
                        
                        if not img_is_square:
                            img_is_landscape = img_width > img_height
                            format_is_landscape = format_width > format_height
                            
                            # Rotate if orientations don't match
                            if img_is_landscape != format_is_landscape:
                                st.write("🔄 Auto-rotating image to match format orientation...")
                                rotated_img = rotate_image(original_img, 90)
                                rotation_applied = True
                                st.info(f"📐 Image rotated 90° to match {format_name} orientation")
                        else:
                            st.info("📐 Square image detected - no rotation needed")
                    
                    # Process the image
                    # Calculate the actual image content area (working backwards from final format)
                    # Recalculate dimensions for processing
                    final_w_mm = format_size[0]
                    final_h_mm = format_size[1]
                    content_w_mm = final_w_mm - (2 * bleed_mm)
                    content_h_mm = final_h_mm - (2 * bleed_mm)
                    image_w_mm = content_w_mm - (2 * padding_mm)
                    image_h_mm = content_h_mm - (2 * padding_mm)
                    
                    # Check if dimensions are valid before processing
                    if image_w_mm <= 0 or image_h_mm <= 0:
                        st.error(f"❌ Cannot process: Padding ({padding_mm}mm) and bleed ({bleed_mm}mm) are too large for {format_name} format!")
                        st.write(f"Maximum total padding + bleed: {min(format_size[0], format_size[1]) / 2:.1f} mm")
                        return
                    
                    # Initialize intermediate images dictionary
                    intermediate_images = {}
                    
                    # Check if shape-aware processing is enabled
                    if shape_aware_mode:
                        st.write("🎯 Detecting shape and applying smart padding...")
                        
                        # Convert color for shape processing
                        shape_bg_color_rgb = tuple(int(shape_bg_color[i:i+2], 16) for i in (1, 3, 5))
                        
                        # Process with shape-aware algorithm
                        content_with_padding_img, shape_intermediates = detect_shape_and_create_smart_padding(
                            rotated_img,
                            content_w_mm + (2 * padding_mm),  # Include padding in target
                            content_h_mm + (2 * padding_mm),
                            dpi,
                            padding_mm,
                            detection_sensitivity,
                            background_handling,
                            shape_bg_color
                        )
                        
                        # Add shape processing intermediates
                        intermediate_images.update(shape_intermediates)
                        
                        st.success(f"✅ Shape detected and processed with {background_handling} background")
                        
                        # Apply bleed and cut lines to the shape-processed image
                        if bleed_mm > 0:
                            st.write(f"📏 Adding {bleed_mm}mm bleed ({bleed_type})...")
                            from bleed import add_bleed
                            bleed_img = add_bleed(content_with_padding_img, bleed_mm, dpi, bleed_type, bleed_color_rgb)
                        else:
                            bleed_img = content_with_padding_img
                        
                        if add_cut_lines_flag and bleed_mm > 0:
                            st.write(f"✂️ Adding {cut_line_color} cut lines...")
                            from cut_lines import add_cut_lines
                            processed_img = add_cut_lines(bleed_img, bleed_mm, dpi, cut_line_color)
                        else:
                            processed_img = bleed_img
                    
                    else:
                        # Traditional processing
                        image_content_size = (image_w_mm, image_h_mm)
                        
                        processed_img, intermediate_images = process_for_print(
                            rotated_img,  # Use rotated image instead of original
                            image_content_size,  # Use calculated image content area instead of format_size
                            dpi, 
                            padding_mm,
                            padding_style,
                            ai_style,
                            custom_prompt,
                            remove_objects,
                            object_removal_sensitivity,
                            bleed_mm,
                            bleed_type,
                            bleed_color_rgb,
                            add_cut_lines_flag,
                            cut_line_color
                        )
                    
                    st.success("✅ Processing complete!")
                    
                    # Show intermediate images if available (especially for AI processing)
                    if intermediate_images:
                        st.markdown("<h3 style='text-align: center;'>🔍 Processing Steps</h3>", unsafe_allow_html=True)
                        
                        # Define step descriptions
                        step_descriptions = {
                            'object_mask': '🎯 Object Detection Mask',
                            'after_object_removal': '🧹 After Object Removal',
                            'processed_for_ai': '🤖 Processed for AI',
                            'padded_rgba': '📦 RGBA with DOUBLE Padding (Buffer Zone)',
                            'openai_input': '⬆️ Input to OpenAI (Centered with Equal Padding)',
                            'openai_mask': '🎭 OpenAI Preservation Mask',
                            'openai_output': '⬇️ OpenAI Generated Result',
                            'scaled_to_buffer': '📐 Scaled Back to Buffer Size',
                            'scaled_back': '✂️ Trimmed to Target Size',
                            'final_result': '✨ Final with Original Content'
                        }
                        
                        # Show intermediate images in a grid
                        cols_per_row = 3
                        current_col = 0
                        cols = st.columns(cols_per_row)
                        
                        for step_key, img in intermediate_images.items():
                            with cols[current_col]:
                                description = step_descriptions.get(step_key, step_key.replace('_', ' ').title())
                                st.image(img, caption=description, use_container_width=True)
                                
                                # Show image info
                                if hasattr(img, 'size'):
                                    st.caption(f"Size: {img.size[0]}×{img.size[1]}")
                                elif hasattr(img, 'shape'):
                                    st.caption(f"Size: {img.shape[1]}×{img.shape[0]}")
                            
                            current_col = (current_col + 1) % cols_per_row
                            if current_col == 0:
                                cols = st.columns(cols_per_row)
                    
                    # Show processed image - centered single column
                    st.markdown("<h3 style='text-align: center;'>🖨️ Print-Ready Image</h3>", unsafe_allow_html=True)
                    
                    # Center the image
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(processed_img, caption="Print-Ready Image", use_container_width=True)
                    
                    # Image info below - centered
                    final_width, final_height = processed_img.size
                    
                    # Build info HTML
                    info_html = f"""
                    <div style='text-align: center;'>
                        <strong>Final Dimensions:</strong> {final_width} × {final_height} px<br>
                        <strong>Print Size:</strong> {format_size[0]} × {format_size[1]} mm<br>
                        <strong>With Bleed:</strong> {final_width_mm:.1f} × {final_height_mm:.1f} mm<br>
                        <strong>DPI:</strong> {dpi}<br>
                    """
                    
                    if padding_mm > 0:
                        # Define styles for display
                        padding_display_styles = {
                            "content_aware": "Content Aware",
                            "soft_shadow": "Soft Shadow",
                            "gradient_fade": "Gradient Fade", 
                            "color_blend": "Color Blend",
                            "vintage_vignette": "Vintage Vignette",
                            "clean_border": "Clean Border",
                            "ai_advanced": "🤖 AI Advanced"
                        }
                        
                        ai_display_styles = {
                            "natural": "Natural Extension",
                            "artistic": "Artistic & Creative",
                            "minimalist": "Minimalist & Clean",
                            "textured": "Textured & Detailed",
                            "blurred": "Blurred & Dreamy"
                        }
                        
                        padding_display = padding_display_styles.get(padding_style, padding_style)
                        if padding_style == "ai_advanced":
                            ai_display = ai_display_styles.get(ai_style, ai_style)
                            info_html += f"<strong>Padding Added:</strong> {padding_mm} mm ({padding_display} - {ai_display})<br>"
                        else:
                            info_html += f"<strong>Padding Added:</strong> {padding_mm} mm ({padding_display})<br>"
                    
                    if bleed_mm > 0:
                        info_html += f"<strong>Bleed Added:</strong> {bleed_mm} mm ({bleed_types[bleed_type]})<br>"
                    
                    if add_cut_lines_flag and bleed_mm > 0:
                        info_html += f"<strong>Cut Lines:</strong> {cut_line_colors[cut_line_color]}<br>"
                    
                    if rotation_applied:
                        info_html += f"<strong>Auto-Rotation:</strong> 90° applied to match format orientation<br>"
                    
                    info_html += "</div>"
                    st.markdown(info_html, unsafe_allow_html=True)
                    
                    # Centered download button
                    output_buffer = io.BytesIO()
                    processed_img.save(output_buffer, format='PNG', dpi=(dpi, dpi))
                    output_buffer.seek(0)
                    
                    # Generate filename
                    original_name = Path(uploaded_file.name).stem
                    output_filename = f"{original_name}_print_{format_name}_{dpi}dpi.png"
                    
                    # Center the download button
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        st.download_button(
                            label="📥 Download Print-Ready File",
                            data=output_buffer.getvalue(),
                            file_name=output_filename,
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # Show processing summary - centered single column
                    st.markdown("<h3 style='text-align: center;'>📋 Processing Summary</h3>", unsafe_allow_html=True)
                    
                    # Center the metrics in a more compact layout
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Create two rows of metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Original Size", f"{orig_width}×{orig_height}px")
                            st.metric("Print Format", format_name)
                        
                        with metric_col2:
                            st.metric("Final Size", f"{final_width}×{final_height}px")
                            st.metric("DPI", dpi)
                        
                        with metric_col3:
                            st.metric("Bleed Type", bleed_types[bleed_type])
                            st.metric("Cut Line Color", cut_line_colors[cut_line_color] if add_cut_lines_flag and bleed_mm > 0 else "None")

if __name__ == "__main__":
    main() 