import streamlit as st
from PIL import Image
import fitz
import io
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def create_pdf_from_image(image, format_size, dpi):
    """Create a PDF document from a processed image with exact page dimensions"""
    # Create new PDF document
    pdf_doc = fitz.open()
    
    # Convert format size from mm to points (PDF unit)
    # 1 mm = 2.834645669 points
    mm_to_points = 2.834645669
    page_width = format_size[0] * mm_to_points
    page_height = format_size[1] * mm_to_points
    
    # Create page with exact dimensions
    page = pdf_doc.new_page(width=page_width, height=page_height)
    
    # Convert PIL Image to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG', dpi=(dpi, dpi))
    img_buffer.seek(0)
    img_bytes = img_buffer.getvalue()
    
    # Insert image to fill entire page
    page.insert_image(
        fitz.Rect(0, 0, page_width, page_height),  # Full page rectangle
        stream=img_bytes
    )
    
    return pdf_doc

# Import from our modules
from print_settings import PRINT_FORMATS, mm_to_inches, resize_image_for_print
from bleed import add_bleed
from cut_lines import add_cut_lines

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
    """Complete print processing pipeline with support for AI Advanced padding and bleed"""
    width_mm, height_mm = format_size
    intermediate_images = {}
    
    # Determine if we're using AI Advanced for padding or bleed
    use_ai_padding = padding_style == "ai_advanced" and padding_mm > 0
    use_ai_bleed = bleed_type == "ai_advanced" and bleed_mm > 0
    
    # Convert mm to pixels for AI processing
    base_dpi = 300  # Standard base resolution for AI processing
    from print_settings import mm_to_pixels
    padding_px = mm_to_pixels(padding_mm, base_dpi) if padding_mm > 0 else 0
    bleed_px = mm_to_pixels(bleed_mm, base_dpi) if bleed_mm > 0 else 0
    
    # AI Advanced workflow - three scenarios
    if use_ai_padding and use_ai_bleed:
        # Scenario 3: Both AI padding and AI bleed
        st.write("🤖 AI Advanced: Padding + Bleed Combined")
        if remove_objects:
            st.write(f"🎯 Removing distracting objects (sensitivity: {object_removal_sensitivity})...")
        st.write(f"📦 Adding {padding_mm}mm AI padding...")
        st.write(f"📏 Adding {bleed_mm}mm AI bleed...")
        
        try:
            from padding import create_ai_padding_and_bleed
            processed_img, intermediate_images = create_ai_padding_and_bleed(
                img, padding_px, bleed_px, custom_prompt, custom_prompt, 
                remove_objects, object_removal_sensitivity
            )
            st.success("✅ AI padding + bleed completed successfully!")
        except Exception as e:
            st.error(f"❌ AI padding + bleed failed: {str(e)}")
            return img, {}
        
        # Resize to final print dimensions
        st.write("🔄 Resizing to target print dimensions...")
        final_img = resize_image_for_print(processed_img, width_mm, height_mm, dpi)
        
    elif use_ai_padding and not use_ai_bleed:
        # Scenario 1: Only AI padding
        st.write("🤖 AI Advanced: Padding Only")
        if remove_objects:
            st.write(f"🎯 Removing distracting objects (sensitivity: {object_removal_sensitivity})...")
        st.write(f"📦 Adding {padding_mm}mm AI padding...")
        
        try:
            from padding import create_ai_padding
            padded_img, intermediate_images = create_ai_padding(
                img, padding_px, ai_style, custom_prompt, 
                remove_objects, object_removal_sensitivity
            )
            st.success("✅ AI padding completed successfully!")
        except Exception as e:
            st.error(f"❌ AI padding failed: {str(e)}")
            return img, {}
        
        # Resize AI-padded image to print dimensions
        st.write("🔄 Resizing to target print dimensions...")
        resized_img = resize_image_for_print(padded_img, width_mm, height_mm, dpi)
        
        # Add traditional bleed if needed
        if bleed_mm > 0:
            st.write(f"📏 Adding {bleed_mm}mm traditional bleed ({bleed_type})...")
            final_img = add_bleed(resized_img, bleed_mm, dpi, bleed_type, bleed_color)
        else:
            final_img = resized_img
            
    elif not use_ai_padding and use_ai_bleed:
        # Scenario 2: Only AI bleed
        st.write("🤖 AI Advanced: Bleed Only")
        
        # First resize and add traditional padding if needed
        st.write("🔄 Resizing to target dimensions...")
        resized_img = resize_image_for_print(img, width_mm, height_mm, dpi)
        
        if padding_mm > 0:
            st.write(f"📦 Adding {padding_mm}mm traditional padding ({padding_style})...")
            try:
                from padding import add_padding
                padded_img, padding_intermediates = add_padding(
                    resized_img, padding_mm, dpi, padding_style, ai_style, 
                    custom_prompt, remove_objects, object_removal_sensitivity
                )
                intermediate_images.update(padding_intermediates)
            except Exception as e:
                st.error(f"Error adding traditional padding: {str(e)}")
                padded_img = resized_img
        else:
            padded_img = resized_img
        
        # Add AI bleed
        st.write(f"📏 Adding {bleed_mm}mm AI bleed...")
        try:
            from padding import create_ai_bleed
            final_img, bleed_intermediates = create_ai_bleed(
                padded_img, bleed_px, custom_prompt, 
                remove_objects, object_removal_sensitivity
            )
            intermediate_images.update(bleed_intermediates)
            st.success("✅ AI bleed completed successfully!")
        except Exception as e:
            st.error(f"❌ AI bleed failed: {str(e)}")
            final_img = padded_img
    
    else:
        # Traditional workflow - no AI
        st.write("🔄 Traditional Processing Workflow")
        st.write("🔄 Resizing to target dimensions...")
        resized_img = resize_image_for_print(img, width_mm, height_mm, dpi)
        
        # Add traditional padding if needed
        if padding_mm > 0:
            st.write(f"📦 Adding {padding_mm}mm padding ({padding_style})...")
            try:
                from padding import add_padding
                padded_img, padding_intermediates = add_padding(
                    resized_img, padding_mm, dpi, padding_style, ai_style, 
                    custom_prompt, remove_objects, object_removal_sensitivity
                )
                intermediate_images.update(padding_intermediates)
            except Exception as e:
                st.error(f"Error adding padding: {str(e)}")
                padded_img = resized_img
        else:
            padded_img = resized_img
        
        # Add traditional bleed if needed
        if bleed_mm > 0:
            st.write(f"📏 Adding {bleed_mm}mm bleed ({bleed_type})...")
            final_img = add_bleed(padded_img, bleed_mm, dpi, bleed_type, bleed_color)
        else:
            final_img = padded_img
    
    # Add cut lines if requested
    if add_cut_lines_flag and bleed_mm > 0:
        st.write(f"✂️ Adding cut lines ({cut_line_color})...")
        final_img = add_cut_lines(final_img, bleed_mm, dpi, cut_line_color)
    
    return final_img, intermediate_images

# Set page config
st.set_page_config(
    page_title="Processor - Print Processor",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🖨️ Print Processor")
    st.markdown("**Professional print preparation tool with DPI adjustment, bleed, and cut lines**")
    
    # Sidebar for settings
    st.sidebar.header("📋 Print Settings")
    
    # Print format selection
    def format_display_func(format_key):
        if format_key == "Custom":
            return "Custom"
        else:
            dimensions = PRINT_FORMATS[format_key]
            return f"{format_key} ({dimensions[0]}×{dimensions[1]}mm)"
    
    format_name = st.sidebar.selectbox(
        "Printing Format",
        list(PRINT_FORMATS.keys()),
        format_func=format_display_func,
        index=0
    )
    
    if format_name == "Custom":
        col1, col2 = st.sidebar.columns(2)
        custom_width = col1.number_input("Width (mm)", min_value=10, max_value=2000, value=210)
        custom_height = col2.number_input("Height (mm)", min_value=10, max_value=2000, value=297)
        format_size = (custom_width, custom_height)
    else:
        format_size = PRINT_FORMATS[format_name]
    
    # DPI selection
    dpi_options = [72, 150, 300, 600, "Custom"]
    
    def dpi_display_func(dpi_value):
        if dpi_value == "Custom":
            return "Custom DPI"
        else:
            return f"{dpi_value} DPI"
    
    selected_dpi = st.sidebar.selectbox(
        "DPI (Print Quality)",
        dpi_options,
        format_func=dpi_display_func,
        index=2,  # Default to 300 DPI
        help="Higher DPI = better quality but larger file size"
    )
    
    # Handle custom DPI input
    if selected_dpi == "Custom":
        dpi = st.sidebar.number_input(
            "Custom DPI Value",
            min_value=50,
            max_value=2400,
            value=300,
            step=25,
            help="Enter your desired DPI (recommended: 150-1200)"
        )
        dpi_description = f"Custom ({dpi} DPI)"
    else:
        dpi = selected_dpi
        # DPI info
        dpi_info = {
            72: "Screen display",
            150: "Draft printing", 
            300: "Good quality printing",
            600: "High quality printing"
        }
        dpi_description = dpi_info[dpi]
    
    st.sidebar.write(f"*{dpi_description}*")
    
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
                
                # AI style is now fixed to "natural" as requested by user
                ai_style = "natural"
                st.sidebar.info("🌿 AI Style: Natural Extension (optimized for gpt-image-1)")
                
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
                    
                st.sidebar.info("🤖 AI-generated contextual padding using OpenAI gpt-image-1")
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
    
    # Bleed type selector with AI Advanced option
    bleed_types = {
        "content_aware": "Content Aware",
        "mirror": "Mirror Edges",
        "edge_extend": "Edge Extend",
        "solid_color": "Solid Color"
    }
    
    # Check OpenAI setup for bleed
    if bleed_mm > 0:
        from padding import OPENAI_AVAILABLE, validate_openai_setup
        openai_ready, openai_status = validate_openai_setup()
        
        # Add AI option if available
        if OPENAI_AVAILABLE:
            bleed_types["ai_advanced"] = "🤖 AI Advanced (OpenAI)"
    
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
        index=4,  # Default to black
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
                
                # Detect input format for output matching
                is_pdf_input = uploaded_file.type == "application/pdf"
                
                if original_img is not None:
                    # Show original image info
                    st.subheader("📸 Original Image")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(original_img, caption="Original Image", use_container_width=True)
                    
                    with col2:
                        orig_width, orig_height = original_img.size
                        st.write(f"**Original Dimensions:** {orig_width} × {orig_height} px")
                        
                        # Estimate original DPI (assuming common screen DPI)
                        estimated_dpi = 72
                        orig_width_inches = orig_width / estimated_dpi
                        orig_height_inches = orig_height / estimated_dpi
                        st.write(f"**Estimated Print Size:** {orig_width_inches:.2f}\" × {orig_height_inches:.2f}\" (at 72 DPI)")
                    
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
                        st.subheader("🔍 Processing Steps")
                        
                        # Define step descriptions for new gpt-image-1 workflow
                        step_descriptions = {
                            # Object removal steps
                            'object_mask': '🎯 Object Detection Mask',
                            'after_object_removal': '🧹 After Object Removal',
                            
                            # AI Padding steps
                            'rgba_with_padding': '📦 RGBA with Transparent Padding Areas',
                            'openai_canvas': '📐 Scaled to OpenAI Canvas',
                            'openai_mask': '🎭 AI Edit Mask (White=Generate, Black=Preserve)',
                            'openai_input_rgb': '⬆️ RGB Input to OpenAI gpt-image-1',
                            'openai_output': '⬇️ AI Generated Result',
                            'extracted_result': '✂️ Extracted from OpenAI Canvas',
                            'scaled_back': '📏 Scaled Back to Target Size',
                            'final_result': '✨ Final with Original Content Preserved',
                            
                            # AI Bleed steps (with prefixes)
                            'bleed_rgba_with_bleed': '📏 RGBA with Transparent Bleed Areas',
                            'bleed_openai_canvas': '📐 Bleed: Scaled to OpenAI Canvas',
                            'bleed_openai_mask': '🎭 Bleed: AI Edit Mask',
                            'bleed_openai_input_rgb': '⬆️ Bleed: RGB Input to OpenAI',
                            'bleed_openai_output': '⬇️ Bleed: AI Generated Result',
                            'bleed_extracted_result': '✂️ Bleed: Extracted Result',
                            'bleed_scaled_back': '📏 Bleed: Scaled Back',
                            'bleed_final_result': '✨ Bleed: Final Result',
                            
                            # Combined padding + bleed steps
                            'padding_rgba_with_padding': '📦 Phase 1: RGBA with Padding',
                            'padding_openai_canvas': '📐 Phase 1: Padding Canvas',
                            'padding_openai_mask': '🎭 Phase 1: Padding Mask',
                            'padding_openai_input_rgb': '⬆️ Phase 1: Padding Input',
                            'padding_openai_output': '⬇️ Phase 1: Padding Output',
                            'padding_extracted_result': '✂️ Phase 1: Padding Extracted',
                            'padding_scaled_back': '📏 Phase 1: Padding Scaled',
                            'padding_final_result': '✨ Phase 1: Padding Complete',
                            
                            # Legacy support for old workflow
                            'processed_for_ai': '🤖 (Legacy) Processed for AI',
                            'padded_rgba': '📦 (Legacy) RGBA Buffer Zone',
                            'openai_input': '⬆️ (Legacy) OpenAI Input',
                            'scaled_to_buffer': '📐 (Legacy) Scaled to Buffer'
                        }
                        
                        # Show intermediate images in a grid with detailed info
                        cols_per_row = 2  # Reduced to 2 for better visibility of details
                        current_col = 0
                        cols = st.columns(cols_per_row)
                        
                        # Sort by step number for proper order
                        sorted_steps = sorted(intermediate_images.items())
                        
                        for step_key, step_data in sorted_steps:
                            with cols[current_col]:
                                # Handle both old format (just image) and new format (dict with image and info)
                                if isinstance(step_data, dict) and 'image' in step_data:
                                    img = step_data['image']
                                    info = step_data['info']
                                    
                                    # Create a cleaner title from step key
                                    step_num = step_key.split('_')[0] if '_' in step_key else ''
                                    step_name = step_key.replace(step_num + '_', '').replace('_', ' ').title()
                                    title = f"{step_num}. {step_name}" if step_num else step_name
                                    
                                    st.image(img, caption=title, use_container_width=True)
                                    st.caption(f"📊 {info}")
                                else:
                                    # Legacy format - just image
                                    img = step_data
                                    description = step_descriptions.get(step_key, step_key.replace('_', ' ').title())
                                    st.image(img, caption=description, use_container_width=True)
                                    
                                    # Show basic image info
                                    if hasattr(img, 'size'):
                                        st.caption(f"📊 Size: {img.size[0]}×{img.size[1]}px")
                                    elif hasattr(img, 'shape'):
                                        st.caption(f"📊 Size: {img.shape[1]}×{img.shape[0]}px")
                            
                            current_col = (current_col + 1) % cols_per_row
                            if current_col == 0:
                                cols = st.columns(cols_per_row)
                    
                    # Show processed image
                    st.subheader("🖨️ Print-Ready Image")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(processed_img, caption="Print-Ready Image", use_container_width=True)
                    
                    with col2:
                        final_width, final_height = processed_img.size
                        st.write(f"**Final Dimensions:** {final_width} × {final_height} px")
                        st.write(f"**Print Size:** {format_size[0]} × {format_size[1]} mm")
                        st.write(f"**With Bleed:** {final_width_mm:.1f} × {final_height_mm:.1f} mm")
                        st.write(f"**DPI:** {dpi}")
                        
                        if padding_mm > 0:
                            # Define styles for display
                            padding_display_styles = {
                                "content_aware": "Content Aware",
                                "soft_shadow": "Soft Shadow",
                                "gradient_fade": "Gradient Fade", 
                                "color_blend": "Color Blend",
                                "vintage_vignette": "Vintage Vignette",
                                "clean_border": "Clean Border",
                                "ai_advanced": "🤖 AI Advanced (gpt-image-1)"
                            }
                            
                            padding_display = padding_display_styles.get(padding_style, padding_style)
                            st.write(f"**Padding Added:** {padding_mm} mm ({padding_display})")
                            
                        if bleed_mm > 0:
                            bleed_display_types = {
                                "content_aware": "Content Aware",
                                "mirror": "Mirror Edges",
                                "edge_extend": "Edge Extend",
                                "solid_color": "Solid Color",
                                "ai_advanced": "🤖 AI Advanced (gpt-image-1)"
                            }
                            bleed_display = bleed_display_types.get(bleed_type, bleed_type)
                            st.write(f"**Bleed Added:** {bleed_mm} mm ({bleed_display})")
                            
                        # Show AI workflow info
                        use_ai_padding = padding_style == "ai_advanced" and padding_mm > 0
                        use_ai_bleed = bleed_type == "ai_advanced" and bleed_mm > 0
                        
                        if use_ai_padding and use_ai_bleed:
                            st.write("**AI Workflow:** Combined Padding + Bleed")
                        elif use_ai_padding:
                            st.write("**AI Workflow:** Padding Only")
                        elif use_ai_bleed:
                            st.write("**AI Workflow:** Bleed Only")
                            
                        if add_cut_lines_flag and bleed_mm > 0:
                            st.write(f"**Cut Lines:** {cut_line_colors[cut_line_color]}")
                        if rotation_applied:
                            st.write(f"**Auto-Rotation:** 90° applied to match format orientation")
                    
                    # Download button - match input format
                    output_buffer = io.BytesIO()
                    original_name = Path(uploaded_file.name).stem
                    
                    if is_pdf_input:
                        # Create PDF output for PDF input
                        pdf_doc = create_pdf_from_image(processed_img, format_size, dpi)
                        pdf_bytes = pdf_doc.tobytes()
                        pdf_doc.close()
                        
                        output_buffer.write(pdf_bytes)
                        output_buffer.seek(0)
                        
                        output_filename = f"{original_name}_print_{format_name}_{dpi}dpi.pdf"
                        mime_type = "application/pdf"
                        file_format_display = "PDF"
                    else:
                        # Create PNG output for image input
                        processed_img.save(output_buffer, format='PNG', dpi=(dpi, dpi))
                        output_buffer.seek(0)
                        
                        output_filename = f"{original_name}_print_{format_name}_{dpi}dpi.png"
                        mime_type = "image/png"
                        file_format_display = "PNG"
                    
                    st.download_button(
                        label=f"📥 Download Print-Ready {file_format_display}",
                        data=output_buffer.getvalue(),
                        file_name=output_filename,
                        mime=mime_type
                    )
                    
                    # Show processing summary
                    st.subheader("📋 Processing Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Original Size", f"{orig_width}×{orig_height}px")
                        st.metric("Input Format", "PDF" if is_pdf_input else "Image")
                    
                    with col2:
                        st.metric("Final Size", f"{final_width}×{final_height}px")
                        st.metric("Print Format", format_name)
                    
                    with col3:
                        st.metric("Output Format", file_format_display)
                        st.metric("DPI", dpi)

if __name__ == "__main__":
    main() 