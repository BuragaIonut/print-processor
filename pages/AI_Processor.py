import streamlit as st
from PIL import Image, ImageDraw
import base64
import io
import os
import json
import fitz  # PyMuPDF for PDF processing
import tempfile
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from gradio_client import Client, handle_file
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_openai import OpenAI as LangChainOpenAI
# Use PyMuPDF for PDF creation (same as Processor.py)
# No need for ReportLab since we already have PyMuPDF working perfectly

# Load environment variables
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(
    page_title="AI Processor - Print Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import functions from other pages
def process_uploaded_file(uploaded_file):
    """Process uploaded file and return PIL Image (handles both images and PDFs)"""
    try:
        if uploaded_file.type == "application/pdf":
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            
            if not pdf_bytes:
                st.error("PDF file appears to be empty")
                return None
            
            uploaded_file.seek(0)
            
            try:
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as pdf_error:
                st.error(f"Could not open PDF: {str(pdf_error)}")
                return None
            
            if len(pdf_document) == 0:
                st.error("PDF has no pages")
                pdf_document.close()
                return None
            
            try:
                page = pdf_document[0]
                mat = fitz.Matrix(2.0, 2.0)  # High resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pdf_document.close()
                
                img = Image.open(io.BytesIO(img_data))
                return img
            except Exception as render_error:
                st.error(f"Could not render PDF page: {str(render_error)}")
                pdf_document.close()
                return None
        else:
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            return img
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def encode_image_from_pil(pil_image):
    """Convert PIL Image to base64 string"""
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        pil_image = pil_image.convert('RGB')
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def analyze_image_with_openai(pil_image, use_case, custom_format=None, custom_dpi=None):
    """Analyze image using OpenAI Vision API and determine print processing needs"""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        base64_image = encode_image_from_pil(pil_image)
        
        width, height = pil_image.size
        aspect_ratio = width / height
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        
        # Build format requirements (make this the PRIMARY focus)
        format_requirement = ""
        dpi_requirement = ""
        
        if custom_format:
            format_requirement = f"""
🚨 MANDATORY FORMAT REQUIREMENT 🚨
The user has EXPLICITLY specified the print format as: '{custom_format}'

YOU MUST USE '{custom_format}' as the recommended_format in your response.
This is NOT a suggestion - this is a REQUIRED constraint that overrides any other format recommendations.
Do NOT suggest alternative formats. The user wants '{custom_format}' specifically.
"""
        
        if custom_dpi:
            dpi_requirement = f"""
🚨 MANDATORY DPI REQUIREMENT 🚨
The user has EXPLICITLY specified the DPI as: {custom_dpi}

YOU MUST USE {custom_dpi} as the dpi_recommendation in your response.
This is NOT a suggestion - this is a REQUIRED constraint.
"""
        
        analysis_prompt = f"""
You are a professional print preparation expert. Analyze this image for printing as: "{use_case}"

{format_requirement}
{dpi_requirement}

Current image details:
- Dimensions: {width} × {height} pixels
- Aspect ratio: {aspect_ratio:.2f}
- Orientation: {orientation}

Your task is to determine the optimal print processing steps for the {'USER-SPECIFIED' if custom_format else 'optimal'} format. Consider:

1. **Format Compliance**: {'MUST use the user-specified format: ' + custom_format if custom_format else 'What print format (A4, A3, A5, Letter, postcard size, etc.) would work best?'}
2. **Bleed Requirements**: How much bleed is recommended for this type of print job?
3. **Padding Needs**: Would padding improve the print quality or aesthetics?
4. **Cut Lines**: Are cut lines necessary for this print job?
5. **Quality Concerns**: Any image quality issues that might affect printing?
6. **Content Analysis**: What's the main subject/content and how does it affect print decisions?

CRITICAL RULES:
- Images will NOT be rotated - work with the current orientation
- If user specified a format, YOU MUST use that exact format in your recommendations
- Focus on making the image work optimally with the {'specified format: ' + custom_format if custom_format else 'best suitable format'}

Provide your analysis in this JSON format:
{{
    "content_description": "Brief description of what's in the image",
    "recommended_actions": {{
        "recommended_format": "format name (e.g., A4, A5, Postcard)",
        "format_dimensions": [width_mm, height_mm],
        "bleed_recommendation": {{
            "needed": true/false,
            "amount_mm": 0-5,
            "type": "content_aware/mirror/edge_extend",
            "reason": "explanation"
        }},
        "padding_recommendation": {{
            "needed": true/false,
            "amount_mm": 0-10,
            "style": "content_aware/clean_border/ai_advanced", 
            "reason": "explanation"
        }},
        "cut_lines": {{
            "needed": true/false,
            "reason": "explanation"
        }},
        "dpi_recommendation": 150/300/600,
        "quality_concerns": ["list of any quality issues"],
        "additional_notes": "any other recommendations"
    }},
    "confidence_score": 0.0-1.0,
    "processing_summary": "One sentence summary of recommended processing"
}}

Be specific and practical in your recommendations based on the intended use case and image content.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content
        
        try:
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                analysis_data = json.loads(json_str)
                return analysis_data
            else:
                return {"error": "Could not parse JSON response", "raw_response": analysis_text}
                
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": analysis_text}
            
    except Exception as e:
        st.error(f"❌ Error analyzing image: {str(e)}")
        return None

def save_image_to_temp(image):
    """Save PIL image to temporary file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name, "PNG")
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    
    # Create a preview image showing the mask
    preview = background.copy().convert('RGBA')
    
    # Create a semi-transparent red overlay
    red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))  # Reduced alpha to 64 (25% opacity)
    
    # Convert black pixels in the mask to semi-transparent red
    red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    red_mask.paste(red_overlay, (0, 0), mask)
    
    # Overlay the red mask on the background
    preview = Image.alpha_composite(preview, red_mask)
    
    return preview

def extend_image_for_padding(image_path, extension_amount_mm, current_format_name):
    """Use gradio client to add padding with GPU-safe dimensions"""
    try:
        client = Client("https://yntm37337cjr7u-7860.proxy.runpod.net/")
        
        # Get current format dimensions and add padding
        base_width_mm, base_height_mm = get_format_dimensions(current_format_name)
        padded_width_mm = base_width_mm + (extension_amount_mm * 2)
        padded_height_mm = base_height_mm + (extension_amount_mm * 2)
        
        # Calculate target aspect ratio
        target_aspect_ratio = padded_width_mm / padded_height_mm
        
        # Use GPU-safe dimensions (max 1536x1536)
        if target_aspect_ratio > 1:
            target_width = min(1536, 1024)
            target_height = int(target_width / target_aspect_ratio)
            if target_height > 1536:
                target_height = 1536
                target_width = int(target_height * target_aspect_ratio)
        else:
            target_height = min(1536, 1024)
            target_width = int(target_height * target_aspect_ratio)
            if target_width > 1536:
                target_width = 1536
                target_height = int(target_width / target_aspect_ratio)
        
        # Ensure minimum size
        target_width = max(target_width, 512)
        target_height = max(target_height, 512)
        
        st.write(f"🖼️ Adding {extension_amount_mm}mm padding: {target_width}×{target_height}px (GPU-safe)")
        
        # Debug: Show all API parameters
        api_params = {
            "image": image_path,
            "width": target_width,
            "height": target_height,
            "overlap_percentage": 3,
            "num_inference_steps": 8,
            "resize_option": "Full",
            "custom_resize_percentage": 50,
            "alignment": "Middle",
            "overlap_left": True,
            "overlap_right": True,
            "overlap_top": True,
            "overlap_bottom": True,
            "api_name": "/infer"
        }
        
        st.write("🔧 **DEBUG: Padding API Call #2**")
        st.json(api_params)
        
        # Show preview of alignment and mask
        with Image.open(image_path) as img:
            preview = preview_image_and_mask(
                img, target_width, target_height, 3, "Full", 50, "Middle",
                True, True, True, True  # All directions for padding
            )
            st.subheader("🔍 Preview: Padding Extension Areas")
            st.image(preview, caption="Red areas will be extended by AI", use_container_width=True)
        
        result = client.predict(
            image=handle_file(image_path),
            width=target_width,
            height=target_height,
            overlap_percentage=3,
            num_inference_steps=8,
            resize_option="Full",
            custom_resize_percentage=50,
            alignment="Middle",
            overlap_left=True,
            overlap_right=True,
            overlap_top=True,
            overlap_bottom=True,
            api_name="/infer"
        )
        
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return result[1]
        elif isinstance(result, (list, tuple)) and len(result) == 1:
            return result[0]
        else:
            return result
        
    except Exception as e:
        st.error(f"Error adding padding: {str(e)}")
        return None

def extend_image_for_bleed(image_path, bleed_amount_mm, current_format_name):
    """Use gradio client to add bleed with GPU-safe dimensions"""
    try:
        client = Client("https://yntm37337cjr7u-7860.proxy.runpod.net/")
        
        # Get current format dimensions and add bleed
        base_width_mm, base_height_mm = get_format_dimensions(current_format_name)
        bleed_width_mm = base_width_mm + (bleed_amount_mm * 2)
        bleed_height_mm = base_height_mm + (bleed_amount_mm * 2)
        
        # Calculate target aspect ratio
        target_aspect_ratio = bleed_width_mm / bleed_height_mm
        
        # Use GPU-safe dimensions (max 1536x1536)
        if target_aspect_ratio > 1:
            target_width = min(1536, 1024)
            target_height = int(target_width / target_aspect_ratio)
            if target_height > 1536:
                target_height = 1536
                target_width = int(target_height * target_aspect_ratio)
        else:
            target_height = min(1536, 1024)
            target_width = int(target_height * target_aspect_ratio)
            if target_width > 1536:
                target_width = 1536
                target_height = int(target_width / target_aspect_ratio)
        
        # Ensure minimum size
        target_width = max(target_width, 512)
        target_height = max(target_height, 512)
        
        st.write(f"🖼️ Adding {bleed_amount_mm}mm bleed: {target_width}×{target_height}px (GPU-safe)")
        
        # Debug: Show all API parameters
        api_params = {
            "image": image_path,
            "width": target_width,
            "height": target_height,
            "overlap_percentage": 3,
            "num_inference_steps": 8,
            "resize_option": "Full",
            "custom_resize_percentage": 50,
            "alignment": "Middle",
            "overlap_left": True,
            "overlap_right": True,
            "overlap_top": True,
            "overlap_bottom": True,
            "api_name": "/infer"
        }
        
        st.write("🔧 **DEBUG: Bleed API Call #3**")
        st.json(api_params)
        
        # Show preview of alignment and mask
        with Image.open(image_path) as img:
            preview = preview_image_and_mask(
                img, target_width, target_height, 3, "Full", 50, "Middle",
                True, True, True, True  # All directions for bleed
            )
            st.subheader("🔍 Preview: Bleed Extension Areas")
            st.image(preview, caption="Red areas will be extended by AI", use_container_width=True)
        
        result = client.predict(
            image=handle_file(image_path),
            width=target_width,
            height=target_height,
            overlap_percentage=3,
            num_inference_steps=8,
            resize_option="Full",
            custom_resize_percentage=50,
            alignment="Middle",
            overlap_left=True,
            overlap_right=True,
            overlap_top=True,
            overlap_bottom=True,
            api_name="/infer"
        )
        
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return result[1]
        elif isinstance(result, (list, tuple)) and len(result) == 1:
            return result[0]
        else:
            return result
        
    except Exception as e:
        st.error(f"Error adding bleed: {str(e)}")
        return None

def get_format_dimensions(format_name):
    """Get dimensions in mm for standard formats"""
    formats = {
        "A4": (210, 297),
        "A3": (297, 420), 
        "A5": (148, 210),
        "Letter": (216, 279),
        "Postcard": (100, 150),
        "Business Card": (85, 55),
        "Photo 4x6": (102, 152),
        "Photo 5x7": (127, 178),
        "Photo 8x10": (203, 254)
    }
    return formats.get(format_name, (210, 297))  # Default to A4

def analyze_aspect_ratios(image_width, image_height, format_width, format_height):
    """
    Analyze image and format aspect ratios to determine extension strategy
    Returns: (strategy, direction)
    - strategy: 'same_orientation', 'extend_to_landscape', 'extend_to_portrait'
    - direction: 'horizontal', 'vertical', 'around' (for padding/bleeding)
    """
    # Calculate aspect ratios
    image_ratio = image_width / image_height
    format_ratio = format_width / format_height
    
    # Define orientation categories
    def get_orientation(ratio):
        if 0.9 <= ratio <= 1.1:  # Square-ish (within 10% of 1:1)
            return "square"
        elif ratio > 1.1:
            return "landscape"
        else:
            return "portrait"
    
    image_orientation = get_orientation(image_ratio)
    format_orientation = get_orientation(format_ratio)
    
    # Decision logic
    if image_orientation == format_orientation:
        # Same orientation - just add padding/bleeding around
        return "same_orientation", "around"
    elif (image_orientation in ["square", "portrait"]) and format_orientation == "landscape":
        # Need to extend horizontally to make it landscape
        return "extend_to_landscape", "horizontal"
    elif (image_orientation in ["square", "landscape"]) and format_orientation == "portrait":
        # Need to extend vertically to make it portrait
        return "extend_to_portrait", "vertical"
    else:
        # Fallback - treat as same orientation
        return "same_orientation", "around"

def extend_image_directional(image_path, target_ratio, direction, format_name):
    """
    Extend image in specific direction to achieve target aspect ratio
    direction: 'horizontal' (extend left/right) or 'vertical' (extend top/bottom)
    """
    try:
        if not os.path.exists(image_path):
            return None
            
        # Create gradio client
        client = Client("https://yntm37337cjr7u-7860.proxy.runpod.net/")
        
        # Read image to get dimensions
        with Image.open(image_path) as img:
            current_width, current_height = img.size
            current_ratio = current_width / current_height
        
        # Calculate target dimensions based on direction
        if direction == "horizontal":
            # Extend horizontally - keep height, adjust width
            target_width = int(current_height * target_ratio)
            target_height = current_height
        else:  # vertical
            # Extend vertically - keep width, adjust height
            target_width = current_width
            target_height = int(current_width / target_ratio)
        
        # Ensure GPU-safe dimensions (max 1536 on longest side)
        max_dim = max(target_width, target_height)
        if max_dim > 1536:
            scale_factor = 1536 / max_dim
            target_width = int(target_width * scale_factor)
            target_height = int(target_height * scale_factor)
        
        # Ensure minimum size
        target_width = max(target_width, 512)
        target_height = max(target_height, 512)
        
        # Set directional overlap based on extension direction
        if direction == "horizontal":
            # Extend left and right only
            overlap_left = True
            overlap_right = True
            overlap_top = False
            overlap_bottom = False
            st.write(f"🔄 Extending horizontally: {target_width}×{target_height}px")
        else:  # vertical
            # Extend top and bottom only
            overlap_left = False
            overlap_right = False
            overlap_top = True
            overlap_bottom = True
            st.write(f"🔄 Extending vertically: {target_width}×{target_height}px")
        
        # Debug: Show all API parameters
        api_params = {
            "image": image_path,
            "width": target_width,
            "height": target_height,
            "overlap_percentage": 5,
            "num_inference_steps": 10,
            "resize_option": "Full",
            "custom_resize_percentage": 50,
            "prompt_input": "",
            "alignment": "Middle",
            "overlap_left": overlap_left,
            "overlap_right": overlap_right,
            "overlap_top": overlap_top,
            "overlap_bottom": overlap_bottom,
            "api_name": "/infer"
        }
        
        st.write("🔧 **DEBUG: Directional Extension API Call #1**")
        st.json(api_params)
        
        # Show preview of alignment and mask
        with Image.open(image_path) as img:
            preview = preview_image_and_mask(
                img, target_width, target_height, 5, "Full", 50, "Middle",
                overlap_left, overlap_right, overlap_top, overlap_bottom
            )
            direction_text = "horizontal (left & right)" if direction == "horizontal" else "vertical (top & bottom)"
            st.subheader(f"🔍 Preview: {direction_text.title()} Extension Areas")
            st.image(preview, caption=f"Red areas will be extended {direction_text}", use_container_width=True)
        
        # Use correct API parameters
        result = client.predict(
            image=handle_file(image_path),
            width=target_width,
            height=target_height,
            overlap_percentage=5,
            num_inference_steps=10,
            resize_option="Full",
            custom_resize_percentage=50,
            prompt_input="",
            alignment="Middle",
            overlap_left=overlap_left,
            overlap_right=overlap_right,
            overlap_top=overlap_top,
            overlap_bottom=overlap_bottom,
            api_name="/infer"
        )
        
        # Handle different result formats
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return result[1]
        elif isinstance(result, (list, tuple)) and len(result) == 1:
            return result[0]
        else:
            return result
        
    except Exception as e:
        st.error(f"Error in directional extension: {str(e)}")
        return None

def resize_image_to_format(image, format_dimensions, dpi):
    """Resize image to exact format dimensions based on DPI"""
    try:
        width_mm, height_mm = format_dimensions
        # Convert mm to pixels using DPI
        width_px = int(width_mm * dpi / 25.4)
        height_px = int(height_mm * dpi / 25.4)
        
        # Resize image to exact dimensions
        resized_image = image.resize((width_px, height_px), Image.Resampling.LANCZOS)
        
        return resized_image
    except Exception as e:
        st.error(f"Error resizing image: {str(e)}")
        return image

def add_cut_lines(image, bleed_mm, dpi, color="red"):
    """Add cut lines to image with bleed"""
    try:
        from PIL import ImageDraw
        
        # Create a copy to draw on
        img_with_lines = image.copy()
        draw = ImageDraw.Draw(img_with_lines)
        
        # Convert bleed mm to pixels
        bleed_px = int(bleed_mm * dpi / 25.4)
        
        # Get image dimensions
        width, height = image.size
        
        # Calculate cut line positions (at the bleed boundary)
        cut_left = bleed_px
        cut_right = width - bleed_px
        cut_top = bleed_px
        cut_bottom = height - bleed_px
        
        # Line width based on DPI
        line_width = max(1, int(dpi / 150))
        
        # Color mapping
        colors = {
            "red": "#FF0000",
            "black": "#000000",
            "blue": "#0000FF", 
            "green": "#00FF00",
            "magenta": "#FF00FF",
            "cyan": "#00FFFF",
            "yellow": "#FFFF00",
            "white": "#FFFFFF"
        }
        line_color = colors.get(color, "#FF0000")
        
        # Draw cut lines
        # Top line
        draw.line([(cut_left, cut_top), (cut_right, cut_top)], fill=line_color, width=line_width)
        # Bottom line
        draw.line([(cut_left, cut_bottom), (cut_right, cut_bottom)], fill=line_color, width=line_width)
        # Left line
        draw.line([(cut_left, cut_top), (cut_left, cut_bottom)], fill=line_color, width=line_width)
        # Right line
        draw.line([(cut_right, cut_top), (cut_right, cut_bottom)], fill=line_color, width=line_width)
        
        return img_with_lines
    except Exception as e:
        st.error(f"Error adding cut lines: {str(e)}")
        return image

def create_pdf_from_image(image_path, output_path, format_dimensions, dpi=300):
    """Create a PDF/X-1 compliant document from a processed image using PyMuPDF (same as Processor.py)"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Create new PDF document using PyMuPDF
        pdf_doc = fitz.open()
        
        # Convert format size from mm to points (PDF unit)
        # 1 mm = 2.834645669 points
        mm_to_points = 2.834645669
        page_width = format_dimensions[0] * mm_to_points
        page_height = format_dimensions[1] * mm_to_points
        
        # Create page with exact dimensions
        page = pdf_doc.new_page(width=page_width, height=page_height)
        
        # Convert image to CMYK for print production
        if image.mode == 'RGB':
            # Note: PIL doesn't have direct CMYK conversion, but we can prepare for print
            # The PDF will embed the RGB image, and the print processor will handle CMYK conversion
            pass
        elif image.mode == 'CMYK':
            # Already in CMYK, perfect for print
            pass
        else:
            # Convert other modes to RGB first
            image = image.convert('RGB')
        
        # Convert PIL Image to bytes with maximum quality for print
        img_buffer = io.BytesIO()
        if image.mode == 'CMYK':
            image.save(img_buffer, format='TIFF', dpi=(dpi, dpi), compression='lzw')
        else:
            image.save(img_buffer, format='PNG', dpi=(dpi, dpi), optimize=False)
        
        img_buffer.seek(0)
        img_bytes = img_buffer.getvalue()
        
        # Insert image to fill entire page with high quality settings
        page.insert_image(
            fitz.Rect(0, 0, page_width, page_height),  # Full page rectangle
            stream=img_bytes,
            keep_proportion=False  # Fill entire page exactly
        )
        
        # Set PDF metadata for PDF/X-1 compliance
        metadata = {
            "title": "Print-Ready Document",
            "author": "AI Print Processor",
            "subject": f"AI processed print document - {format_dimensions[0]}x{format_dimensions[1]}mm at {dpi}DPI",
            "creator": "AI Print Processor v1.0",
            "producer": "PyMuPDF with AI Print Processor",
        }
        pdf_doc.set_metadata(metadata)
        
        # Save PDF to output path
        pdf_doc.save(output_path)
        pdf_doc.close()
        
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return False

# LangChain Tools with proper Pydantic v2 annotations
class ImagePaddingTool(BaseTool):
    name: str = "image_padding"
    description: str = "Add padding to an image using AI. Input should be JSON with image_path, padding_mm, and format_name."
    
    def _run(self, input_data: str) -> str:
        try:
            data = json.loads(input_data)
            result_path = extend_image_for_padding(
                data["image_path"],
                data["padding_mm"],
                data["format_name"]
            )
            return f"Padded image saved to: {result_path}"
        except Exception as e:
            return f"Error adding padding: {str(e)}"
    
    def _arun(self, input_data: str) -> str:
        raise NotImplementedError("Async not implemented")

class ImageBleedTool(BaseTool):
    name: str = "image_bleed"
    description: str = "Add bleed to an image using AI. Input should be JSON with image_path, bleed_mm, and format_name."
    
    def _run(self, input_data: str) -> str:
        try:
            data = json.loads(input_data)
            result_path = extend_image_for_bleed(
                data["image_path"],
                data["bleed_mm"],
                data["format_name"]
            )
            return f"Bleed image saved to: {result_path}"
        except Exception as e:
            return f"Error adding bleed: {str(e)}"
    
    def _arun(self, input_data: str) -> str:
        raise NotImplementedError("Async not implemented")

class PDFCreationTool(BaseTool):
    name: str = "pdf_creation"
    description: str = "Create a PDF from a processed image. Input should be JSON with image_path, output_path, format_dimensions, and dpi."
    
    def _run(self, input_data: str) -> str:
        try:
            data = json.loads(input_data)
            success = create_pdf_from_image(
                data["image_path"],
                data["output_path"],
                data["format_dimensions"],
                data.get("dpi", 300)
            )
            return f"PDF creation {'successful' if success else 'failed'}"
        except Exception as e:
            return f"Error creating PDF: {str(e)}"
    
    def _arun(self, input_data: str) -> str:
        raise NotImplementedError("Async not implemented")

# Note: Rotation tool removed - images will not be rotated anymore

def create_processing_agent():
    """Create LangChain agent with image processing tools"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    
    llm = LangChainOpenAI(openai_api_key=api_key, temperature=0)
    
    tools = [
        ImagePaddingTool(),
        ImageBleedTool(),
        PDFCreationTool()
        # Note: ImageRotationTool removed - no more rotation
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

def process_image_with_agent(agent, analysis_data, original_image_path, use_case):
    """Use LangChain agent to process image according to analysis"""
    try:
        actions = analysis_data.get('recommended_actions', {})
        
        # Create processing plan (no rotation)
        processing_steps = []
        
        # Step 1: Padding if needed
        if actions.get('padding_recommendation', {}).get('needed', False):
            padding_mm = actions['padding_recommendation'].get('amount_mm', 5)
            processing_steps.append(f"Add {padding_mm}mm padding using AI extension")
        
        # Step 2: Bleed if needed
        if actions.get('bleed_recommendation', {}).get('needed', False):
            bleed_mm = actions['bleed_recommendation'].get('amount_mm', 3)
            processing_steps.append(f"Add {bleed_mm}mm bleed using AI extension")
        
        # Step 3: Create final PDF
        format_name = actions.get('recommended_format', 'A4')
        dpi = actions.get('dpi_recommendation', 300)
        processing_steps.append(f"Create PDF in {format_name} format at {dpi} DPI")
        
        # Create comprehensive prompt for agent
        agent_prompt = f"""
        Process an image for printing (no rotation will be applied).
        
        Original image path: {original_image_path}
        
        Required processing steps:
        {chr(10).join([f"- {step}" for step in processing_steps])}
        
        Analysis recommendations:
        {json.dumps(actions, indent=2)}
        
        Please execute these steps in order and create a final PDF ready for printing.
        Use temporary files for intermediate steps and return the path to the final PDF.
        Note: Do not rotate the image - maintain original orientation.
        """
        
        # Execute with agent
        result = agent.run(agent_prompt)
        return result
        
    except Exception as e:
        return f"Error in agent processing: {str(e)}"

def display_analysis_results(analysis_data):
    """Display the AI analysis results in a compact format"""
    if not analysis_data or "error" in analysis_data:
        return
    
    actions = analysis_data.get('recommended_actions', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📏 Bleed**")
        bleed = actions.get('bleed_recommendation', {})
        if bleed.get('needed', False):
            st.success(f"✅ {bleed.get('amount_mm', 3)}mm")
        else:
            st.info("❌ Not needed")
    
    with col2:
        st.write("**📦 Padding**")
        padding = actions.get('padding_recommendation', {})
        if padding.get('needed', False):
            st.success(f"✅ {padding.get('amount_mm', 5)}mm")
        else:
            st.info("❌ Not needed")
    
    # Additional info
    format_name = actions.get('recommended_format', 'A4')
    dpi = actions.get('dpi_recommendation', 300)
    st.write(f"**📄 Format:** {format_name} | **🎨 DPI:** {dpi}")
    st.info("📐 Note: Image will maintain current orientation (no rotation)")

def main():
    st.title("🤖 AI Processor")
    st.markdown("**Intelligent automated print processing using AI analysis and LangChain agents**")
    
    # Check dependencies
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        st.sidebar.success("✅ OpenAI API Connected")
    else:
        st.sidebar.error("❌ OpenAI API Key Required")
        st.error("Please set your OpenAI API key to use this feature.")
        return
    
    # Mode selection
    st.subheader("🎯 Processing Mode")
    mode = st.radio(
        "Choose processing mode:",
        ["🤖 Full Auto", "⚙️ Partially Auto"],
        index=0,
        help="Full Auto: AI decides everything. Partially Auto: You set format and DPI."
    )
    
    # File upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="Upload the image or PDF you want to process for printing"
    )
    
    if uploaded_file is not None:
        # Display uploaded file
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            try:
                image = process_uploaded_file(uploaded_file)
                if image is None:
                    return
                
                st.image(image, caption="Original Image", use_container_width=True)
                
                width, height = image.size
                st.write(f"**Dimensions:** {width} × {height} pixels")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        # Configuration based on mode
        with col2:
            st.subheader("⚙️ Configuration")
            
            # Use case (required for both modes)
            use_case = st.text_input(
                "Print use case",
                placeholder="e.g., Book Cover, Postcard, Poster...",
                help="Describe your intended print use case"
            )
            
            custom_format = None
            custom_dpi = None
            
            if mode == "⚙️ Partially Auto":
                st.write("**Manual Overrides:**")
                
                # Format selection
                custom_format = st.selectbox(
                    "Print Format",
                    ["A4", "A3", "A5", "Letter", "Postcard", "Business Card", "Photo 4x6", "Photo 5x7", "Photo 8x10"],
                    index=0
                )
                
                # DPI selection
                custom_dpi = st.selectbox(
                    "Print DPI",
                    [150, 300, 600],
                    index=1
                )
                
                st.info(f"📋 Using {custom_format} format at {custom_dpi} DPI")
            
            # Start processing button
            if st.button("🚀 Start AI Processing", type="primary", use_container_width=True):
                if not use_case.strip():
                    st.error("Please enter a use case first!")
                    return
                
                # Step 1: AI Analysis
                st.subheader("🔍 Step 1: AI Analysis")
                with st.spinner("Analyzing image with AI..."):
                    analysis_result = analyze_image_with_openai(image, use_case, custom_format, custom_dpi)
                
                if not analysis_result or "error" in analysis_result:
                    st.error("Failed to analyze image. Please try again.")
                    return
                
                st.success("✅ Analysis completed!")
                display_analysis_results(analysis_result)
                
                # Step 2: AI Processing
                st.subheader("🤖 Step 2: AI Processing")
                
                with st.spinner("Creating processing agent..."):
                    agent = create_processing_agent()
                
                if not agent:
                    st.error("Failed to create processing agent.")
                    return
                
                # Save original image to temp file
                original_temp_path = save_image_to_temp(image)
                if not original_temp_path:
                    st.error("Failed to save original image.")
                    return
                
                with st.spinner("Processing image with AI agent..."):
                    try:
                        # Get processing parameters
                        actions = analysis_result.get('recommended_actions', {})
                        current_image = image
                        processing_log = []
                        intermediate_images = []  # Store intermediate results for display
                        
                        # Get format and processing settings
                        format_name = actions.get('recommended_format', custom_format or 'A4')
                        format_dims = get_format_dimensions(format_name)
                        padding_mm = actions.get('padding_recommendation', {}).get('amount_mm', 0)
                        bleed_mm = actions.get('bleed_recommendation', {}).get('amount_mm', 0)
                        dpi = custom_dpi or actions.get('dpi_recommendation', 300)
                        
                        # Show original image dimensions and format target
                        orig_width, orig_height = current_image.size
                        st.write(f"📊 Original: {orig_width}×{orig_height}px | Target: {format_name} ({format_dims[0]}×{format_dims[1]}mm)")
                        
                        # Step 1: Analyze aspect ratios and determine strategy
                        strategy, direction = analyze_aspect_ratios(orig_width, orig_height, format_dims[0], format_dims[1])
                        
                        # Display strategy
                        if strategy == "same_orientation":
                            st.info(f"🎯 Strategy: Same orientation - adding content around the image")
                        elif strategy == "extend_to_landscape":
                            st.info(f"🎯 Strategy: Extending horizontally to match landscape format")
                        elif strategy == "extend_to_portrait":
                            st.info(f"🎯 Strategy: Extending vertically to match portrait format")
                        
                        processing_log.append(f"✅ Determined strategy: {strategy}")
                        
                        # Step 2: Apply orientation extension if needed
                        if strategy != "same_orientation":
                            st.write(f"🔄 Extending image to match format orientation...")
                            
                            # Calculate target ratio for the format
                            target_ratio = format_dims[0] / format_dims[1]
                            
                            # Save current image and extend it
                            temp_path = save_image_to_temp(current_image)
                            if temp_path:
                                extended_path = extend_image_directional(temp_path, target_ratio, direction, format_name)
                                if extended_path and os.path.exists(extended_path):
                                    current_image = Image.open(extended_path)
                                    processing_log.append(f"✅ Extended image {direction}ly to match {format_name} orientation")
                                    
                                    # Show intermediate result
                                    st.subheader("🔄 Step 2: Orientation Extension")
                                    st.image(current_image, caption=f"After {direction} extension", use_container_width=True)
                                    intermediate_images.append(("Orientation Extension", current_image.copy()))
                                    
                                    # Cleanup
                                    try:
                                        os.unlink(temp_path)
                                        os.unlink(extended_path)
                                    except:
                                        pass
                                else:
                                    st.warning("Failed to extend image orientation, continuing with original")
                        
                        # Step 3: Add padding if needed
                        if actions.get('padding_recommendation', {}).get('needed', False):
                            st.write(f"📦 Adding {padding_mm}mm padding...")
                            temp_path = save_image_to_temp(current_image)
                            if temp_path:
                                padded_path = extend_image_for_padding(temp_path, padding_mm, format_name)
                                if padded_path and os.path.exists(padded_path):
                                    current_image = Image.open(padded_path)
                                    processing_log.append(f"✅ Added {padding_mm}mm padding")
                                    
                                    # Show intermediate result
                                    st.subheader("📦 Step 3: Padding Added")
                                    st.image(current_image, caption=f"After {padding_mm}mm padding", use_container_width=True)
                                    intermediate_images.append(("Padding Added", current_image.copy()))
                                    
                                    # Cleanup
                                    try:
                                        os.unlink(temp_path)
                                        os.unlink(padded_path)
                                    except:
                                        pass
                        
                        # Step 4: Add bleed if needed
                        if actions.get('bleed_recommendation', {}).get('needed', False):
                            st.write(f"📏 Adding {bleed_mm}mm bleed...")
                            temp_path = save_image_to_temp(current_image)
                            if temp_path:
                                bleed_path = extend_image_for_bleed(temp_path, bleed_mm, format_name)
                                if bleed_path and os.path.exists(bleed_path):
                                    current_image = Image.open(bleed_path)
                                    processing_log.append(f"✅ Added {bleed_mm}mm bleed")
                                    
                                    # Show intermediate result
                                    st.subheader("📏 Step 4: Bleed Added")
                                    st.image(current_image, caption=f"After {bleed_mm}mm bleed", use_container_width=True)
                                    intermediate_images.append(("Bleed Added", current_image.copy()))
                                    
                                    # Cleanup
                                    try:
                                        os.unlink(temp_path)
                                        os.unlink(bleed_path)
                                    except:
                                        pass
                        
                        # Step 5: Resize to final format dimensions with DPI
                        # Calculate final dimensions including bleed
                        final_width_mm = format_dims[0] + (bleed_mm * 2) if bleed_mm > 0 else format_dims[0]
                        final_height_mm = format_dims[1] + (bleed_mm * 2) if bleed_mm > 0 else format_dims[1]
                        
                        st.write(f"📏 Resizing to final dimensions: {final_width_mm}×{final_height_mm}mm at {dpi} DPI")
                        current_image = resize_image_to_format(current_image, (final_width_mm, final_height_mm), dpi)
                        processing_log.append(f"✅ Resized to {final_width_mm}×{final_height_mm}mm at {dpi} DPI")
                        
                        # Show intermediate result
                        st.subheader("📏 Step 5: Final Sizing")
                        st.image(current_image, caption=f"Resized to {final_width_mm}×{final_height_mm}mm at {dpi}DPI", use_container_width=True)
                        intermediate_images.append(("Final Sizing", current_image.copy()))
                        
                        # Step 6: Add cut lines if bleed was added
                        if bleed_mm > 0:
                            st.write("✂️ Adding cut lines...")
                            current_image = add_cut_lines(current_image, bleed_mm, dpi, "red")
                            processing_log.append(f"✅ Added cut lines")
                            
                            # Show intermediate result
                            st.subheader("✂️ Step 6: Cut Lines Added")
                            st.image(current_image, caption="With cut lines for bleed", use_container_width=True)
                            intermediate_images.append(("Cut Lines Added", current_image.copy()))
                        
                        # Show final dimensions info
                        current_width, current_height = current_image.size
                        st.write(f"📊 Final image: {current_width}×{current_height}px")
                        
                        # Save final processed image
                        final_image_path = save_image_to_temp(current_image)
                        
                        # Create PDF using final dimensions (with bleed if applicable)
                        pdf_path = tempfile.mktemp(suffix=".pdf")
                        pdf_dimensions = (final_width_mm, final_height_mm)
                        if create_pdf_from_image(final_image_path, pdf_path, pdf_dimensions, dpi):
                            processing_log.append(f"✅ Created PDF: {format_name} at {dpi} DPI")
                            
                            # Show results
                            st.success("🎉 Processing completed successfully!")
                            
                            # Display processing log
                            st.write("**Processing Steps:**")
                            for log_entry in processing_log:
                                st.write(log_entry)
                            
                            # Show final image
                            st.subheader("🖼️ Final Processed Image")
                            st.image(current_image, caption="Ready for printing", use_container_width=True)
                            
                            # Download buttons
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                # Download PDF
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=pdf_file.read(),
                                        file_name=f"processed_{uploaded_file.name.split('.')[0]}.pdf",
                                        mime="application/pdf",
                                        type="primary",
                                        use_container_width=True
                                    )
                            
                            with col_b:
                                # Download processed image
                                img_buffer = io.BytesIO()
                                current_image.save(img_buffer, format='PNG')
                                st.download_button(
                                    label="📥 Download Image",
                                    data=img_buffer.getvalue(),
                                    file_name=f"processed_{uploaded_file.name}",
                                    mime="image/png",
                                    type="secondary",
                                    use_container_width=True
                                )
                            
                            # Cleanup
                            try:
                                os.unlink(final_image_path)
                                os.unlink(pdf_path)
                            except:
                                pass
                        else:
                            st.error("Failed to create PDF")
                        
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                
                # Cleanup
                try:
                    os.unlink(original_temp_path)
                except:
                    pass
    
    else:
        # Help section
        st.info("👆 Upload an image to get started!")
        
        st.subheader("💡 How AI Processor Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 Full Auto Mode
            1. **📤 Upload** your image
            2. **✍️ Describe** the print use case
            3. **🔍 AI analyzes** the image automatically
            4. **🤖 Agent processes** according to recommendations
            5. **📄 Download** ready-to-print PDF
            
            Perfect when you want the AI to handle everything!
            """)
        
        with col2:
            st.markdown("""
            ### ⚙️ Partially Auto Mode  
            1. **📤 Upload** your image
            2. **✍️ Describe** the print use case
            3. **📐 Set** custom format and DPI
            4. **🔍 AI analyzes** with your constraints
            5. **🤖 Agent processes** with your preferences
            6. **📄 Download** ready-to-print PDF
            
            Perfect when you have specific requirements!
            """)
        
        st.subheader("🛠️ AI Processing Capabilities")
        st.markdown("""
        The AI Processor can automatically:
        - **🔄 Rotate** images to match print orientation
        - **📏 Add bleed** for professional printing
        - **📦 Add padding** for better composition
        - **✂️ Plan cut lines** when needed
        - **📄 Generate print-ready PDFs** in standard formats
        - **🎨 Optimize DPI** for the intended use case
        
        **Powered by:**
        - 🧠 OpenAI GPT-4 Vision for intelligent analysis
        - 🎨 AI image extension for padding and bleed
        - 🤖 LangChain agents for automated processing
        - 📄 Professional PDF generation
        """)

if __name__ == "__main__":
    main() 