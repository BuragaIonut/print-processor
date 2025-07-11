import streamlit as st
from PIL import Image, ImageDraw
import base64
import io
import os
import json
import subprocess
import fitz  # PyMuPDF for PDF processing
import tempfile
import shutil
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from gradio_client import Client, handle_file

# Load environment variables
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(
    page_title="Final Processor - Print Processor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def get_format_dimensions(format_name):
    """Get format dimensions in mm for standard formats"""
    formats = {
        "A4": {"width_mm": 210, "height_mm": 297},
        "A3": {"width_mm": 297, "height_mm": 420}, 
        "A5": {"width_mm": 148, "height_mm": 210},
        "Letter": {"width_mm": 216, "height_mm": 279},
        "Legal": {"width_mm": 216, "height_mm": 356},
        "Postcard": {"width_mm": 100, "height_mm": 150},
        "Business Card": {"width_mm": 85, "height_mm": 55},
        "Photo 4x6": {"width_mm": 102, "height_mm": 152},
        "Photo 5x7": {"width_mm": 127, "height_mm": 178},
        "Photo 8x10": {"width_mm": 203, "height_mm": 254},
        "Poster A2": {"width_mm": 420, "height_mm": 594},
        "Poster A1": {"width_mm": 594, "height_mm": 841},
        "Square 20x20": {"width_mm": 200, "height_mm": 200},
        "Banner 24x36": {"width_mm": 610, "height_mm": 914}
    }
    return formats.get(format_name, {"width_mm": 210, "height_mm": 297})  # Default to A4

def calculate_print_requirements(paper_format, dpi, padding_mm, bleeding_mm):
    """Calculate all print-related dimensions based on print_requirements.md methodology"""
    
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
    padding_h_percent = (padding_mm / paper_format['width_mm']) * 100
    padding_v_percent = (padding_mm / paper_format['height_mm']) * 100
    bleeding_h_percent = (bleeding_mm / paper_format['width_mm']) * 100
    bleeding_v_percent = (bleeding_mm / paper_format['height_mm']) * 100
    
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

def determine_api_extension_strategy(original_image_size, padding_mm, bleeding_mm, target_format="A4", subject_occupancy_percent=50):
    """Determine API-based extension strategy using 1024x1024, 1280x720, 720x1280 constraints
    
    Args:
        subject_occupancy_percent: How much of the image the subject occupies (0-100%)
                                 Only use 2 API calls if >90% to preserve quality
    """
    
    orig_w, orig_h = original_image_size
    orig_aspect = orig_w / orig_h
    
    # Calculate total bleeding/padding effect needed
    total_bleeding_percent = (padding_mm + bleeding_mm) / max(padding_mm + bleeding_mm, 10) * 100  # Normalize
    
    # Determine API format category
    if orig_aspect > 1.5:  # Landscape-ish
        api_format = "landscape"
        api_size = (1280, 720)
        format_category = "landscape"
    elif orig_aspect < 0.7:  # Portrait-ish  
        api_format = "portrait"
        api_size = (720, 1280)
        format_category = "portrait"
    else:  # Square-ish
        api_format = "square"
        api_size = (1024, 1024)
        format_category = "square"
    
    # Determine target format orientation
    format_dimensions = get_format_dimensions(target_format)
    target_aspect = format_dimensions['width_mm'] / format_dimensions['height_mm']
    target_is_portrait = target_aspect < 1.0
    target_is_landscape = target_aspect > 1.2
    target_is_square = not target_is_portrait and not target_is_landscape
    
    # Determine extension sequence based on target format orientation
    # Only use 2 API calls if subject occupies >90% of the image
    use_dual_api_calls = subject_occupancy_percent > 90
    
    # Calculate extension sequence - prioritize single API call for better quality
    # Strategy: extend in the direction that brings us closer to target format
    extension_sequence = []
    
    if format_category == "portrait":
        if target_is_portrait:
            # Portrait → Portrait: extend horizontally for bleeding
            extension_sequence = [
                {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"}
            ]
        elif target_is_landscape:
            # Portrait → Landscape: need significant extension
            if use_dual_api_calls and (bleeding_mm > 3 or padding_mm > 10):
                extension_sequence = [
                    {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"},
                    {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"}
                ]
            else:
                extension_sequence = [
                    {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"}
                ]
        else:  # target_is_square
            # Portrait → Square: extend horizontally
            extension_sequence = [
                {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"}
            ]
    
    elif format_category == "landscape":
        if target_is_landscape:
            # Landscape → Landscape: extend vertically for bleeding
            extension_sequence = [
                {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"}
            ]
        elif target_is_portrait:
            # Landscape → Portrait: need significant extension
            if use_dual_api_calls and (bleeding_mm > 3 or padding_mm > 10):
                extension_sequence = [
                    {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"},
                    {"from": "square", "to": "portrait", "size": (720, 1280), "adds": "vertical"}
                ]
            else:
                extension_sequence = [
                    {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"}
                ]
        else:  # target_is_square
            # Landscape → Square: extend vertically
            extension_sequence = [
                {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"}
            ]
    
    else:  # square
        if target_is_portrait:
            # Square → Portrait: extend vertically (THIS WAS THE BUG!)
            extension_sequence = [
                {"from": "square", "to": "portrait", "size": (720, 1280), "adds": "vertical"}
            ]
        elif target_is_landscape:
            # Square → Landscape: extend horizontally
            extension_sequence = [
                {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"}
            ]
        else:  # target_is_square
            # Square → Square: minimal extension for bleeding
            if use_dual_api_calls and (bleeding_mm > 3 or padding_mm > 10):
                extension_sequence = [
                    {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"},
                    {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"}
                ]
            else:
                extension_sequence = [
                    {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"}
                ]
    
    # Calculate preprocessing steps
    preprocessing_steps = []
    
    # Step 1: Resize to API format (if needed)
    api_w, api_h = api_size
    scale_to_api = min(api_w / orig_w, api_h / orig_h)
    
    if scale_to_api != 1.0:
        fitted_w = int(orig_w * scale_to_api)
        fitted_h = int(orig_h * scale_to_api)
        preprocessing_steps.append({
            "action": "resize_to_api_format",
            "from_size": (orig_w, orig_h),
            "to_size": (fitted_w, fitted_h),
            "scale_factor": scale_to_api
        })
        
        # Step 2: Pad to exact API dimensions (if needed)
        # REMOVED based on user feedback that it adds unnecessary padding
    
    # Calculate final crop size - should match target format aspect ratio, not original
    # This preserves the extended content while preparing for target format
    # (format_dimensions and target_aspect already calculated above)
    
    # Determine the last extension size to crop from
    if extension_sequence:
        last_ext_size = extension_sequence[-1]['size']
        last_ext_w, last_ext_h = last_ext_size
    else:
        last_ext_w, last_ext_h = api_size
    
    # Calculate crop size that matches target aspect ratio
    # We want to crop as little as possible while matching the target format
    if target_aspect > (last_ext_w / last_ext_h):
        # Target is wider than extended image - crop height
        crop_w = last_ext_w
        crop_h = int(crop_w / target_aspect)
    else:
        # Target is taller than extended image - crop width  
        crop_h = last_ext_h
        crop_w = int(crop_h * target_aspect)
    
    # Ensure crop size is not larger than extended image
    crop_w = min(crop_w, last_ext_w)
    crop_h = min(crop_h, last_ext_h)
    
    # Calculate bleeding factor for display
    bleeding_crop_factor = min(crop_w / orig_w, crop_h / orig_h)
    final_crop_size = (crop_w, crop_h)
    
    return {
        'original_size': (orig_w, orig_h),
        'original_aspect': orig_aspect,
        'format_category': format_category,
        'api_format': api_format,
        'api_size': api_size,
        'target_format': target_format,
        'target_aspect': target_aspect,
        'target_orientation': "portrait" if target_is_portrait else "landscape" if target_is_landscape else "square",
        'preprocessing_steps': preprocessing_steps,
        'extension_sequence': extension_sequence,
        'total_api_calls': len(extension_sequence),
        'final_crop_size': final_crop_size,
        'bleeding_crop_factor': bleeding_crop_factor,
        'complexity': "simple" if len(extension_sequence) == 1 else "complex",
        'subject_occupancy_percent': subject_occupancy_percent,
        'quality_strategy': "single_api_optimized" if len(extension_sequence) == 1 else "dual_api_high_occupancy"
    } 

def analyze_image_with_openai(pil_image, use_case, target_format, target_dpi=300):
    """Advanced print analysis using OpenAI Vision API with print_requirements.md methodology"""
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        return None
    
    try:
        # Get image properties
        width, height = pil_image.size
        aspect_ratio = width / height
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        
        # Calculate detailed print requirements
        paper_format = get_format_dimensions(target_format)
        
        # AI will recommend optimal padding and bleeding
        client = OpenAI(api_key=api_key)
        base64_image = encode_image_from_pil(pil_image)
        
        # First, get AI recommendations for padding, bleeding, and subject occupancy
        initial_prompt = f"""
Analyze this image for print quality and recommend optimal settings for {target_format} format.

Image: {width}×{height}px, {orientation} orientation
Target: {target_format} ({paper_format['width_mm']}×{paper_format['height_mm']}mm)
Use case: {use_case}

Analyze and recommend:
1. Padding (0-15mm) based on image content and edge quality
2. Bleeding (0-5mm) based on print use case requirements
3. Subject occupancy percentage (0-100%) - how much of the image area is occupied by the main subject/content

CRITICAL: Subject occupancy affects API processing quality. High occupancy (>90%) may require multiple API calls but reduces quality. Most images should use single API call strategy.

Respond with only: padding_mm,bleeding_mm,subject_occupancy_percent (e.g., "10,3,75")
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": initial_prompt},
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
            max_tokens=50,
            temperature=0.1
        )
        
        # Parse padding, bleeding, and subject occupancy recommendations
        try:
            rec_text = response.choices[0].message.content.strip()
            if ',' in rec_text:
                parts = rec_text.split(',')
                if len(parts) >= 3:
                    padding_mm, bleeding_mm, subject_occupancy_percent = map(float, parts[:3])
                elif len(parts) == 2:
                    padding_mm, bleeding_mm = map(float, parts)
                    subject_occupancy_percent = 60.0  # Default moderate occupancy
                else:
                    padding_mm, bleeding_mm, subject_occupancy_percent = 8.0, 3.0, 60.0
            else:
                padding_mm, bleeding_mm, subject_occupancy_percent = 8.0, 3.0, 60.0  # Safe defaults
        except:
            padding_mm, bleeding_mm, subject_occupancy_percent = 8.0, 3.0, 60.0  # Safe defaults
        
        # Calculate comprehensive print requirements
        print_requirements = calculate_print_requirements(paper_format, target_dpi, padding_mm, bleeding_mm)
        extension_strategy = determine_api_extension_strategy((width, height), padding_mm, bleeding_mm, target_format, subject_occupancy_percent)
        
        # Generate detailed processing steps
        processing_steps = generate_processing_steps(
            (width, height), paper_format, print_requirements, extension_strategy, target_dpi, target_format
        )
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
You are a professional print preparation expert. Analyze this image for API-based extension processing.

CURRENT IMAGE:
- Dimensions: {width} × {height} pixels
- Aspect ratio: {aspect_ratio:.2f}
- Orientation: {orientation}
- Subject occupancy: {subject_occupancy_percent:.0f}% (affects API quality strategy)

TARGET PRINT:
- Format: {target_format} ({paper_format['width_mm']}×{paper_format['height_mm']}mm)
- DPI: {target_dpi}
- Use case: {use_case}

API PROCESSING STRATEGY (QUALITY-OPTIMIZED):
- Image category: {extension_strategy['format_category']} → Target: {extension_strategy['target_orientation']}
- API format: {extension_strategy['api_format']} ({extension_strategy['api_size'][0]}×{extension_strategy['api_size'][1]}px)
- Extension strategy: {extension_strategy['format_category']} → {extension_strategy['target_orientation']} (extends toward target format)
- Total API calls: {extension_strategy['total_api_calls']} ({"single call for quality" if extension_strategy['total_api_calls'] == 1 else "dual calls for high occupancy"})
- Processing complexity: {extension_strategy['complexity']}
- Crop factor: {extension_strategy['bleeding_crop_factor']:.1%} (for bleeding effect)
- Quality strategy: {"Optimized for quality - single API call" if extension_strategy['total_api_calls'] == 1 else "High occupancy detected - dual API calls"}

PRINT REQUIREMENTS:
- Final print size: {print_requirements['bleed_size_px'][0]}×{print_requirements['bleed_size_px'][1]}px at {target_dpi} DPI
- Padding: {padding_mm}mm, Bleeding: {bleeding_mm}mm

Analyze the image content and provide professional recommendations for this quality-optimized API workflow.

Respond in JSON format:
{{
    "content_description": "Brief description of image content and print suitability",
    "print_analysis": {{
        "format_compatibility": "How well the image suits {target_format} format",
        "api_processing_assessment": "Suitability for {extension_strategy['api_format']} API processing",
        "quality_impact": "Expected quality impact from {extension_strategy['total_api_calls']} API call(s) with {subject_occupancy_percent:.0f}% subject occupancy",
        "bleeding_effectiveness": "How well the {extension_strategy['bleeding_crop_factor']:.1%} crop will create bleeding effect"
    }},
    "recommended_actions": {{
        "padding_mm": {padding_mm},
        "bleeding_mm": {bleeding_mm},
        "optimal_dpi": {target_dpi},
        "api_strategy": "{extension_strategy['format_category']}_to_various",
        "processing_complexity": "{extension_strategy['complexity']}",
        "subject_occupancy_percent": {subject_occupancy_percent:.0f},
        "quality_concerns": ["list any quality issues from API processing"],
        "professional_notes": "Expert recommendations for this quality-optimized API workflow"
    }},
    "processing_summary": "One sentence summary of the quality-optimized API-based print preparation approach"
}}
"""
        
        # Make comprehensive analysis API call
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
        
        # Parse JSON response
        try:
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                analysis_data = json.loads(json_str)
                
                # Add calculated data to response
                analysis_data['calculated_requirements'] = print_requirements
                analysis_data['extension_strategy'] = extension_strategy
                analysis_data['processing_steps'] = processing_steps
                analysis_data['target_format'] = target_format
                analysis_data['target_dpi'] = target_dpi
                analysis_data['use_case'] = use_case
                
                return analysis_data
            else:
                return {"error": "Could not parse JSON response", "raw_response": analysis_text}
                
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": analysis_text}
                
    except Exception as e:
        st.error(f"❌ Error analyzing image: {str(e)}")
        return None

def generate_processing_steps(image_size, paper_format, print_requirements, extension_strategy, dpi, target_format="A4"):
    """Generate detailed step-by-step processing instructions based on API constraints"""
    
    orig_w, orig_h = image_size
    steps = []
    step_counter = 1
    
    # Step 1: Preprocessing (resize to API format)
    preprocessing = extension_strategy['preprocessing_steps']
    if preprocessing:
        for prep_step in preprocessing:
            if prep_step['action'] == 'resize_to_api_format':
                from_w, from_h = prep_step['from_size']
                to_w, to_h = prep_step['to_size']
                scale_factor = prep_step['scale_factor']
                
                steps.append({
                    "step": step_counter,
                    "action": "Resize to API format",
                    "description": f"Resize from {from_w}×{from_h}px to {to_w}×{to_h}px ({scale_factor:.1%} scaling)",
                    "purpose": f"Prepare image for {extension_strategy['api_format']} API format",
                    "api_calls": 0,
                    "type": "preprocessing",
                    "from_size": (from_w, from_h),
                    "to_size": (to_w, to_h),
                    "scale_factor": scale_factor
                })
                step_counter += 1
    
    # Step 2: Extension steps (API calls)
    extension = extension_strategy['extension_sequence']
    if extension:
        for i, ext_step in enumerate(extension):
            from_format = ext_step['from']
            to_format = ext_step['to']
            size = ext_step['size']
            adds_direction = ext_step['adds']
            
            steps.append({
                "step": step_counter,
                "action": f"Extend {from_format} → {to_format}",
                "description": f"API call: {from_format} to {size[0]}×{size[1]}px ({to_format})",
                "purpose": f"Add {adds_direction} content for bleeding/padding",
                "api_calls": 1,
                "type": "extension",
                "api_details": {
                    "input_format": from_format,
                    "output_format": to_format,
                    "output_resolution": f"{size[0]}×{size[1]}",
                    "content_added": adds_direction
                }
            })
            step_counter += 1
    
    # Step N: Crop to target format aspect ratio (preserving extended content)
    crop_w, crop_h = extension_strategy['final_crop_size']
    crop_factor = extension_strategy['bleeding_crop_factor']
    
    # Calculate what this crop achieves
    target_aspect = crop_w / crop_h
    paper_aspect = paper_format['width_mm'] / paper_format['height_mm']
    
    steps.append({
        "step": step_counter,
        "action": "Smart crop for target format",
        "description": f"Crop to {crop_w}×{crop_h}px (aspect ratio {target_aspect:.2f} for {paper_format['width_mm']}×{paper_format['height_mm']}mm)",
        "purpose": f"Preserve extended content while matching {target_format} format proportions",
        "api_calls": 0,
        "type": "cropping",
        "details": {
            "crop_factor": f"{crop_factor:.1%}",
            "bleeding_effect": f"Extended content preserved with {target_format} aspect ratio",
            "target_aspect": target_aspect,
            "paper_aspect": paper_aspect
        }
    })
    step_counter += 1
    
    # Step N+2: Final print preparation
    bleed_w, bleed_h = print_requirements['bleed_size_px']
    steps.append({
        "step": step_counter,
        "action": "Print preparation",
        "description": f"Scale to print size: {bleed_w}×{bleed_h}px at {dpi} DPI",
        "purpose": f"Prepare final print-ready file for {paper_format['width_mm']}×{paper_format['height_mm']}mm",
        "api_calls": 0,
        "type": "finalization",
        "target_size": (bleed_w, bleed_h),
        "sub_steps": [
            f"Scale to {bleed_w}×{bleed_h}px ({dpi} DPI)",
            "Add cut lines for bleeding areas",
            "Create PDF/X-1a compliant file using Ghostscript",
            "Generate professional print-ready downloads"
        ]
    })
    
    return steps

def get_temp_folder():
    """Get or create a temporary folder for this session"""
    if 'temp_folder' not in st.session_state:
        st.session_state['temp_folder'] = tempfile.mkdtemp(prefix="final_processor_")
    return st.session_state['temp_folder']

def save_image_to_temp_folder(image, filename):
    """Save PIL image to temporary folder with specific filename"""
    try:
        temp_folder = get_temp_folder()
        filepath = os.path.join(temp_folder, filename)
        image.save(filepath, "PNG")
        
        # Verify file was created
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            st.write(f"🔍 **Debug:** Saved {filename} to {filepath} ({file_size} bytes)")
            return filepath
        else:
            st.error(f"❌ File was not created: {filepath}")
            return None
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def load_image_from_temp_folder(filename):
    """Load PIL image from temporary folder"""
    try:
        temp_folder = get_temp_folder()
        filepath = os.path.join(temp_folder, filename)
        st.write(f"🔍 **Debug:** Trying to load {filename} from {filepath}")
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            st.write(f"🔍 **Debug:** File exists, size: {file_size} bytes")
            image = Image.open(filepath)
            st.write(f"🔍 **Debug:** Loaded image size: {image.size}")
            return image
        else:
            st.error(f"Image file not found: {filename} at {filepath}")
            
            # List all files in temp folder for debugging
            if os.path.exists(temp_folder):
                files = os.listdir(temp_folder)
                st.write(f"🔍 **Debug:** Files in temp folder: {files}")
            
            return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def cleanup_temp_folder():
    """Clean up temporary folder"""
    try:
        if 'temp_folder' in st.session_state:
            temp_folder = st.session_state['temp_folder']
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
            del st.session_state['temp_folder']
    except Exception as e:
        st.warning(f"Could not clean up temp folder: {str(e)}")

def save_image_to_temp(image):
    """Save PIL image to temporary file (legacy function for compatibility)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name, "PNG")
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def create_pdf_x1(image, target_format, dpi, use_case, filename_prefix="print_ready"):
    """Create PDF/X-1a compliant file from processed image using PyMuPDF + Ghostscript"""
    try:
        # Get format dimensions
        format_dims = get_format_dimensions(target_format)
        page_width_mm = format_dims['width_mm']
        page_height_mm = format_dims['height_mm']
        
        # Convert mm to points (1 mm = 72/25.4 points)
        page_width_pt = page_width_mm * 72 / 25.4
        page_height_pt = page_height_mm * 72 / 25.4
        
        # Create PDF document with PyMuPDF
        doc = fitz.open()
        
        # Add page with exact dimensions
        page = doc.new_page(width=page_width_pt, height=page_height_pt)
        
        # Save image to temporary file for insertion
        temp_image_path = save_image_to_temp(image)
        if not temp_image_path:
            return None
        
        # Calculate image placement to fill page
        img_w, img_h = image.size
        img_aspect = img_w / img_h
        page_aspect = page_width_pt / page_height_pt
        
        # Scale image to fill page completely
        if img_aspect > page_aspect:
            # Image is wider - fit to page height
            scaled_height = page_height_pt
            scaled_width = scaled_height * img_aspect
        else:
            # Image is taller - fit to page width
            scaled_width = page_width_pt
            scaled_height = scaled_width / img_aspect
        
        # Center the image on the page
        x_offset = (page_width_pt - scaled_width) / 2
        y_offset = (page_height_pt - scaled_height) / 2
        
        # Insert image
        rect = fitz.Rect(x_offset, y_offset, x_offset + scaled_width, y_offset + scaled_height)
        page.insert_image(rect, filename=temp_image_path)
        
        # Set basic metadata
        doc.set_metadata({
            "title": f"{use_case} - {target_format} Print Ready",
            "author": "Print Processor",
            "subject": f"Print-ready {target_format} document for {use_case}",
            "keywords": f"print-ready, {target_format}, {dpi}dpi, {use_case}",
            "creator": "Print Processor - PDF/X-1a Generator",
            "producer": "PyMuPDF + Ghostscript"
        })
        
        # Save initial PDF to temporary file
        temp_pdf_path = tempfile.mktemp(suffix=".pdf")
        doc.save(temp_pdf_path, 
                 garbage=4,  # Clean up
                 deflate=True,  # Compress
                 pretty=True)  # Pretty print
        
        doc.close()
        
        # Clean up temp image
        try:
            os.unlink(temp_image_path)
        except:
            pass
        
        # Convert to PDF/X-1a using Ghostscript
        output_path = tempfile.mktemp(suffix=".pdf")
        
        gs_command = [
            "gs",
            "-dPDFX",
            "-dBATCH",
            "-dNOPAUSE",
            "-dUseCIEColor",
            "-sProcessColorModel=DeviceCMYK",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={output_path}",
            temp_pdf_path
        ]
        
        try:
            # Run Ghostscript command
            result = subprocess.run(
                gs_command,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                # Ghostscript succeeded
                st.success("✅ PDF/X-1a created successfully with Ghostscript")
                
                # Clean up temp PyMuPDF file
                try:
                    os.unlink(temp_pdf_path)
                except:
                    pass
                
                return output_path
            else:
                # Ghostscript failed, fall back to PyMuPDF version
                st.warning(f"⚠️ Ghostscript conversion failed (code {result.returncode}), using PyMuPDF version")
                if result.stderr:
                    st.write(f"Ghostscript error: {result.stderr}")
                
                # Clean up failed output
                try:
                    os.unlink(output_path)
                except:
                    pass
                
                return temp_pdf_path
                
        except subprocess.TimeoutExpired:
            st.warning("⚠️ Ghostscript conversion timed out, using PyMuPDF version")
            # Clean up failed output
            try:
                os.unlink(output_path)
            except:
                pass
            return temp_pdf_path
            
        except FileNotFoundError:
            st.warning("⚠️ Ghostscript not found in system PATH, using PyMuPDF version")
            # Clean up failed output
            try:
                os.unlink(output_path)
            except:
                pass
            return temp_pdf_path
            
        except Exception as gs_error:
            st.warning(f"⚠️ Ghostscript error: {str(gs_error)}, using PyMuPDF version")
            # Clean up failed output
            try:
                os.unlink(output_path)
            except:
                pass
            return temp_pdf_path
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def call_image_extension_api(current_image, target_width, target_height, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Call the real image extension API"""
    try:
        # Save current image to temp file for API call
        temp_image_path = save_image_to_temp(current_image)
        if not temp_image_path:
            st.error("❌ Failed to save image for API call")
            return None
        
        # Create gradio client
        client = Client("https://yntm37337cjr7u-7860.proxy.runpod.net/")
        
        st.write(f"🔗 **API Call:** Connecting to image extension service...")
        
        # API parameters
        api_params = {
            "image": temp_image_path,
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
        
        st.write("🔧 **API Parameters:**")
        st.json(api_params)
        
        # Make API call
        with st.spinner("🤖 Calling image extension API..."):
            result = client.predict(
                image=handle_file(temp_image_path),
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
        
        # Handle API response
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            result_path = result[1]
        elif isinstance(result, (list, tuple)) and len(result) == 1:
            result_path = result[0]
        else:
            result_path = result
        
        # Load result image
        if result_path and os.path.exists(result_path):
            result_image = Image.open(result_path)
            st.success("✅ API call completed successfully!")
            
            # Cleanup temp files
            try:
                os.unlink(temp_image_path)
                os.unlink(result_path)
            except:
                pass
            
            return result_image
        else:
            st.error("❌ API call failed or returned invalid result")
            return None
            
    except Exception as e:
        st.error(f"❌ API call error: {str(e)}")
        return None

def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Prepare image and mask for API extension preview"""
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
    """Create a preview image showing the extension mask"""
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

def show_crop_preview(image, crop_factor):
    """Show image with crop lines overlay"""
    width, height = image.size
    
    # Calculate crop dimensions
    crop_width = int(width * crop_factor)
    crop_height = int(height * crop_factor)
    
    # Calculate crop position (center crop)
    x_offset = (width - crop_width) // 2
    y_offset = (height - crop_height) // 2
    
    # Create preview with crop lines
    preview = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw crop rectangle
    draw.rectangle([
        (x_offset, y_offset),
        (x_offset + crop_width, y_offset + crop_height)
    ], outline=(255, 0, 0, 255), width=3)
    
    # Add semi-transparent overlay outside crop area
    # Top
    draw.rectangle([(0, 0), (width, y_offset)], fill=(0, 0, 0, 100))
    # Bottom
    draw.rectangle([(0, y_offset + crop_height), (width, height)], fill=(0, 0, 0, 100))
    # Left
    draw.rectangle([(0, y_offset), (x_offset, y_offset + crop_height)], fill=(0, 0, 0, 100))
    # Right
    draw.rectangle([(x_offset + crop_width, y_offset), (width, y_offset + crop_height)], fill=(0, 0, 0, 100))
    
    # Combine preview with overlay
    preview = Image.alpha_composite(preview, overlay)
    
    return preview, (x_offset, y_offset, x_offset + crop_width, y_offset + crop_height)

def show_scaling_comparison(original_image, target_size, description):
    """Show before/after comparison for scaling operations"""
    # Create scaled version
    scaled_image = original_image.resize(target_size, Image.LANCZOS)
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Scaling:**")
        st.image(original_image, caption=f"Original: {original_image.size[0]}×{original_image.size[1]}px", use_container_width=True)
    
    with col2:
        st.write("**After Scaling:**")
        st.image(scaled_image, caption=f"Scaled: {target_size[0]}×{target_size[1]}px", use_container_width=True)
    
    st.info(f"📐 {description}")
    
    return scaled_image

def execute_processing_step_with_visuals(step, current_image, analysis_data):
    """Execute a processing step and show visual feedback"""
    
    # Ensure we have a valid image
    if current_image is None:
        st.error("❌ No image provided for processing step")
        return None
    
    step_type = step.get('type', 'unknown')
    
    st.subheader(f"🔄 Step {step['step']}: {step['action']}")
    st.write(f"**Description:** {step['description']}")
    st.write(f"**Purpose:** {step['purpose']}")
    
    if step_type == "preprocessing":
        # Show scaling comparison for preprocessing
        if "Resize to API format" in step['action']:
            from_size = step.get('from_size', current_image.size)
            to_size = step.get('to_size', current_image.size)
            
            st.write(f"🔍 **Debug:** Resize preprocessing - from {from_size} to {to_size}")
            
            result_image = show_scaling_comparison(
                current_image, 
                to_size, 
                f"Resizing to API format: {to_size[0]}×{to_size[1]}px"
            )
            
            st.write(f"🔍 **Debug:** Resize result: {result_image.size if result_image else 'None'}")
            return result_image
        
        elif "Pad to API dimensions" in step['action']:
            # Show padding to API dimensions
            from_size = step.get('from_size', current_image.size)
            to_size = step.get('to_size', current_image.size)
            
            st.write(f"🔍 **Debug:** Pad preprocessing - from {from_size} to {to_size}")
            
            # Create padded image
            padded_image = Image.new('RGB', to_size, (255, 255, 255))
            x_offset = (to_size[0] - from_size[0]) // 2
            y_offset = (to_size[1] - from_size[1]) // 2
            padded_image.paste(current_image, (x_offset, y_offset))
            
            # Show comparison but return the actual padded image
            show_scaling_comparison(
                current_image,
                to_size,
                f"Padding to exact API dimensions: {to_size[0]}×{to_size[1]}px"
            )
            
            st.write(f"🔍 **Debug:** Pad result: {padded_image.size if padded_image else 'None'}")
            return padded_image
        
        else:
            # Catch any other preprocessing steps that don't match
            st.error(f"❌ Unknown preprocessing action: '{step['action']}'")
            st.write(f"🔍 **Debug:** Available actions: 'Resize to API format', 'Pad to API dimensions'")
            return None
    
    elif step_type == "extension":
        # Show extension preview with mask
        api_details = step.get('api_details', {})
        output_resolution = api_details.get('output_resolution', '1024×1024')
        content_added = api_details.get('content_added', 'all directions')
        
        # Parse resolution
        try:
            w_str, h_str = output_resolution.split('×')
            target_width, target_height = int(w_str), int(h_str)
        except:
            target_width, target_height = 1024, 1024
        
        # Determine overlap directions based on content added
        overlap_left = overlap_right = overlap_top = overlap_bottom = False
        
        if content_added == "horizontal":
            overlap_left = overlap_right = True
        elif content_added == "vertical":
            overlap_top = overlap_bottom = True
        else:  # all directions
            overlap_left = overlap_right = overlap_top = overlap_bottom = True
        
        st.write("**🔍 Extension Preview:**")
        
        # Show input image
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Input Image:**")
            if current_image is not None:
                st.image(current_image, caption=f"Input: {current_image.size[0]}×{current_image.size[1]}px", use_container_width=True)
            else:
                st.error("❌ No input image available")
        
        with col2:
            st.write("**Extension Mask Preview:**")
            if current_image is not None:
                mask_preview = preview_image_and_mask(
                    current_image, target_width, target_height, 5, "Full", 50, "Middle",
                    overlap_left, overlap_right, overlap_top, overlap_bottom
                )
                st.image(mask_preview, caption=f"Red areas will be extended with AI", use_container_width=True)
            else:
                st.error("❌ Cannot generate mask preview without input image")
        
        # Make real API call
        st.info(f"🤖 **Making API Call:** Calling image extension service to generate content in the red areas")
        
        result_image = call_image_extension_api(
            current_image, 
            target_width, target_height,
            overlap_left, overlap_right, overlap_top, overlap_bottom
        )
        
        if result_image is None:
            st.error("❌ API call failed, using fallback (resized original)")
            result_image = current_image.resize((target_width, target_height), Image.LANCZOS)
            
            st.write("**Fallback Result:**")
            st.image(result_image, caption=f"Fallback: {target_width}×{target_height}px", use_container_width=True)
        else:
            st.success("✅ API extension completed successfully!")
            st.write("**API Result:**")
            st.image(result_image, caption=f"Extended: {target_width}×{target_height}px", use_container_width=True)
        
        return result_image
    
    elif step_type == "cropping":
        # Smart crop that preserves extended content while matching target aspect ratio
        target_aspect = step['details'].get('target_aspect', 1.0)
        paper_aspect = step['details'].get('paper_aspect', 1.0)
        
        st.write("**🔍 Smart Crop Preview:**")
        st.write(f"**Target Aspect Ratio:** {target_aspect:.3f} (matching paper format: {paper_aspect:.3f})")
        
        # Calculate smart crop dimensions based on target aspect ratio
        img_w, img_h = current_image.size
        img_aspect = img_w / img_h
        
        if target_aspect > img_aspect:
            # Target is wider - crop height, keep width
            crop_w = img_w
            crop_h = int(crop_w / target_aspect)
        else:
            # Target is taller - crop width, keep height
            crop_h = img_h
            crop_w = int(crop_h * target_aspect)
        
        # Center the crop
        x1 = (img_w - crop_w) // 2
        y1 = (img_h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Show crop preview
        crop_preview = current_image.copy()
        draw = ImageDraw.Draw(crop_preview)
        
        # Draw crop area outline
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(2, min(img_w, img_h) // 200))
        
        # Draw semi-transparent overlay on areas being cropped
        overlay = Image.new('RGBA', current_image.size, (0, 0, 0, 100))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Fill areas outside crop with dark overlay
        if x1 > 0:
            overlay_draw.rectangle([0, 0, x1, img_h], fill=(0, 0, 0, 100))
        if x2 < img_w:
            overlay_draw.rectangle([x2, 0, img_w, img_h], fill=(0, 0, 0, 100))
        if y1 > 0:
            overlay_draw.rectangle([0, 0, img_w, y1], fill=(0, 0, 0, 100))
        if y2 < img_h:
            overlay_draw.rectangle([0, y2, img_w, img_h], fill=(0, 0, 0, 100))
        
        crop_preview = Image.alpha_composite(crop_preview.convert('RGBA'), overlay).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Smart Crop:**")
            st.image(crop_preview, caption=f"Red outline shows crop area for {target_aspect:.3f} aspect ratio", use_container_width=True)
        
        # Perform actual crop
        cropped_image = current_image.crop((x1, y1, x2, y2))
        
        with col2:
            st.write("**After Smart Crop:**")
            st.image(cropped_image, caption=f"Cropped: {cropped_image.size[0]}×{cropped_image.size[1]}px (aspect: {cropped_image.size[0]/cropped_image.size[1]:.3f})", use_container_width=True)
        
        bleeding_effect = step['details']['bleeding_effect']
        st.success(f"✅ **Smart Crop:** {bleeding_effect}")
        
        # Show what was preserved
        preserved_w = min(crop_w, img_w)
        preserved_h = min(crop_h, img_h)
        st.info(f"📐 **Preserved Content:** {preserved_w}×{preserved_h}px from extended image, matching target format aspect ratio")
        
        return cropped_image
    
    elif step_type == "scaling":
        # Show scaling comparison
        target_size = step.get('to_size', current_image.size)
        
        result_image = show_scaling_comparison(
            current_image,
            target_size,
            step['description']
        )
        return result_image
    
    elif step_type == "finalization":
        # Show final print preparation
        st.write("**📄 Final Print Preparation:**")
        
        sub_steps = step.get('sub_steps', [])
        for i, sub_step in enumerate(sub_steps, 1):
            st.write(f"{i}. {sub_step}")
        
        # Get target print dimensions from step
        target_size = step.get('target_size', current_image.size)
        
        # Show final scaling
        result_image = show_scaling_comparison(
            current_image,
            target_size,
            f"Final print size with bleed: {target_size[0]}×{target_size[1]}px"
        )
        
        # Add cut lines visualization
        st.write("**✂️ Cut Lines Preview:**")
        
        # Create cut lines preview
        cut_lines_preview = result_image.copy()
        draw = ImageDraw.Draw(cut_lines_preview)
        
        # Calculate bleed area (approximate)
        bleeding_mm = analysis_data.get('recommended_actions', {}).get('bleeding_mm', 3)
        target_dpi = analysis_data.get('target_dpi', 300)
        bleed_px = int(bleeding_mm * target_dpi / 25.4)
        
        width, height = cut_lines_preview.size
        
        # Draw cut lines (where paper should be cut)
        line_color = (255, 0, 0)
        line_width = max(2, int(target_dpi / 150))
        
        # Cut lines at bleed boundary
        cut_left = bleed_px
        cut_right = width - bleed_px
        cut_top = bleed_px
        cut_bottom = height - bleed_px
        
        # Draw cut lines
        draw.line([(cut_left, cut_top), (cut_right, cut_top)], fill=line_color, width=line_width)  # Top
        draw.line([(cut_left, cut_bottom), (cut_right, cut_bottom)], fill=line_color, width=line_width)  # Bottom
        draw.line([(cut_left, cut_top), (cut_left, cut_bottom)], fill=line_color, width=line_width)  # Left
        draw.line([(cut_right, cut_top), (cut_right, cut_bottom)], fill=line_color, width=line_width)  # Right
        
        st.image(cut_lines_preview, caption="Red lines show where to cut after printing", use_container_width=True)
        
        # Create PDF/X-1 for download
        st.write("**📄 PDF/X-1 Creation:**")
        
        # Get metadata from analysis
        target_format = analysis_data.get('target_format', 'A4')
        target_dpi = analysis_data.get('target_dpi', 300)
        use_case = analysis_data.get('use_case', 'Print')
        
        with st.spinner("Creating PDF/X-1 compliant file..."):
            pdf_path = create_pdf_x1(result_image, target_format, target_dpi, use_case)
        
        if pdf_path:
            st.success("✅ PDF/X-1 file created successfully!")
            
            # Read PDF for download
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            
            # Create filename
            pdf_filename = f"print_ready_{target_format}_{target_dpi}dpi.pdf"
            
            # Download button
            st.download_button(
                label="📥 Download PDF/X-1 Print-Ready File",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                help="Download the PDF/X-1 compliant file ready for professional printing"
            )
            
            # Show PDF info
            format_dims = get_format_dimensions(target_format)
            st.info(f"📋 **PDF Details:**\n"
                   f"- Format: PDF/X-1a:2001\n"
                   f"- Page Size: {target_format} ({format_dims['width_mm']}×{format_dims['height_mm']}mm)\n"
                   f"- Resolution: {target_dpi} DPI\n"
                   f"- Use Case: {use_case}")
            
            # Clean up temp PDF
            try:
                os.unlink(pdf_path)
            except:
                pass
        else:
            st.error("❌ Failed to create PDF/X-1 file")
        
        return result_image
    
    else:
        # Default: just return the current image
        st.info("📋 This step doesn't require visual changes")
        st.write(f"🔍 **Debug:** Unknown step type '{step_type}' for action '{step['action']}'")
        return current_image 

def main():
    st.title("🎯 Final Processor")
    st.markdown("""
    **Visual Step-by-Step Print Processing with PDF/X-1 Output**
    
    This page combines comprehensive AI analysis with visual step-by-step execution, showing how your image evolves through each processing stage and creates professional print-ready files.
    
    **Features:**
    - 🔍 **Comprehensive Analysis**: Detailed print analysis with API-optimized processing strategy
    - 👁️ **Visual Feedback**: See how your image changes at each step
    - 📐 **Scaling Previews**: Before/after comparisons for all scaling operations
    - ✂️ **Crop Visualization**: See exactly where crops will be applied
    - 🎨 **Extension Previews**: Input/mask/result visualization for API calls
    - 📋 **Step-by-Step Execution**: Process one step at a time or run all automatically
    - 📄 **PDF/X-1 Output**: Professional print-ready PDF files with exact format compliance
    """)
    
    # Check OpenAI availability
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        st.sidebar.success("✅ OpenAI API Connected")
    else:
        st.sidebar.error("❌ OpenAI API Key Required")
        st.sidebar.markdown("""
        To use Final Processor, you need to set your OpenAI API key:
        1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
        2. Set the `OPENAI_API_KEY` environment variable
        3. Restart the application
        """)
        return
    
    # Main interface - single column layout
    st.header("📤 Upload & Configure")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="Upload the image or PDF you want to process for printing"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Process uploaded file
        try:
            image = process_uploaded_file(uploaded_file)
            if image is None:
                return
            
            if uploaded_file.type == "application/pdf":
                st.info("📄 PDF processed - analyzing first page")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    # Configuration
    if uploaded_file is not None:
        st.subheader("⚙️ Print Configuration")
        
        # Format selection
        format_options = [
            "A4", "A3", "A5", "Letter", "Legal", "Postcard", "Business Card",
            "Photo 4x6", "Photo 5x7", "Photo 8x10", "Poster A2", "Poster A1",
            "Square 20x20", "Banner 24x36"
        ]
        target_format = st.selectbox(
            "Target Format",
            format_options,
            index=0,
            help="Choose the print format you want to target"
        )
        
        # DPI selection
        target_dpi = st.selectbox(
            "Target DPI",
            [150, 300, 600],
            index=1,
            help="Print resolution quality"
        )
        
        # Use case (optional)
        use_case = st.text_input(
            "Print use case (optional)",
            placeholder="e.g., Book Cover, Postcard, Poster, Flyer...",
            help="Optional: Helps AI provide better analysis, but not required"
        )
        
        if not use_case.strip():
            st.caption("💡 Use case is optional - analysis will use 'General Print' if left empty")
        
        # Show format info
        format_dims = get_format_dimensions(target_format)
        st.info(f"📏 {target_format}: {format_dims['width_mm']}×{format_dims['height_mm']}mm at {target_dpi} DPI")
        
        # Analysis section moved here
        st.subheader("🔍 Analysis & Processing")
        
        # Analysis button
        if st.button("🤖 Analyze Image", type="primary", use_container_width=True):
            # Clean up any previous temp files
            cleanup_temp_folder()
            
            # Use default use case if empty
            analysis_use_case = use_case.strip() if use_case.strip() else "General Print"
            
            with st.spinner("Analyzing image with AI..."):
                analysis_result = analyze_image_with_openai(image, analysis_use_case, target_format, target_dpi)
            
            if not analysis_result or "error" in analysis_result:
                st.error("Failed to analyze image. Please try again.")
                if "raw_response" in analysis_result:
                    with st.expander("Error Details"):
                        st.text(analysis_result["raw_response"])
                return
            
            # Store analysis in session state and save original image to temp folder
            st.session_state['final_analysis'] = analysis_result
            st.session_state['final_use_case'] = analysis_use_case
            st.session_state['final_format'] = target_format
            st.session_state['final_dpi'] = target_dpi
            
            # Save original image to temp folder
            original_image_path = save_image_to_temp_folder(image, "original.png")
            if original_image_path:
                st.session_state['final_original_image_path'] = original_image_path
                st.success("✅ Analysis completed!")
            else:
                st.error("❌ Failed to save original image")
        
        # Processing Status - moved under analyze button
        if uploaded_file is not None:
            st.subheader("📋 Processing Status")
            
            if 'final_analysis' not in st.session_state:
                st.info("👆 Click 'Analyze Image' to start")
            else:
                st.success("✅ Ready for processing!")
                
                # Show quick processing status
                analysis_data = st.session_state['final_analysis']
                processing_steps = analysis_data.get('processing_steps', [])
                
                if processing_steps:
                    st.write(f"**Ready to execute {len(processing_steps)} processing steps:**")
                    
                    # Show all steps in a compact format
                    for step in processing_steps:
                        step_type = step.get('type', 'unknown')
                        api_calls = step.get('api_calls', 0)
                        
                        if api_calls > 0:
                            st.write(f"🤖 Step {step['step']}: {step['action']} *({api_calls} API call)*")
                        else:
                            st.write(f"📐 Step {step['step']}: {step['action']}")
                    
                    st.info("💡 Scroll down to start visual processing!")
                else:
                    st.warning("⚠️ No processing steps generated")
        
        # Show analysis results if available
        if 'final_analysis' in st.session_state:
            st.divider()
            
            analysis_data = st.session_state['final_analysis']
            
            # Quick summary
            st.subheader("📊 Analysis Summary")
            
            calc_req = analysis_data.get('calculated_requirements', {})
            ext_strategy = analysis_data.get('extension_strategy', {})
            actions = analysis_data.get('recommended_actions', {})
            
            # Metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                padding_mm = actions.get('padding_mm', 0)
                st.metric("Padding", f"{padding_mm}mm")
            
            with col_b:
                bleeding_mm = actions.get('bleeding_mm', 0)
                st.metric("Bleeding", f"{bleeding_mm}mm")
            
            with col_c:
                total_api_calls = ext_strategy.get('total_api_calls', 0)
                quality_strategy = ext_strategy.get('quality_strategy', 'unknown')
                api_color = "🟢" if total_api_calls == 1 else "🟡"
                st.metric("API Calls", f"{api_color} {total_api_calls}", help="🟢 Single call (quality optimized) | 🟡 Dual calls (high occupancy)")
            
            with col_d:
                subject_occupancy = actions.get('subject_occupancy_percent', ext_strategy.get('subject_occupancy_percent', 0))
                st.metric("Subject Fill", f"{subject_occupancy:.0f}%", help="How much of the image area is occupied by the main subject")
            
            # Processing summary
            processing_summary = analysis_data.get('processing_summary', '')
            if processing_summary:
                st.info(f"💡 **Strategy:** {processing_summary}")
            
            # Show original image under analysis summary
            st.subheader("📸 Original Image")
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Show image info
            width, height = image.size
            st.write(f"**Dimensions:** {width} × {height} pixels")
            aspect_ratio = width / height
            orientation = "Landscape" if width > height else "Portrait" if height > width else "Square"
            st.write(f"**Orientation:** {orientation} ({aspect_ratio:.2f})")
            
            # Complete processing steps overview
            processing_steps = analysis_data.get('processing_steps', [])
            if processing_steps:
                st.subheader(f"📋 Complete Processing Plan ({len(processing_steps)} steps)")
                
                for step in processing_steps:  # Show all steps
                    step_type = step.get('type', 'unknown')
                    api_calls = step.get('api_calls', 0)
                    
                    if api_calls > 0:
                        st.write(f"🤖 **Step {step['step']}:** {step['action']} *({api_calls} API call)*")
                    else:
                        st.write(f"📐 **Step {step['step']}:** {step['action']}")
                    
                    st.caption(f"   {step['description']}")
                    st.caption(f"   Purpose: {step['purpose']}")
                    
                    if step_type == 'finalization':
                        sub_steps = step.get('sub_steps', [])
                        if sub_steps:
                            for i, sub_step in enumerate(sub_steps, 1):
                                st.caption(f"     {i}. {sub_step}")

    # Visual Processing Section (full width)
    if 'final_analysis' in st.session_state:
        st.divider()
        st.header("🎬 Visual Step-by-Step Processing")
        
        analysis_data = st.session_state['final_analysis']
        
        # Load original image from temp folder
        original_image_path = st.session_state.get('final_original_image_path')
        if not original_image_path or not os.path.exists(original_image_path):
            st.error("❌ Original image not found. Please re-analyze the image.")
            return
            
        original_image = load_image_from_temp_folder("original.png")
        if original_image is None:
            st.error("❌ Failed to load original image from temp folder.")
            return
            
        processing_steps = analysis_data.get('processing_steps', [])
        
        if not processing_steps:
            st.error("No processing steps found in analysis")
            return
        
        # Processing mode selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            processing_mode = st.radio(
                "Processing Mode:",
                ["🔄 Step by Step", "🚀 Run All Steps"],
                index=0,
                horizontal=True
            )
        
        with col2:
            if st.button("🔄 Reset Processing", help="Start over with original image"):
                # Clear processing step files from temp folder
                temp_folder = get_temp_folder()
                for filename in os.listdir(temp_folder):
                    if filename.startswith('step_') and filename.endswith('.png'):
                        try:
                            os.remove(os.path.join(temp_folder, filename))
                        except:
                            pass
                
                # Clear any processing state from session state
                for key in list(st.session_state.keys()):
                    if key.startswith('final_step_'):
                        del st.session_state[key]
                st.rerun()
        
        if processing_mode == "🔄 Step by Step":
            # Step-by-step processing
            st.subheader("🎯 Execute Steps One by One")
            
            # Show step selector
            step_numbers = [f"Step {step['step']}: {step['action']}" for step in processing_steps]
            selected_step_index = st.selectbox(
                "Select step to execute:",
                range(len(processing_steps)),
                format_func=lambda x: step_numbers[x],
                help="Choose which step to execute and visualize"
            )
            
            selected_step = processing_steps[selected_step_index]
            step_key = f"final_step_{selected_step['step']}"
            
            # Show step details
            with st.expander(f"📋 Step {selected_step['step']} Details", expanded=True):
                st.write(f"**Action:** {selected_step['action']}")
                st.write(f"**Description:** {selected_step['description']}")
                st.write(f"**Purpose:** {selected_step['purpose']}")
                st.write(f"**Type:** {selected_step.get('type', 'unknown')}")
                
                # Debug: Show step data
                if selected_step.get('type') == 'preprocessing':
                    st.write(f"**From Size:** {selected_step.get('from_size', 'Not set')}")
                    st.write(f"**To Size:** {selected_step.get('to_size', 'Not set')}")
                    st.write(f"**Scale Factor:** {selected_step.get('scale_factor', 'Not set')}")
                
                if selected_step.get('api_calls', 0) > 0:
                    st.warning(f"⚠️ This step requires {selected_step['api_calls']} API call(s)")
            
            # Execute step button
            if st.button(f"▶️ Execute Step {selected_step['step']}", type="primary"):
                with st.spinner(f"Executing step {selected_step['step']}..."):
                    # Get current image (either original or result from previous step)
                    if selected_step['step'] == 1:
                        current_image = original_image
                        st.info(f"🔍 **Debug:** Using original image for step 1: {current_image.size}")
                    else:
                        # Look for previous step result in temp folder
                        prev_step_filename = f"step_{selected_step['step'] - 1}.png"
                        prev_step_key = f"final_step_{selected_step['step'] - 1}"
                        st.write(f"🔍 **Debug:** Looking for previous step result: {prev_step_filename}")
                        st.write(f"🔍 **Debug:** Previous step key in session: {prev_step_key in st.session_state}")
                        
                        if prev_step_key in st.session_state:
                            current_image = load_image_from_temp_folder(prev_step_filename)
                            if current_image is None:
                                current_image = original_image
                                st.warning("⚠️ Previous step result not found. Using original image.")
                                st.write(f"🔍 **Debug:** Failed to load {prev_step_filename}, using original")
                            else:
                                st.success(f"✅ Loaded previous step result: {current_image.size}")
                        else:
                            current_image = original_image
                            st.warning("⚠️ Previous step not executed. Using original image.")
                            st.write(f"🔍 **Debug:** No session state for {prev_step_key}, using original")
                    
                    # Ensure we have a valid image
                    if current_image is None:
                        st.error("❌ No valid image available for processing")
                        return
                    
                    # Debug: Show what image we're processing
                    st.write(f"🔍 **Debug:** Processing with image size: {current_image.size if current_image else 'None'}")
                    
                    # Execute step with visuals
                    result_image = execute_processing_step_with_visuals(
                        selected_step, 
                        current_image, 
                        analysis_data
                    )
                    
                    # Debug: Show result
                    st.write(f"🔍 **Debug:** Step result image size: {result_image.size if result_image else 'None'}")
                    
                    # Ensure result is valid before storing
                    if result_image is not None:
                        # Save result to temp folder
                        step_filename = f"step_{selected_step['step']}.png"
                        step_filepath = save_image_to_temp_folder(result_image, step_filename)
                        if step_filepath:
                            st.session_state[step_key] = step_filename  # Store filename, not image
                            st.success(f"✅ Step {selected_step['step']} completed! Saved as {step_filename}")
                            st.write(f"🔍 **Debug:** Saved to: {step_filepath}")
                        else:
                            st.error(f"❌ Failed to save step {selected_step['step']} result")
                    else:
                        st.error(f"❌ Step {selected_step['step']} failed to produce a result")
            
            # Show executed steps
            executed_steps = []
            for step in processing_steps:
                step_key = f"final_step_{step['step']}"
                if step_key in st.session_state:
                    executed_steps.append(step['step'])
            
            if executed_steps:
                st.write(f"**✅ Executed Steps:** {', '.join(map(str, executed_steps))}")
                
                # Show progression
                if len(executed_steps) > 1:
                    st.subheader("🔄 Processing Progression")
                    
                    cols = st.columns(min(len(executed_steps), 4))
                    for i, step_num in enumerate(executed_steps[:4]):
                        step_filename = f"step_{step_num}.png"
                        step_image = load_image_from_temp_folder(step_filename)
                        with cols[i]:
                            if step_image is not None:
                                st.image(
                                    step_image, 
                                    caption=f"After Step {step_num}",
                                    use_container_width=True
                                )
                            else:
                                st.error(f"Step {step_num} image not found")
                    
                    if len(executed_steps) > 4:
                        st.write(f"... and {len(executed_steps) - 4} more steps executed")
        
        else:
            # Run all steps automatically
            st.subheader("🚀 Automatic Processing")
            
            if st.button("🚀 Execute All Steps", type="primary", use_container_width=True):
                current_image = original_image
                
                # Ensure we have a valid starting image
                if current_image is None:
                    st.error("❌ No valid original image available for processing")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, step in enumerate(processing_steps):
                    progress = (i + 1) / len(processing_steps)
                    progress_bar.progress(progress)
                    status_text.text(f"Executing Step {step['step']}: {step['action']}")
                    
                    # Execute step with visuals
                    result_image = execute_processing_step_with_visuals(
                        step, 
                        current_image, 
                        analysis_data
                    )
                    
                    # Check if step was successful
                    if result_image is None:
                        st.error(f"❌ Step {step['step']} failed. Stopping processing.")
                        break
                    
                    current_image = result_image
                    
                    # Save intermediate result to temp folder
                    step_filename = f"step_{step['step']}.png"
                    step_filepath = save_image_to_temp_folder(current_image, step_filename)
                    if step_filepath:
                        step_key = f"final_step_{step['step']}"
                        st.session_state[step_key] = step_filename  # Store filename, not image
                    else:
                        st.error(f"❌ Failed to save step {step['step']} result. Stopping processing.")
                        break
                    
                    st.divider()
                
                progress_bar.progress(1.0)
                status_text.text("✅ All steps completed!")
                
                st.success("🎉 Processing completed successfully!")

    # Help section
    if 'final_analysis' not in st.session_state:
        st.divider()
        st.subheader("💡 How Final Processor Works")
        
        st.markdown("""
        ### 🎯 Visual Processing Pipeline
        
        **1. Comprehensive Analysis** 🔍
        - AI analyzes your image for the target print format
        - Calculates optimal padding and bleeding requirements
        - Determines API-optimized processing strategy
        - Generates detailed step-by-step processing plan
        
        **2. Visual Step Execution** 👁️
        - **Preprocessing**: See before/after scaling to API formats
        - **Smart Extension**: AI extends in the direction that matches your target format (e.g., square → portrait for A4)
        - **Extension**: View input image, extension mask, and actual API results
        - **Cropping**: Visualize crop areas with red outlines and dark overlays
        - **Scaling**: Compare original vs scaled images side-by-side
        - **Finalization**: Preview cut lines and final print preparation
        
        **3. Processing Modes** ⚙️
        - **Step by Step**: Execute and visualize one step at a time
        - **Run All**: Automatic execution with progress tracking
        
        **4. Key Visual Features** 🎨
        - 📐 **Scaling Comparisons**: Before/after views for all resize operations
        - ✂️ **Crop Previews**: Red outlines show exact crop areas
        - 🎨 **Extension Masks**: Red areas show where AI will add content
        - 🧠 **Smart Extension**: AI automatically extends in the direction that matches your target format
        - 📄 **Cut Lines**: Red lines show professional cutting guides
        - 🔄 **Progression Views**: See how image evolves through processing
        
        **5. Professional Output** 📄
        - 📸 **High-Quality Images**: Download processed images in PNG format
        - 📄 **PDF/X-1a Compliance**: Professional print-ready PDF files using Ghostscript for maximum compatibility
        - 🎯 **Print Optimization**: Exact format dimensions with bleeding
        - ⚡ **Instant Download**: Ready for professional printing services
        """)

if __name__ == "__main__":
    main() 