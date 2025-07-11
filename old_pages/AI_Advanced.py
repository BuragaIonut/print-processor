import streamlit as st
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import json
import fitz  # PyMuPDF for PDF processing

# Load environment variables
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(
    page_title="AI Advanced - Print Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return PIL Image (handles both images and PDFs)"""
    try:
        if uploaded_file.type == "application/pdf":
            # Handle PDF - ensure we read the bytes properly
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            
            # Check if we actually got bytes
            if not pdf_bytes:
                st.error("PDF file appears to be empty")
                return None
            
            # Reset file pointer for potential future reads
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
                # Get first page as image
                page = pdf_document[0]
                mat = fitz.Matrix(2.0, 2.0)  # High resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pdf_document.close()
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                return img
            except Exception as render_error:
                st.error(f"Could not render PDF page: {str(render_error)}")
                pdf_document.close()
                return None
        else:
            # Handle image files - also reset file pointer
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            return img
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def encode_image_from_pil(pil_image):
    """Convert PIL Image to base64 string"""
    # Convert to RGB if needed
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        pil_image = pil_image.convert('RGB')
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

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

def determine_api_extension_strategy(original_image_size, padding_mm, bleeding_mm):
    """Determine API-based extension strategy using 1024x1024, 1280x720, 720x1280 constraints"""
    
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
    
    # Determine extension sequence based on needed bleeding directions
    h_bleeding_needed = bleeding_mm > 0 or padding_mm > 0  # Always need some extension
    v_bleeding_needed = bleeding_mm > 0 or padding_mm > 0
    
    # Calculate extension sequence
    extension_sequence = []
    
    if format_category == "portrait":
        if h_bleeding_needed and v_bleeding_needed:
            # Both directions: portrait → square → portrait
            extension_sequence = [
                {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"},
                {"from": "square", "to": "portrait", "size": (720, 1280), "adds": "vertical"}
            ]
        elif h_bleeding_needed:
            # Horizontal only: portrait → square
            extension_sequence = [
                {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"}
            ]
        else:
            # Vertical only: portrait → square → portrait
            extension_sequence = [
                {"from": "portrait", "to": "square", "size": (1024, 1024), "adds": "horizontal"},
                {"from": "square", "to": "portrait", "size": (720, 1280), "adds": "vertical"}
            ]
    
    elif format_category == "landscape":
        if h_bleeding_needed and v_bleeding_needed:
            # Both directions: landscape → square → landscape
            extension_sequence = [
                {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"},
                {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"}
            ]
        elif v_bleeding_needed:
            # Vertical only: landscape → square
            extension_sequence = [
                {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"}
            ]
        else:
            # Horizontal only: landscape → square → landscape
            extension_sequence = [
                {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"},
                {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"}
            ]
    
    else:  # square
        if h_bleeding_needed and v_bleeding_needed:
            # Both directions: square → landscape → square (or alternative path)
            extension_sequence = [
                {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"},
                {"from": "landscape", "to": "square", "size": (1024, 1024), "adds": "vertical"}
            ]
        elif h_bleeding_needed:
            # Horizontal only: square → landscape
            extension_sequence = [
                {"from": "square", "to": "landscape", "size": (1280, 720), "adds": "horizontal"}
            ]
        else:
            # Vertical only: square → portrait
            extension_sequence = [
                {"from": "square", "to": "portrait", "size": (720, 1280), "adds": "vertical"}
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
        if fitted_w != api_w or fitted_h != api_h:
            preprocessing_steps.append({
                "action": "pad_to_api_dimensions", 
                "from_size": (fitted_w, fitted_h),
                "to_size": api_size,
                "padding_needed": True
            })
    
    # Calculate final crop and scale back
    bleeding_crop_factor = 1 - ((padding_mm + bleeding_mm) / 100)  # Approximate
    final_crop_size = (int(orig_w * bleeding_crop_factor), int(orig_h * bleeding_crop_factor))
    
    return {
        'original_size': (orig_w, orig_h),
        'original_aspect': orig_aspect,
        'format_category': format_category,
        'api_format': api_format,
        'api_size': api_size,
        'preprocessing_steps': preprocessing_steps,
        'extension_sequence': extension_sequence,
        'total_api_calls': len(extension_sequence),
        'final_crop_size': final_crop_size,
        'bleeding_crop_factor': bleeding_crop_factor,
        'complexity': "simple" if len(extension_sequence) == 1 else "complex"
    }

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

def analyze_image_with_openai(pil_image, use_case, target_format=None, target_dpi=300):
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
        if target_format:
            paper_format = get_format_dimensions(target_format)
            
            # AI will recommend optimal padding and bleeding
            client = OpenAI(api_key=api_key)
            base64_image = encode_image_from_pil(pil_image)
            
            # First, get AI recommendations for padding and bleeding
            initial_prompt = f"""
Analyze this image for print quality and recommend optimal padding and bleeding for {target_format} format.

Image: {width}×{height}px, {orientation} orientation
Target: {target_format} ({paper_format['width_mm']}×{paper_format['height_mm']}mm)
Use case: {use_case}

Recommend padding (0-15mm) and bleeding (0-5mm) amounts based on:
1. Image content and edge quality
2. Print use case requirements
3. Professional printing standards

Respond with only: padding_mm,bleeding_mm (e.g., "10,3")
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
            
            # Parse padding and bleeding recommendations
            try:
                rec_text = response.choices[0].message.content.strip()
                if ',' in rec_text:
                    padding_mm, bleeding_mm = map(float, rec_text.split(','))
                else:
                    padding_mm, bleeding_mm = 8.0, 3.0  # Safe defaults
            except:
                padding_mm, bleeding_mm = 8.0, 3.0  # Safe defaults
            
            # Calculate comprehensive print requirements
            print_requirements = calculate_print_requirements(paper_format, target_dpi, padding_mm, bleeding_mm)
            extension_strategy = determine_api_extension_strategy((width, height), padding_mm, bleeding_mm)
            
            # Generate detailed processing steps
            processing_steps = generate_processing_steps(
                (width, height), paper_format, print_requirements, extension_strategy, target_dpi
            )
            
                        # Create comprehensive analysis prompt
        analysis_prompt = f"""
You are a professional print preparation expert. Analyze this image for API-based extension processing.

CURRENT IMAGE:
- Dimensions: {width} × {height} pixels
- Aspect ratio: {aspect_ratio:.2f}
- Orientation: {orientation}

TARGET PRINT:
- Format: {target_format} ({paper_format['width_mm']}×{paper_format['height_mm']}mm)
- DPI: {target_dpi}
- Use case: {use_case}

API PROCESSING STRATEGY:
- Image category: {extension_strategy['format_category']}
- API format: {extension_strategy['api_format']} ({extension_strategy['api_size'][0]}×{extension_strategy['api_size'][1]}px)
- Total API calls: {extension_strategy['total_api_calls']}
- Processing complexity: {extension_strategy['complexity']}
- Crop factor: {extension_strategy['bleeding_crop_factor']:.1%} (for bleeding effect)

PRINT REQUIREMENTS:
- Final print size: {print_requirements['bleed_size_px'][0]}×{print_requirements['bleed_size_px'][1]}px at {target_dpi} DPI
- Padding: {padding_mm}mm, Bleeding: {bleeding_mm}mm

Analyze the image content and provide professional recommendations for this API-based workflow.

Respond in JSON format:
{{
    "content_description": "Brief description of image content and print suitability",
    "print_analysis": {{
        "format_compatibility": "How well the image suits {target_format} format",
        "api_processing_assessment": "Suitability for {extension_strategy['api_format']} API processing",
        "quality_impact": "Expected quality impact from {extension_strategy['total_api_calls']} API call(s)",
        "bleeding_effectiveness": "How well the {extension_strategy['bleeding_crop_factor']:.1%} crop will create bleeding effect"
    }},
    "recommended_actions": {{
        "padding_mm": {padding_mm},
        "bleeding_mm": {bleeding_mm},
        "optimal_dpi": {target_dpi},
        "api_strategy": "{extension_strategy['format_category']}_to_various",
        "processing_complexity": "{extension_strategy['complexity']}",
        "quality_concerns": ["list any quality issues from API processing"],
        "professional_notes": "Expert recommendations for this API workflow"
    }},
    "processing_summary": "One sentence summary of the API-based print preparation approach"
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
                    
                return analysis_data
            else:
                    return {"error": "Could not parse JSON response", "raw_response": analysis_text}
                    
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing error: {str(e)}", "raw_response": analysis_text}
        
        else:
            # Original analysis without specific format (for format recommendation)
            client = OpenAI(api_key=api_key)
            base64_image = encode_image_from_pil(pil_image)
            
            analysis_prompt = f"""
You are a professional print preparation expert. Analyze this image and recommend the optimal print format.

Current image details:
- Dimensions: {width} × {height} pixels  
- Aspect ratio: {aspect_ratio:.2f}
- Orientation: {orientation}
- Use case: {use_case}

Recommend the best print format and processing approach. Consider standard formats like A4, A3, A5, Letter, Postcard, Photo sizes, etc.

Respond in JSON format:
{{
    "content_description": "Brief description of what's in the image",
    "recommended_format": "Best format name (e.g., A4, Photo 5x7)",
    "format_dimensions": [width_mm, height_mm],
    "recommended_dpi": 150/300/600,
    "orientation_match": "How well image orientation matches format",
    "quality_concerns": ["list any quality issues"],
    "processing_summary": "One sentence summary"
}}
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
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = analysis_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                return {"error": "Could not parse JSON response", "raw_response": analysis_text}
                
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": analysis_text}
            
    except Exception as e:
        st.error(f"❌ Error analyzing image: {str(e)}")
        return None

def generate_processing_steps(image_size, paper_format, print_requirements, extension_strategy, dpi):
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
                    "api_calls": 0
                })
                step_counter += 1
            
            elif prep_step['action'] == 'pad_to_api_dimensions':
                from_w, from_h = prep_step['from_size']
                to_w, to_h = prep_step['to_size']
                
                steps.append({
                    "step": step_counter,
                    "action": "Pad to API dimensions",
                    "description": f"Pad from {from_w}×{from_h}px to {to_w}×{to_h}px",
                    "purpose": f"Match exact API requirements for {extension_strategy['api_format']} format",
                    "api_calls": 0
                })
                step_counter += 1
    
    # Step 2+: Extension sequence
    extension_sequence = extension_strategy['extension_sequence']
    if extension_sequence:
        for i, ext_step in enumerate(extension_sequence):
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
                "api_details": {
                    "input_format": from_format,
                    "output_format": to_format,
                    "output_resolution": f"{size[0]}×{size[1]}",
                    "content_added": adds_direction
                }
            })
            step_counter += 1
    
    # Step N: Crop with bleeding effect
    crop_w, crop_h = extension_strategy['final_crop_size']
    crop_factor = extension_strategy['bleeding_crop_factor']
    
    steps.append({
        "step": step_counter,
        "action": "Apply bleeding crop",
        "description": f"Crop center region to {crop_w}×{crop_h}px ({crop_factor:.1%} of original)",
        "purpose": "Create bleeding effect by cropping smaller than original",
        "api_calls": 0,
        "details": {
            "crop_factor": f"{crop_factor:.1%}",
            "bleeding_effect": f"{(1-crop_factor)*100:.1f}% bleeding margin"
        }
    })
    step_counter += 1
    
    # Step N+1: Scale back to original size
    steps.append({
        "step": step_counter,
        "action": "Scale to original size",
        "description": f"Scale {crop_w}×{crop_h}px back to {orig_w}×{orig_h}px",
        "purpose": "Restore original dimensions with bleeding content",
        "api_calls": 0
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
        "sub_steps": [
            f"Scale to {bleed_w}×{bleed_h}px ({dpi} DPI)",
            "Add cut lines for bleeding areas",
            "Create PDF/X-1 compliant output"
        ]
    })
    
    return steps

def display_analysis_results(analysis_data):
    """Display the comprehensive AI analysis results"""
    
    if not analysis_data:
        return
    
    if "error" in analysis_data:
        st.error(f"Analysis Error: {analysis_data['error']}")
        if "raw_response" in analysis_data:
            with st.expander("Raw Response"):
                st.text(analysis_data['raw_response'])
        return
    
    # Content Description
    st.subheader("🖼️ Image Analysis")
    st.write(f"**Content:** {analysis_data.get('content_description', 'N/A')}")
    
    # Processing Summary
    if 'processing_summary' in analysis_data:
        st.info(f"💡 **Summary:** {analysis_data['processing_summary']}")
    
    # Check if this is a comprehensive analysis (with target format) or basic analysis
    if 'calculated_requirements' in analysis_data:
        # Comprehensive Analysis with Calculations
        st.subheader("📐 Print Calculations")
        
        calc_req = analysis_data['calculated_requirements']
        ext_strategy = analysis_data['extension_strategy']
        target_format = analysis_data.get('target_format', 'Unknown')
        target_dpi = analysis_data.get('target_dpi', 300)
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
            st.metric("Format", target_format)
            st.metric("DPI", f"{target_dpi}")
        
        with col2:
            paper_w, paper_h = calc_req['paper_size_px']
            st.metric("Paper Size", f"{paper_w}×{paper_h}px")
            
        with col3:
            bleed_w, bleed_h = calc_req['bleed_size_px']
            st.metric("Bleed Size", f"{bleed_w}×{bleed_h}px")
            
        with col4:
            content_w, content_h = calc_req['content_size_px']
            st.metric("Content Area", f"{content_w}×{content_h}px")
        
        # Print Analysis Details
        if 'print_analysis' in analysis_data:
            st.subheader("🔍 Print Quality Analysis")
            print_analysis = analysis_data['print_analysis']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Format Compatibility:**")
                st.info(print_analysis.get('format_compatibility', 'N/A'))
                
                st.write("**API Processing Assessment:**")
                st.info(print_analysis.get('api_processing_assessment', 'N/A'))
            
            with col2:
                st.write("**Quality Impact:**")
                st.info(print_analysis.get('quality_impact', 'N/A'))
                
                st.write("**Bleeding Effectiveness:**")
                st.info(print_analysis.get('bleeding_effectiveness', 'N/A'))
        
        # Extension Strategy
        st.subheader("🔄 API Extension Strategy")
        format_category = ext_strategy['format_category']
        api_format = ext_strategy['api_format']
        api_size = ext_strategy['api_size']
        total_api_calls = ext_strategy['total_api_calls']
        complexity = ext_strategy['complexity']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Image Category:** {format_category.title()}")
            st.write(f"**API Format:** {api_format} ({api_size[0]}×{api_size[1]}px)")
            st.write(f"**Total API Calls:** {total_api_calls}")
            
        with col2:
            st.write(f"**Processing Complexity:** {complexity.title()}")
            crop_factor = ext_strategy['bleeding_crop_factor']
            st.write(f"**Crop Factor:** {crop_factor:.1%}")
            bleeding_effect = (1 - crop_factor) * 100
            st.write(f"**Bleeding Effect:** {bleeding_effect:.1f}%")
            
            if complexity == "simple":
                st.success("✅ Simple processing (1 API call)")
        else:
                st.info("ℹ️ Complex processing (2 API calls)")
        
        # Show extension sequence
        if ext_strategy['extension_sequence']:
            st.write("**Extension Sequence:**")
            for i, step in enumerate(ext_strategy['extension_sequence'], 1):
                st.write(f"{i}. {step['from'].title()} → {step['to'].title()} ({step['size'][0]}×{step['size'][1]}px) - adds {step['adds']} content")
        
        # Processing Steps
        if 'processing_steps' in analysis_data:
            st.subheader("📋 Detailed Processing Steps")
            
            steps = analysis_data['processing_steps']
            total_api_calls = sum(step.get('api_calls', 0) for step in steps)
            
            st.info(f"**Total API Calls Required:** {total_api_calls}")
            
            for step in steps:
                with st.expander(f"Step {step['step']}: {step['action']}", expanded=True):
                    st.write(f"**Description:** {step['description']}")
                    st.write(f"**Purpose:** {step['purpose']}")
                    
                    api_calls = step.get('api_calls', 0)
                    if api_calls > 0:
                        st.write(f"**API Calls:** {api_calls}")
                    
                    if 'sub_steps' in step:
                        st.write("**Sub-steps:**")
                        for sub_step in step['sub_steps']:
                            st.write(f"• {sub_step}")
        
        # Recommended Settings
        st.subheader("⚙️ Recommended Settings")
        actions = analysis_data.get('recommended_actions', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Padding:** {actions.get('padding_mm', 0)}mm")
            st.write(f"**Bleeding:** {actions.get('bleeding_mm', 0)}mm")
            st.write(f"**DPI:** {actions.get('optimal_dpi', target_dpi)}")
    
    with col2:
            complexity = actions.get('processing_complexity', 'moderate')
            st.write(f"**Complexity:** {complexity.title()}")
            
            if complexity == "simple":
                st.success("✅ Simple processing")
            elif complexity == "complex":
                st.warning("⚠️ Complex processing required")
        else:
                st.info("ℹ️ Moderate processing")
        
        # Professional Notes
        prof_notes = actions.get('professional_notes', '')
        if prof_notes:
            st.subheader("👨‍💼 Professional Recommendations")
            st.info(prof_notes)
            
        else:
        # Basic Analysis (Format Recommendation Mode)
        st.subheader("📄 Format Recommendation")
        
        recommended_format = analysis_data.get('recommended_format', 'Not specified')
        format_dims = analysis_data.get('format_dimensions', [])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Recommended Format:** {recommended_format}")
            if format_dims:
                st.write(f"**Dimensions:** {format_dims[0]} × {format_dims[1]} mm")
                
        with col2:
            dpi_rec = analysis_data.get('recommended_dpi', 300)
            st.write(f"**Recommended DPI:** {dpi_rec}")
            
            orientation_match = analysis_data.get('orientation_match', '')
            if orientation_match:
                st.write(f"**Orientation Match:** {orientation_match}")
    
    # Quality Concerns (for both modes)
    quality_concerns = analysis_data.get('recommended_actions', {}).get('quality_concerns', [])
    if not quality_concerns:
        quality_concerns = analysis_data.get('quality_concerns', [])
        
    if quality_concerns:
        st.subheader("⚠️ Quality Concerns")
        for concern in quality_concerns:
            st.warning(f"• {concern}")

def main():
    st.title("🤖 AI Advanced Print Analyzer")
    st.markdown("""
    This advanced tool uses OpenAI's vision capabilities to analyze your images and PDFs for optimal print processing.
    
    **Key Features:**
    - 🎯 **API-Optimized Processing**: Designed for 1024×1024, 1280×720, and 720×1280 image extension formats
    - 🔄 **Smart Extension Strategy**: Determines optimal API call sequence based on image orientation
    - 📐 **Precise Print Calculations**: Calculates exact bleeding/padding requirements for any print format
    - 📊 **Step-by-Step Workflow**: Shows detailed processing steps including API calls needed
    - 📄 **PDF Support**: Analyzes first page of PDFs for print optimization
    
    Choose between format recommendation or specific format analysis for detailed processing plans.
    """)
    
    # Check OpenAI availability
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        st.sidebar.success("✅ OpenAI API Connected")
    else:
        st.sidebar.error("❌ OpenAI API Key Required")
        st.sidebar.markdown("""
        To use AI Advanced features, you need to set your OpenAI API key:
        1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
        2. Set the `OPENAI_API_KEY` environment variable
        3. Restart the application
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload & Configure")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image or PDF file",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Upload the image or PDF you want to prepare for printing"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Show file info
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**File type:** {uploaded_file.type}")
            
            # Process and display uploaded file
            try:
                image = process_uploaded_file(uploaded_file)
                if image is None:
                    return
                
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Show image dimensions
                width, height = image.size
                st.write(f"**Dimensions:** {width} × {height} pixels")
                aspect_ratio = width / height
                orientation = "Landscape" if width > height else "Portrait" if height > width else "Square"
                st.write(f"**Orientation:** {orientation} ({aspect_ratio:.2f})")
                
                # Show source type
                if uploaded_file.type == "application/pdf":
                    st.info("📄 PDF processed - analyzing first page")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        # Use case input
        st.subheader("🎯 Print Use Case")
        use_case = st.text_input(
            "Describe how you want to print this image",
            placeholder="e.g., postcard, poster, business card, photo print, book cover...",
            help="Describe your intended print use case. Be as specific as possible."
        )
        
        # Format selection
        st.subheader("📐 Target Format (Optional)")
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["🤖 Format Recommendation", "🎯 Specific Format Analysis"],
            help="Choose whether you want AI to recommend a format or analyze for a specific format"
        )
        
        target_format = None
        target_dpi = 300
        
        if analysis_mode == "🎯 Specific Format Analysis":
            col_fmt, col_dpi = st.columns([2, 1])
            
            with col_fmt:
                format_options = [
                    "A4", "A3", "A5", "Letter", "Legal", "Postcard", "Business Card",
                    "Photo 4x6", "Photo 5x7", "Photo 8x10", "Poster A2", "Poster A1",
                    "Square 20x20", "Banner 24x36"
                ]
                target_format = st.selectbox(
                    "Select target format:",
                    format_options,
                    index=0,
                    help="Choose the specific format you want to print"
                )
            
            with col_dpi:
                target_dpi = st.selectbox(
                    "Target DPI:",
                    [150, 300, 600],
                    index=1,
                    help="Print resolution quality"
                )
            
            if target_format:
                format_dims = get_format_dimensions(target_format)
                st.info(f"📏 {target_format}: {format_dims['width_mm']}×{format_dims['height_mm']}mm at {target_dpi} DPI")
        
        # Common use case suggestions
        st.write("**Quick suggestions:**")
        suggestions = ["Postcard", "Photo Print", "Poster", "Business Card", "Flyer", "Book Cover", "Art Print", "Banner", "Document Print", "Presentation Slide"]
        
        # Display suggestions as labels
        suggestion_text = " • ".join(suggestions)
        st.caption(f"💡 Examples: {suggestion_text}")
        st.caption("📝 Be as specific as possible for better AI recommendations")
        st.caption("📄 PDFs supported - first page will be analyzed for print optimization")
    
    with col2:
        st.header("🔍 AI Analysis")
        
        if uploaded_file is not None and use_case.strip() and api_key:
            # Determine button text based on analysis mode
            if analysis_mode == "🎯 Specific Format Analysis" and target_format:
                button_text = f"🤖 Analyze for {target_format}"
                button_help = f"Analyze image for {target_format} format at {target_dpi} DPI"
            else:
                button_text = "🤖 Analyze & Recommend Format"
                button_help = "Analyze image and get format recommendations"
            
            if st.button(button_text, type="primary", help=button_help):
                with st.spinner("Analyzing image with AI..."):
                    # Load and analyze image
                    try:
                        image = process_uploaded_file(uploaded_file)
                        if image is None:
                            st.error("Failed to process uploaded file")
                            return
                            
                        # Call analysis with appropriate parameters
                        analysis_result = analyze_image_with_openai(
                            image, 
                            use_case, 
                            target_format=target_format if analysis_mode == "🎯 Specific Format Analysis" else None,
                            target_dpi=target_dpi if analysis_mode == "🎯 Specific Format Analysis" else 300
                        )
                        
                        if analysis_result:
                            # Store analysis in session state for persistence
                            st.session_state['analysis_result'] = analysis_result
                            st.session_state['analyzed_image'] = image
                            st.session_state['use_case'] = use_case
                            st.session_state['analysis_mode'] = analysis_mode
                            st.session_state['target_format'] = target_format
                            st.session_state['target_dpi'] = target_dpi
                            st.session_state['source_type'] = "PDF" if uploaded_file.type == "application/pdf" else "Image"
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        elif not api_key:
            st.warning("⚠️ OpenAI API key required for AI analysis")
        elif not uploaded_file:
            st.info("ℹ️ Please upload an image first")
        elif not use_case.strip():
            st.info("ℹ️ Please describe your print use case")
    
    # Display analysis results if available
    if 'analysis_result' in st.session_state:
        st.divider()
        
        # Show header with mode and source type
        source_type = st.session_state.get('source_type', 'Image')
        analysis_mode = st.session_state.get('analysis_mode', 'Format Recommendation')
        target_format = st.session_state.get('target_format')
        
        if target_format:
            st.header(f"📊 {target_format} Analysis Results ({source_type})")
        else:
        st.header(f"📊 Analysis Results ({source_type})")
        
        # Show analysis info
        col1, col2 = st.columns([3, 1])
        with col1:
        if source_type == "PDF":
            st.info("📄 Analysis based on first page of PDF document")
            
            if analysis_mode == "🎯 Specific Format Analysis":
                target_dpi = st.session_state.get('target_dpi', 300)
                st.info(f"🎯 Specific format analysis for {target_format} at {target_dpi} DPI")
            else:
                st.info("🤖 AI format recommendation mode")
        
        with col2:
            if st.button("🔄 New Analysis", help="Start a new analysis"):
                # Clear session state
                for key in ['analysis_result', 'analyzed_image', 'use_case', 'analysis_mode', 'target_format', 'target_dpi', 'source_type']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        display_analysis_results(st.session_state['analysis_result'])
        
        # Show calculated requirements summary if available
        if 'calculated_requirements' in st.session_state['analysis_result']:
        st.divider()
            st.success("✅ **Ready for AI Processor:** This analysis provides all the data needed for automated processing!")
            
            with st.expander("💾 Export Analysis Data"):
                st.json(st.session_state['analysis_result'], expanded=False)
        else:
            st.divider()
            st.info("💡 **Tip:** Use 'Specific Format Analysis' mode to get detailed processing calculations for the AI Processor!")

if __name__ == "__main__":
    main() 