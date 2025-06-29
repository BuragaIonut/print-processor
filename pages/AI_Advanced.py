import streamlit as st
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import json

# Load environment variables
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(
    page_title="AI Advanced - Print Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def analyze_image_with_openai(pil_image, use_case):
    """Analyze image using OpenAI Vision API and determine print processing needs"""
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Encode image
        base64_image = encode_image_from_pil(pil_image)
        
        # Get image dimensions
        width, height = pil_image.size
        aspect_ratio = width / height
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        
        # Create analysis prompt
        analysis_prompt = f"""
You are a professional print preparation expert. Analyze this image for printing as: "{use_case}"

Current image details:
- Dimensions: {width} × {height} pixels
- Aspect ratio: {aspect_ratio:.2f}
- Orientation: {orientation}

Please analyze the image and determine the optimal print processing steps. Consider:

1. **Rotation**: Does the image need rotation to match the intended print format?
2. **Format Recommendation**: What print format (A4, A3, A5, Letter, postcard size, etc.) would work best?
3. **Orientation Match**: Does the image orientation match the typical orientation for this use case?
4. **Bleed Requirements**: How much bleed is recommended for this type of print job?
5. **Padding Needs**: Would padding improve the print quality or aesthetics?
6. **Cut Lines**: Are cut lines necessary for this print job?
7. **Quality Concerns**: Any image quality issues that might affect printing?
8. **Content Analysis**: What's the main subject/content and how does it affect print decisions?

Provide your analysis in this JSON format:
{{
    "content_description": "Brief description of what's in the image",
    "recommended_actions": {{
        "rotation_needed": true/false,
        "rotation_degrees": 0/90/180/270,
        "rotation_reason": "explanation",
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

        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 with vision capabilities
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
        
        # Parse response
        analysis_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                analysis_data = json.loads(json_str)
                return analysis_data
            else:
                # If no JSON found, return the raw text
                return {"error": "Could not parse JSON response", "raw_response": analysis_text}
                
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": analysis_text}
            
    except Exception as e:
        st.error(f"❌ Error analyzing image: {str(e)}")
        return None

def display_analysis_results(analysis_data):
    """Display the AI analysis results in a user-friendly format"""
    
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
    
    if 'confidence_score' in analysis_data:
        confidence = analysis_data['confidence_score']
        st.metric("AI Confidence", f"{confidence:.1%}")
    
    # Processing Summary
    if 'processing_summary' in analysis_data:
        st.info(f"💡 **Summary:** {analysis_data['processing_summary']}")
    
    # Recommended Actions
    st.subheader("🎯 Recommended Processing Steps")
    
    actions = analysis_data.get('recommended_actions', {})
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Rotation
        st.write("### 🔄 Rotation")
        if actions.get('rotation_needed', False):
            rotation_degrees = actions.get('rotation_degrees', 0)
            rotation_reason = actions.get('rotation_reason', 'No reason provided')
            st.success(f"✅ Rotate {rotation_degrees}°")
            st.caption(f"Reason: {rotation_reason}")
        else:
            st.info("ℹ️ No rotation needed")
        
        # Format Recommendation
        st.write("### 📄 Print Format")
        recommended_format = actions.get('recommended_format', 'Not specified')
        format_dims = actions.get('format_dimensions', [])
        st.write(f"**Format:** {recommended_format}")
        if format_dims:
            st.write(f"**Dimensions:** {format_dims[0]} × {format_dims[1]} mm")
        
        # DPI Recommendation
        st.write("### 🎨 Print Quality")
        dpi_rec = actions.get('dpi_recommendation', 300)
        st.write(f"**Recommended DPI:** {dpi_rec}")
        
        dpi_descriptions = {
            150: "Good for draft prints",
            300: "Standard high quality",
            600: "Premium quality"
        }
        if dpi_rec in dpi_descriptions:
            st.caption(dpi_descriptions[dpi_rec])
    
    with col2:
        # Bleed Recommendation
        st.write("### 📏 Bleed Settings")
        bleed = actions.get('bleed_recommendation', {})
        if bleed.get('needed', False):
            bleed_amount = bleed.get('amount_mm', 3)
            bleed_type = bleed.get('type', 'content_aware')
            bleed_reason = bleed.get('reason', 'No reason provided')
            st.success(f"✅ Add {bleed_amount}mm bleed ({bleed_type})")
            st.caption(f"Reason: {bleed_reason}")
        else:
            st.info("ℹ️ No bleed needed")
        
        # Padding Recommendation
        st.write("### 📦 Padding Settings")
        padding = actions.get('padding_recommendation', {})
        if padding.get('needed', False):
            padding_amount = padding.get('amount_mm', 0)
            padding_style = padding.get('style', 'content_aware')
            padding_reason = padding.get('reason', 'No reason provided')
            st.success(f"✅ Add {padding_amount}mm padding ({padding_style})")
            st.caption(f"Reason: {padding_reason}")
        else:
            st.info("ℹ️ No padding needed")
        
        # Cut Lines
        st.write("### ✂️ Cut Lines")
        cut_lines = actions.get('cut_lines', {})
        if cut_lines.get('needed', False):
            cut_reason = cut_lines.get('reason', 'No reason provided')
            st.success("✅ Add cut lines")
            st.caption(f"Reason: {cut_reason}")
        else:
            st.info("ℹ️ No cut lines needed")
    
    # Quality Concerns
    quality_concerns = actions.get('quality_concerns', [])
    if quality_concerns:
        st.subheader("⚠️ Quality Concerns")
        for concern in quality_concerns:
            st.warning(f"• {concern}")
    
    # Additional Notes
    additional_notes = actions.get('additional_notes', '')
    if additional_notes:
        st.subheader("📝 Additional Recommendations")
        st.info(additional_notes)

def main():
    st.title("🤖 AI Advanced Print Analyzer")
    st.markdown("**Intelligent print preparation using AI image analysis**")
    
    st.markdown("""
    This advanced tool uses OpenAI's vision capabilities to analyze your image and automatically determine 
    the best print processing steps based on your intended use case.
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
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload the image you want to prepare for printing"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Show file info
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**File type:** {uploaded_file.type}")
            
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Show image dimensions
                width, height = image.size
                st.write(f"**Dimensions:** {width} × {height} pixels")
                aspect_ratio = width / height
                orientation = "Landscape" if width > height else "Portrait" if height > width else "Square"
                st.write(f"**Orientation:** {orientation} ({aspect_ratio:.2f})")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return
        
        # Use case input
        st.subheader("🎯 Print Use Case")
        use_case = st.text_input(
            "Describe how you want to print this image",
            placeholder="e.g., postcard, poster, business card, photo print, book cover...",
            help="Describe your intended print use case. Be as specific as possible."
        )
        
        # Common use case suggestions
        st.write("**Quick suggestions:**")
        suggestions = ["Postcard", "Photo Print", "Poster", "Business Card", "Flyer", "Book Cover", "Art Print", "Banner"]
        
        # Create buttons for quick selection
        cols = st.columns(4)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 4]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    use_case = suggestion
                    st.rerun()
    
    with col2:
        st.header("🔍 AI Analysis")
        
        if uploaded_file is not None and use_case.strip() and api_key:
            if st.button("🤖 Analyze Image", type="primary"):
                with st.spinner("Analyzing image with AI..."):
                    # Load and analyze image
                    try:
                        image = Image.open(uploaded_file)
                        analysis_result = analyze_image_with_openai(image, use_case)
                        
                        if analysis_result:
                            # Store analysis in session state for persistence
                            st.session_state['analysis_result'] = analysis_result
                            st.session_state['analyzed_image'] = image
                            st.session_state['use_case'] = use_case
                            
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
        st.header("📊 Analysis Results")
        display_analysis_results(st.session_state['analysis_result'])
        
        # Future: Add buttons to apply recommendations automatically
        st.divider()
        st.info("🚀 **Coming Soon:** Automatic application of AI recommendations to the Processor page!")

if __name__ == "__main__":
    main() 