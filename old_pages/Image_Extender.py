import streamlit as st
from PIL import Image
import tempfile
import os
from gradio_client import Client, handle_file
import time

st.set_page_config(
    page_title="Image Extender - Print Processor",
    page_icon="🖨️",
    layout="wide"
)

def save_uploaded_image(uploaded_file):
    """Save uploaded image to temporary file for gradio_client"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def extend_image_with_gradio(image_path, width, height, overlap_percentage, num_inference_steps, 
                           resize_option, custom_resize_percentage, prompt_input, alignment,
                           overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Use gradio client to extend the image"""
    try:
        with st.spinner("🤖 Connecting to AI image extension service..."):
            client = Client("https://yntm37337cjr7u-7860.proxy.runpod.net/")
        
        with st.spinner("🎨 Extending your image... This may take a moment..."):
            # The infer function is a generator that yields intermediate results
            # We need to iterate through all results to get the final one
            result_generator = client.predict(
                image=handle_file(image_path),
                width=width,
                height=height,
                overlap_percentage=overlap_percentage,
                num_inference_steps=num_inference_steps,
                resize_option=resize_option,
                custom_resize_percentage=custom_resize_percentage,
                prompt_input=prompt_input,
                alignment=alignment,
                overlap_left=overlap_left,
                overlap_right=overlap_right,
                overlap_top=overlap_top,
                overlap_bottom=overlap_bottom,
                api_name="/infer"
            )
            
            # Debug: Log what we received
            st.write("🔍 **Debug Info:**")
            st.write(f"Result type: {type(result_generator)}")
            if hasattr(result_generator, '__len__'):
                st.write(f"Result length: {len(result_generator)}")
            
            # The API returns a tuple of (cnet_image, generated_image)
            # We want the second element which is the final generated image
            if isinstance(result_generator, (list, tuple)) and len(result_generator) >= 2:
                st.write(f"✅ Got tuple with {len(result_generator)} elements")
                st.write(f"Element 0 (cnet): {result_generator[0]}")
                st.write(f"Element 1 (generated): {result_generator[1]}")
                # Return both the intermediate (cnet) and final (generated) images
                return result_generator
            else:
                st.write(f"⚠️ Unexpected result format: {result_generator}")
                # Fallback - return the result as is
                return result_generator
        
    except Exception as e:
        st.error(f"Error during image extension: {str(e)}")
        st.error(f"Full error details: {str(e)}")
        return None

def main():
    st.title("🎨 Image Extender")
    st.markdown("Extend your images intelligently using AI to create larger compositions perfect for printing.")
    
    # File upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload the image you want to extend"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption=f"Original: {original_image.size[0]} x {original_image.size[1]} pixels", use_container_width=True)
        
        # Extension settings
        st.subheader("⚙️ Extension Settings")
        
        # Basic dimensions
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider(
                "🔲 Target Width", 
                min_value=256, 
                max_value=2048, 
                value=720, 
                step=32,
                help="Final width of the extended image"
            )
        
        with col2:
            height = st.slider(
                "📏 Target Height", 
                min_value=256, 
                max_value=2048, 
                value=1280, 
                step=32,
                help="Final height of the extended image"
            )
        
        # Advanced settings in an expander
        with st.expander("🔧 Advanced Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                overlap_percentage = st.slider(
                    "🎭 Mask Overlap (%)", 
                    min_value=0, 
                    max_value=50, 
                    value=10,
                    help="Percentage of overlap between original and extended areas"
                )
                
                num_inference_steps = st.slider(
                    "🔄 Inference Steps", 
                    min_value=1, 
                    max_value=50, 
                    value=8,
                    help="Number of denoising steps (more steps = higher quality but slower)"
                )
                
                resize_option = st.radio(
                    "📐 Resize Input Image",
                    ['Full', '50%', '33%', '25%', 'Custom'],
                    index=0,
                    help="How to resize your input image before processing"
                )
                
                if resize_option == 'Custom':
                    custom_resize_percentage = st.slider(
                        "📊 Custom Resize (%)", 
                        min_value=10, 
                        max_value=200, 
                        value=50,
                        help="Custom resize percentage when 'Custom' is selected"
                    )
                else:
                    custom_resize_percentage = 50
            
            with col2:
                alignment = st.selectbox(
                    "🎯 Alignment",
                    ['Middle', 'Left', 'Right', 'Top', 'Bottom'],
                    index=0,
                    help="How to align the original image within the extended canvas"
                )
                
                st.write("🔗 **Overlap Directions**")
                col_left, col_right = st.columns(2)
                with col_left:
                    overlap_left = st.checkbox("⬅️ Overlap Left", value=True)
                    overlap_top = st.checkbox("⬆️ Overlap Top", value=True)
                with col_right:
                    overlap_right = st.checkbox("➡️ Overlap Right", value=True)
                    overlap_bottom = st.checkbox("⬇️ Overlap Bottom", value=True)
        
        # Prompt input
        st.subheader("✍️ Style Prompt")
        prompt_input = st.text_area(
            "Describe the style or content for the extended areas",
            value="",
            placeholder="e.g., 'elegant marble texture', 'natural landscape background', 'abstract geometric pattern'...",
            help="Optional: Describe how you want the extended areas to look. Leave empty for automatic content-aware extension."
        )
        
        # Process button
        if st.button("🚀 Extend Image", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            temp_image_path = save_uploaded_image(uploaded_file)
            
            if temp_image_path:
                try:
                    # Show processing info
                    st.info(f"🎨 Extending image from {original_image.size[0]}x{original_image.size[1]} to {width}x{height} pixels...")
                    
                    # Show current parameters for debugging
                    with st.expander("🔧 Processing Parameters", expanded=False):
                        st.json({
                            "width": width,
                            "height": height,
                            "overlap_percentage": overlap_percentage,
                            "num_inference_steps": num_inference_steps,
                            "resize_option": resize_option,
                            "custom_resize_percentage": custom_resize_percentage,
                            "prompt_input": prompt_input,
                            "alignment": alignment,
                            "overlap_left": overlap_left,
                            "overlap_right": overlap_right,
                            "overlap_top": overlap_top,
                            "overlap_bottom": overlap_bottom
                        })
                    
                    # Process the image
                    result = extend_image_with_gradio(
                        temp_image_path, width, height, overlap_percentage, 
                        num_inference_steps, resize_option, custom_resize_percentage,
                        prompt_input, alignment, overlap_left, overlap_right, 
                        overlap_top, overlap_bottom
                    )
                    
                    if result:
                        st.success("✅ Image extension completed successfully!")
                        
                        # Display result
                        if isinstance(result, (list, tuple)) and len(result) >= 2:
                            # The API returns a tuple: (cnet_image_path, generated_image_path)
                            # We want the second element which is the final generated image
                            cnet_image_path = result[0]  # Control image with mask (for debugging)
                            generated_image_path = result[1]  # Final generated result
                            
                            if generated_image_path and os.path.exists(generated_image_path):
                                with col2:
                                    st.subheader("🎨 Extended Image")
                                    extended_image = Image.open(generated_image_path)
                                    st.image(extended_image, caption=f"Extended: {extended_image.size[0]} x {extended_image.size[1]} pixels", use_container_width=True)
                                    
                                    # Download button
                                    with open(generated_image_path, "rb") as file:
                                        st.download_button(
                                            label="📥 Download Extended Image",
                                            data=file.read(),
                                            file_name=f"extended_{uploaded_file.name}",
                                            mime="image/png",
                                            type="secondary",
                                            use_container_width=True
                                        )
                                        
                                    # Show size comparison
                                    st.metric(
                                        "📊 Size Increase", 
                                        f"{extended_image.size[0]}x{extended_image.size[1]}", 
                                        f"+{extended_image.size[0] - original_image.size[0]}x{extended_image.size[1] - original_image.size[1]}"
                                    )
                                    
                                # Optional: Show the control image for debugging
                                if st.checkbox("🔍 Show debug info (control image with mask)", value=False):
                                    if cnet_image_path and os.path.exists(cnet_image_path):
                                        st.subheader("🛠️ Control Image (Debug)")
                                        cnet_image = Image.open(cnet_image_path)
                                        st.image(cnet_image, caption="Control image showing masked areas", use_container_width=True)
                            else:
                                st.error("❌ Generated image file not found or inaccessible.")
                        elif isinstance(result, (list, tuple)) and len(result) == 1:
                            # Fallback for single result
                            result_path = result[0]
                            if result_path and os.path.exists(result_path):
                                with col2:
                                    st.subheader("🎨 Extended Image")
                                    extended_image = Image.open(result_path)
                                    st.image(extended_image, caption=f"Extended: {extended_image.size[0]} x {extended_image.size[1]} pixels", use_container_width=True)
                                    
                                    # Download button
                                    with open(result_path, "rb") as file:
                                        st.download_button(
                                            label="📥 Download Extended Image",
                                            data=file.read(),
                                            file_name=f"extended_{uploaded_file.name}",
                                            mime="image/png",
                                            type="secondary",
                                            use_container_width=True
                                        )
                        else:
                            st.warning("⚠️ Unexpected result format from the AI service. Please try again.")
                            st.write("Debug info:", type(result), len(result) if hasattr(result, '__len__') else 'No length')
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    try:
                        if os.path.exists(temp_image_path):
                            os.unlink(temp_image_path)
                    except:
                        pass
    
    else:
        # Help section when no image is uploaded
        st.info("👆 Upload an image above to get started!")
        
        st.subheader("💡 How it works")
        st.markdown("""
        The **Image Extender** uses advanced AI to intelligently expand your images:
        
        1. **📤 Upload** your image (PNG, JPG, JPEG)
        2. **📐 Set target dimensions** for the extended image
        3. **🎯 Choose alignment** to position your original image
        4. **🔗 Select overlap directions** to control which sides get extended
        5. **✍️ Add a style prompt** (optional) to guide the extension style
        6. **🚀 Click "Extend Image"** and let AI do the magic!
        
        **Perfect for:**
        - 🖼️ Creating larger print formats from smaller images
        - 📱 Converting portrait images to landscape (or vice versa)
        - 🎨 Adding artistic backgrounds to product photos
        - 📄 Preparing images for specific print dimensions
        - 🖨️ Print preparation with custom aspect ratios
        """)
        
        st.subheader("⚙️ Parameter Guide")
        with st.expander("🔍 Click to learn about each setting"):
            st.markdown("""
            **🔲 Target Width/Height**: Final dimensions of your extended image
            
            **🎭 Mask Overlap (%)**: How much the AI blends the original with extended areas
            - Lower values = sharper boundary
            - Higher values = smoother blending
            
            **🔄 Inference Steps**: Quality vs speed trade-off
            - More steps = higher quality but slower processing
            - 8-12 steps usually provide good results
            
            **📐 Resize Input Image**: Pre-processing option
            - Full: Use original resolution
            - 50%/33%/25%: Reduce size for faster processing
            - Custom: Specify your own percentage
            
            **🎯 Alignment**: Where to place your original image
            - Middle: Center the image
            - Left/Right/Top/Bottom: Align to specific edge
            
            **🔗 Overlap Directions**: Which sides to extend
            - Enable directions where you want new content generated
            - Disable to keep original edges intact
            
            **✍️ Style Prompt**: Guide the AI's creative process
            - Describe textures, colors, or themes
            - Leave empty for automatic content-aware extension
            """)

if __name__ == "__main__":
    main() 
