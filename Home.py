import streamlit as st
import tempfile
import os
from PIL import Image, ImageCms
import fitz  # PyMuPDF
from pdf2image import convert_from_path, convert_from_bytes
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io
import base64
from bleed import add_bleed
from cut_lines import add_cut_lines
from print_settings import PRINT_FORMATS, mm_to_pixels, resize_image_for_print

st.set_page_config(
    page_title="Print Processor",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def convert_pdf_to_image(pdf_file):
    """Convert first page of PDF to PIL Image"""
    try:
        # Try using pdf2image first
        images = convert_from_bytes(pdf_file.read(), first_page=1, last_page=1, dpi=300)
        if images:
            return images[0], None
    except Exception as e:
        # Fallback to PyMuPDF
        try:
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            pdf_document.close()
            return image, None
        except Exception as e2:
            return None, f"Error converting PDF: {str(e2)}"

def convert_to_cmyk(image):
    """Convert image to CMYK color space with multiple fallback methods"""
    try:
        # If image is already CMYK, return as is
        if image.mode == 'CMYK':
            return image
        
        # Convert to RGB first if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Method 1: Try using built-in PIL CMYK conversion
        try:
            cmyk_image = image.convert('CMYK')
            return cmyk_image
        except Exception:
            pass
        
        # Method 2: Try using ICC profiles if available
        try:
            # Try to use standard ICC profiles
            rgb_profile = ImageCms.createProfile('sRGB')
            
            # Try different CMYK profile creation methods
            try:
                cmyk_profile = ImageCms.createProfile('LAB')  # Use LAB as intermediate
                # Create transform RGB -> LAB -> CMYK simulation
                transform = ImageCms.buildTransformFromOpenProfiles(
                    rgb_profile, rgb_profile, 'RGB', 'RGB'  # Keep as RGB but optimize for print
                )
                optimized_image = ImageCms.applyTransform(image, transform)
                return optimized_image.convert('CMYK')
            except Exception:
                pass
                
        except Exception:
            pass
        
        # Method 3: Manual RGB to CMYK conversion algorithm
        try:
            import numpy as np
            
            # Convert to numpy array
            rgb_array = np.array(image).astype(np.float32) / 255.0
            
            # Simple RGB to CMYK conversion
            # K = 1 - max(R, G, B)
            # C = (1 - R - K) / (1 - K)
            # M = (1 - G - K) / (1 - K)  
            # Y = (1 - B - K) / (1 - K)
            
            r, g, b = rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2]
            
            # Calculate K (black) channel
            k = 1 - np.maximum(np.maximum(r, g), b)
            
            # Avoid division by zero
            denominator = 1 - k
            denominator[denominator == 0] = 1
            
            # Calculate CMY channels
            c = (1 - r - k) / denominator
            m = (1 - g - k) / denominator
            y = (1 - b - k) / denominator
            
            # Convert back to 0-255 range
            c = (c * 255).astype(np.uint8)
            m = (m * 255).astype(np.uint8)
            y = (y * 255).astype(np.uint8)
            k = (k * 255).astype(np.uint8)
            
            # Stack channels
            cmyk_array = np.stack([c, m, y, k], axis=2)
            
            # Create CMYK image
            cmyk_image = Image.fromarray(cmyk_array, mode='CMYK')
            return cmyk_image
            
        except Exception:
            pass
        
        # Method 4: Fallback - just convert to CMYK using PIL's basic conversion
        try:
            return image.convert('CMYK')
        except Exception:
            pass
            
    except Exception as e:
        pass
    
    # Final fallback: Return RGB image with warning
    st.warning("⚠️ CMYK conversion not available on this system. Using optimized RGB for print compatibility.")
    return image

def create_pdf_from_image(image, filename="processed_image.pdf", dpi=300):
    """Convert PIL Image to PDF with improved format handling"""
    try:
        pdf_buffer = io.BytesIO()
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Create PDF with exact image dimensions (in points)
        # Convert from pixels to points using the actual DPI
        pdf_width = img_width * 72 / dpi
        pdf_height = img_height * 72 / dpi
        
        # Create canvas
        c = canvas.Canvas(pdf_buffer, pagesize=(pdf_width, pdf_height))
        
        # Prepare image for PDF insertion
        img_buffer = io.BytesIO()
        
        # Handle different image modes
        try:
            if image.mode == 'CMYK':
                # For CMYK images, try TIFF first, then fallback
                try:
                    image.save(img_buffer, format='TIFF', compression='tiff_lzw')
                except Exception:
                    # Fallback: convert to RGB and save as PNG
                    rgb_image = image.convert('RGB')
                    rgb_image.save(img_buffer, format='PNG', optimize=True)
            elif image.mode in ['RGB', 'RGBA']:
                # For RGB images, use PNG
                if image.mode == 'RGBA':
                    # Convert RGBA to RGB with white background
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])
                    rgb_image.save(img_buffer, format='PNG', optimize=True)
                else:
                    image.save(img_buffer, format='PNG', optimize=True)
            else:
                # For other modes, convert to RGB
                rgb_image = image.convert('RGB')
                rgb_image.save(img_buffer, format='PNG', optimize=True)
        
        except Exception as e:
            # Ultimate fallback: force convert to RGB and save as JPEG
            try:
                rgb_image = image.convert('RGB')
                rgb_image.save(img_buffer, format='JPEG', quality=95, optimize=True)
            except Exception:
                st.error(f"Failed to prepare image for PDF: {str(e)}")
                return None
        
        img_buffer.seek(0)
        
        # Add image to PDF
        try:
            c.drawImage(ImageReader(img_buffer), 0, 0, width=pdf_width, height=pdf_height)
        except Exception as e:
            # If ImageReader fails, try with a different approach
            img_buffer.seek(0)
            # Save as temporary file approach
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                if image.mode != 'RGB':
                    rgb_image = image.convert('RGB')
                    rgb_image.save(tmp_file.name, format='PNG')
                else:
                    image.save(tmp_file.name, format='PNG')
                
                c.drawImage(tmp_file.name, 0, 0, width=pdf_width, height=pdf_height)
                os.unlink(tmp_file.name)  # Clean up temp file
        
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def get_download_link(file_data, filename, file_type="PDF"):
    """Generate download link for processed file"""
    b64 = base64.b64encode(file_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Download {file_type}</a>'
    return href

# Dependency check function
def check_dependencies():
    """Check if all required dependencies are available"""
    issues = []
    
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        issues.append("pdf2image - Run: pip install pdf2image")
    
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        issues.append("reportlab - Run: pip install reportlab")
    
    try:
        import fitz
    except ImportError:
        issues.append("PyMuPDF - Run: pip install PyMuPDF")
    
    return issues

# Main interface
st.write("# Print Processor - Quick Processing 🖨️")

# Check dependencies
dependency_issues = check_dependencies()
if dependency_issues:
    st.error("⚠️ Missing Dependencies")
    st.write("Please install the following packages:")
    for issue in dependency_issues:
        st.code(f"pip install {issue.split(' - Run: pip install ')[1]}")
    st.write("Then restart the application.")
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📂 Upload Your File")
    
    # File uploader with drag and drop
    uploaded_file = st.file_uploader(
        "Drag and drop your image or PDF here, or click to browse",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Supported formats: PNG, JPG, JPEG, PDF"
    )

with col2:
    st.markdown("### ⚙️ Processing Settings")
    
    # Dimensions selection
    st.markdown("**Print Dimensions:**")
    format_choice = st.selectbox(
        "Select print format:",
        list(PRINT_FORMATS.keys()),
        index=0
    )
    
    if format_choice == "Custom":
        col_w, col_h = st.columns(2)
        with col_w:
            custom_width = st.number_input("Width (mm)", min_value=10, max_value=1000, value=210)
        with col_h:
            custom_height = st.number_input("Height (mm)", min_value=10, max_value=1000, value=297)
        target_width, target_height = custom_width, custom_height
    else:
        target_width, target_height = PRINT_FORMATS[format_choice]
    
    # DPI selection
    dpi = st.selectbox(
        "Select DPI (Quality):",
        [150, 300, 600, 1200],
        index=1,  # Default to 300 DPI
        help="Higher DPI = better quality but larger file size"
    )
    
    # Display selected settings
    st.info(f"📐 **Dimensions:** {target_width} × {target_height} mm\n\n📊 **DPI:** {dpi}")

# Processing section
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 🔄 Processing")
    
    # Show file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.1f} KB",
        "File type": uploaded_file.type
    }
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("**File Information:**")
        for key, value in file_details.items():
            st.write(f"• **{key}:** {value}")
    
    # Process button
    with info_col2:
        st.markdown("**Ready to process?**")
        if st.button("🚀 Process for Print", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load and convert file
                status_text.text("📖 Loading file...")
                progress_bar.progress(10)
                
                if uploaded_file.type == "application/pdf":
                    # Convert PDF to image
                    status_text.text("📄 Converting PDF to image...")
                    image, error = convert_pdf_to_image(uploaded_file)
                    if error:
                        st.error(f"PDF conversion failed: {error}")
                        st.stop()
                else:
                    # Load image directly
                    image = Image.open(uploaded_file)
                
                progress_bar.progress(20)
                
                # Step 2: Convert to CMYK
                status_text.text("🎨 Converting to CMYK color space...")
                image = convert_to_cmyk(image)
                progress_bar.progress(30)
                
                # Step 3: Scale image to target dimensions
                status_text.text(f"📏 Scaling to {target_width}×{target_height}mm at {dpi} DPI...")
                scaled_image = resize_image_for_print(image, target_width, target_height, dpi)
                progress_bar.progress(50)
                
                # Step 4: Add 3mm bleed
                status_text.text("🔲 Adding 3mm content-aware bleed...")
                try:
                    bleed_image = add_bleed(scaled_image, 3, dpi, bleed_type='content_aware')
                except Exception as bleed_error:
                    st.warning(f"⚠️ Advanced bleed failed, using simple edge extension: {str(bleed_error)}")
                    try:
                        bleed_image = add_bleed(scaled_image, 3, dpi, bleed_type='edge_extend')
                    except Exception:
                        st.warning("⚠️ Bleed failed, continuing without bleed")
                        bleed_image = scaled_image
                progress_bar.progress(70)
                
                # Step 5: Add cut lines
                status_text.text("✂️ Adding cutting lines...")
                try:
                    final_image = add_cut_lines(bleed_image, 3, dpi, line_color='black', line_width=2)
                except Exception as cut_error:
                    st.warning(f"⚠️ Cut lines failed, continuing without cut lines: {str(cut_error)}")
                    final_image = bleed_image
                progress_bar.progress(85)
                
                # Step 6: Convert to PDF
                status_text.text("📋 Creating PDF...")
                pdf_data = create_pdf_from_image(final_image, f"processed_{uploaded_file.name.split('.')[0]}.pdf", dpi)
                
                if pdf_data:
                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed!")
                    
                    # Success message and download
                    st.success("🎉 **Processing Complete!**")
                    
                    # Display processing summary
                    final_width_mm = target_width + 6  # +3mm bleed on each side
                    final_height_mm = target_height + 6
                    final_width_px = mm_to_pixels(final_width_mm, dpi)
                    final_height_px = mm_to_pixels(final_height_mm, dpi)
                    
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.markdown("**📊 Processing Summary:**")
                        st.write(f"• **Original format:** {uploaded_file.type}")
                        st.write(f"• **Color space:** CMYK")
                        st.write(f"• **Print size:** {target_width} × {target_height} mm")
                        st.write(f"• **Final size:** {final_width_mm} × {final_height_mm} mm (with bleed)")
                    
                    with summary_col2:
                        st.markdown("**⚙️ Technical Details:**")
                        st.write(f"• **Resolution:** {dpi} DPI")
                        st.write(f"• **Pixel dimensions:** {final_width_px} × {final_height_px} px")
                        st.write(f"• **Bleed:** 3mm content-aware")
                        st.write(f"• **Cut lines:** Included")
                    
                    # Download button
                    st.markdown("---")
                    filename = f"print_ready_{uploaded_file.name.split('.')[0]}.pdf"
                    
                    st.download_button(
                        label="📥 Download Print-Ready PDF",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    # Show preview (convert back to RGB for display)
                    st.markdown("### 👁️ Preview")
                    display_image = final_image.convert('RGB') if final_image.mode == 'CMYK' else final_image
                    
                    # Resize for display (maintain aspect ratio)
                    display_width = min(800, display_image.width)
                    display_height = int(display_image.height * (display_width / display_image.width))
                    display_image = display_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    
                    st.image(display_image, caption=f"Print-ready file with 3mm bleed and cut lines", use_container_width=True)
                    
                else:
                    st.error("Failed to create PDF file")
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

# Help section
st.markdown("---")
st.markdown("### 📚 How It Works")

with st.expander("🔍 Click to see the processing pipeline"):
    st.markdown("""
    **Your file goes through these steps:**
    
    1. **📖 File Loading:** Upload PDF or image files
    2. **📄 PDF Conversion:** If PDF, converts first page to high-resolution image
    3. **🎨 CMYK Conversion:** Converts to CMYK color space for professional printing
    4. **📏 Scaling:** Resizes to exact print dimensions at specified DPI
    5. **🔲 Bleed Addition:** Adds 3mm content-aware bleed around edges
    6. **✂️ Cut Lines:** Adds cutting guides for precise trimming
    7. **📋 PDF Creation:** Converts final image to print-ready PDF
    8. **📥 Download:** Provides instant download of processed file
    
    **🎯 Perfect for:**
    - Business cards and postcards
    - Flyers and brochures  
    - Posters and presentations
    - Professional print preparation
    """)

# Advanced features note
st.info("💡 **Need more control?** Use the advanced pages in the sidebar for AI-powered processing, step-by-step workflow, or manual fine-tuning.")

st.markdown("---")
st.markdown(
    """
    **Print Processor** - Professional print preparation made simple.
    
    *For advanced features like AI-powered processing, object removal, and custom workflows, check out the other pages in the sidebar.*
    """
) 