import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Import from our modules
from print_settings import PRINT_FORMATS, mm_to_inches, resize_image_for_print, mm_to_pixels
from bleed import add_bleed
from cut_lines import add_cut_lines

# Set page config
st.set_page_config(
    page_title="Step by Step - Print Processor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    if pil_image.mode == 'RGB':
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif pil_image.mode == 'RGBA':
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
    else:
        # Convert to RGB first
        rgb_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    if len(cv2_image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(cv2_image)

def detect_main_object(image, method="contours"):
    """Detect the main object/logo in the image using OpenCV"""
    cv_image = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    if method == "contours":
        # Use edge detection and contours to find main object
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the main object)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Create result image with bounding box
            result_image = cv_image.copy()
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            return cv2_to_pil(result_image), (x, y, w, h)
    
    elif method == "threshold":
        # Use adaptive thresholding for better logo detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area to remove noise
            min_area = (gray.shape[0] * gray.shape[1]) * 0.01  # At least 1% of image
            large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if large_contours:
                largest_contour = max(large_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                result_image = cv_image.copy()
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                
                return cv2_to_pil(result_image), (x, y, w, h)
    
    elif method == "grabcut":
        # Use GrabCut algorithm for more sophisticated object detection
        mask = np.zeros(gray.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle around the center area (assuming object is roughly centered)
        height, width = gray.shape
        rect = (width//4, height//4, width//2, height//2)
        
        cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask for foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Find bounding box of the detected object
        coords = np.column_stack(np.where(mask2 > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            result_image = cv_image.copy()
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            
            return cv2_to_pil(result_image), (x_min, y_min, x_max - x_min, y_max - y_min)
    
    elif method == "sobel_extremes":
        # Custom method: Use Sobel edge detection to find extreme edge points
        # Apply Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        
        # Combine gradients
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        sobel_combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Apply threshold to get binary edge image
        edge_threshold = 30  # Adjustable threshold for edge sensitivity
        _, binary_edges = cv2.threshold(sobel_combined, edge_threshold, 255, cv2.THRESH_BINARY)
        
        # Find all white pixels (edge pixels)
        edge_points = np.column_stack(np.where(binary_edges > 0))
        
        if len(edge_points) > 0:
            # Find extreme points
            y_coords = edge_points[:, 0]
            x_coords = edge_points[:, 1]
            
            x_min = np.min(x_coords)
            x_max = np.max(x_coords)
            y_min = np.min(y_coords)
            y_max = np.max(y_coords)
            
            # Calculate bounding rectangle
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            
            # Create result image with visualization
            result_image = cv_image.copy()
            
            # Draw the main bounding rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 165, 0), 3)  # Orange rectangle
            
            # Mark the extreme points
            cv2.circle(result_image, (x_min, int((y_min + y_max) / 2)), 8, (0, 255, 255), -1)  # Left extreme (cyan)
            cv2.circle(result_image, (x_max, int((y_min + y_max) / 2)), 8, (0, 255, 255), -1)  # Right extreme (cyan)
            cv2.circle(result_image, (int((x_min + x_max) / 2), y_min), 8, (255, 0, 255), -1)  # Top extreme (magenta)
            cv2.circle(result_image, (int((x_min + x_max) / 2), y_max), 8, (255, 0, 255), -1)  # Bottom extreme (magenta)
            
            # Add text annotations
            cv2.putText(result_image, f'Sobel Extremes', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(result_image, f'Size: {w}x{h}', (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            cv2.putText(result_image, f'Edge Points: {len(edge_points)}', (x, y-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            return cv2_to_pil(result_image), (x, y, w, h), cv2_to_pil(binary_edges)
    
    return image, None

def detect_edges(image, method="canny"):
    """Detect edges in the image using various OpenCV methods"""
    cv_image = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    if method == "canny":
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
    elif method == "sobel":
        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
    elif method == "laplacian":
        # Laplacian edge detection
        edges = cv2.Laplacian(gray, cv2.CV_8U)
        
    # Create colored edge image for better visualization
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    
    return cv2_to_pil(edges), cv2_to_pil(edges_colored)

def detect_outer_contour(image, method="adaptive"):
    """Detect only the outer contour of the main object"""
    cv_image = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    if method == "adaptive":
        # Use adaptive thresholding for better contour detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
    elif method == "otsu":
        # Use Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    elif method == "canny":
        # Use Canny edge detection for contour finding
        thresh = cv2.Canny(gray, 50, 150)
        
    # Find contours - only external ones
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create result image
    result_image = cv_image.copy()
    contour_info = None
    
    if contours:
        # Filter contours by area to remove noise
        min_area = (gray.shape[0] * gray.shape[1]) * 0.005  # At least 0.5% of image area
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if large_contours:
            # Find the largest contour (main object)
            largest_contour = max(large_contours, key=cv2.contourArea)
            
            # Draw the outer contour
            cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 3)
            
            # Get contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calculate contour properties
            total_area = gray.shape[0] * gray.shape[1]
            area_percentage = (area / total_area) * 100
            
            # Approximate contour to get number of vertices
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            contour_info = {
                'area': area,
                'perimeter': perimeter,
                'area_percentage': area_percentage,
                'vertices': len(approx),
                'bounding_box': (x, y, w, h),
                'aspect_ratio': w / h if h > 0 else 0
            }
            
            # Add text annotations
            cv2.putText(result_image, f'Area: {area:.0f}px', (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, f'Vertices: {len(approx)}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return cv2_to_pil(result_image), contour_info, cv2_to_pil(thresh)

def rotate_image_precise(img, angle):
    """Rotate image by specified angle with high quality"""
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
        # For custom angles, use rotate with expand=True
        return img.rotate(-angle, expand=True, resample=Image.LANCZOS, fillcolor='white')

def main():
    st.title("⚙️ Step by Step Processing")
    st.markdown("**Individual control over each processing step**")
    
    st.markdown("""
    This page allows you to work through each processing step individually, giving you complete control 
    and the ability to fine-tune each aspect of your print preparation.
    """)
    
    # Initialize session state for processed images
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    # Sidebar controls
    st.sidebar.header("🎛️ Processing Controls")
    
    # Reset button
    if st.sidebar.button("🔄 Reset to Original", help="Reset to the original uploaded image"):
        if st.session_state.original_image:
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.processing_history = []
            st.rerun()
    
    # Show processing history
    if st.session_state.processing_history:
        st.sidebar.subheader("📋 Processing History")
        for i, step in enumerate(st.session_state.processing_history, 1):
            st.sidebar.write(f"{i}. {step}")
    
    # Main upload section
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload the image you want to process step by step"
    )
    
    if uploaded_file is not None:
        # Load and store original image
        try:
            original_image = Image.open(uploaded_file)
            if original_image.mode in ('RGBA', 'LA', 'P'):
                original_image = original_image.convert('RGB')
                
            st.session_state.original_image = original_image
            if st.session_state.current_image is None:
                st.session_state.current_image = original_image.copy()
            
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Show current image
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(st.session_state.current_image, caption="Current Image", use_container_width=True)
            
            with col2:
                width, height = st.session_state.current_image.size
                st.write(f"**Dimensions:** {width} × {height} px")
                aspect_ratio = width / height
                orientation = "Landscape" if width > height else "Portrait" if height > width else "Square"
                st.write(f"**Orientation:** {orientation}")
                st.write(f"**Aspect Ratio:** {aspect_ratio:.2f}")
                
                # File size estimation
                img_buffer = io.BytesIO()
                st.session_state.current_image.save(img_buffer, format='PNG')
                size_kb = len(img_buffer.getvalue()) / 1024
                st.write(f"**Est. Size:** {size_kb:.1f} KB")
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return
    
    if st.session_state.current_image is None:
        st.info("👆 Please upload an image to start processing")
        return
    
    # Processing sections
    st.divider()
    
    # Section 1: Object Detection
    with st.expander("🎯 Object Detection", expanded=False):
        st.markdown("**Identify the main object or logo in your image**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            detection_method = st.selectbox(
                "Detection Method",
                ["contours", "threshold", "grabcut", "sobel_extremes"],
                format_func=lambda x: {
                    "contours": "Contours (General)",
                    "threshold": "Threshold (Logos/Text)",
                    "grabcut": "GrabCut (Complex Objects)",
                    "sobel_extremes": "Sobel Extremes (Edge-based)"
                }[x],
                help="Choose the best method for your image type"
            )
            
            if st.button("🔍 Detect Object", key="detect_obj"):
                with st.spinner("Detecting main object..."):
                    result = detect_main_object(st.session_state.current_image, detection_method)
                    
                    # Handle different return formats
                    if detection_method == "sobel_extremes" and len(result) == 3:
                        detected_image, bbox, sobel_edges = result
                        show_sobel_edges = True
                    else:
                        detected_image, bbox = result[0], result[1] if len(result) > 1 else None
                        sobel_edges = None
                        show_sobel_edges = False
                    
                    # Display results
                    if show_sobel_edges:
                        # For Sobel Extremes method, show both detection result and edge image
                        st.subheader("Sobel Extremes Detection Results")
                        tab1, tab2 = st.tabs(["Detection Result", "Sobel Edges"])
                        
                        with tab1:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(detected_image, caption="Extremal Points Detection", use_container_width=True)
                                st.caption("🟠 Orange: Bounding rectangle | 🔵 Cyan: Left/Right extremes | 🟣 Magenta: Top/Bottom extremes")
                            
                            with col_b:
                                if bbox:
                                    x, y, w, h = bbox
                                    st.success("✅ Extreme points detected!")
                                    st.write(f"**Bounding Rectangle:**")
                                    st.write(f"- Position: ({x}, {y})")
                                    st.write(f"- Size: {w} × {h} px")
                                    st.write(f"- Area: {(w * h) / (st.session_state.current_image.size[0] * st.session_state.current_image.size[1]) * 100:.1f}% of image")
                                    
                                    st.write(f"**Extreme Points:**")
                                    st.write(f"- Left edge: x = {x}")
                                    st.write(f"- Right edge: x = {x + w}")
                                    st.write(f"- Top edge: y = {y}")
                                    st.write(f"- Bottom edge: y = {y + h}")
                                    
                                    # Analysis specific to Sobel method
                                    st.write(f"**Edge Analysis:**")
                                    st.info("📊 This method finds the outermost edge pixels detected by Sobel gradient analysis")
                                    
                                    # Best fit suggestions
                                    img_w, img_h = st.session_state.current_image.size
                                    obj_center_x = x + w // 2
                                    obj_center_y = y + h // 2
                                    img_center_x = img_w // 2
                                    img_center_y = img_h // 2
                                    
                                    st.write(f"**Best Fit Analysis:**")
                                    if abs(obj_center_x - img_center_x) < img_w * 0.1 and abs(obj_center_y - img_center_y) < img_h * 0.1:
                                        st.info("✅ Object is well-centered")
                                    else:
                                        st.warning("⚠️ Object is off-center - consider cropping or repositioning")
                                else:
                                    st.warning("⚠️ No edge points detected. Image may be too smooth or uniform.")
                        
                        with tab2:
                            st.image(sobel_edges, caption="Sobel Edge Detection (Binary)", use_container_width=True)
                            st.caption("White pixels represent detected edges that were used to find extreme points")
                    
                    else:
                        # For other methods, show single result
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(detected_image, caption="Object Detection Result", use_container_width=True)
                        
                        with col_b:
                            if bbox:
                                x, y, w, h = bbox
                                st.success("✅ Object detected!")
                                st.write(f"**Bounding Box:**")
                                st.write(f"- Position: ({x}, {y})")
                                st.write(f"- Size: {w} × {h} px")
                                st.write(f"- Area: {(w * h) / (st.session_state.current_image.size[0] * st.session_state.current_image.size[1]) * 100:.1f}% of image")
                                
                                # Best fit suggestions
                                img_w, img_h = st.session_state.current_image.size
                                obj_center_x = x + w // 2
                                obj_center_y = y + h // 2
                                img_center_x = img_w // 2
                                img_center_y = img_h // 2
                                
                                st.write(f"**Best Fit Analysis:**")
                                if abs(obj_center_x - img_center_x) < img_w * 0.1 and abs(obj_center_y - img_center_y) < img_h * 0.1:
                                    st.info("✅ Object is well-centered")
                                else:
                                    st.warning("⚠️ Object is off-center - consider cropping or repositioning")
                                
                                # Suggest optimal crop
                                margin = 50  # pixels
                                crop_x1 = max(0, x - margin)
                                crop_y1 = max(0, y - margin)
                                crop_x2 = min(img_w, x + w + margin)
                                crop_y2 = min(img_h, y + h + margin)
                                
                                st.write(f"**Suggested Crop:** ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
                                
                            else:
                                st.warning("⚠️ No clear object detected. Try a different method.")
        
        with col2:
            st.info("""
            **Detection Methods:**
            
            🔹 **Contours** - Good for objects with clear edges
            🔹 **Threshold** - Best for logos, text, and high-contrast objects  
            🔹 **GrabCut** - Advanced method for complex objects
            🔹 **Sobel Extremes** - 🆕 Custom edge-based bounding detection
            
            **Sobel Extremes Method:**
            - Uses Sobel gradient detection to find edges
            - Locates the most extreme edge points (leftmost, rightmost, topmost, bottommost)
            - Creates tight bounding rectangle from these extremes
            - Perfect for logos and objects with defined boundaries
            - Shows actual edge pixels used for detection
            
            **When to Use Sobel Extremes:**
            ✅ High-contrast logos or graphics
            ✅ Objects with clear, defined edges
            ✅ When you need the tightest possible bounding box
            ✅ Images with minimal background noise
            
            **Tips:**
            - Use high-contrast images for better detection
            - Ensure the main object is clearly visible
            - Try different methods if first attempt fails
            - Sobel Extremes works best with sharp, well-defined objects
            """)
            
            if detection_method == "sobel_extremes":
                st.warning("""
                **Sobel Extremes Specific Notes:**
                - Finds outermost edge pixels using gradient analysis
                - May include background edges if object boundaries aren't clear
                - Edge threshold is set to 30 (adjustable in code)
                - Best results with logos, text, and geometric shapes
                """)
    
    # Section 2: Edge Detection
    with st.expander("🔲 Edge Detection", expanded=False):
        st.markdown("**Detect and analyze edges to understand image structure**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔍 Edge Detection")
            edge_method = st.selectbox(
                "Edge Detection Method",
                ["canny", "sobel", "laplacian"],
                format_func=lambda x: {
                    "canny": "Canny (Balanced)",
                    "sobel": "Sobel (Directional)",
                    "laplacian": "Laplacian (All Directions)"
                }[x],
                help="Choose edge detection algorithm"
            )
            
            if st.button("🔍 Detect Edges", key="detect_edges"):
                with st.spinner("Detecting edges..."):
                    edges_bw, edges_colored = detect_edges(st.session_state.current_image, edge_method)
                    
                    st.subheader("Edge Detection Results")
                    tab1, tab2 = st.tabs(["Black & White", "Colored"])
                    
                    with tab1:
                        st.image(edges_bw, caption="B&W Edge Detection", use_container_width=True)
                    
                    with tab2:
                        st.image(edges_colored, caption="Colored Edge Visualization", use_container_width=True)
            
            st.divider()
            
            st.subheader("🔲 Outer Contour Detection")
            contour_method = st.selectbox(
                "Contour Detection Method",
                ["adaptive", "otsu", "canny"],
                format_func=lambda x: {
                    "adaptive": "Adaptive Threshold (Best for most images)",
                    "otsu": "Otsu Threshold (High contrast images)",
                    "canny": "Canny Edges (Sharp boundaries)"
                }[x],
                help="Choose method for finding the outer contour",
                key="contour_method"
            )
            
            if st.button("🔲 Detect Contour", key="detect_contour"):
                with st.spinner("Detecting outer contour..."):
                    contour_result, contour_info, threshold_image = detect_outer_contour(st.session_state.current_image, contour_method)
                    
                    st.subheader("Outer Contour Results")
                    tab1, tab2 = st.tabs(["Contour Detection", "Threshold Used"])
                    
                    with tab1:
                        st.image(contour_result, caption="Outer Contour (Green) + Bounding Box (Blue)", use_container_width=True)
                        
                        if contour_info:
                            # Display contour analysis
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Area", f"{contour_info['area']:.0f} px²")
                                st.metric("Perimeter", f"{contour_info['perimeter']:.0f} px")
                                st.metric("Vertices", contour_info['vertices'])
                            
                            with col_b:
                                st.metric("Area %", f"{contour_info['area_percentage']:.1f}%")
                                st.metric("Aspect Ratio", f"{contour_info['aspect_ratio']:.2f}")
                                x, y, w, h = contour_info['bounding_box']
                                st.metric("Bounds", f"{w}×{h} at ({x},{y})")
                            
                            # Shape analysis
                            vertices = contour_info['vertices']
                            if vertices <= 4:
                                shape_type = "Simple geometric shape"
                            elif vertices <= 8:
                                shape_type = "Moderate complexity"
                            else:
                                shape_type = "Complex/organic shape"
                            
                            st.info(f"**Shape Analysis:** {shape_type} ({vertices} vertices)")
                            
                            # Contour quality assessment
                            area_pct = contour_info['area_percentage']
                            if area_pct > 50:
                                st.warning("⚠️ Contour covers >50% of image - may include background")
                            elif area_pct < 5:
                                st.warning("⚠️ Contour is very small - may be noise or partial detection")
                            else:
                                st.success("✅ Good contour detection - appropriate size")
                        else:
                            st.warning("⚠️ No clear outer contour detected. Try a different method.")
                    
                    with tab2:
                        st.image(threshold_image, caption="Threshold Image Used for Contour Detection", use_container_width=True)
                        st.caption("White areas are detected as foreground objects")
        
        with col2:
            st.info("""
            **Edge Detection Uses:**
            
            🔹 **Quality Assessment** - Check for blur or noise
            🔹 **Object Boundaries** - Find precise edges for cropping
            🔹 **Bleed Planning** - Understand where content ends
            🔹 **Print Optimization** - Ensure crisp edges in print
            
            **Methods:**
            - **Canny**: Best general-purpose edge detector
            - **Sobel**: Good for detecting specific directions
            - **Laplacian**: Detects edges in all directions
            """)
            
            st.info("""
            **Outer Contour Detection:**
            
            🔹 **Shape Analysis** - Understand object geometry
            🔹 **Precise Boundaries** - Get exact outer edges
            🔹 **Cropping Guide** - Perfect for tight cropping
            🔹 **Logo Processing** - Ideal for logo isolation
            
            **Contour Methods:**
            - **Adaptive**: Best for varying lighting
            - **Otsu**: Great for high contrast images
            - **Canny**: Sharp, clean boundaries
            
            **Contour Info:**
            - **Green Line**: Detected outer contour
            - **Blue Box**: Bounding rectangle
            - **Metrics**: Area, perimeter, shape complexity
            """)
    
    # Section 3: Rotation
    with st.expander("🔄 Rotation", expanded=False):
        st.markdown("**Rotate your image for optimal orientation**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            rotation_type = st.radio(
                "Rotation Type",
                ["Quick (90° increments)", "Custom angle"],
                help="Choose rotation method"
            )
            
            if rotation_type == "Quick (90° increments)":
                angle = st.selectbox(
                    "Rotation Angle",
                    [0, 90, 180, 270],
                    format_func=lambda x: f"{x}° {'(No rotation)' if x == 0 else '(Clockwise)'}",
                    help="Select rotation angle"
                )
            else:
                angle = st.slider(
                    "Custom Angle",
                    min_value=-180,
                    max_value=180,
                    value=0,
                    step=5,
                    help="Enter custom rotation angle (-180 to 180)"
                )
            
            if st.button("🔄 Apply Rotation", key="rotate"):
                if angle != 0:
                    with st.spinner("Rotating image..."):
                        rotated_image = rotate_image_precise(st.session_state.current_image, angle)
                        st.session_state.current_image = rotated_image
                        st.session_state.processing_history.append(f"Rotated {angle}°")
                        st.success(f"✅ Image rotated {angle}°")
                        st.rerun()
                else:
                    st.info("No rotation applied (0°)")
        
        with col2:
            # Show orientation analysis
            width, height = st.session_state.current_image.size
            current_orientation = "Landscape" if width > height else "Portrait" if height > width else "Square"
            
            st.write(f"**Current Orientation:** {current_orientation}")
            st.write(f"**Current Aspect Ratio:** {width/height:.2f}")
            
            # Suggest rotations for common formats
            st.write("**Format Suggestions:**")
            if current_orientation == "Landscape":
                st.info("📄 Good for: A4 Landscape, Postcards, Banners")
                st.write("💡 Rotate 90° for portrait formats")
            elif current_orientation == "Portrait":
                st.info("📄 Good for: A4 Portrait, Posters, Book covers")
                st.write("💡 Rotate 90° for landscape formats")
            else:
                st.info("📄 Good for: Square prints, Social media")
    
    # Section 4: Resizing
    with st.expander("📏 Resizing", expanded=False):
        st.markdown("**Resize your image for specific print dimensions**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            resize_method = st.radio(
                "Resize Method",
                ["Print Format", "Custom Dimensions", "Scale Percentage"],
                help="Choose how to specify new size"
            )
            
            if resize_method == "Print Format":
                format_name = st.selectbox(
                    "Print Format",
                    list(PRINT_FORMATS.keys()),
                    help="Select standard print format"
                )
                
                dpi = st.selectbox(
                    "DPI",
                    [150, 300, 600],
                    index=1,
                    help="Print quality setting"
                )
                
                if st.button("📏 Resize to Format", key="resize_format"):
                    with st.spinner("Resizing image..."):
                        format_size = PRINT_FORMATS[format_name]
                        resized_image = resize_image_for_print(st.session_state.current_image, format_size[0], format_size[1], dpi)
                        st.session_state.current_image = resized_image
                        st.session_state.processing_history.append(f"Resized to {format_name} at {dpi} DPI")
                        st.success(f"✅ Resized to {format_name}")
                        st.rerun()
            
            elif resize_method == "Custom Dimensions":
                new_width = st.number_input("Width (pixels)", min_value=100, max_value=10000, value=st.session_state.current_image.size[0])
                new_height = st.number_input("Height (pixels)", min_value=100, max_value=10000, value=st.session_state.current_image.size[1])
                
                maintain_ratio = st.checkbox("Maintain aspect ratio", value=True)
                
                if st.button("📏 Resize to Custom", key="resize_custom"):
                    with st.spinner("Resizing image..."):
                        if maintain_ratio:
                            resized_image = st.session_state.current_image.resize((new_width, new_height), Image.LANCZOS)
                        else:
                            # Calculate height to maintain ratio
                            ratio = st.session_state.current_image.size[1] / st.session_state.current_image.size[0]
                            calc_height = int(new_width * ratio)
                            resized_image = st.session_state.current_image.resize((new_width, calc_height), Image.LANCZOS)
                        
                        st.session_state.current_image = resized_image
                        st.session_state.processing_history.append(f"Resized to {new_width}×{new_height if not maintain_ratio else calc_height}")
                        st.success("✅ Image resized")
                        st.rerun()
            
            else:  # Scale Percentage
                scale_percent = st.slider("Scale Percentage", min_value=10, max_value=500, value=100, step=10)
                
                if st.button("📏 Scale Image", key="resize_scale"):
                    if scale_percent != 100:
                        with st.spinner("Scaling image..."):
                            current_size = st.session_state.current_image.size
                            new_width = int(current_size[0] * scale_percent / 100)
                            new_height = int(current_size[1] * scale_percent / 100)
                            
                            scaled_image = st.session_state.current_image.resize((new_width, new_height), Image.LANCZOS)
                            st.session_state.current_image = scaled_image
                            st.session_state.processing_history.append(f"Scaled to {scale_percent}%")
                            st.success(f"✅ Image scaled to {scale_percent}%")
                            st.rerun()
                    else:
                        st.info("No scaling applied (100%)")
        
        with col2:
            current_width, current_height = st.session_state.current_image.size
            st.write(f"**Current Size:** {current_width} × {current_height} px")
            
            # Size recommendations
            st.write("**Size Guidelines:**")
            st.info("""
            📏 **Print DPI Guidelines:**
            - 150 DPI: Draft quality
            - 300 DPI: Standard print quality  
            - 600 DPI: High quality/professional
            
            📐 **Common Print Sizes:**
            - A4: 210×297mm
            - A5: 148×210mm  
            - Postcard: 148×105mm
            - Letter: 216×279mm
            """)
    
    # Section 5: Padding
    with st.expander("📦 Padding", expanded=False):
        st.markdown("**Add padding around your image**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            padding_mm = st.number_input(
                "Padding (mm)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                help="Amount of padding to add around the image"
            )
            
            if padding_mm > 0:
                padding_style = st.selectbox(
                    "Padding Style",
                    ["content_aware", "soft_shadow", "gradient_fade", "color_blend", "clean_border"],
                    format_func=lambda x: {
                        "content_aware": "Content Aware",
                        "soft_shadow": "Soft Shadow",
                        "gradient_fade": "Gradient Fade",
                        "color_blend": "Color Blend",
                        "clean_border": "Clean Border"
                    }[x],
                    help="Choose padding generation method"
                )
                
                if st.button("📦 Apply Padding", key="add_padding"):
                    with st.spinner("Adding padding..."):
                        try:
                            from padding import add_padding
                            dpi = 300  # Default DPI for padding calculation
                            padded_image, _ = add_padding(
                                st.session_state.current_image, 
                                padding_mm, 
                                dpi, 
                                padding_style, 
                                "natural", 
                                None, 
                                False, 
                                0.3
                            )
                            st.session_state.current_image = padded_image
                            st.session_state.processing_history.append(f"Added {padding_mm}mm {padding_style} padding")
                            st.success(f"✅ Padding added: {padding_mm}mm")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding padding: {str(e)}")
            else:
                st.info("Set padding > 0 to add padding")
        
        with col2:
            st.info("""
            **Padding Styles:**
            
            🔹 **Content Aware** - Intelligent padding based on image content
            🔹 **Soft Shadow** - Adds subtle shadow effect
            🔹 **Gradient Fade** - Fades from edge colors
            🔹 **Color Blend** - Blends colors from image palette
            🔹 **Clean Border** - Simple, clean border
            
            **When to Use Padding:**
            - Creating breathing room around your design
            - Preventing important content from being cut
            - Adding aesthetic appeal to prints
            """)
    
    # Section 6: Bleeding
    with st.expander("📐 Bleeding", expanded=False):
        st.markdown("**Add bleed area for professional printing**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            bleed_mm = st.number_input(
                "Bleed (mm)",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Amount of bleed to add around the image"
            )
            
            if bleed_mm > 0:
                bleed_type = st.selectbox(
                    "Bleed Type",
                    ["content_aware", "mirror", "edge_extend", "solid_color"],
                    format_func=lambda x: {
                        "content_aware": "Content Aware",
                        "mirror": "Mirror Edges",
                        "edge_extend": "Edge Extend",
                        "solid_color": "Solid Color"
                    }[x],
                    help="Choose how to fill the bleed area"
                )
                
                bleed_color = (255, 255, 255)  # Default white
                if bleed_type == "solid_color":
                    bleed_color_hex = st.color_picker("Bleed Color", value="#FFFFFF")
                    bleed_color = tuple(int(bleed_color_hex[i:i+2], 16) for i in (1, 3, 5))
                
                if st.button("📐 Apply Bleed", key="add_bleed"):
                    with st.spinner("Adding bleed..."):
                        try:
                            dpi = 300  # Default DPI for bleed calculation
                            bleed_image = add_bleed(st.session_state.current_image, bleed_mm, dpi, bleed_type, bleed_color)
                            st.session_state.current_image = bleed_image
                            st.session_state.processing_history.append(f"Added {bleed_mm}mm {bleed_type} bleed")
                            st.success(f"✅ Bleed added: {bleed_mm}mm")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding bleed: {str(e)}")
            else:
                st.info("Set bleed > 0 to add bleed area")
        
        with col2:
            st.info("""
            **Bleed Explained:**
            
            Bleed is extra area around your design that gets cut off during printing. It ensures your design extends to the very edge without white borders.
            
            **Bleed Types:**
            🔹 **Content Aware** - Intelligently extends content
            🔹 **Mirror** - Mirrors edge content outward
            🔹 **Edge Extend** - Extends edge pixels
            🔹 **Solid Color** - Fills with chosen color
            
            **Standard Bleed:** 3mm for most print jobs
            """)
    
    # Section 7: Cut Lines
    with st.expander("✂️ Cut Lines", expanded=False):
        st.markdown("**Add cut lines for trimming guidance**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            add_cut_lines_flag = st.checkbox(
                "Add Cut Lines",
                value=False,
                help="Add guide lines showing where to cut"
            )
            
            if add_cut_lines_flag:
                cut_line_color = st.selectbox(
                    "Cut Line Color",
                    ["black", "red", "blue", "green", "magenta", "cyan", "white"],
                    format_func=lambda x: x.title(),
                    index=1,  # Default to red
                    help="Color for the cut guide lines"
                )
                
                # This assumes there's bleed - check if bleed was added
                bleed_present = any("bleed" in step.lower() for step in st.session_state.processing_history)
                
                if bleed_present:
                    if st.button("✂️ Add Cut Lines", key="add_cut_lines"):
                        with st.spinner("Adding cut lines..."):
                            try:
                                # Assume 3mm bleed for cut line calculation
                                dpi = 300
                                bleed_mm = 3.0
                                cut_image = add_cut_lines(st.session_state.current_image, bleed_mm, dpi, cut_line_color)
                                st.session_state.current_image = cut_image
                                st.session_state.processing_history.append(f"Added {cut_line_color} cut lines")
                                st.success("✅ Cut lines added")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error adding cut lines: {str(e)}")
                else:
                    st.warning("⚠️ Add bleed first before cut lines")
            else:
                st.info("Enable cut lines to add trimming guides")
        
        with col2:
            st.info("""
            **Cut Lines Purpose:**
            
            Cut lines show exactly where to trim your print for perfect results.
            
            **When to Use:**
            ✅ Professional printing with bleed
            ✅ Business cards, postcards, flyers
            ✅ Any print job requiring precise cutting
            
            **Colors:**
            - **Red**: Most visible, industry standard
            - **Black**: Professional, high contrast
            - **Blue/Green**: Alternative visibility options
            """)
    
    # Final section: Download Results
    if st.session_state.processing_history:
        st.divider()
        st.header("💾 Download Processed Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Show final result
            st.subheader("Final Result")
            st.image(st.session_state.current_image, caption="Processed Image", use_container_width=True)
        
        with col2:
            # Download options
            st.subheader("Download Options")
            
            final_width, final_height = st.session_state.current_image.size
            st.write(f"**Final Dimensions:** {final_width} × {final_height} px")
            
            # Create download buffer
            output_buffer = io.BytesIO()
            st.session_state.current_image.save(output_buffer, format='PNG', dpi=(300, 300))
            output_buffer.seek(0)
            
            # Generate filename based on processing history
            base_name = "processed_image"
            if st.session_state.processing_history:
                # Create a descriptive filename
                steps = len(st.session_state.processing_history)
                base_name = f"processed_{steps}_steps"
            
            filename = f"{base_name}.png"
            
            st.download_button(
                label="📥 Download PNG",
                data=output_buffer.getvalue(),
                file_name=filename,
                mime="image/png"
            )
            
            # Show processing summary
            st.write("**Processing Steps Applied:**")
            for i, step in enumerate(st.session_state.processing_history, 1):
                st.write(f"{i}. {step}")

if __name__ == "__main__":
    main() 