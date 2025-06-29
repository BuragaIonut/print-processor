import streamlit as st

st.set_page_config(
    page_title="Print Processor",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("# Welcome to Print Processor! 🖨️")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    **Print Processor** is a professional print preparation tool designed to help you prepare images and PDFs for high-quality printing.

    **👈 Select a page from the sidebar** to get started:
    - **Processor** - Manual configuration with full control
    - **AI Advanced** - Intelligent analysis with AI recommendations  
    - **Step by Step** - Granular control over individual processing steps

    ## 🚀 Features

    ### 📐 **Print Format Support**
    - A4, A3, A5, Letter, Legal, Tabloid formats
    - Custom dimensions support
    - Auto-rotation to match format orientation

    ### 🎨 **Advanced Padding**
    - Content-aware padding
    - Soft shadow effects
    - Gradient fade options
    - Color blend techniques
    - Vintage vignette effects
    - **🤖 AI Advanced padding** (OpenAI integration)

    ### 📏 **Professional Bleed**
    - Content-aware bleed generation
    - Mirror edge effects
    - Edge extension
    - Solid color fills
    - **🤖 AI Advanced bleed** (OpenAI integration)

    ### ✂️ **Cut Lines & Quality**
    - Customizable cut guide lines
    - Multiple DPI options (72-2400)
    - Professional print specifications
    - PDF and PNG output formats

    ### 🧹 **Object Removal** (Experimental)
    - AI-powered object detection and removal
    - Perfect for cleaning up logos, text, and distracting elements
    - Adjustable sensitivity settings

    ## 🤖 AI Features

    ### **AI Advanced Page** 🆕
    - **Intelligent Analysis** - Upload an image or PDF and describe your print use case (e.g., "postcard", "poster", "business card")
    - **Smart Recommendations** - AI analyzes your content and suggests optimal processing steps
    - **Expert Guidance** - Get professional advice on rotation, format, bleed, padding, and quality settings
    - **Use Case Optimization** - Tailored recommendations based on your specific printing needs
    - **PDF Support** - First page analysis for document and presentation printing

    ### **Step by Step Processing** 🆕
    - **Object Detection** - Use OpenCV to identify main objects and logos in your images
    - **Edge Detection** - Analyze image structure with advanced edge detection algorithms
    - **Individual Controls** - Fine-tune rotation, resizing, padding, bleed, and cut lines separately
    - **Processing History** - Track every step applied to your image with undo capabilities
    - **Best Fit Analysis** - Get suggestions for optimal cropping and positioning

    ### **AI Processing Options**
    Our **AI Advanced** processing options use OpenAI's latest image generation models to create contextual padding and bleed areas that seamlessly extend your image content. This results in more natural-looking print preparations compared to traditional algorithmic methods.

    ## 📋 How to Use

    ### **Option 1: AI Advanced (Recommended for beginners)**
    1. **Navigate to AI Advanced** using the sidebar
    2. **Upload your image or PDF** and **describe your use case** (e.g., "postcard", "document print")
    3. **Get AI recommendations** for optimal print settings
    4. **Apply suggestions manually** or use them as guidance

    ### **Option 2: Step by Step (Granular control)**
    1. **Navigate to Step by Step** using the sidebar
    2. **Upload your image** and see current status
    3. **Work through each section** (object detection, rotation, resizing, etc.)
    4. **Apply changes individually** with real-time preview
    5. **Download the final result** when satisfied

    ### **Option 3: Manual Processor (Full control)**
    1. **Navigate to the Processor page** using the sidebar
    2. **Upload your image or PDF** file
    3. **Configure your print settings** (format, DPI, padding, bleed)
    4. **Click "Process for Print"** to generate your print-ready file
    5. **Download the result** in your preferred format

    ## 🔧 Technical Specifications

    - **Supported Input Formats:** JPG, PNG, PDF (all pages supported)
    - **Output Formats:** PNG, PDF (matches input type) 
    - **PDF Processing:** First page analysis for AI Advanced, all pages for manual processing
    - **DPI Range:** 72-2400 DPI
    - **Maximum Bleed:** 10mm
    - **Maximum Padding:** 20mm
    - **AI Integration:** OpenAI gpt-image-1 for advanced processing

    ---

    *Ready to process your files? Click on **Processor** in the sidebar to get started!*
    """
) 