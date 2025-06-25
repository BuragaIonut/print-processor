import io
from typing import Optional, Tuple, List
from PIL import Image
from enum import Enum

# Try to import rembg with graceful fallback
try:
    import rembg
    from rembg import remove, session_factory
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    rembg = None

class BackgroundRemovalModel(Enum):
    """Available background removal models from rembg"""
    U2NET = "u2net"  # General purpose, good for most images
    U2NET_HUMAN_SEG = "u2net_human_seg"  # Optimized for human segmentation
    U2NET_CLOTH_SEG = "u2net_cloth_seg"  # Optimized for clothing
    SILUETA = "silueta"  # Good for objects and products
    ISNET_DIS = "isnet-general-use"  # High accuracy general purpose
    SAM = "sam"  # Segment Anything Model (if available)

class BackgroundRemovalQuality(Enum):
    """Quality levels for background removal"""
    FAST = "fast"  # Quick processing, lower quality
    BALANCED = "balanced"  # Good balance of speed and quality
    HIGH = "high"  # High quality, slower processing

class BackgroundRemovalResult:
    """Result object containing the processed image and metadata"""
    
    def __init__(self, 
                 image_with_transparent_bg: Image.Image,
                 original_image: Image.Image,
                 mask: Optional[Image.Image] = None,
                 model_used: str = "unknown",
                 processing_time: float = 0.0,
                 confidence_score: Optional[float] = None):
        self.image_with_transparent_bg = image_with_transparent_bg
        self.original_image = original_image
        self.mask = mask
        self.model_used = model_used
        self.processing_time = processing_time
        self.confidence_score = confidence_score
    
    def get_foreground_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box of the foreground object (x, y, width, height)"""
        if self.image_with_transparent_bg.mode != 'RGBA':
            raise ValueError("Image must have alpha channel")
        
        # Get alpha channel
        alpha = self.image_with_transparent_bg.split()[-1]
        
        # Find bounding box of non-transparent pixels
        bbox = alpha.getbbox()
        if bbox:
            x, y, x2, y2 = bbox
            return (x, y, x2 - x, y2 - y)
        else:
            return (0, 0, 0, 0)
    
    def create_image_with_solid_background(self, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Create version with solid color background instead of transparency"""
        if self.image_with_transparent_bg.mode != 'RGBA':
            return self.image_with_transparent_bg.convert('RGB')
        
        # Create background
        background = Image.new('RGB', self.image_with_transparent_bg.size, background_color)
        
        # Paste the image with transparency
        background.paste(self.image_with_transparent_bg, mask=self.image_with_transparent_bg.split()[-1])
        
        return background
    
    def save_result(self, output_path: str, format: str = "PNG", background_color: Optional[Tuple[int, int, int]] = None):
        """Save the result to file"""
        if background_color:
            # Save with solid background
            img_to_save = self.create_image_with_solid_background(background_color)
        else:
            # Save with transparency (PNG only)
            if format.upper() != "PNG":
                raise ValueError("Transparency requires PNG format")
            img_to_save = self.image_with_transparent_bg
        
        img_to_save.save(output_path, format=format)

class BackgroundRemover:
    """Advanced background removal using rembg with multiple models and options"""
    
    def __init__(self):
        """Initialize the background remover"""
        if not REMBG_AVAILABLE:
            raise ImportError("rembg is not installed. Install with: pip install rembg")
        
        self._sessions = {}  # Cache for model sessions
        
    def _get_session(self, model: BackgroundRemovalModel):
        """Get or create a session for the specified model"""
        model_name = model.value
        
        if model_name not in self._sessions:
            try:
                self._sessions[model_name] = session_factory(model_name)
            except Exception as e:
                # Fallback to default model
                print(f"Warning: Could not load model {model_name}, falling back to u2net: {e}")
                model_name = "u2net"
                if model_name not in self._sessions:
                    self._sessions[model_name] = session_factory(model_name)
        
        return self._sessions[model_name]
    
    def remove_background(self, 
                         image: Image.Image,
                         model: BackgroundRemovalModel = BackgroundRemovalModel.U2NET,
                         quality: BackgroundRemovalQuality = BackgroundRemovalQuality.BALANCED,
                         return_mask: bool = False) -> BackgroundRemovalResult:
        """
        Remove background from image
        
        Args:
            image: PIL Image to process
            model: Background removal model to use
            quality: Quality level for processing
            return_mask: Whether to return the segmentation mask
            
        Returns:
            BackgroundRemovalResult object with processed image and metadata
        """
        import time
        start_time = time.time()
        
        # Convert image to RGB if needed (rembg expects RGB)
        if image.mode != 'RGB':
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # Apply quality settings
        processed_image = self._apply_quality_settings(rgb_image, quality)
        
        try:
            # Get the appropriate session
            session = self._get_session(model)
            
            # Convert to bytes for rembg
            img_bytes = io.BytesIO()
            processed_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Remove background
            result_bytes = remove(img_bytes.getvalue(), session=session)
            
            # Convert back to PIL Image
            result_image = Image.open(io.BytesIO(result_bytes))
            
            # Ensure RGBA mode
            if result_image.mode != 'RGBA':
                result_image = result_image.convert('RGBA')
            
            # Scale back to original size if we resized for quality
            if processed_image.size != image.size:
                result_image = result_image.resize(image.size, Image.Resampling.LANCZOS)
            
            processing_time = time.time() - start_time
            
            # Extract mask if requested
            mask = None
            if return_mask:
                mask = result_image.split()[-1]  # Alpha channel as mask
            
            return BackgroundRemovalResult(
                image_with_transparent_bg=result_image,
                original_image=image,
                mask=mask,
                model_used=model.value,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Background removal failed: {str(e)}")
    
    def _apply_quality_settings(self, image: Image.Image, quality: BackgroundRemovalQuality) -> Image.Image:
        """Apply quality-specific preprocessing"""
        if quality == BackgroundRemovalQuality.FAST:
            # Resize for faster processing
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                return image.resize(new_size, Image.Resampling.LANCZOS)
        
        elif quality == BackgroundRemovalQuality.HIGH:
            # Ensure minimum quality
            min_size = 1024
            if max(image.size) < min_size:
                ratio = min_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                return image.resize(new_size, Image.Resampling.LANCZOS)
        
        # BALANCED or no change needed
        return image
    
    def remove_background_batch(self, 
                               images: List[Image.Image],
                               model: BackgroundRemovalModel = BackgroundRemovalModel.U2NET,
                               quality: BackgroundRemovalQuality = BackgroundRemovalQuality.BALANCED) -> List[BackgroundRemovalResult]:
        """Remove background from multiple images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.remove_background(image, model, quality)
                results.append(result)
                print(f"Processed image {i+1}/{len(images)} in {result.processing_time:.2f}s")
            except Exception as e:
                print(f"Failed to process image {i+1}: {e}")
                # Add a placeholder result with the original image
                results.append(BackgroundRemovalResult(
                    image_with_transparent_bg=image.convert('RGBA'),
                    original_image=image,
                    model_used="failed",
                    processing_time=0.0
                ))
        
        return results
    
    def suggest_best_model(self, image: Image.Image) -> BackgroundRemovalModel:
        """Suggest the best model based on image characteristics"""
        width, height = image.size
        aspect_ratio = width / height
        
        # Simple heuristics - could be enhanced with AI analysis
        if aspect_ratio > 0.7 and aspect_ratio < 1.3:  # Roughly square, likely portrait
            return BackgroundRemovalModel.U2NET_HUMAN_SEG
        elif max(width, height) > 2000:  # High resolution
            return BackgroundRemovalModel.ISNET_DIS
        else:  # General purpose
            return BackgroundRemovalModel.U2NET
    
    def create_cutout_with_background(self, 
                                    result: BackgroundRemovalResult,
                                    new_background: Image.Image,
                                    position: str = "center") -> Image.Image:
        """Composite the cutout onto a new background"""
        foreground = result.image_with_transparent_bg
        
        # Resize background to match foreground if needed
        if new_background.size != foreground.size:
            new_background = new_background.resize(foreground.size, Image.Resampling.LANCZOS)
        
        # Ensure background is RGB
        if new_background.mode != 'RGB':
            new_background = new_background.convert('RGB')
        
        # Position the foreground
        if position == "center":
            # Center the foreground on the background
            paste_x = (new_background.size[0] - foreground.size[0]) // 2
            paste_y = (new_background.size[1] - foreground.size[1]) // 2
        else:
            paste_x, paste_y = 0, 0  # Top-left by default
        
        # Create the composite
        composite = new_background.copy()
        if foreground.mode == 'RGBA':
            composite.paste(foreground, (paste_x, paste_y), foreground.split()[-1])
        else:
            composite.paste(foreground, (paste_x, paste_y))
        
        return composite

def quick_remove_background(image: Image.Image, 
                          model_name: str = "u2net",
                          quality: str = "balanced") -> BackgroundRemovalResult:
    """Quick background removal function for convenience"""
    if not REMBG_AVAILABLE:
        raise ImportError("rembg is not installed. Install with: pip install rembg")
    
    remover = BackgroundRemover()
    
    # Convert string parameters to enums
    model = BackgroundRemovalModel(model_name)
    quality_enum = BackgroundRemovalQuality(quality)
    
    return remover.remove_background(image, model, quality_enum)

def get_available_models() -> List[str]:
    """Get list of available models"""
    return [model.value for model in BackgroundRemovalModel]

def check_rembg_installation() -> Tuple[bool, str]:
    """Check if rembg is properly installed"""
    if not REMBG_AVAILABLE:
        return False, "rembg is not installed. Install with: pip install rembg"
    
    try:
        # Test basic functionality
        session_factory("u2net")
        return True, "rembg is installed and working"
    except Exception as e:
        return False, f"rembg installation issue: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    # Test the module
    installed, status = check_rembg_installation()
    print(f"Status: {status}")
    
    if installed:
        print("Available models:", get_available_models())
        
        # Example usage (commented out - requires actual image)
        try:
            remover = BackgroundRemover()
            
            # Load test image
            test_image = Image.open(r"C:\Users\burag\Downloads\Lucid_Realism_A_dynamic_mountain_landscape_complete_with_lush__1.jpg")
            
            # Remove background
            result = remover.remove_background(test_image, 
                                              BackgroundRemovalModel.U2NET,
                                              BackgroundRemovalQuality.BALANCED,
                                              return_mask=True)
            
            print(f"Processing completed in {result.processing_time:.2f} seconds")
            print(f"Model used: {result.model_used}")
            print(f"Foreground bounds: {result.get_foreground_bounds()}")
            
            # Save result
            result.save_result("output_transparent.png")
            result.save_result("output_white_bg.jpg", "JPEG", (255, 255, 255))
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("BackgroundRemover module loaded successfully!") 