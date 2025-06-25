import base64
import io
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
from openai import OpenAI
from enum import Enum
import json

class ImageType(Enum):
    """Image classification types for print processing optimization"""
    OBJECT_ON_BACKGROUND = "object_on_background"  # Logos, isolated objects with clear background
    FULL_IMAGE = "full_image"  # Complex scenes, photos that fill the frame
    PRODUCT_PHOTO = "product_photo"  # Product on clean/studio background
    LOGO_EMBLEM = "logo_emblem"  # Company logos, team emblems, brand marks
    DOCUMENT_GRAPHIC = "document_graphic"  # Charts, diagrams, text-based graphics
    ARTWORK = "artwork"  # Paintings, illustrations, artistic content

class BackgroundType(Enum):
    """Background classification for processing decisions"""
    SOLID_COLOR = "solid_color"  # Uniform solid background
    GRADIENT = "gradient"  # Gradient or smooth transition background
    TEXTURED = "textured"  # Textured but uniform background
    COMPLEX_SCENE = "complex_scene"  # No clear background separation
    TRANSPARENT = "transparent"  # Already has transparency
    STUDIO_BACKGROUND = "studio_background"  # Professional photo background

class ContentComplexity(Enum):
    """Content complexity for processing strategy"""
    SIMPLE = "simple"  # Simple shapes, minimal details
    MODERATE = "moderate"  # Some detail but clear structure
    COMPLEX = "complex"  # High detail, intricate elements
    VERY_COMPLEX = "very_complex"  # Extremely detailed or busy

class ImageAnalysisResult:
    """Structured result from image analysis"""
    
    def __init__(self, 
                 image_type: ImageType,
                 background_type: BackgroundType,
                 complexity: ContentComplexity,
                 has_text: bool,
                 dominant_colors: List[str],
                 confidence: float,
                 recommendations: Dict[str, str],
                 raw_analysis: str):
        self.image_type = image_type
        self.background_type = background_type
        self.complexity = complexity
        self.has_text = has_text
        self.dominant_colors = dominant_colors
        self.confidence = confidence
        self.recommendations = recommendations
        self.raw_analysis = raw_analysis
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for easy serialization"""
        return {
            "image_type": self.image_type.value,
            "background_type": self.background_type.value,
            "complexity": self.complexity.value,
            "has_text": self.has_text,
            "dominant_colors": self.dominant_colors,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "raw_analysis": self.raw_analysis
        }
    
    def __str__(self) -> str:
        return f"Image Type: {self.image_type.value}, Background: {self.background_type.value}, Complexity: {self.complexity.value}"

class ImageAnalyzer:
    """AI-powered image analyzer for print processing optimization"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def encode_image_from_pil(self, pil_image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string for OpenAI API"""
        # Convert to RGB if needed (for JPEG compatibility)
        if format.upper() == "JPEG" and pil_image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for JPEG
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        
        # Save to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format=format, quality=95 if format.upper() == "JPEG" else None)
        img_buffer.seek(0)
        
        return base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
    def encode_image_from_path(self, image_path: str) -> str:
        """Convert image file to base64 string for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def create_analysis_prompt(self) -> str:
        """Create the analysis prompt for OpenAI"""
        return """Analyze this image for print processing optimization. Provide a structured analysis in JSON format with the following information:

1. IMAGE_TYPE: Classify as one of:
   - "object_on_background": Logos, isolated objects, items with clear background separation
   - "full_image": Complex scenes, photographs, artwork that fills the entire frame
   - "product_photo": Products on clean/studio backgrounds
   - "logo_emblem": Company logos, team emblems, brand marks
   - "document_graphic": Charts, diagrams, text-based graphics
   - "artwork": Paintings, illustrations, artistic content

2. BACKGROUND_TYPE: Classify as one of:
   - "solid_color": Uniform solid background
   - "gradient": Gradient or smooth transition background  
   - "textured": Textured but uniform background
   - "complex_scene": No clear background separation
   - "transparent": Already has transparency
   - "studio_background": Professional photo background

3. COMPLEXITY: Rate content complexity:
   - "simple": Simple shapes, minimal details
   - "moderate": Some detail but clear structure
   - "complex": High detail, intricate elements
   - "very_complex": Extremely detailed or busy

4. HAS_TEXT: Boolean - does the image contain readable text?

5. DOMINANT_COLORS: List of 3-5 dominant colors in hex format (e.g., ["#FF0000", "#00FF00"])

6. CONFIDENCE: Float 0-1 indicating confidence in the classification

7. RECOMMENDATIONS: Object with processing recommendations:
   - "shape_aware": Boolean - recommend shape-aware processing
   - "padding_style": Recommended padding style
   - "background_handling": Recommended background handling
   - "notes": Additional processing notes

Respond ONLY with valid JSON in this exact format:
{
  "image_type": "category",
  "background_type": "category", 
  "complexity": "level",
  "has_text": boolean,
  "dominant_colors": ["#color1", "#color2"],
  "confidence": 0.0-1.0,
  "recommendations": {
    "shape_aware": boolean,
    "padding_style": "style_name",
    "background_handling": "handling_method",
    "notes": "additional notes"
  }
}"""

    def analyze_image(self, image_input, max_retries: int = 3) -> ImageAnalysisResult:
        """
        Analyze an image using OpenAI's vision API
        
        Args:
            image_input: PIL Image object or path to image file
            max_retries: Number of retry attempts for API calls
            
        Returns:
            ImageAnalysisResult object with structured analysis
        """
        
        # Prepare image data
        if isinstance(image_input, str):
            # File path
            base64_image = self.encode_image_from_path(image_input)
            image_format = "jpeg"
        elif isinstance(image_input, Image.Image):
            # PIL Image
            base64_image = self.encode_image_from_pil(image_input, "PNG")
            image_format = "png"
        else:
            raise ValueError("image_input must be either a file path (str) or PIL Image object")
        
        # Create the API request
        prompt = self.create_analysis_prompt()
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Use the latest vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_format};base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.1  # Low temperature for consistent analysis
                )
                
                # Parse the response
                raw_response = response.choices[0].message.content.strip()
                
                # Try to extract JSON from the response
                try:
                    # Find JSON in the response (in case there's extra text)
                    json_start = raw_response.find('{')
                    json_end = raw_response.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = raw_response[json_start:json_end]
                        analysis_data = json.loads(json_str)
                    else:
                        # Fallback: try to parse the entire response as JSON
                        analysis_data = json.loads(raw_response)
                    
                    # Create structured result
                    return self._create_analysis_result(analysis_data, raw_response)
                    
                except json.JSONDecodeError as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed, return fallback result
                        return self._create_fallback_result(raw_response, f"JSON parsing failed: {e}")
                    else:
                        continue  # Retry
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, return error result
                    return self._create_fallback_result("", f"API call failed: {e}")
                else:
                    continue  # Retry
        
        # Should not reach here, but just in case
        return self._create_fallback_result("", "All retry attempts failed")
    
    def _create_analysis_result(self, data: Dict, raw_response: str) -> ImageAnalysisResult:
        """Create ImageAnalysisResult from parsed JSON data"""
        try:
            # Parse enums with fallbacks
            image_type = ImageType(data.get("image_type", "full_image"))
            background_type = BackgroundType(data.get("background_type", "complex_scene"))
            complexity = ContentComplexity(data.get("complexity", "moderate"))
            
            # Extract other fields with defaults
            has_text = data.get("has_text", False)
            dominant_colors = data.get("dominant_colors", ["#808080"])
            confidence = float(data.get("confidence", 0.7))
            recommendations = data.get("recommendations", {
                "shape_aware": False,
                "padding_style": "content_aware",
                "background_handling": "extend_background",
                "notes": "Standard processing recommended"
            })
            
            return ImageAnalysisResult(
                image_type=image_type,
                background_type=background_type,
                complexity=complexity,
                has_text=has_text,
                dominant_colors=dominant_colors,
                confidence=confidence,
                recommendations=recommendations,
                raw_analysis=raw_response
            )
            
        except (ValueError, KeyError) as e:
            # Fallback if enum parsing fails
            return self._create_fallback_result(raw_response, f"Data parsing failed: {e}")
    
    def _create_fallback_result(self, raw_response: str, error_msg: str) -> ImageAnalysisResult:
        """Create a fallback result when analysis fails"""
        return ImageAnalysisResult(
            image_type=ImageType.FULL_IMAGE,
            background_type=BackgroundType.COMPLEX_SCENE,
            complexity=ContentComplexity.MODERATE,
            has_text=False,
            dominant_colors=["#808080"],
            confidence=0.3,
            recommendations={
                "shape_aware": False,
                "padding_style": "content_aware",
                "background_handling": "extend_background",
                "notes": f"Analysis failed: {error_msg}. Using safe defaults."
            },
            raw_analysis=raw_response or f"Error: {error_msg}"
        )

def quick_analyze(image_input, api_key: Optional[str] = None) -> ImageAnalysisResult:
    """Quick analysis function for convenience"""
    analyzer = ImageAnalyzer(api_key)
    return analyzer.analyze_image(image_input)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        analyzer = ImageAnalyzer()
        
        # Test with a sample image (you would replace this with actual image)
        result = analyzer.analyze_image(r"C:\Users\burag\Downloads\Lucid_Realism_A_dynamic_mountain_landscape_complete_with_lush__1.jpg")
        print(f"Analysis Result: {result}")
        print(f"Recommendations: {result.recommendations}")
        
        print("ImageAnalyzer module loaded successfully!")
        print("Available image types:", [t.value for t in ImageType])
        print("Available background types:", [t.value for t in BackgroundType])
        print("Available complexity levels:", [t.value for t in ContentComplexity])
        
    except Exception as e:
        print(f"Error initializing ImageAnalyzer: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set") 