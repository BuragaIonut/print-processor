from rembg import remove
from PIL import Image
  
# Path to input image
input_path =  r"C:\Users\burag\Downloads\Screenshot 2025-06-25 144952.png" 
  
# Path to output Image
output_path = r"C:\Users\burag\Downloads\Screen.png"
  
# Open the input image using Pillow
inp = Image.open(input_path)

# Removing the background from the given Image using rembg
output = remove(inp)
  
# Saving the image to the given path 
output.save(output_path)