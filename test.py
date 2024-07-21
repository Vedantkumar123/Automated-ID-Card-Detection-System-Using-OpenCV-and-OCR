from PIL import Image
import pytesseract

# Path to the Tesseract OCR executable
tesseract_exe_path = r'C:\Users\Acer\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'




# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path

# Open the image using Pillow (PIL)
image = Image.open('test_img.png')

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(image)

# Print the extracted text
print(extracted_text)
