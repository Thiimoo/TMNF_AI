import cv2
import pytesseract
import numpy as np

# Optional: Specify the path to the Tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_speed_image(image_path):
    # Load the image from the specified file
    speed_image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if speed_image is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")

    # Get image dimensions
    height, width, _ = speed_image.shape

    # Crop the bottom right corner (e.g., last 100 pixels from both height and width)
    crop_height = 60
    crop_width = 180
    cropped_image = speed_image[height - crop_height:height, width - crop_width:width]

    return cropped_image

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to binarize the image
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh_image

def extract_speed_from_image(thresh_image):
    # Customize the configuration

    # Use Pytesseract to extract text from the thresholded image
    speed_text = pytesseract.image_to_string(thresh_image)
    print("Extracted Text:", speed_text)
    # Clean up the extracted text to get only the speed value
    speed_value = ''.join(filter(str.isdigit, speed_text))

    return speed_value


def display_image(image, window_name='Captured Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

# Main testing workflow
if __name__ == "__main__":
    # Specify the path to the test image
    image_path = 'test.png'  # Replace with the correct path if necessary

    # Capture the speed image
    speed_image = capture_speed_image(image_path)

    # Display the captured image
    print("Captured speed_image shape:", speed_image.shape)
    display_image(speed_image, 'Captured Speed Image (Bottom Right Corner)')

    # Preprocess the image
    thresh_image = preprocess_image(speed_image)


    # Display the thresholded image
    display_image(thresh_image, 'Thresholded Speed Image (Bottom Right Corner)')

    # Extract speed from the thresholded image
    speed_value = extract_speed_from_image(thresh_image)

    print("Extracted Speed Value:", speed_value)


