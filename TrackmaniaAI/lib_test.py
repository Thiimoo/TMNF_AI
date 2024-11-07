import cv2
import os

# Load the font image
font_image_path = 'Led_00.png'
font_image = cv2.imread(font_image_path)

# Check if the image was loaded successfully
if font_image is None:
    raise FileNotFoundError(f"Could not load image from path: {font_image_path}")

# Parameters for character extraction
character_height = 48  # Adjust according to your font size
character_width = 33    # Adjust according to your font size
num_characters = 8     # Adjust based on the number of characters in the font

# Starting row for the second row (adjust based on your image structure)
# If your first row is at height 0, the second row would start at height 48
starting_row = 132  # This assumes there's no gap between rows

# Create a directory to save character images
if not os.path.exists('characters'):
    os.makedirs('characters')

# Extract and save characters from the second row
for i in range(num_characters):
    # Crop character (adjust row/column indices as needed)
    character_image = font_image[starting_row:starting_row + character_height,
                      i * character_width:(i + 1) * character_width]

    # Save the character image
    cv2.imwrite(f'characters/character_{i + num_characters}.png', character_image)  # Offset the filenames

print("Character images from the second row extracted and saved.")


# Create a .box file for Tesseract
box_file_content = ""
characters = "23456789"  # Adjust according to the actual characters

for i, char in enumerate(characters):
    box_file_content += f"character_{i}.png 0 0 {character_width} {character_height} 0 0 0\n"

# Save the box file
with open("characters/characters.box", "w") as f:
    f.write(box_file_content)

print("Box file created.")

