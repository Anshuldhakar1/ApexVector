"""Create test images for vectorization."""
import numpy as np
from PIL import Image
import os

def create_test_images():
    """Create simple test images."""
    os.makedirs('testing', exist_ok=True)
    
    # Test 1: Simple colored squares
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :128] = [255, 0, 0]  # Red
    img[:128, 128:] = [0, 255, 0]  # Green
    img[128:, :128] = [0, 0, 255]  # Blue
    img[128:, 128:] = [255, 255, 0]  # Yellow
    Image.fromarray(img).save('testing/squares.png')
    print("Created: testing/squares.png")
    
    # Test 2: Gradient
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[:, i] = [int(i), int(255-i), 128]
    Image.fromarray(img).save('testing/gradient.png')
    print("Created: testing/gradient.png")
    
    # Test 3: Circle
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    center = 128
    radius = 80
    for y in range(256):
        for x in range(256):
            if (x - center)**2 + (y - center)**2 < radius**2:
                img[y, x] = [255, 100, 100]
    Image.fromarray(img).save('testing/circle.png')
    print("Created: testing/circle.png")

if __name__ == '__main__':
    create_test_images()
    print("\nTest images created successfully!")
