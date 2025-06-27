from PIL import Image

# Open the two images
img1 = Image.open("/home/sumukh/Desktop/Face Aging Model/Section 4: Transformers/Pillow(PIL)/Resources/Photos/cat.jpg")
img2 = Image.open("/home/sumukh/Desktop/Face Aging Model/Section 4: Transformers/Pillow(PIL)/Resources/Photos/cats.jpg")

# Resize img2 to match the size of img1
img2 = img2.resize(img1.size)

# Now blend the images
img = Image.blend(img1, img2, alpha=0.5)

# Show the blended image
img.show()
