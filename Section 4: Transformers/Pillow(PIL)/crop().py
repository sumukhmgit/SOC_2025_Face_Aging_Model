from PIL import Image

img = Image.open("/home/sumukh/Desktop/Face Aging Model/Section 4: Transformers/Pillow(PIL)/Resources/Photos/cat.jpg")

imgCropped = img.crop(box=(20,20,200,200))

imgCropped.show()