from PIL import Image

filename = "/home/sumukh/Desktop/Face Aging Model/Section 4: Transformers/Pillow(PIL)/Resources/Photos/cat.jpg"

img = Image.open(filename)

img = img.rotate(angle=45, expand=True, fillcolor="blue", resample=Image.BICUBIC)

img.show()