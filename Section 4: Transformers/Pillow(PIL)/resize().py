from PIL import Image

filename = "/home/sumukh/Desktop/Face Aging Model/Section 4: Transformers/Pillow(PIL)/Resources/Photos/cat.jpg"

img = Image.open(filename)
print(img.width, img.height)
img.show()

img=img.resize((int(img.width*2),int(img.height*2)), resample=Image.LANCOZ)
print(img.width, img.height)
img.show()