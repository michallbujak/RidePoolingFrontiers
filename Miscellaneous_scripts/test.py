from PIL import Image

img = Image.open("pic.jpeg")
img.save("pic2.jpeg", quality=50)

