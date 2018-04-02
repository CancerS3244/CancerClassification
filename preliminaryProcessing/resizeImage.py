from PIL import Image
from multiprocessing import Pool
import os

def resize(file):
    size = 70, 50
    if os.path.exists("ResizedImages/"+file+".jpg"):
        return
    im = Image.open("Images/"+file+".jpg").convert("RGB")
    im = im.resize(size, Image.LANCZOS)
    im.save("ResizedImages/"+file+".jpg")

if __name__ == "__main__":
    filenames = os.listdir("Descriptions")
    with Pool(6) as p:
        p.map(resize, filenames)
