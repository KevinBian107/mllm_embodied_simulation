

import pandas as pd
import os
from PIL import Image


def convert_pic_to_jpg(pic_path):
    img = Image.open(pic_path)
    img.save(pic_path.replace('.pic', '.jpg'))


def convert_all_pics_to_bmps(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith('.pic'):
            convert_pic_to_bmp(os.path.join(dir_path, file))


dir_path = "pecher2012/Orientation"
convert_all_pics_to_bmps(dir_path)

pic_path = "pecher2012/Orientation/squirl.pic"
img = Image.open(pic_path)

paxman_fnames = os.listdir(
    "orientation/Sentence Picture Verification Task Stimuli")
paxman_fnames = [fname.replace(".bmp", "") for fname in paxman_fnames]


df = pd.read_csv("pecher2012/data/raw/items.csv")
# change bmp to jpg in match and mismatch cols
df["match"] = df["match"].apply(lambda x: x.replace(".bmp", ".jpg"))
df["mismatch"] = df["mismatch"].apply(lambda x: x.replace(".bmp", ".jpg"))
item_names = set(list(df["match"]) + list(df["mismatch"]))
len(item_names)

fnames = os.listdir("pecher2012/data/raw/images")
len(fnames)

# Check all item_names are in fnames
for fname in item_names:
    if fname not in fnames:
        print(fname)


# delete files in images that are not in item_names
for fname in fnames:
    if fname not in item_names:
        os.remove(os.path.join("pecher2012/data/raw/images", fname))
