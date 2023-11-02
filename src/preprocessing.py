
import pandas as pd
import os
from PIL import Image

"""
Muraki
"""
df = pd.read_csv("muraki2021/data/raw/items.csv")
df
item_names = list(set(df["image"].tolist()))

fnames = os.listdir("muraki2021/data/raw/images")
len(fnames)

# Check all item_names are in fnames
for fname in item_names:
    if fname not in fnames:
        print(fname)

# delete files in images that are not in item_names
for fname in fnames:
    if fname not in item_names:
        os.remove(os.path.join("muraki2021/data/raw/images", fname))


# convert all pics to jpgs
def convert_pic_to_jpg(pic_path):
    img = Image.open(pic_path)
    img = img.convert('RGB')
    img.save(pic_path.replace('.bmp', '.jpg'))

# convert images in muraki2021/data/raw/images to jpg
dir_path = "muraki2021/data/raw/images"
for file in os.listdir(dir_path):
    if file.endswith('.bmp'):
        convert_pic_to_jpg(os.path.join(dir_path, file))

# delete bmps
for file in os.listdir(dir_path):
    if file.endswith('.bmp'):
        os.remove(os.path.join(dir_path, file))

# change file extensions in items.csv
df["image"] = df["image"].apply(lambda x: x.replace(".bmp", ".jpg"))


"""Reorganize"""
import re
df = df.sort_values("image")

df["object"] = df["image"].apply(lambda x: re.search("[a-z]+", x).group(0))
df["img_o"] = df["image"].apply(lambda x: re.search("[HV]", x).group(0))
df["match"] = df["match"].apply(lambda x: x.strip())

# H if img_o=H & match=Match
reverse = {"H": "V", "V": "H"}
df["sentence_o"] = df.apply(lambda x: x["img_o"] if x["match"] == "Match" else reverse[x["img_o"]], axis=1)

df_h = df[df["sentence_o"] == "H"][["sentence", "object"]]
df_h = df_h.drop_duplicates()
df_h = df_h.rename(columns={"sentence": "sentence_h"})

df_v = df[df["sentence_o"] == "V"][["sentence", "object"]]
df_v = df_v.drop_duplicates()
df_v = df_v.rename(columns={"sentence": "sentence_v"})

df_merge = pd.merge(df_h, df_v, on=["object"])
df_merge["image_h"] = df_merge["object"] + "H.jpg"
df_merge["image_v"] = df_merge["object"] + "V.jpg"

df_merge.to_csv("muraki2021/data/processed/items.csv", index=False)