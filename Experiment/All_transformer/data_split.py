import os, random
import shutil
from tqdm import tqdm
from glob import glob

files = glob('../dataset/processed/*.npz')
files.sort()

last_name = None
in_train = None
for f in tqdm(files):
    file_name = os.path.split(f)[-1]
    cur_name = file_name.split('_')[0]

    if last_name == cur_name:
        if in_train:
            shutil.move(f, '../dataset/processed_train/'+file_name)
        else:
            shutil.move(f, '../dataset/processed_val/'+file_name)
    else:
        last_name = cur_name
        if random.random() > 0.8:
            in_train = False
        else:
            in_train = True

