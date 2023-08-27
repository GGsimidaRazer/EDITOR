import time, threading
import os
import random
import shutil
lst = os.listdir('/home/jason/Data/CelebA/CelebAMask-HQ/CelebA-HQ-img')
length = len(lst)
print(length)
if len(lst) > 50000:
    raise ValueError('Too long!')
result=random.sample(population=lst, k=50000)
for i in result:
    try:
        shutil.copy('/home/jason/Data/CelebA/CelebAMask-HQ/CelebA-HQ-img/'+i,'/home/jason/Data/CelebA/CelebAMask-HQ/CelebA-HQ-smallset')
    except:
        raise ValueError('Cannot read file!')
