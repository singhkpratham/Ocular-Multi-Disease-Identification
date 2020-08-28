import os
from sklearn.model_selection import KFold
import glob
import numpy as np
import csv

image_list = glob.glob('/extendspace/yizhou/DR/ODIR-5K/images/*')
write_path = '/extendspace/yizhou/DR/ODIR-5K/'

X = np.arange(len(image_list))
kf = KFold(n_splits=5)
kf.get_n_splits(X)

kf = KFold(n_splits=5, shuffle=True, random_state=5)
i = 1
for train_index, test_index in kf.split(X):
    print(i, len(train_index), len(test_index))
    train_file = os.path.join(write_path, 'train_%02d.csv' % i)
    with open(train_file, "w", newline='') as f:
        writer =csv.writer(f)
        for j in range(len(train_index)):
            writer.writerow([str(train_index[j])])
        f.close()

    test_file = os.path.join(write_path, 'test_%02d.csv' % i)
    with open(test_file, "w", newline='') as f:
        writer =csv.writer(f)
        for j in range(len(test_index)):
            writer.writerow([str(test_index[j])])
        f.close()
    i += 1


