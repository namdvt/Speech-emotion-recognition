import os
import random

train_path = 'data/radvess/trainval/'
test_path = 'data/radvess/test/'


if __name__ == '__main__':
    emotions = os.listdir(train_path)
    for emotion in emotions:
        files = os.listdir(train_path + emotion)
        random.shuffle(files)

        if ~os.path.exists(test_path + emotion):
            os.mkdir(test_path + emotion)

        for file in files[0: int(len(files) * 0.2)]:
            os.rename(train_path + emotion + '/' + file, test_path + emotion + '/' + file)
            print()