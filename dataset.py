import numpy as np
import cv2
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset, GeneratorDataset
from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset
import os



class CUB:
    def __init__(self,train=True):
        self.is_train = train
        self.image_root='/4T/dataset/Cross_CUB/CUB_200_2011/images/'
        path='/4T/dataset/Cross_CUB'

        if self.is_train:
            self.video_root = os.path.join(path, 'video_img', 'train')
            with open('/4T/yuht/Mind/train.txt') as f:  # 因为有中文，所以encoding='utf-8'
                names = f.readlines()
            self.list = [[] for i in range(len(names))]
            for i in range(len(names)):
                for word in names[i].split():
                    self.list[i].append(word)
            self.text=np.load('cub_train.npy')

        if not self.is_train:
            self.video_root = os.path.join(path, 'video_img', 'test')
            with open('/4T/yuht/Mind/test.txt') as f:  # 因为有中文，所以encoding='utf-8'
                names = f.readlines()
            self.list = [[] for i in range(len(names))]
            for i in range(len(names)):
                for word in names[i].split():
                    self.list[i].append(word)
            self.text = np.load('cub_test.npy')


    def __getitem__(self, index):
        image_path = self.list[index][0]
        class_id = self.list[index][1]
        video_name = self.list[index][2]
        text_path = self.list[index][3]
        audio_path = self.list[index][4]

        # 读取image
        image = cv2.imread(self.image_root+image_path)

        # 读取audio
        audio = cv2.imread(audio_path)

        # 读取video
        video_path = self.video_root + '/' + video_name + '_' + str(12) + '.jpg'
        video = cv2.imread(video_path)

        # video_image = []
        # for i in range(25):
        #     # video_path = self.video_root + '/' + '--3UbM_4b7k' + '_' + str(i) + '.jpg'#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     video_path = self.video_root + '/'+video_name+'_'+str(i)+'.jpg'
        #     video_image.append(cv2.imread(video_path))

        # 读取text
        text=self.text[index]

        class_id=int(class_id)



        return image,audio,text,video,class_id

        # return image,class_id

    def __len__(self):
        return len(self.list)

def data(train=True,batch_size=64,num_worker=4):
    TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
    TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
    TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
    TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]
    train_composed = transforms.Compose(
        [
            vision.Resize([448, 448]),
            vision.RandomHorizontalFlip(),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
            vision.ToTensor()
        ]
    )
    test_composed = transforms.Compose(
        [
            vision.Resize(448),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=TEST_MEAN, std=TEST_STD),
            vision.ToTensor()
        ]
    )
    if train:
        train_dataset = CUB(train=True)
        train_dataloader = GeneratorDataset(source=train_dataset, column_names=["image", "audio", "text", "video", "label"])
        train_dataloader = train_dataloader.map(
            operations=train_composed,
            input_columns=['image'],
            num_parallel_workers=num_worker
        )
        train_dataloader = train_dataloader.map(
            operations=train_composed,
            input_columns=['audio'],
            num_parallel_workers=num_worker
        )
        train_dataloader = train_dataloader.map(
            operations=train_composed,
            input_columns=['video'],
            num_parallel_workers=num_worker
        )
        train_dataloader = train_dataloader.batch(batch_size=batch_size)
        print('train')
        return train_dataloader
    else:
        test_dataset = CUB(train=False)
        test_dataloader = GeneratorDataset(source=test_dataset, column_names=["image", "audio", "text", "video", "label"])
        test_dataloader = test_dataloader.map(
            operations=test_composed,
            input_columns=['image'],
            num_parallel_workers=num_worker
        )
        test_dataloader = test_dataloader.map(
            operations=test_composed,
            input_columns=['audio'],
            num_parallel_workers=num_worker
        )
        test_dataloader = test_dataloader.map(
            operations=test_composed,
            input_columns=['video'],
            num_parallel_workers=num_worker
        )
        test_dataloader = test_dataloader.batch(batch_size=batch_size)
        print('test')
        return test_dataloader

if __name__ == '__main__':
    num_worker=4
    batch_size=2
    TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
    TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
    TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
    TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]
    train_composed = transforms.Compose(
        [
            vision.Resize([448, 448]),
            vision.RandomHorizontalFlip(),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
            vision.ToTensor()
        ]
    )
    test_composed = transforms.Compose(
        [
            vision.Resize(448),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=TEST_MEAN, std=TEST_STD),
            vision.ToTensor()
        ]
    )

    train_dataset = CUB(train=True)
    train_dataloader = GeneratorDataset(source=train_dataset, column_names=["image", "audio", "text", "video", "label"])
    train_dataloader = train_dataloader.map(
        operations=train_composed,
        input_columns=['image'],
        num_parallel_workers=num_worker
    )
    train_dataloader = train_dataloader.map(
        operations=train_composed,
        input_columns=['audio'],
        num_parallel_workers=num_worker
    )
    train_dataloader = train_dataloader.map(
        operations=train_composed,
        input_columns=['video'],
        num_parallel_workers=num_worker
    )
    train_dataloader = train_dataloader.batch(batch_size=batch_size)

    for idx, (image, audio, text, video, class_id) in enumerate(train_dataloader):
        print('stop')
        image=image




