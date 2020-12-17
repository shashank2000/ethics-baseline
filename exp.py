# example commands
# python test_representation.py config/cocotransfer.json cur_checkpoint 100 config/pretraining_on_coco.json --gpu-device 5
# python test_representation.py config/cocotransfer.json /mnt/fs5/shashank2000/experiments/coco_pretraining_pure/checkpoints 100 config/pretraining_on_coco.json --gpu-device 5
#  python test_representation.py config/cifar10transfer.json /mnt/fs5/shashank2000/experiments/cifar_pretraining_cos_decay/checkpoints 20541 config/pretraining_on_cifar10_new.json --gpu-device 4
# python test_representation.py config/dumbjeoptest.json /mnt/fs5/shashank2000/experiments/full_lstm_jeop/checkpoints/epoch=10.ckpt 20541 config/new_jeopardy_model.json --gpu-device 6

# Steps
# rerun model with transforms
# run simclr only with transforms
# image-image contrast + language-language contrast
# mcq task
from torch import tensor
from PIL import ImageFilter, Image
import random
from torchvision import transforms
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'dataset')))
# from dataset.cifar10 import CIFAR10Modified

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

from mscoco import MSCOCO, BaseMSCOCO

# perhaps the issue is with the RandomResizedCrop - try getting rid of this
# use just regular crop instead???
# transforms.Resize(224)
train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1,2.])], p=0.5), # perhaps this blur is too much
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261])

    ])
# run on every image in the dataset, see where it breaks
# dataset = MSCOCO(train=True, image_transforms=train_transform)
# i = 0
# while i < 500:
#     print("i is {}".format(i))
#     print(dataset.__getitem__(i)[2])
#     i += 1
from tqdm import tqdm
# # dataset = BaseMSCOCO(train=True, image_transforms=train_transform)
# # for i in tqdm(range(len(dataset))):
# #     print(i)
#     print(dataset.__getitem__(i))
# breakpoint()
# print(dataset[27299])
bbox = [181.6, 339.13, 182.47, 347.65999999999997]
image = Image.open("/Users/shashankrammoorthy/000000171360.jpg").convert(mode='RGB').crop(bbox)
breakpoint()
image.save("cropped.jpg")
# print(train_transform(image))

# print(dataset.__getitem__(198))
# # already_found = set()
# # for elem in dataset:
# #     label = elem[3]
# #     if label not in already_found:
# #         # are there bounding boxes here? 
# #         im = elem[2]
# #         im.save('crap/new_' + str(label) + '.png')
# #         already_found.add(label)
# #     if len(already_found) == 80:
# #         break
# # breakpoint()
# # for i in range(100):
# #     if dataset[i][3] == 3:
# #         im = dataset[i][1]
# #         im.save('crap/cat' + str(i) + '.png')

# im = dataset[10][1]

# # im = transforms.ToPILImage()(im).convert("RGB")
# im.save('crap/cocotest.png')

# im = dataset[0][2]

# # im = transforms.ToPILImage()(im).convert("RGB")
# im.save('crap/cocotest2.png')

# print(dataset[0][3])
# # path = '/data5/wumike/coco/train2017/000000558840.jpg'
# # im = Image.open(path).convert(mode='RGB')
# # im = train_transform(im)
# # im = transforms.ToPILImage()(im).convert("RGB")
# # im.save('test_images/customblur2.png')

