import torch
from PIL import ImageEnhance
from torchvision import transforms
from . import CUB200

transform_type_dict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                           Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
	def __init__(self, transform_dict):
		self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

	def __call__(self, img):
		out = img
		randtensor = torch.rand(len(self.transforms))
		for i, (transformer, alpha) in enumerate(self.transforms):
			r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
			out = transformer(out).enhance(r).convert('RGB')
		return out


TRAIN_TRANSFORMS = {
	CUB200: transforms.Compose([
		transforms.RandomResizedCrop(size=299),
		ImageJitter({"Brightness": 0.4, "Contrast": 0.4, "Color": 0.4}),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
}
VAL_TRANSFORMS = {
	CUB200: transforms.Compose([
		transforms.CenterCrop(size=299),
		transforms.Resize(size=299),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

}


def get_transform(dataset, data_augmentation=False):
	if data_augmentation:
		return TRAIN_TRANSFORMS[dataset]
	else:
		return VAL_TRANSFORMS[dataset]


