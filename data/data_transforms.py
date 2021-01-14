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


def get_transform(dataset, data_augmentation=False, inception=False):
	size = 299 if inception else 224
	if dataset == CUB200:
		if data_augmentation:
			transform = transforms.Compose([
					transforms.RandomResizedCrop(size=size),
					ImageJitter({"Brightness": 0.4, "Contrast": 0.4, "Color": 0.4}),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
		else:
			transform = transforms.Compose([
				transforms.CenterCrop(size=size),
				transforms.Resize(size=size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
	else:
		raise NotImplementedError()
	return transform


