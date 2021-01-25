from torchvision import transforms
from . import CUB200


def get_transform(dataset, data_augmentation=False, inception=False) -> transforms.Compose:
	size = 299 if inception else 224
	if dataset == CUB200:
		if data_augmentation:
			transform = transforms.Compose([
				# transforms.RandomResizedCrop(size=size),
				transforms.Resize(size=size),
				transforms.CenterCrop(size=size),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.4, saturation=0.4),
				transforms.RandomRotation(0.4),
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		else:
			transform = transforms.Compose([
				transforms.Resize(size=size),
				transforms.CenterCrop(size=size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
	else:
		raise NotImplementedError()
	return transform
