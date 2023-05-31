# from importlib import import_module

# from dataloader import MSDataLoader
from symbacdata import SRCNNDataset
from symbacdata import get_datasets, get_dataloaders
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        kwargs = {}
        train_image_paths = args.train_lr
        train_label_paths = args.train_hr
        valid_image_path = args.test_lr
        valid_label_paths = args.test_hr
        self.loader_train = None
        if not args.test_only:
            trainset, testset = get_datasets(train_image_paths,train_label_paths,valid_image_path,valid_label_paths)
            self.loader_train, self.loader_test = get_dataloaders(trainset,testset)
        else:
            trainset,testset = get_datasets(train_image_paths,train_label_paths,valid_image_path,valid_label_paths)
            self.loader_test = get_dataloaders(trainset,testset)[1]
