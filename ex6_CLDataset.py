from ex1_prepare_data import *
from ex6_load_cifar10 import load_cifar10

class CLDataset(Dataset):
    def __init__(self, x_data, y_data, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data

        assert transform_augment is not None, 'set transform_augment'
        # TODO: pass your code
        self.transform_augment = transform_augment

    def __len__(self):
        # TODO: pass your code
        return len(self.x_data)

    def __getitem__(self, item):
        image = self.x_data[item]
        image = (image * 255).astype(np.uint8)
        label = self.y_data[item]

        # TODO: pass your code
        x1 = self.transform_augment(image=image)['image']
        x2 = self.transform_augment(image=image)['image']

        image = torch.tensor(image).permute(2, 0, 1)

        return x1, x2, label, image

def plot_batch(x, STD, MEAN):
    plt.figure(figsize=[12, 10])
    plt.axis('off')

    grid = torchvision.utils.make_grid(x, 4, )
    grid = grid.numpy().transpose((1, 2, 0))
    grid = grid * STD + MEAN
    plt.imshow(grid)

    plt.tight_layout()
    

def get_cropped_data_idxs(data, crop_coef: float = 1.0):
    crop_coef = np.clip(crop_coef, 0, 1)

    init_data_size = len(data)
    final_data_size = int(init_data_size * crop_coef)

    random_idxs = np.random.choice(tuple(range(init_data_size)), final_data_size, replace=False)
    return random_idxs

def load_datasets(X_train, y_train, X_val, y_val, crop_coef=0.2):
    train_idxs = get_cropped_data_idxs(X_train, crop_coef=crop_coef)
    train_data = X_train[train_idxs]
    train_labels = y_train[train_idxs]

    valid_idxs = get_cropped_data_idxs(X_val, crop_coef=crop_coef)
    valid_data = X_val[valid_idxs]
    valid_labels = y_val[valid_idxs]

    train_dataset = CLDataset(train_data, train_labels, transform_augment=train_transform)
    valid_dataset = CLDataset(valid_data, valid_labels, transform_augment=valid_transform)

    return train_dataset, valid_dataset
    

if __name__ == "__main__":
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10("cifar_data", channels_last=True)
    
    class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'])
    
    MEAN = list(np.mean(X_train, axis=(0, 1, 2), keepdims=True).squeeze())
    STD = list(np.std(X_train, axis=(0, 1, 2), keepdims=True).squeeze())
    
    train_transform = A.Compose([
        A.OneOf([
            A.ColorJitter(),
            A.ToGray(),
    #         A.GaussNoise(),
        ]),
    #     A.OneOf([
    #         A.Cutout(num_holes=1, max_h_size=10, max_w_size=10),
    #         A.RandomResizedCrop(32, 32),
    #         A.GaussianBlur(),
    #     ]),
        A.HorizontalFlip(),
    #     A.RandomRotate90(),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    
    train_dataset = CLDataset(X_train, y_train, transform_augment=train_transform)
    valid_dataset = CLDataset(X_val, y_val, transform_augment=valid_transform)
    
    batch_size = 32
    n_workers = 0

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=n_workers)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=n_workers)
    
    