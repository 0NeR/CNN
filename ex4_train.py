import torch.optim as optim
from tqdm.notebook import tnrange

import neptune
import argparse

from ex1_prepare_data import *
from ex3_ProtoNet import load_protonet_conv


def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size, experiment):
    """
    Trains the protonet
    Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
    """

    experiment["parameters/n_way"] = n_way
    experiment["parameters/n_support"] = n_support
    experiment["parameters/n_query"] = n_query
    experiment["parameters/max_epoch"] = max_epoch
    experiment["parameters/epoch_size"] = epoch_size
    experiment["parameters/optimizer"] = type(optimizer).__name__
    experiment["parameters/learning_rate"] = optimizer.param_groups[0]['lr']
    
    #divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tnrange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']

            experiment[f"epoch_{epoch}_episode_loss"].log(output['loss'])
            experiment[f"epoch_{epoch}_episode_acc"].log(output['acc'])
            
            loss.backward()
            optimizer.step()
        
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size

        experiment["train/epoch_loss"].log(epoch_loss)
        experiment["train/epoch_acc"].log(epoch_acc)
        experiment["train/learning_rate"].log(optimizer.param_groups[0]['lr'])
        
        print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
        epoch += 1
        scheduler.step()
        
    experiment["final/train_loss"] = epoch_loss
    experiment["final/train_acc"] = epoch_acc
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Prototypical Network")
    parser.add_argument('--n_way', type=int, default=220, help='Number of classes per episode')
    parser.add_argument('--n_support', type=int, default=10, help='Support samples per class')
    parser.add_argument('--n_query', type=int, default=10, help='Query samples per class')
    parser.add_argument('--max_epoch', type=int, default=5, help='Maximum number of epochs')
    parser.add_argument('--epoch_size', type=int, default=2000, help='Number of episodes per epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--api_token', type=str, required=True, help='Your Neptune API token')
    parser.add_argument('--project_name', type=str, required=True, help='Neptune project name (e.g., "user/project")')

    args = parser.parse_args()

    trainx, trainy = read_images('images_background')
    model = load_protonet_conv(x_dim=(3, 28, 28), hid_dim=64, z_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run = neptune.init_run(
        project="eduard-andreev/Images-homework-3",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiN2QxMjE1NS01YjYwLTQ0NWMtYThjYy1jOWFjYmU0ODUwNjEifQ==",
    )

    train(model, optimizer, trainx, trainy,
          args.n_way, args.n_support, args.n_query,
          args.max_epoch, args.epoch_size, experiment=run)
    