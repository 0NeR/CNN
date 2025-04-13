import torch
import torch.nn as nn
import torch.nn.functional as F

from ex1_prepare_data import *


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.layer(x)


class ImageEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=64):
        super(ImageEncoder, self).__init__()
        self.net = nn.Sequential(
            Layer(in_dim, hidden_dim),
            Layer(hidden_dim, hidden_dim),
            Layer(hidden_dim, hidden_dim),
            Layer(hidden_dim, out_dim),
            nn.Flatten())
        
    def forward(self, x):
        # x shape: [batch_size, n_classes, C, H, W] или [n_way, n_support+n_query, C, H, W]
        original_shape = x.shape
        x = x.reshape(-1, *x.shape[-3:])  # [batch_size*n_classes, C, H, W]
        x = self.net(x)  # [batch_size*n_classes, D]
        return x.reshape(original_shape[0], original_shape[1], -1)  # [batch_size, n_classes, D]
    

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda()

    def set_forward_loss(self, sample):
        """
        Args:
            sample (dict): {
                'images': tensor of shape (n_way, n_support+n_query, C, H, W),
                'n_way': int,
                'n_support': int,
                'n_query': int
            }
        Returns:
            tuple: (loss, {
                'loss': float,
                'acc': float,
                'y_hat': tensor
            })
        """
        sample_images = sample['images'].cuda()  # [n_way, n_support+n_query, C, H, W]
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']
        
        embeddings = self.encoder(sample_images)  # [n_way*(n_support+n_query), D]
        
        support = embeddings[:, :n_support, :]  # [n_way, n_support, D]
        queries = embeddings[:, n_support:, :]  # [n_way, n_query, D]
        
        prototypes = torch.mean(support, dim=1)  # [n_way, D]
        
        distances = torch.cdist(
            queries.reshape(-1, queries.size(-1)),  # [n_way*n_query, D]
            prototypes,  # [n_way, D]
            p=2
        )  # [n_way*n_query, n_way]
        
        logits = -distances
        labels = torch.arange(n_way, device=prototypes.device)
        labels = labels.repeat_interleave(n_query)  # [n_way*n_query]
        
        loss = F.cross_entropy(logits, labels)
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        
        return loss, {'loss': loss.item(), 'acc': acc.item(), 'y_hat': y_hat}


def load_protonet_conv(x_dim=(3, 28, 28), hid_dim=64, z_dim=64):
    """
    Loads the prototypical network model
    Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (Class ProtoNet)
    """
    encoder = ImageEncoder(in_dim=x_dim[0], hidden_dim=hid_dim, out_dim=z_dim)
    return ProtoNet(encoder)



class ProtoNet2(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet2, self).__init__()
        self.encoder = encoder.cuda()

    def set_forward_loss(self, sample):
        """
        Args:
            sample (dict): {
                'images': tensor of shape (n_way, n_support+n_query, C, H, W),
                'n_way': int,
                'n_support': int,
                'n_query': int
            }
        Returns:
            tuple: (loss, {
                'loss': float,
                'acc': float,
                'y_hat': tensor
            })
        """
        sample_images = sample['images'].cuda()  # [n_way, n_support+n_query, C, H, W]
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']
        
        x = sample_images.reshape(-1, *sample_images.shape[-3:])  # [n_way*(n_support+n_query), C, H, W]
        embeddings = self.encoder(x)  # [n_way*(n_support+n_query), D]
        embeddings = embeddings.reshape(n_way, n_support + n_query, -1)  # [n_way, n_support+n_query, D]
        
        support = embeddings[:, :n_support, :]  # [n_way, n_support, D]
        queries = embeddings[:, n_support:, :]  # [n_way, n_query, D]
        
        prototypes = torch.mean(support, dim=1)  # [n_way, D]
        
        distances = torch.cdist(
            queries.reshape(-1, queries.size(-1)),  # [n_way*n_query, D]
            prototypes,  # [n_way, D]
            p=2
        )  # [n_way*n_query, n_way]
        
        logits = -distances
        labels = torch.arange(n_way, device=prototypes.device)
        labels = labels.repeat_interleave(n_query)  # [n_way*n_query]
        
        loss = F.cross_entropy(logits, labels)
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        
        return loss, {'loss': loss.item(), 'acc': acc.item(), 'y_hat': y_hat}