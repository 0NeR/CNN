from tqdm.notebook import tnrange
import argparse

from ex1_prepare_data import *
from ex3_ProtoNet import load_protonet_conv
from ex4_train import model


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in tnrange(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']
        
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Prototypical Network")
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_support', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=5)
    parser.add_argument('--test_episode', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    testx, testy = read_images('images_evaluation')
    model = load_protonet_conv(x_dim=(3, 28, 28), hid_dim=64, z_dim=64)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        
    test(model, testx, testy, args.n_way, args.n_support, args.n_query, args.test_episode)