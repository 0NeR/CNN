import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from ex1_prepare_data import *
from ex4_train import model

def visualize_prediction_episode(model, test_x, test_y, n_way, n_support, n_query, class_names=None):
    """
    Визуализирует предсказания ProtoNet на одном тестовом эпизоде.
    """
    model.eval()
    sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    with torch.no_grad():
        loss, output = model.set_forward_loss(sample)

    images = sample['images'] 
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']

    support_images = images[:, :n_support]
    query_images = images[:, n_support:]

    y_true = torch.arange(n_way).repeat_interleave(n_query)
    y_pred = output['y_hat'].cpu()

    inv_transform = transforms.ToPILImage()

    fig, axs = plt.subplots(n_way, n_query + n_support, figsize=(2 * (n_query + n_support), 2 * n_way))

    for i in range(n_way):
        for j in range(n_support):
            img = inv_transform(support_images[i, j].cpu() / 255.)
            axs[i, j].imshow(img)
            axs[i, j].set_title(f"Support\nClass {i}" if class_names is None else f"Support\n{class_names[i]}")
            axs[i, j].axis("off")

        for j in range(n_query):
            idx = i * n_query + j
            pred = y_pred[idx].item()
            true = y_true[idx].item()
            color = "green" if pred == true else "red"
            img = inv_transform(query_images[i, j].cpu() / 255.)
            axs[i, j + n_support].imshow(img)
            title = f"Pred {pred}" if class_names is None else f"Pred {class_names[pred]}"
            axs[i, j + n_support].set_title(title, color=color)
            axs[i, j + n_support].axis("off")

    plt.tight_layout()
    plt.show()
    
    

if __name__ == "__main__":
    
    trainx, trainy = read_images('images_background')
    testx, testy = read_images('images_evaluation')
    test_x = testx
    test_y = testy
    
    visualize_prediction_episode(model, test_x, test_y, n_way=5, n_support=5, n_query=3)
