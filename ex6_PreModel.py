from ex1_prepare_data import *

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = nn.Linear(self.in_features,
                                self.out_features,
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x
    
    
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super().__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, use_bias=False, use_bn=True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, use_bias=True, use_bn=True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, use_bias=False, use_bn=True))

    def forward(self,x):
        x = l2_norm(x)
        x = self.layers(x)
        return x


class PreModel(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained model
        model = torchvision.models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*tuple(model.children())[:-1])

        emb_size = tuple(model.children())[-1].in_features

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.projector = ProjectionHead(emb_size, 2048, 10)

    def forward(self,x):
        out = self.encoder(x)

        xp = self.projector(torch.squeeze(out))

        return xp
    


def plot_features(model, dataloader, device='cpu'):
    feats = []
    labels = []
    model.eval()

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc='collect feats')
        for x1, x2, label, _ in pbar:
            x1 = x1.to(device)
            out = model(x1)
            out = out.cpu().data.numpy()
            feats.append(out)
            labels.append(label.cpu().data.numpy())

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    print('Train TSNE ...')
    tsne = TSNE(n_components=2, perplexity=50, verbose=0, n_jobs=4)
    x_feats = tsne.fit_transform(feats)

    print('Plot labels ...')
    num_classes = len(np.unique(labels))
    fig = plt.figure(figsize=(6.4 * 2, 4.8 * 1))

    for i in range(num_classes):
        label_idxs = np.argwhere(labels == i)
        plt.scatter(x_feats[label_idxs, 1],x_feats[label_idxs, 0])

    plt.legend([str(i) for i in range(num_classes)])
    plt.axis('off')
    plt.margins(0)
    plt.tight_layout()
    plt.close()
    return fig
    