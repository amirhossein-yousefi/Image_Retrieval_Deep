import numpy as np
import torch

from dataset_svhn import SVHNDataset
from utils import load_data, get_default_device, DeviceDataLoader, to_device
from sklearn.neighbors import NearestNeighbors
from autoencoder_second_app import Autoencoder
import torchvision.transforms as transforms

device = get_default_device()


# %%
def create_embedding(encoder, full_loader, embedding_dim):
    encoder.eval()
    # print(embedding.shape)
    embed = []
    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            enc_output = encoder(train_img).cpu()
            embed.append(enc_output.reshape(enc_output.shape[0], -1))
            # print(embedding.shape)
    embedding = torch.cat(embed, dim=0)
    return embedding


MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
trans = transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     ]
)
full_dataset = SVHNDataset("./data", 'unlabelled_task3', transform=trans)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=False)
train_dl = DeviceDataLoader(full_loader, device)
model = Autoencoder()
to_device(model, device)
model.load_state_dict(torch.load('model_autoenc_app2_activation_relu_tanhenc.pt', map_location='cpu'))
embedding = create_embedding(encoder=model.encoder,
                             full_loader=train_dl,
                             embedding_dim=128 * 4)

test_imgs, test_labels = load_data('./data', 'test_task3')
unlabelled_imgs, unlabelled_labels = load_data('./data', 'unlabelled_task3')
img_accuracies = []
K = 5
N = test_imgs.shape[-1]
test_imgs_T = test_imgs.reshape(-1, test_imgs.shape[-1]).T
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(embedding.numpy())

for img_idx in range(0, N):
    img = test_imgs_T[img_idx, :]
    encoder = model.encoder
    encoder.eval()
    with torch.no_grad():
        query_embedding = encoder(
            trans(test_imgs[..., img_idx].astype(np.uint8)).unsqueeze(0).cuda()).cpu().detach().numpy()
    flattened_embedding = query_embedding.reshape((query_embedding.shape[0], -1))
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    similar_img_labels = unlabelled_labels[indices_list]
    img_label = test_labels[img_idx]
    image_accuracy = (similar_img_labels == img_label).mean()
    img_accuracies.append(image_accuracy)
    print('Accuracy for test sample {idx} with label {label}  in TOP{K} retrieved images: {acc}'.format(idx=img_idx,
                                                                                                        label=img_label,
                                                                                                        K=K,
                                                                                                        acc=image_accuracy))

# Compute average accuracy over testing set
average_acc = np.asarray(img_accuracies).mean()
print('Average Accuracy over the testing set: {acc}'.format(acc=average_acc))
