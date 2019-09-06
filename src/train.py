from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trans_NFCM import TransNFCM
from optimizer.radam import RAdam
from feature.data_loader_for_NFCM import FMNISTDataset, loader


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def main(args):
    n_relational_embeddings = args.n_class**2
    n_tag_embeddings = args.n_class
    in_ch, out_ch = 1, 128
    model = TransNFCM(in_ch, out_ch,
                      n_relational_embeddings, n_tag_embeddings,
                      embedding_dim=128).to(device)

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    optimizer = RAdam(model.parameters(), weight_decay=1e-3)

    train_dataset = FMNISTDataset(n_class=args.n_class, train=True)
    test_dataset = FMNISTDataset(n_class=args.n_class, train=False)

    train_loader = loader(train_dataset, args.batch_size)
    test_loader = loader(test_dataset, 1, shuffle=False)

    train(args, model, optimizer, train_loader)
    test(args, model, test_loader)


def train(args, model, optimizer, data_loader):
    model.train()

    pre_loss = 1e6
    min_loss = 1e6
    no_implove_cnt = 0
    for epoch in range(args.epochs):
        for i, (image, cat, near_image, near_cat, far_image, far_cat, near_relation, far_relation) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)
            near_image = near_image.to(device)
            near_cat = near_cat.to(device)
            far_image = far_image.to(device)
            far_cat = far_cat.to(device)
            near_relation = near_relation.to(device)
            far_relation = far_relation.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            loss = model(image, near_image, image, far_image,
                         cat, near_cat, cat, far_cat,
                         near_relation, far_relation).sum()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()

            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        if epoch % 50 == 0:
            torch.save(model.state_dict(),
                       '{}/model.pth'.format(args.out_dir))
    torch.save(model.state_dict(),
               '{}/model.pth'.format(args.out_dir))


def test(args, model, data_loader):
    model.eval()
    with torch.no_grad():
        for i, (image, cat) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)

            embedded_vec = model.predict(image)
        print("done test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-class', type=int, default=10, help='number of class')
    parser.add_argument('--resume-model', default='./results/model_.pth', help='path to trained model')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--out-dir', default='./results', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
