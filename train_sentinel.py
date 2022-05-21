import bisect
import glob
import argparse
import os
import re
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import pytorch_mask_rcnn as pmr
import wandb
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "c6ea42f5f183e325a719b86d84e7aed50b2dfd5c"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda":
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    gpu_num = torch.cuda.device_count()
    args.batch_size *= gpu_num
    print(args)
    wandb.init(project='sentinel', config=args, entity="hudsonchen")

    # ---------------------- prepare data loader ------------------------------- #
    train_dataset = pmr.Sentinel_Dataset(args, train=True)
    test_dataset = pmr.Sentinel_Dataset(args, train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    # -------------------------------------------------------------------------- #
    if args.model == 'fcn':
        vgg_model = pmr.VGGNet(requires_grad=True, remove_fc=True).to(device)
        model = pmr.FCNs(pretrained_net=vgg_model, n_class=1).to(device)
    elif args.model == 'deeplab':
        model = pmr.deeplab(n_class=2).to(device)
    else:
        raise NotImplementedError(args.model)
    model = torch.nn.DataParallel(model)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------- train ----------------------------- #
    for epoch in range(0, args.epochs):
        model.eval()
        with torch.no_grad():
            loss_test = 0
            acc_test = 0
            iou_test = 0
            for data in test_loader:
                image = data['images'].to(device).float()
                target = data['target'].to(device).float()
                outputs = model(image)[:, 0, :]
                loss_test += criterion(outputs, target)
                acc_test += pmr.get_acc(outputs, target)
                iou_test += pmr.get_iou(outputs, target)
            loss_test /= len(test_loader)
            acc_test /= len(test_loader)
            iou_test /= len(test_loader)

            loss_train = 0
            acc_train = 0
            iou_train = 0
            for data in train_loader:
                image = data['images'].to(device).float()
                target = data['target'].to(device).float()
                outputs = model(image)[:, 0, :]
                loss_train += criterion(outputs, target)
                acc_train += pmr.get_acc(outputs, target)
                iou_train += pmr.get_iou(outputs, target)
            loss_train /= len(train_loader)
            acc_train /= len(train_loader)
            iou_train /= len(train_loader)

        wandb.log({"Train Loss": loss_train,
                   "Train Acc": acc_train,
                   "Train IoU": iou_train,
                   "Test Loss": loss_test,
                   "Test Acc": acc_test,
                   "Test IoU": iou_test})

        for data in train_loader:
            image = data['images'].to(device).float()
            target = data['target'].to(device).float()
            model.train()
            outputs = model(image)[:, 0, :]
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 5 == 0 or epoch == (args.epochs - 1):
            model.eval()
            with torch.no_grad():
                fig_train = pmr.visualize(args, data, outputs, "train")
                for data in test_loader:
                    image = data['images'].to(device).float()
                    target = data['target'].to(device).float()
                    outputs = model(image)[:, 0, :]
                    fig_test = pmr.visualize(args, data, outputs, "test")
                    break
            wandb.log({"fig_train": fig_train,
                       "fig_test": fig_test})
        break

    print("Training finished.")
    model.eval()
    final_mask = {}
    with torch.no_grad():
        for data in train_loader:
            image = data['images'].to(device).float()
            target = data['target'].to(device).float()
            outputs = model(image)[:, 0, :]
            mask = (torch.sigmoid(outputs) > 0.5).float()
            final_mask = pmr.merge(data, mask, final_mask)
        for data in test_loader:
            image = data['images'].to(device).float()
            target = data['target'].to(device).float()
            outputs = model(image)[:, 0, :]
            mask = (torch.sigmoid(outputs) > 0.5).float()
            final_mask = pmr.merge(data, mask, final_mask)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")

    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--model", default="fcn", help="fcn or deeplab")
    parser.add_argument("--data-dir", default="/home/xzhoubi/hudson/data/cocostuff")
    parser.add_argument("--ckpt-path", default="/home/xzhoubi/hudson/maskrcnn/checkpoints")
    parser.add_argument("--results")

    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.02 * 1 / 16  # lr should be 'batch_size / 16 * 0.02'
    # if args.ckpt_path is None:
    #     args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")

    main(args)
