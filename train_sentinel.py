import bisect
import glob
import os
import re
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import pytorch_mask_rcnn as pmr
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #

    image_array, target_list = pmr.load_sentinel(args)

    # -------------------------------------------------------------------------- #
    print(args)
    num_classes = 2  # including background class
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    start_epoch = 0

    # ------------------------------- train ------------------------------------ #
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))

        for i in np.arange(image.shape[0]):
            image = image_array[i, :][None, :, :, :].to(device)
            target = target_list[i]
            target = {k: v.to(device) for k, v in target.items()}

            args.lr_epoch = lr_lambda(epoch) * args.lr
            print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))

            model.train()
            losses = model(image, target)
            total_loss = sum(losses.values())
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            output = model(image)
            map_test = target['masks']
            map_test_hat = output['masks']
            map_test_plot = map_test_hat.sum(0)

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image.cpu().numpy().transpose((1, 2, 0)))
            axs[1].imshow(map_test_plot.cpu().numpy(), cmap='gray')
            plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="/home/xzhoubi/hudson/data/cocostuff")
    parser.add_argument("--ckpt-path", default="/home/xzhoubi/hudson/maskrcnn/checkpoints")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    # if args.ckpt_path is None:
    #     args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)
