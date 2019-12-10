import os
import torch
import torch.nn as nn
import numpy as np
import logging

from torch.utils.data import DataLoader, Dataset

from main import noisy_circle, iou, Model, count_parameters

def to_device(x, gpu=True):
    if isinstance(x, list):
        return [to_device(t, gpu) for t in x]
    elif isinstance(x, dict):
        for key in x:
            x[key] = to_device(x[key], gpu)
        return x
    elif isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
        return x.cuda() if gpu else x.cpu()
    else:
        raise NotImplementedError

class NoisyCircleDataset(Dataset):
    def __init__(self, size):
        super(NoisyCircleDataset, self).__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        params, img = noisy_circle(size=200, radius=50, noise=2)
        img = torch.from_numpy(img)
        return {"labels": params, "img": img}

class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, out_params, gt_params):
        loss_dict = {}
        gt_params = [x.to(torch.float) for x in gt_params]
        loss_dict['row_reg_loss'] = self.reg_loss(out_params[:, 0], gt_params[0] / 199)
        loss_dict['col_reg_loss'] = self.reg_loss(out_params[:, 1], gt_params[1] / 199)
        loss_dict['rad_reg_loss'] = self.reg_loss(out_params[:, 2], (gt_params[2] - 10) / 39)
        loss_dict['loss'] = loss_dict['row_reg_loss'] + loss_dict['col_reg_loss'] + loss_dict['rad_reg_loss']
        return loss_dict


def save_checkpoint(state, out_file_name):
    outdir = os.path.dirname(out_file_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    torch.save(state, out_file_name)

def train(detect_net, reg_loss, optimizer, dataloader, logger, use_gpu=False):
    detect_net = to_device(detect_net, gpu=use_gpu)

    # training loop
    for i, data in enumerate(dataloader, 1):
        data = to_device(data, gpu=use_gpu)

        # forward pass
        detection = detect_net(data['img'].to(torch.float).unsqueeze(1))

        # compute loss and update weights
        loss_dict = reg_loss(detection, data['labels'])
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        # logging loss values
        log_out = "Step {}/{}:\n".format(i, len(dataloader))
        for name, loss in loss_dict.items():
            log_out += "{}: {:.4f}, ".format(name, loss.item())

        # compute IOU of detection and ground truth
        iou_total = []
        for bi in range(detection.size(0)):
            pred_params = [detection[bi,j].item() for j in range(3)]
            gt_params = [data['labels'][j][bi].item() for j in range(3)]
            pred_params[0] = pred_params[0] * 199
            pred_params[1] = pred_params[1] * 199
            pred_params[2] = pred_params[2] * 39 + 10
            
            iou_total.append(iou(pred_params, gt_params))
        iou_total = np.array(iou_total)
        log_out += "IOU: {:.4f}\n".format(iou_total.mean())

        logger.info(log_out)

        # save checkpoint
        if i % 12500 == 0:
            state = {"state_dict": detect_net.state_dict()}
            save_checkpoint(state, os.path.join(os.curdir, "checkpoints", "ckpt_{}".format(i)))


if __name__ == "__main__":

    # init logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(os.curdir, "train.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler) 
    
    # dataloader initialization
    dataset = NoisyCircleDataset(10000000)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8)
    
    # model and optimizer
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # print num learnable parameters
    logger.info("Model Parameters: {}".format(count_parameters(model)))

    criterion = RegressionLoss()

    train(model, criterion, optimizer, dataloader, logger, use_gpu=torch.cuda.is_available())