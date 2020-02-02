import argparse

from torch.utils.data import DataLoader
from torch import multiprocessing
import torch.nn.functional as F
from torchvision import transforms

from bdcn import BDCN
from datasets import PoseLoadDataset, TensorProducer, NYUDataset
from utils import *
from PIL import Image
from net import Cycle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainer unity server setup.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size of images to read into memory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers for data loader')
    parser.add_argument('--save', action='store_true',
                        help='Save positions and images to disk (default: no)')
    parser.add_argument('--save-preds', action='store_true',
                        help='Save predictions to disk (default: no)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Run on device (default: cpu)')
    parser.add_argument('--show', action='store_true',
                        help='Show incoming batches (default: no)')
    parser.add_argument('--compress', action='store_true',
                        help='Compressed communication between unity and python (default: no)')
    parser.add_argument('--filename', type=str,
                        help='Filename for saved model (will be appended to models directory)')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='whether resume from some, default is None')
    parser.add_argument('--dataset', action='store_true',
                        help='Train or test on stored dataset')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-8,
                        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
                        help='the weight_decay of net')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='the decay of learning rate, default 0.1')
    parser.add_argument('--max_iter', type=int, default=40000,
                        help='max iters to train network, default is 40000')
    parser.add_argument('--iter_size', type=int, default=10,
                        help='iter size equal to the batch size, default 10')
    parser.add_argument('--side_weight', type=float, default=0.5,
                        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse_weight', type=float, default=1.1,
                        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--average_loss', type=int, default=50,
                        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1000,
                        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step_size', type=int, default=10000,
                       help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--val_step_size', type=int, default=100,
                        help='every n steps doing a validation')
    parser.add_argument('--display', type=int, default=20,
                        help='how many iters display one time, default is 20')
    parser.add_argument('--param_dir', type=str, default='models',
                        help='the directory to store the params')

    args = parser.parse_args()

    np.random.seed(42)

    device = torch.device(args.device)

    params = {'batch_size': args.batch_size,
              'num_workers': args.workers,
              'pin_memory': True}

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    path = "./models/70_net_G_A.pth"
    t = Cycle(3, 3).cuda()
    t.load_state_dict(torch.load(path))
    t.eval()

    def gan(data):
        o = t(data)
        for i, d in enumerate(o):
            o[i] = normalize((o[i] * 128 + 128).clamp(0, 255))
        return o

    #dataset = NYUDataset('data/unity_test', augment=False)
    dataset = NYUDataset('data/nyudv2/test', augment=False)
    #dataset = NYUDataset('data/bsds500', augment=False)
    loader = DataLoader(dataset, **params, shuffle=False)
    print("Test samples: {}".format(len(loader)*args.batch_size))

    model = BDCN(pretrained=True).to(device)
    model_state = torch.load(args.filename)
    model.load_state_dict(model_state)

    detransform = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    print("Testing...")
    for nr, (inputs, outputs) in enumerate(loader):
        inputs = inputs.to(device)
        #inputs = gan(inputs)
        scales = [0.5, 0.75, 1.0]#[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]#[0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.65, 0.7, 0.75]#[0.25, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.75]#[0.5, 1.0, 1.5]#[0.25, 0.375, 0.4375, 0.5]
        preds = []
        for s in scales:  # multiscale testing
            preds += [nn.functional.interpolate(nn.functional.sigmoid(
                    model(nn.functional.interpolate(inputs, scale_factor=s, mode='bilinear'))[-1]),
                    size=(inputs.shape[2], inputs.shape[3]), mode='bilinear').data.cpu()]
            #pred = model(nn.functional.interpolate(input, scale_factor=s, mode='bilinear'))
            #for p in pred:
            #    preds += [nn.functional.interpolate(nn.functional.sigmoid(p),
            #        size=(inputs.shape[2], inputs.shape[3]), mode='bilinear').data.cpu()]

        preds = torch.mean(torch.stack(preds), dim=0)

        for i in range(len(inputs)):
            image = detransform(inputs[i]).data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            pred_lines = preds[i, 0].data.cpu().numpy()
            gt_lines = outputs[0][i].data.numpy()[0]

            if args.save_preds:
                name = dataset.inputs[nr * len(inputs) + i].split('.')[0]
                im = Image.fromarray(image)
                im.save("results/rgb/{}.png".format(name))
                im = Image.fromarray((gt_lines * 255).astype(np.uint8))
                im.save("results/gt/{}.png".format(name))
                im = Image.fromarray(np.floor(pred_lines * 255).astype(np.uint8))
                im.save("results/pred/{}.png".format(name))
