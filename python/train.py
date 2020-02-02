import argparse
import re

from torch.utils.data import DataLoader, Subset
from torch import multiprocessing
import torch.nn.functional as F

from net import Cycle
from bdcn import BDCN
from datasets import PoseLoadDataset, TensorProducer, NYUDataset
from utils import *

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
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-6,
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

    def combined_loss(preds, targets):
        mask = (targets[0] > 0.5).float()
        b, c, h, w = mask.shape
        num_p = torch.sum(mask, dim=[1, 2, 3])
        num_n = c * h * w - num_p
        weight = torch.empty_like(mask)
        for i, w in enumerate(weight):
            w[mask[i] == 0] = num_p[i] * 1.2 / (num_p[i] + num_n[i])
            w[mask[i] != 0] = num_n[i] / (num_p[i] + num_n[i])

        losses = []
        for pred in preds[:-1]:
            losses += [args.side_weight * F.binary_cross_entropy_with_logits(
                pred.float(), targets[0].float(), weight=weight, reduction='sum')]
        losses += [args.fuse_weight * F.binary_cross_entropy_with_logits(
            preds[-1].float(), targets[0].float(), weight=weight, reduction='sum')]

        return sum([loss / b for loss in losses])


    def adjust_learning_rate(optimizer, gamma=0.1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
            print("LR", param_group['lr'])

    detransform = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    params = {'batch_size': args.batch_size,
              'num_workers': args.workers,
              'pin_memory': True}

    send_queue = multiprocessing.Queue()
    receive_queue = multiprocessing.Queue()

    translators = []
    for path in ["./models/50_net_G_A.pth", "./models/70_net_G_A.pth", "./models/100_net_G_A.pth"]:
        t = Cycle(3, 3).cuda()
        t.load_state_dict(torch.load(path))
        t.eval()
        translators += [t]
    t = TensorProducer(*command_line_args, input=send_queue, output=receive_queue,
                           translator=translators, show=False, save=False, compress=False)

    vtrain = PoseLoadDataset("data/ego-motion-train.csv", send_queue=send_queue, receive_queue=receive_queue, randomize=True)
    vval = PoseLoadDataset("data/ego-motion-val.csv", send_queue=send_queue, receive_queue=receive_queue, randomize=False)

    vtrain_loader = DataLoader(vtrain, **params, shuffle=False)
    vval_loader = DataLoader(vval, **params, shuffle=False)

    dataset = NYUDataset('data/nyudv2/train', augment=True)
    rtrain = Subset(dataset, range(0, int(0.9*len(dataset))))
    rval = Subset(dataset, range(int(0.9*len(dataset)), len(dataset)))

    rtrain_loader = DataLoader(rtrain, **params, shuffle=True)
    rval_loader = DataLoader(rval, **params, shuffle=False)

    #print("Train samples: {}".format(len(train_loader)*args.batch_size))
    #print("Validation samples: {}".format(len(val_loader) * args.batch_size))
    #print("Test samples: {}".format(len(test_loader)*args.batch_size))

    model = BDCN(pretrained=True).to(device)

    params_dict = dict(model.named_parameters())
    base_lr = args.base_lr
    weight_decay = args.weight_decay

    params = []
    for key, v in params_dict.items():
        if re.match(r'conv[1-5]_[1-3]_down', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 0.1, 'weight_decay': weight_decay * 1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 0.2, 'weight_decay': weight_decay * 0, 'name': key}]
        elif re.match(r'.*conv[1-4]_[1-3]', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
        elif re.match(r'.*conv5_[1-3]', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 100, 'weight_decay': weight_decay * 1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 200, 'weight_decay': weight_decay * 0, 'name': key}]
        elif re.match(r'score_dsn[1-5]', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 0.01, 'weight_decay': weight_decay * 1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 0.02, 'weight_decay': weight_decay * 0, 'name': key}]
        elif re.match(r'upsample_[248](_5)?', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
        elif re.match(r'.*msblock[1-5]_[1-3]\.conv', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
        else:
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr * 0.001, 'weight_decay': weight_decay * 1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr * 0.002, 'weight_decay': weight_decay * 0, 'name': key}]

    optimizer = torch.optim.SGD(params, momentum=args.momentum, lr=args.base_lr, weight_decay=args.weight_decay)
    start_step = 1
    mean_loss = []
    mean_loss_lst = []
    val_mean_loss = []
    cur = 0
    rcur = len(rtrain_loader)
    vcur = len(vtrain_loader)
    pos = 0
    val_pos = 0
    print('*' * 40)
    print('train images in all are %d ' % (rcur+vcur))
    print('*' * 40)
    for param_group in optimizer.param_groups:
        print('%s: %s' % (param_group['name'], param_group['lr']))
    start_time = time.time()
    if args.resume:
        print('resume from %s' % args.resume)
        state = torch.load(args.resume)
        print('*' * 40)
        print('*' * 40)
        start_step = state['step']
        optimizer.load_state_dict(state['solver'])
        model.load_state_dict(state['param'])
    model.train()
    batch_size = args.iter_size * args.batch_size
    print("Training...")
    for step in range(start_step, args.max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(args.iter_size):
            if rcur == len(rtrain_loader):
                rcur = 0
                rdata_iter = iter(rtrain_loader)
            if vcur == len(vtrain_loader):
                vcur = 0
                vdata_iter = iter(vtrain_loader)
            if cur % 10 == 0:
                images, labels = next(vdata_iter)
                vcur += 1
            else:
                images, labels = next(rdata_iter)
                #images = F.interpolate(images, scale_factor=(1./2.), mode='bilinear')
                #labels = [F.max_pool2d(labels[0], (2,2), 2)]
                rcur += 1
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            out = model(images)
            loss = combined_loss(out, labels)
            loss.backward()
            batch_loss += loss.item()
            cur += 1

        # update parameter
        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, args.gamma)
        if step % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/%s_%d.pth' % (args.param_dir, args.filename, step))
            state = {'step': step + 1, 'param': model.state_dict(), 'solver': optimizer.state_dict()}
            torch.save(state, '%s/%s_%d.pth.tar' % (args.param_dir, args.filename, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            print('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)'
                  % (step, optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm / args.display))
            start_time = time.time()
        # VALIDATION
        if vval_loader is not None and step % args.val_step_size == 0:
            model.train(False)
            model.eval()
            print('mode: validation')
            val_batch_loss = 0
            for i, data in enumerate(rval_loader):
                val_images, val_labels = data
                val_images = val_images.to(device)
                val_labels = [val_label.to(device) for val_label in val_labels]
                val_out = model(val_images)
                val_loss = combined_loss(val_out, val_labels)
                val_loss.backward()
                val_batch_loss += val_loss.item()
            if len(val_mean_loss) < args.average_loss:
                val_mean_loss.append(val_batch_loss)
            else:
                val_mean_loss[val_pos] = val_batch_loss
                val_pos = (val_pos + 1) % args.average_loss
            mean_loss_lst.append([step, np.mean(mean_loss), np.mean(val_mean_loss)])
            for entry in mean_loss_lst:
                print('iter: %d, loss: %f, val_loss: %f' % (entry[0], entry[1], entry[2]))
            print('mode: training')
            model.train()
