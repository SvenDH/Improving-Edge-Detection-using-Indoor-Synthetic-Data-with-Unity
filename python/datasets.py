import threading
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from api import Simulation
from utils import *


class TensorProducer(Simulation):

    def __init__(self, *argv, input, output, save=False, show=False, compress=False, translator=None):
        Simulation.__init__(self, *argv, save=save, compress=compress)

        self.show = show
        self.image_buffer = {}
        self.index = 0
        self.input = input
        self.output = output
        self.translator = translator

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        threading.Thread(target=self._watcher, daemon=True).start()

    def image_callback(self, data, idx):
        if self.show:
            plt.ion()
            plt.tight_layout()
            ax = plt.subplot(2, 3, 1)
            ax.axis('off')
            plt.imshow(data[0, :, :, :3])
            ax = plt.subplot(2, 3, 2)
            ax.axis('off')
            plt.imshow(data[1, :, :, :3])
            ax = plt.subplot(2, 3, 3)
            ax.axis('off')
            plt.imshow(np.reshape(data[2].flatten().view(np.single), (data.shape[1], data.shape[2])))
            ax = plt.subplot(2, 3, 4)
            ax.axis('off')
            plt.imshow(data[3, :, :, :3])
            ax = plt.subplot(2, 3, 5)
            ax.axis('off')
            plt.imshow(1.0 - (data[3, :, :, 3] / 255))
            plt.show()
            plt.pause(0.01)

        if self.translator and np.random.rand() > 0.1:
            i = np.random.choice(len(self.translator))
            rgb = self.normalize(torch.from_numpy(data[0, :, :, :3].transpose((2, 0, 1))).float())
            rgb = self.translator[i](rgb.unsqueeze(0).cuda()).squeeze().cpu()
            data[0, :, :, :3] = (rgb * 128 + 128).clamp(0, 255).data.permute(1, 2, 0).numpy().astype(np.uint8)

        self.output.put(data)

    def _watcher(self):
        while True:
            pose = self.input.get()
            self.move_to(pose, self.index)
            self.index += 1


class PoseLoadDataset(Dataset):
    def __init__(self, poses, send_queue, receive_queue, augment=True, randomize=False,
                 crop_size=400, bgr=True):
        if isinstance(poses, str):
            self.poses = load_positions(poses)
        else:
            self.poses = poses

        self.inputs = []
        for i in range(len(self.poses)):
            self.inputs += ['{}.png'.format(i)]

        self.send_queue = send_queue
        self.receive_queue = receive_queue

        self.randomize = randomize
        self.crop_size = crop_size
        self.augment = augment
        self.bgr = bgr

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.id = 0

    def get_image(self, pose):
        self.send_queue.put(pose)
        # outside mechanics need to fill receive_queue with images
        return self.receive_queue.get()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx]
        if self.randomize:
            pose += np.random.normal(0, 0.2, 6)
        data = self.get_image(pose)

        if True:
            #name = self.inputs[idx].split('.')[0]
            name = self.id
            self.id += 1
            image = Image.fromarray(data[0, :, :, :3])
            image.save("images/rgb-{}.png".format(name))
            image = Image.fromarray(data[1, :, :, :3])
            image.save("images/normal-{}.png".format(name))
            image = Image.fromarray((np.reshape(data[2].flatten().view(np.single), (data.shape[1], data.shape[2])) * 255 * 3).astype(np.uint8))
            image.save("images/depth-{}.png".format(name))
            image = Image.fromarray(data[3, :, :, :3])
            image.save("images/segmentation-{}.png".format(name))
            image = Image.fromarray(255 - data[3, :, :, 3])
            image.save("images/edge-{}.png".format(name))

        input = np.float32(data[0, :, :, :3])
        #input = scipy.ndimage.gaussian_filter(input, 1)
        if self.bgr:
            input = input[:,:,::-1].copy()
        input -= np.array([104.00699, 116.66877, 122.67892])
        input = torch.from_numpy(input.transpose((2, 0, 1)))

        output = 1.0 - (np.float32(data[3, :, :, 3]) / 255)
        output = torch.from_numpy(output).unsqueeze(0).float()

        if self.augment:
            x = np.random.randint(0, np.maximum(0, input.size()[1] - self.crop_size))
            y = np.random.randint(0, np.maximum(0, input.size()[2] - self.crop_size))

            input = input[:,x:x+self.crop_size,y:y+self.crop_size]
            output = output[:,x:x+self.crop_size,y:y+self.crop_size]

            if np.random.rand() > 0.5:
                input = torch.flip(input,[2])
                output = torch.flip(output,[2])

        return input, [output]


class NYUDataset(Dataset):
    def __init__(self, path, augment=False, crop_size=400, bgr=True):
        self.root_dir = path
        self.inputs = []
        self.outputs = []

        self.augment = augment
        self.crop_size = crop_size
        self.bgr = bgr

        for i, filename in enumerate(os.listdir(os.path.join(path, 'rgb'))):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.inputs.append(filename)

        for i, filename in enumerate(os.listdir(os.path.join(path, 'edge'))):
            if filename.endswith(".mat") or filename.endswith(".png"):
                self.outputs.append(filename)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = np.float32(Image.open(os.path.join(self.root_dir, 'rgb', self.inputs[idx])))
        if self.bgr:
            input = input[:,:,::-1]
        input -= np.array([104.00699, 116.66877, 122.67892])
        input = torch.from_numpy(input.transpose((2, 0, 1)).copy()).float()

        output = os.path.join(self.root_dir, 'edge', self.outputs[idx])
        if output.endswith(".mat"):
            output = scipy.io.loadmat(output)
            output = output["groundTruth"][0][0][0][0][1].astype(np.bool)
        else:
            output = Image.open(output)

        output = transforms.ToTensor()(output).float()

        if self.augment:
            if np.random.rand() > 0.5:
                input = torch.flip(input,[2])
                output = torch.flip(output,[2])

            x = np.random.randint(0, np.maximum(0, input.size()[1] - self.crop_size))
            y = np.random.randint(0, np.maximum(0, input.size()[2] - self.crop_size))

            input = input[:, x:x + self.crop_size, y:y + self.crop_size]
            output = output[:, x:x + self.crop_size, y:y + self.crop_size]

        return input, [output]




