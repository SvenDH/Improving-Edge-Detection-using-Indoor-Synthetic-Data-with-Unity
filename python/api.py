import queue
import select
import socket
import struct
import subprocess
import threading
from PIL import Image

from const import *
from utils import *


class Simulation(object):

    def __init__(self, *argv,
                 save=False,
                 host="localhost",
                 port=49500,
                 width=640,
                 height=480,
                 FOV=45.0,
                 near=0.01,
                 far=100.0,
                 compress=False,
                 run_process=False):

        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.FOV = FOV
        self.near = near
        self.far = far
        self.channels = 4
        self.save = save
        self.compress = compress
        self.run_process = run_process
        self.pose_buffer = []
        self._buffer_size = 4194304
        self._internal_send_queue = queue.Queue()
        self._internal_receive_queue = queue.Queue()

        if self.run_process:
            print("Starting Unity simulation with {} ...".format(
                argv))
            self._client_process = subprocess.Popen(argv) #"-screen-width " + str(width), "-screen-height " + str(height),
        try:
            self._rsock, self._ssock = socket.socketpair()
            self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except:
            raise Exception("Couldn't start socket.")

        threading.Thread(target=self._connection, daemon=True).start()
        threading.Thread(target=self._update, daemon=True).start()

    def move_to(self, pose, idx):
        bytes = struct.pack("i7f", PLAY_CMD, idx, *pose)
        self._internal_send_queue.put(bytes)
        self._ssock.send(b"\x00")

    def pose_callback(self, pose, idx):
        print("received pose {}".format(idx))

    def image_callback(self, data, idx):
        print("received image {}".format(idx))

    def _connect(self):
        for _ in range(100000):
            try:
                self._conn.connect((self.host, self.port))
                print("Connected to Unity")
                break
            except ConnectionRefusedError:
                if self.run_process:
                    time.sleep(1)
                else:
                    raise Exception("Couldn't connect to Unity")

    def _connection(self):
        self._connect()
        data = struct.pack("3i3f?", INIT_CMD, self.width, self.height, self.FOV, self.near, self.far, self.compress)
        self._send(data)
        time.sleep(1)  # give unity time to prepare
        while True:
            rlist, _, _ = select.select([self._conn, self._rsock], [], [])
            for ready_socket in rlist:
                if ready_socket is self._conn:
                    self._internal_receive_queue.put(self._receive())
                else:
                    self._rsock.recv(1)
                    self._send(self._internal_send_queue.get())
                    self._internal_send_queue.task_done()

    def _update(self):
        while True:
            data = self._internal_receive_queue.get()
            command = struct.unpack("I", data[:4])[0]
            data = data[4:]
            if command == INIT_CMD:
                self._settings = [*struct.unpack("2i3f?", data)]
                print(self._settings)
            elif command == START_REC_CMD:
                if self.save:  # clear motion file
                    open(data_fn, 'w').close()
                self.pose_buffer = []
            elif command == REC_CMD:
                # receive 3d position and rotation angles
                pose = struct.unpack("6f", data)
                self.pose_callback(pose, len(self.pose_buffer))
                self.pose_buffer.append(pose)
            elif command == DONE_REC_CMD:
                # append motion file
                if self.save:
                    with open(data_fn, "a") as f:
                        for pose in self.pose_buffer:
                            f.write(','.join(str(e) for e in pose) + '\n')
            elif command == CAPTURE_CMD:
                # receive header
                seq_nr = struct.unpack("I", bytearray(data[:4]))[0]
                idx = 4
                if self.compress:
                    output = np.empty((len(texture_types), self.height, self.width, self.channels), dtype=np.uint8)
                    for i, type in enumerate(texture_types):
                        length = struct.unpack("I", bytearray(data[idx:idx+4]))[0]
                        image = np.array(
                            Image.open(io.BytesIO(data[idx + 4:idx + 4 + length])), dtype=np.uint8)

                        output[i, :, :, :] = image
                        if i is not len(texture_types) - 1:
                            idx = idx + 4 + length

                else:
                    output = np.array(np.frombuffer(data[idx:], dtype=np.uint8).reshape(
                        (len(texture_types), self.height, self.width, self.channels))[:, ::-1, :, :])

                if self.save:  # save to png file
                    for i, type in enumerate(texture_types):
                        im_save = Image.fromarray(output[i, :, :, :texture_channels[i]])
                        im_save.save(os.path.join("images", type + "-" + str(seq_nr) + ".png"))

                self.image_callback(output, seq_nr)

            self._internal_receive_queue.task_done()

    def _receive(self):
        try:
            data = self._conn.recv(self._buffer_size)
            if len(data) > 0:
                length = struct.unpack("I", bytearray(data[:4]))[0]
                data = data[4:]
                while len(data) < length:
                    data += self._conn.recv(min(self._buffer_size, length - len(data)))
        except socket.timeout as e:
            raise Exception("The environment took too long to respond.")
        return data

    def _send(self, data):
        self._conn.sendall(struct.pack("I", len(data)) + data)

    def __del__(self):
        self._conn.close()
        if self.run_process:
            self._client_process.kill()
            time.sleep(5)


# For testing the simulation
if __name__ == "__main__":
    simulation = Simulation(unity_path, *command_line_args)

    pos = [0.0,0.0,0.0,0.0,0.0,0.0]
    simulation.move_to(pos, 0)
