import os
texture_types = ["rgb", "normals", "depth", "segmentation"]
texture_channels = [3, 3, 4, 4]
if os.name == 'nt':
    unity_path = "../UnitySimulationWindows/UnitySimulator.exe"
else:
    unity_path = "../UnitySimulationLinux/Simulator.x86_64"
command_line_args = ["./" + unity_path, "-batchmode", "-nolog"]

nyu_path = "data/nyudv2/nyu_depth_v2_labeled.mat"
nyu_split_path = "data/nyudv2/splits.mat"

INIT_CMD = 0
START_REC_CMD = 1
REC_CMD = 2
DONE_REC_CMD = 3
PLAY_CMD = 4
DONE_PLAY_CMD = 5
CAPTURE_CMD = 6

import numpy as np

class_labels = np.array([
[0x00, 0x00, 0x00],
#"Roof", "Bar", "House"
[0xff, 0x09, 0xe0],
#"Wall"
[0x78, 0x78, 0x78],
#Ivy", "Tree", "Branch", "Plant"
[0x04, 0xc8, 0x03],
#"Decor"
[0x00, 0xff, 0xcc],
#"Floor"
[0x50, 0x32, 0x32],
#"Ground"
[0x78, 0x78, 0x46],
#"Cloth", "Shoe"
[0x00, 0x70, 0xff],
#"Mug", "Bowl", "Can", "Kettle", "Dish"
[0x00, 0xff, 0x0a],
#"Book" },
[0xff, 0xa3, 0x00],
#"Table", "Console" },
[0xff, 0x06, 0x52],
#"Desk" },
[0x0a, 0xff, 0x47],
#"Chair" },
[0xcc, 0x46, 0x03],
#"Armchair" },
[0x08, 0xff, 0xd6],
#"Bed" },
[0xcc, 0x05, 0xff],
#"Bookcase" },
[0x00, 0xff, 0xf5],
#"Commode", "Drawer", "Cupboard" },
[0xe0, 0x05, 0xff],
#"Closet", "Wardrobe" },
[0x07, 0xff, 0xff],
#"Displaycase" },
[0x00, 0x00, 0xff],
#"Couch", "Sofa" },
[0x0b, 0x66, 0xff],
#"Mirror" },
[0xdc, 0xdc, 0xdc],
#"Lamp", "Chandelier", "Sconce", "Light", "Spotlight" },
[0xe0, 0xff, 0x08],
#"Pissoir", "Toilet", "WC" },
[0x00, 0xff, 0x85],
#"Sink" },
[0x00, 0xa3, 0xff],
#"Candle" },
[0xff, 0xad, 0x00],
#"Picture" },
[0xff, 0x06, 0x33],
#"Frame" },
[0x7f, 0x00, 0x00],      #Custom
#"Vase" },
[0x00, 0xff, 0xcc],
#"Box" },
[0x00, 0xff, 0x14],
#"Basket", "Cart" },
[0x5c, 0xff, 0x00],
#"Carpet" },
[0xff, 0x09, 0x5c],
#"Trashcan" },
[0xad, 0x00, 0xff],
#"Footstool" },
[0xff, 0x99, 0x00],
#"Door" },
[0x08, 0xff, 0x33],
#"Stair" },
[0x1f, 0x00, 0xff],
#"Pillow" },
[0x00, 0xeb, 0xff],
#"Blanket", "Cover" },
[0x14, 0x00, 0xff],
#"Cable" },
[0xff, 0xff, 0x3f],       #Custom
#"Curtain" },
[0xff, 0x33, 0x07],
#"Bench" },
[0xc2, 0xff, 0x00],
#"Fireplace" },
[0xfa, 0x0a, 0x0f],
#"Fridge" },
[0x14, 0xff, 0x00],
#"Kitchen" },
[0x00, 0xff, 0x29],
#"Heater", "Radiator"
[0xff, 0xd6, 0x00]
]).astype(np.uint8)
"""
class_labels = np.round((np.array([
    [0.0, 0.0, 0.0],     # nothing
    [0.0, 0.5, 0.0],     # vegetation
    [1.0, 0.25, 1.0],    # decoration
    [1.0, 1.0, 0.5],     # dishes
    [1.0, 1.0, 0.0],     # books
    [1.0, 0.0, 0.0],     # table
    [0.0, 1.0, 0.0],     # chair
    [0.0, 1.0, 1.0],     # bed
    [0.0, 1.0, 0.5],     # drawers
    [0.0, 0.0, 1.0],     # couch
    [1.0, 1.0, 1.0],     # mirror
    [1.0, 0.0, 1.0],     # lamp
    [0.5, 0.5, 0.5],     # toilet
    [0.0, 0.0, 0.5],     # sink
    [0.0, 0.5, 1.0],     # candle
    [1.0, 0.0, 0.5],     # picture
    [0.5, 0.0, 0.0],     # frame
    [0.5, 0.0, 1.0],     # vase
    [1.0, 0.5, 0.0],     # storage
    [0.0, 0.0, 0.25],    # carpet
    [0.5, 0.5, 1.0],     # trashcan
    [0.5, 1.0, 0.5],     # footstool
    [0.5, 0.0, 0.5],     # door
    [0.5, 0.5, 0.0],     # stair
    [0.25, 0.0, 0.0],    # pillow
    [1.0, 1.0, 0.25],    # cable
    [0.25, 0.0, 0.25],   # roof
    [0.0, 0.25, 0.25],   # curtain
    [0.25, 0.25, 0.25],  # bench
    [0.25, 0.25, 0.0],   # fridge
    [0.25, 1.0, 0.25],   # kitchen
    ]) * 255) - 0.001).astype(np.uint8)
"""