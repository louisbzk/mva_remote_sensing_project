# MVA - Remote Sensing Project

## Baseline run

To run the training procedure, the code works with the following directory tree

```bash
.
├── checkpoint
├── data
│   ├── sample
│   │   └── run_0
│   │       ├── test
│   │       └── val
│   ├── test
│   │   ├── gt
│   │   │   ├── npy
│   │   │   └── png
│   │   └── raw
│   ├── train
│   │   ├── gt
│   │   │   ├── npy
│   │   │   └── png
│   │   └── raw
│   │       └── npy
│   └── val
│       ├── gt
│       │   ├── npy
│       │   └── png
│       └── raw
```

and the command to execute is

`python src/main.py --training_set data/train/raw/npy`

executed from the project's root directory (`mva_remote_sensing_project`)

To run training while supplying line maps, run the command

```
python src/main.py
--training_set data/train/gt/npy
--checkpoint_dir <checkpoint_dir>
--loss l1
--train_line_detection_path <path to train lines>
--test_line_detection_path <path to test lines>
```

The line maps are `.npy` files which must have exactly the same name as their image counterpart, e.g. image
`data/train/gt/npy/lely.npy` has a line map `data/train/lines/lely.npy`, the same height and width, and values between
0 and 1. They must be grayscale images (i.e. they are 2D arrays, not multi-channel images).
