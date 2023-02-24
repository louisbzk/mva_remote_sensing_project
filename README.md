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
