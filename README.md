# behavior
A collection of tools for analyzing behavioral data


## Installation

First you'll have to install the `git` package in order to access the code on github. Follow the directions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for your specific OS.
Then, in the command line, navigate to where you'd like to install the `behavior` package and move into that directory:
```
$: git clone https://github.com/themattinthehatt/behavior
$: cd behavior
```

Next, follow the directions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install the `conda` package for managing development environments. 
Then, create a conda environment:

```
$: conda create --name=behavior python=3.6
$: conda activate behavior
(behavior) $: pip install -r requirements.txt 
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `behavior` directory:

```
(behavior) $: pip install -e .
```

To be able to use this environment for jupyter notebooks:

```
(behavior) $: python -m ipykernel install --user --name behavior
``` 

To install ssm, `cd` to any directory where you would like to keep the ssm code and run the following:

```
(behavior) $: git clone git@github.com:slinderman/ssm.git
(behavior) $: cd ssm
(behavior) $: pip install cython
(behavior) $: pip install -e .
```

## Set paths

Next, you should create a file in the `behavior` package named `paths.py` that looks like the following:

```python
# labels should be stored in the following format:
# DATA_PATH/labels/[expt_id]_labeled.[csv/h5]
#
# videos should be stored in the following format:
# DATA_PATH/videos_cropped/[expt_id].avi

DATA_PATH = '/top/level/data/path/'
RESULTS_PATH = '/top/level/results/path/'
```

This file contains the local paths on your machine, and will not be synced with github.
