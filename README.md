# behavior
A collection of tools for analyzing behavioral data

## Environment Set-Up

Create a conda environment:

```
$: conda create --name=behavior python=3.6
$: source activate behavior
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
