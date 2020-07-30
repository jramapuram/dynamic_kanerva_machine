# Dynamic Kanerva Machine (DKM) Pytorch

An implementation of Dynamic Kanerva Machines in pytorch.  
Ported from the [tensorflow / sonnet implementation](https://github.com/deepmind/dynamic-kanerva-machines).


## Setup

Clone the repo **WITH** the submodules. If you get an error about permissions setup your [github key](https://docs.github.com/en/enterprise/2.15/user/articles/adding-a-new-ssh-key-to-your-github-account)

``` bash
git clone --recursive git+ssh://git@github.com/jramapuram/dynamic_kanerva_machine.git     # clone the repo WITH submodules
```

The dependencies are fully captured in a docker container:

```bash
docker pull jramapuram/pytorch:1.6.0-cuda10.1
```

#### Usage Binarized MNIST

Change the transforms in `supervised_main.py` appropriately for cifar10 and:

``` bash
sh docker/run.sh "python main.py" 0  # tailing 0 runs on cuda device 0
```
