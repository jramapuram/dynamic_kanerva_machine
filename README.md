# Dynamic Kanerva Machine (DKM) Pytorch

An (unofficial) implementation of Dynamic Kanerva Machines in pytorch.  
Ported from the [tensorflow / sonnet implementation](https://github.com/deepmind/dynamic-kanerva-machines).


## Setup

Clone the repo **WITH** the submodules. If you get an error about permissions setup your [github key](https://docs.github.com/en/enterprise/2.15/user/articles/adding-a-new-ssh-key-to-your-github-account).

``` bash
git clone --recursive git+ssh://git@github.com/jramapuram/dynamic_kanerva_machine.git
```

The dependencies are fully captured in a docker container:

```bash
docker pull jramapuram/pytorch:1.6.0-cuda10.1
```

If you want to setup your own environment use:

  - environment.yml (conda) in addition to
  - requirements.txt (pip)


#### Usage Binarized MNIST

``` bash
sh docker/run.sh "python main.py" 0  # tailing 0 runs on cuda device 0
```

#### Citation

Cite the original authors on doing some great work:

```
@inproceedings{wu2018learning,
  title={Learning attractor dynamics for generative memory},
  author={Wu, Yan and Wayne, Gregory and Gregor, Karol and Lillicrap, Timothy},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9379--9388},
  year={2018}
}
```

Like this replication? Buy me [a beer](https://github.com/sponsors/jramapuram).

