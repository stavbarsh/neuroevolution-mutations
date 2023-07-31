# Towards Improving Mutations in Neuroevolution

This is the code implementation for the paper.

The paper investigate the prevalent mutation technique of additive Gaussian noise and suggest to improve it.

The algorithm is based on the neuroevolution algorithm by [Uber AI](https://arxiv.org/abs/1712.06567).

The code is base on the reference implementations by [uber](https://github.com/uber-research/deep-neuroevolution)
, [openai](https://github.com/openai/evolution-strategies-starter) and a reference implementation using
MPI [here](https://github.com/sash-a/es_pytorch).

### How to run

* Make sure to have Mujoco installed.
* Install conda environment: `conda install -n env -f env.yml`
* Select configuration from `config/`

run a single example using:

```
conda activate env
mpirun -np {num_procs} python gym_experiment.py configs/chsw.json
```

run all experiments using:

```
bash loop.sh
```
