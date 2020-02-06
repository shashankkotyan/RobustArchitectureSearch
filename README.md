# Robust Architecture Search

This github repository contains the official code for the paper,

> [Evolving Robust Neural Architectures to Defend from Adversarial Attacks](https://arxiv.org/abs/1906.11667)\
> Danilo Vasconcellos Vargas, Shashank Kotyan\
> _arXiv:1906.11667_.
 
# IMPORTANT

**In the current version of the code, the robustness evalution is not implemented in the code. We are currently testing for the compatibility and reproducibility of the adversarial examples and results before adding it in the repository to ensure the quality of the code remains intact.**

**In the meantime, one can generate their own adversarial samples using the repository of [Dual Quality Assessment](https://github.com/shashankkotyan/DualQualityAssessment) and make necessary changes in fitness evaluation of the evolved models in `run_model` function in `worker.py` file. Fitness of the evolved models is calculated in the `line 113`**

## Citation

If this work helps your research and/or project in anyway, please cite:

```
@article{vargas2019evolving,
  title   = {Evolving Robust Neural Architectures to Defend from Adversarial Attacks},
  author  = {Vargas, Danilo Vasconcellos and Kotyan, Shashank},
  journal = {arXiv preprint arXiv:1906.11667},
  year    = {2019}
}
```

## Testing Environment 

The code is tested on Ubuntu 18.04.3 with Python 3.7.4.

## Getting Started

### Requirements

To run the code in the tutorial locally, it is recommended, 
- a dedicated GPU suitable for running, and
- install Anaconda. 

The following python packages are required to run the code. 
- `GPUtil==1.4.0`
- `matplotlib==3.1.1`
- `networkx==2.3`
- `numpy==1.17.2`
- `scipy==1.4.1`
- `tensorflow==2.1.0`

---

### Steps

1. Clone the repository.

```bash
git clone https://github.com/shashankkotyan/RobustArchitectureSearch.git
cd ./RobustArchitectureSearch
```

2. Create a virtual environment 

```bash
conda create --name ras python=3.7.4
conda activate ras
```

3. Install the python packages in `requirements.txt` if you don't have them already.

```bash
pip install -r ./requirements.txt
```

4. Run the Robust Architecture Search Code with the following command.

```bash
python -u run_evolution.py [ARGS] > run.txt
```

5. Calculate the statstics for the evolution.

```bash
python -u run_stats.py > run_stats.txt     
```

## Arguments to run run_evolution.py

TBD

## Notes

- It is recommended to run the code on a multi-gpu system to ensure faster evolution. However, changing the number of workers in `num_workers.txt` to 1 will ensure the evolution on a single GPU system. 
- Setting `num_workers.txt` to number of GPUs your system has will run the code optimally utilising maximum performance by the GPUs.

## Milestones

- [ ] **Include Robustness Evaluation**
- [ ] Toy Example for Evolutionary Strategy
- [ ] Addition of Comments in the Code
- [ ] Cross Platform Compatibility
- [ ] Description of Method in Readme File

## License

Robust Architecture Search is licensed under the MIT license. 
Contributors agree to license their contributions under the MIT license.

## Contributors and Acknowledgements

TBD

## Reaching out

You can reach me at shashankkotyan@gmail.com or [\@shashankkotyan](https://twitter.com/shashankkotyan).
If you tweet about Robust Architecture Search, please use the tag `#RAS` and/or mention me ([\@shashankkotyan](https://twitter.com/shashankkotyan)) in the tweet.
For bug reports, questions, and suggestions, use [Github issues](https://github.com/shashankkotyan/RobustArchitectureSearch/issues).
