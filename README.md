# Deep reinforcement learning for dynamic scheduling of a flexible job shop

This repository includes the code of algorithms used in the following paper: 

**Liu, R.**, Piplani, R., & Toro, C. (2022). Deep reinforcement learning for dynamic scheduling of a flexible job shop. International Journal of Production Research 2022 Vol. 60 Issue 13 Pages 4049-4069. https://doi.org/10.1080/00207543.2022.2058432

Free Eprint link: https://www.tandfonline.com/eprint/ITWCDGRSWFXS7G37PGBQ/full?target=10.1080/00207543.2022.2058432 

Feel free to revise the code, and good luck with your research!

If you make use of the code in your work, please consider citing our paper (Bibtex below):

    @article{liu2022deep,
      title={Deep reinforcement learning for dynamic scheduling of a flexible job shop},
      author={Liu, Renke and Piplani, Rajesh and Toro, Carlos},
      journal={International Journal of Production Research},
      volume={60}
      number={13}
      pages={4049--4069},
      year={2022},
      publisher={Taylor \& Francis}
    }

## Repository Overview

This repo includes the code of simulation model, learning algorithm, and experimentation. Those files can be identified by their prefix or suffix:
> 1. "agent": assets on shop floor: machines and work centers;
> 2. "brain": learning algorithm for job sequencing and machine routing;
> 3. "creation": dynamic events on shop floor, such as job arrival and machine breakdown;
> 4. "main": actual simulation processes, to train, or validate deep MARL-based algorithms;
> 5. "routing" and "sequencing: priority rules
> 6. "validation": modules to import trained parameters for experiment.

Data and trained parameters can be found in folders:
> 1. "experiment_result": results of experiments in dynamic environments;
> 2. "routing models": trained routing agent parameters;
> 3. "sequencing_models": trained sequencing agent parameters.

## User Guide

To use our code as the benchmark, kindly refer to "routing_models" and "sequencing_models" folder for the trained parameters; use the class "network_validated" within "brain_machine_S.py" file to build the sequencing neural network; use "build_network_small" to "build_network_large" in "brain_workcenter-R" to build routing networks of different size. The state and action functions can also be found in "brain_...py" files.

An alternative way is to test your approach in our simulation model and context, you may create your algorithm and run the simulation in "main_experiment.py" for comparison. Please refer to the comments in each module to see how to interact with the simulation model.
