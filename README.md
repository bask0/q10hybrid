
Author: B. Kraft [bkraf@bgc-jena.mpg.de]

<div align="center">

# Hybrid modeling of ecosystem respiration temperature sensitivity

</div><br><br>

## Description

Q10 hybrid modeling experiment for a book chapter.

## How to run

First, install dependencies.

```bash
# clone project
git clone https://github.com/bask0/q10hybrid

# Optional: create and activate new conda environment.
conda create --yes --name q10hybrid python=3.6
conda activate q10hybrid

# install project
cd q10hybrid
pip install -e .
pip install -r requirements.txt
```

## Q10 hybrid modeling experiment

Base respiration is simulated using observed short-wave irradiation and the delta thereof. Ecosyste respiration is calculated using the [Q10 approach](https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)).

<img src="https://render.githubusercontent.com/render/math?math=Rb_\mathrm{syn} = f(W_\mathrm{in, pot}, \Delta SW_\mathrm{in, pot})"><br>

<img src="https://render.githubusercontent.com/render/math?math=RECO_\mathrm{syn} = Rb_\mathrm{syn} \cdot 1.5^{0.1 \cdot (TA - 15.0)}">

## Experiment

Estimate Q10 in two different setups:

* Rb=NN(SW_in, dSW_in)
* Rb=NN(SW_in, dSW_in, T)

We investigate wheter we can estimate Q10 in both cases robustly and how model hyperparameters (here: dropout={0.0, 0.2, 0.4, 0.6}) impact the results.

![data](/analysis/plots/data.png)

### Usage

Run experiments:

```bash
# Create a new study (delete old runs).
python experiments/experiment_01.py --create_study
```

```bash
# Start first process on GPU 0.
CUDA_VISIBLE_DEVICES="0" python experiments/experiment_01.py
```

To work on independent runs in parallel, just call the study again from another terminal!

```bash
# Start a second process on GPU 1.
CUDA_VISIBLE_DEVICES="1" python experiments/experiment_01.py
```

Alternatively, you can use `run_experiment.py` to create a new study and spawn multiple processes, for example with 12 jobs distributed
on 4 GPUs (0,1,2,3). 
```bash
# Start a second process on GPU 1.
CUDA_VISIBLE_DEVICES="0,1,2,3" python run_experiment.py --num_jobs 12
```

Use `analysis/analysis.ipynb` for evaluation.

## Note

> From the `optuna` doc: `GridSampler` automatically stops the optimization if all combinations in the passed `search_space` have already been evaluated, internally invoking the `stop()` method.

The grid search runs too many combinations, they are cleane in `analysis/analysis.ipynb`.

### Results 

Q10 estimation **without** (top) and **with** (bottom) air temperature as predictor:

![training progress](/analysis/plots/q10_training.png)

Q10 estimation and loss for different HPs.

![Q10 interactions](/analysis/plots/q10_interactions.png)

## Citation

```tex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
