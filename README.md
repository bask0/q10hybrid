
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

Estimate Q10 and Rb=NN(SW_in, dSW_in) vs Rb=NN(SW_in, dSW_in, T). Due to equifinality, the variant with T is supposed to produe unstable Q10 estimates.

### Usage

Run experiments:

```bash
# start first process on GPU 0 (--restart deletes existing runs)
CUDA_VISIBLE_DEVICES=0 python experiments/experiment_01.py --restart
```

To work on independent runs in parallel, just call the study again from another terminal, **without `--restart`**!

```bash
# start a second process on GPU 1
CUDA_VISIBLE_DEVICES=1 python experiments/experiment_01.py
```

Use `analysis/analysis.ipynb` for evaluation.

## Note

> From the `optuna` doc: `GridSampler` automatically stops the optimization if all combinations in the passed `search_space` have already been evaluated, internally invoking the `stop()` method.

The grid search runs too many combinations, they are cleane in `analysis/analysis.ipynb`.

### Results 

Stable Q10 estimation **without** air temperature as predictor:

![training progress](/analysis/plots/q10_wo_ta.png)

Instable Q10 estimation **with** air temperature as predictor:

![training progress](/analysis/plots/q10_wi_ta.png)

Q10 estimation and loss for different HPs.

![training progress](/analysis/plots/q10_interactions.png)

## Citation

```tex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
