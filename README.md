
Author: B. Kraft [bkraf@bgc-jena.mpg.de]

<div align="center">

# Hybrid modeling of ecosystem respiration temperature sensiticity

</div><br><br>

## Description

Q10 hybrid modeling experiment for a book chapter.

## How to run

First, install dependencies.

```bash
# clone project
git clone https://github.com/bask0/q10hybrid

# Optional: create and activate new conda environment.
conda create --name q10hybrid
conda activate q10hybrid

# install project
cd q10hybrid
pip install -e .   
pip install -r requirements.txt
```

Next, navigate to any file and run it.

```bash
# module folder
cd experiments

# run module (example: mnist as your main contribution)   
python experiment_01.py    
```

## Q10 hybrid modeling experiment

Base respiratino is simulated using observed short-wave irradiation and the delta thereof. Ecosyste respiration is calculated using the [Q10 approach](https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)).

<img src="https://render.githubusercontent.com/render/math?math=Rb_\mathrm{syn} = f(W_\mathrm{in, pot}, \Delta SW_\mathrm{in, pot})"><br>

<img src="https://render.githubusercontent.com/render/math?math=RECO_\mathrm{syn} = Rb_\mathrm{syn} \cdot 1.5^{0.1 \cdot (TA - 15.0)}">

## Experiment 1

[in development]

Estimate Q10 and Rb=NN(SW_in, dSW_in).

#### Usage

Run model: `python train.py`

No options implemented yet, just hybrid model run. Outputs will be written to `lightning_logs/version_xx/`. Predictions are saved as `lightning_logs/version_xx/predictions.nc`.

Use `analysis/analysis.ipynb` for evaluation.

#### First results

![training progress](/analysis/plots/predictions.png)

![training progress](/analysis/plots/q10.png)

### Experiment 2



### Citation

```tex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
