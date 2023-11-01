# Multiple Instance Learning Experiments

## Details
This implementation only supports CPU training.

## Reference
```
@inproceedings{ilse2018attention,
  title={Attention-based deep multiple instance learning},
  author={Ilse, Maximilian and Tomczak, Jakub and Welling, Max},
  booktitle={International conference on machine learning},
  pages={2127--2136},
  year={2018},
  organization={PMLR}
}
```
## Dataset

### MNIST
We have code to construct the dataset. We ensure that half the bags are positive and half are negative. We then use $1000$ test bags sampled from the distribution we are interseted in.

### Colon Cancer
Download dataset from https://github.com/utayao/Atten_Deep_MIL.


## Quick Experiment

### MNIST
```
python train_count.py --lr=5e-4 --weight_decay=1e-4 --mean_bag_size=10 --variance_bags=2 --num_train_bags=50 --epochs=200 --file_name=experiment_results
```

### Colon Cancer
```
python train.py --lr=1e-4 --L2=5e-4 --file_name=colon_results --epochs=100
```

## Baseline Code
[Ref Implementation](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)


