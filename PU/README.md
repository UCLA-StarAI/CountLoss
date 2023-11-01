# PU Learning Experiments

## References
```
@inproceedings{garg2021PUlearning,
    title={Mixture Proportion Estimation and {PU} Learning: A Modern Approach},
    author={Garg, Saurabh and Wu, Yifan and Smola, Alex and Balakrishnan, Sivaraman and Lipton, Zachary},
    year={2021},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)} 
}
```
## Quick Experiments
```
# Distribution
python train.py --data-type="cifar_DogCat" --train-method="Count" --net-type="ResNet_out1" --epochs=1000 --alpha=0.5 --beta=0.5

# Penalize EV
python train.py --data-type="mnist_17" --train-method="Count_EV" --net-type="MLP1" --epochs=1000 --alpha=0.5 --beta=0.5
```
## Baseline Code
[Ref Implementation](https://github.com/acmi-lab/PU_learning)