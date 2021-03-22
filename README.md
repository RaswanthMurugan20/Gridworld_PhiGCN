# Reward Progpagation using Graph Convolutional Networks in GridWorld

The repository contains the code for running the experiments in the paper [Reward Propagation using Graph Convolutional Networks](https://arxiv.org/abs/2010.02474) using the Proto Value Functions. The implementation is based on a few source codes: Thomas Kipf's [pytorch GCN implementation](https://github.com/tkipf/pygcn). The environment currently is only a GridWorld and all the results have been produced using this environment. The actor critic network implementation was not from any library but our own implementation using linear function approximators.

## Getting Started

For a quick start clone the repository, and type the following command.
```
$ git clone <repo link>
```

```
$ python AC.py 
```


### Installing

```python
# PyTorch
$ conda install pytorch torchvision -c soumith

# Other requirements
$ pip install -r requirements.txt

#Installing PyGCN
$ python setup_gcn.py install
```



## Usage

### GridWorld


```
$ python AC.py --n 15 --m 15 --nt 14 --mt 14 --episodes 1000 --hidden 64 --gcn_lambda 2 --lr 0.01 --gcn_epochs 100
```

## Results

![Reward Propagation](/images/1.png)
![Regret Plot](/images/2.png)
![Loss Plot](/images/3.png)


## Built With

* [Pytorch GCN implementation](https://github.com/tkipf/pygcn)
* [Python](https://python.org)

## Authors

* **Raswanth Murugan** - [RaswanthMurugan20](https://github.com/RaswanthMurugan20)

* **Alisetti Sai Vamsi** - [Vamsi995](https://github.com/Vamsi995)

## Acknowledgments

* **Chandra Shekar Lakshminarayan**
