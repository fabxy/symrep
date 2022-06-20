# symrep

## Representation learning via symbolic regression

This project is created as a part of the [2022 Flatiron Machine Learning x Science Summer School](https://www.simonsfoundation.org/grant/2022-flatiron-machine-learning-x-science-summer-school). 

The goal is using deep learning to accelerate symbolic regression and model discovery. We aim to break down complex symbolic regression problems into simpler subproblems. We train deep neural networks on the data-of-interest in order to learn disentangled latent representations. These representations are then learned (and combined) via symbolic regression.

## Plan

1. Create generic data from algebraic equations

2. Run symbolic regression on generic data

3. Train plain MLP

4. Train MLP with $L_1$ regularization on latent features

5. Train [DSN](https://astroautomata.com/data/sjnn_paper.pdf) with $L_1$ regularization on latent features

6. Run symbolic regression on latent feature predictions

7. Implement symbolic discriminator

## TODO

- [] Nothing