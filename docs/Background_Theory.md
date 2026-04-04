# Background Theory

## Overview 
This document outlines the core theoretical mechanics of the generative and ensemble algorithms utilized within the CreditCardFraudRnD framework. This theory specifically focuses on how deep learning and adversarial data generation can be leveraged to secure modern financial ecosystems.

## 1. Variational Autoencoders (VAEs)
A Variational Autoencoder (VAE) is an unsupervised deep learning model capable of capturing complex probability distributions of input data. Traditional autoencoders compress data into a static bottleneck (latent space) and attempt to reconstruct it. A VAE diverges from this by mapping the input into a **probabilistic distribution** characterized by a mean ($\mu$) and a standard deviation ($\sigma$).

### Why VAEs for Fraud Detection?
Legitimate transactions typically follow a highly structured (but complex) mathematical distribution. Fraud, however, consists of anomalies that fall *outside* this normal distribution. 
By primarily training a VAE on millions of legitimate transactions, the networking learns to compress and decompress normal payments with low error. When a fraudulent vector (such as a synthetic identity) enters the network, the VAE attempts to reconstruct it based on the rules of normal transactions. Because the fraud vector is out-of-distribution, the VAE fails to decode it perfectly, resulting in a **high Reconstruction Error**. This error acts as a highly effective anomaly score.

## 2. Graph Attention Networks (GATs)
Money laundering and CNP (Card-Not-Present) fraud rarely happen in isolation; they involve interconnected networks of IP addresses, merchant IDs, and compromised cards. Graph Neural Networks (GNNs) analyze these relationships.

Our framework integrates a **Graph Attention (GAT)** mechanism via PyTorch Multihead Attention. Unlike standard Graph Convolutional Networks (GCNs) that assign equal weight to all neighboring nodes, GATs use self-attention coefficients to weigh the influence of surrounding transactions dynamically. A transaction sharing a device fingerprint with known fraudulent nodes will receive "attention" from those nodes during matrix multiplication, immediately spiking its localized risk parameters before it passes into the VAE bottleneck.

## 3. Wasserstein GAN with Gradient Penalty (WGAN-GP)
Generative Adversarial Networks (GANs) consist of a Generator (creating fake data) and a Critic. The Critic attempts to distinguish real from fake data. 

### Overcoming Class Imbalance in Finance
In modern payment networks, fraud accounts for $<0.2\%$ of all transactions. Deep learning classifiers trained on heavily imbalanced datasets tend to naturally predict "Normal" for everything, as it yields 99.8% accuracy mathematically.

To solve this, the framework uses a WGAN-GP trained exclusively on the minority class (fraud). Unlike SMOTE or random oversampling, which merely interpolate or duplicate points, the WGAN-GP synthesizes entirely new, hyper-realistic, complex fraud feature vectors. By generating 5,000 algorithmic "fake" fraud behaviors and mixing them into the training set, we dramatically balance the classification boundary, allowing the downstream classifier to learn distinct discriminative rules without bias.

## 4. Continuous Learning (Streaming SGD)
In traditional systems, ML models are trained on historical data and deployed in batches. However, as attackers pivot into new "zero-day" attack vectors (Concept Drift), batch models become rapidly obsolete.

To secure realtime payments, the backend employs **Streaming Machine Learning**. Using a Stochastic Gradient Descent (SGD) framework parameterized via logarithmic loss, the model operates in a state of continuous adaptation. It does not train once; instead, it processes every incoming transaction incrementally (via `partial_fit`). If a fraudster switches from exploiting a VPN node to launching a velocity attack, the decision bounds natively shift within seconds to intercept the new trajectory without requiring human intervention or overnight retraining.
