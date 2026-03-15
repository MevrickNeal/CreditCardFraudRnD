# Research Methodology

## 3.1 Research Philosophy and Design
This study employs a **Design Science Research (DSR)** methodology, which prioritizes the creation and evaluation of artificial artifacts to solve identified real-world problems. The objective is to develop a functional, high-accuracy fintech fraud detection framework. The research design integrates a **Pragmatic Approach**, focusing on empirical validation via quantitative performance metrics and real-time operational feasibility.

The framework adheres to three core design principles:
1. **Real-time Feasibility**: Ensuring sub-millisecond inference latency for frictionless authorization.
2. **Proactive Resilience**: Using generative modeling to anticipate "zero-day" fraud vectors before they manifest in historical logs.
3. **Explainability (XAI)**: Providing transparent, feature-level insights for regulatory compliance and auditability.

---

## 3.2 Data Selection and Environment
The research leverages the **IEEE-CIS Fraud Detection Dataset** (sourced from Vesta Corporation via Kaggle). This dataset is chosen for its high dimensionality (400+ features) and realistic representation of complex digital payment behavior.

| Property | Detail |
| :--- | :--- |
| **Primary Source** | IEEE-CIS / Vesta Corporation |
| **Observations** | ~590,000 Transactions (joined with Identity metadata) |
| **Class Imbalance** | ~3.5% Fraud Prevalence (Minority Class) |
| **Complexity** | Sparse categorical variables, heavy missing values, extreme dimensionality |

---

## 3.3 Data Preprocessing Pipeline
To transform raw multi-modal payment data into ML-ready tensors, a modular five-stage pipeline was implemented:

1. **Deterministic Merging**: Relational join between transaction and identity databases using a unique `TransactionID` key.
2. **Stratified Sampling**: A 10% stratified sample was utilized for local training efficiency while preserving the critical 3.5% fraud distribution.
3. **Missing Value Management**: Sentinel imputation (`-999`) was applied to numerical features, while categorical nulls were treated as an explicit "missing" class to preserve structural information.
4. **Categorical Encoding**: High-cardinality nominal features were transformed via **Label Encoding** to enable tensor compatibility.
5. **Feature Standardization**: Global feature scaling was applied via **StandardScaler** to ensure uniform gradient propagation across deep learning layers.

---

## 3.4 Hybrid Generative Framework Architecture
The core architectural contribution is a multi-stage hybrid engine that synthesizes generative modeling with continuous learning.

### 3.4.1 Unsupervised Anomaly Modeling (VAE)
The **Variational Autoencoder (VAE)** is utilized for its ability to model complex, high-dimensional probability distributions of legitimate transaction data. The architecture consists of:
- **Encoder Network ($q_\phi(z|x)$)**: A multi-layer perceptron that compresses the 432-dimensional input feature vector into a lower-dimensional latent space. It outputs two vectors: mean ($\mu$) and standard deviation ($\sigma$).
- **Reparameterization Trick**: To enable backpropagation through stochastic layers, the latent variable $z$ is sampled as $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
- **Decoder Network ($p_\theta(x|z)$)**: A symmetric network that attempts to reconstruct the original transaction features from the latent $z$.
- **Loss Function**: The model is optimized using the **Evidence Lower Bound (ELBO)**, combining:
  1. **Reconstruction Loss**: Mean Squared Error (MSE) between the input $x$ and reconstruction $x'$, measuring how well the model understands "normal" patterns.
  2. **KL Divergence**: Regularizes the latent space toward a standard Gaussian distribution, preventing overfitting to specific transaction identities.

Transactions with high reconstruction error are flagged as anomalies, as they represent behaviors that the VAE cannot "explain" based on its training on legitimate data.

### 3.4.2 Adversarial Synthetic Data Generation (WGAN-GP)
To address the critical "Card-Not-Present" (CNP) fraud data scarcity, a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** is employed. Unlike standard GANs which suffer from mode collapse, WGAN-GP provides a more stable training signal:
- **Generator ($G$)**: Learns to map random noise and conditional attack labels (e.g., "VPN Anomaly") to realistic transaction feature vectors.
- **Critic ($D$)**: Unlike a traditional discriminator, the Critic outputs a scalar value representing the "realness" or quality of the sample. It is trained to estimate the Wasserstein-1 distance between the real and synthetic distributions.
- **Gradient Penalty ($\lambda_{gp}$)**: A 1-Lipschitz continuity constraint is enforced via a gradient penalty on the Critic's norm. This ensures that the gradient doesn't vanish or explode, which is vital when synthesizing high-dimensional tabular data like the IEEE-CIS features.

The resulting synthetic samples are used to "stress-test" the detection framework and pre-train the ensemble on anticipated "Frankenstein" attack patterns.

### 3.4.3 Continuous Learning Ensemble (Streaming SGD)
Acknowledging that fraud patterns are non-stationary (**Concept Drift**), a **Streaming Ensemble** is implemented to provide "Online Learning":
- **Incremental Training**: Utilizing an **SGDClassifier** (Stochastic Gradient Descent) or a **Hoeffding Tree**, the model processes transactions one by one ($N=1$ batch size). 
- **Feedback Loop**: When a transaction is finalized (or labeled via chargeback/manual review), the model performs a partial fit (`fit_one`). This allows the system's decision boundary to shift in real-time, adapting to a new attack IP or device fingerprint within seconds, rather than waiting for a nightly batch retraining cycle.
- **Ensemble Fusion**: The final risk score is a weighted fusion of the VAE anomaly score (unsupervised) and the Streaming Learner's probability (supervised). This ensures high accuracy against both known and entirely novel fraud profiles.

---

## 3.5 Explainability and Decision Engine
Integration of **SHAP (Shapley Additive Explanations)** ensures that every fraud score is decomposes into individual feature contributions. The decision engine employs a tiered risk threshold:
- **Critical (Risk > 85%)**: Automated high-velocity block with "Fraud" reason code.
- **Moderate (40-85%)**: Flagged for adaptive friction (e.g., 2FA or analyst review).
- **Normal (Risk < 15%)**: Immediate authorization.

---

## 3.6 Evaluation Metrics
Performance is measured through a recall-prioritized lens, recognizing the asymmetric cost of fraud:
- **Recall (Detection Rate)**: The primary metric, maximizing the capture of fraudulent transactions.
- **Precision**: Minimizing false positives to prevent legitimate customer friction.
- **AUC-ROC**: Evaluation of discriminative power across varying risk thresholds.
- **F1-Score**: The harmonic mean utilized for final model benchmarks.
