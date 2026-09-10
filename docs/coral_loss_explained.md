# 📐 Mathematical Foundation of CORAL Ordinal Regression

Diabetic Retinopathy (DR) grading follows the International Clinical Diabetic Retinopathy Disease Severity Scale:

* **Grade 0:** No apparent retinopathy
* **Grade 1:** Mild non-proliferative DR (microaneurysms only)
* **Grade 2:** Moderate non-proliferative DR (more than microaneurysms, but less than severe)
* **Grade 3:** Severe non-proliferative DR (any of: >20 intraretinal hemorrhages in each of 4 quadrants, definite venous beading in 2+ quadrants, prominent IRMA in 1+ quadrant)
* **Grade 4:** Proliferative DR (neovascularization, vitreous/preretinal hemorrhage)

---

## The Flaw of Standard Multi-Class Cross-Entropy

In standard Categorical Cross-Entropy, the model outputs an unconstrained probability vector $\mathbf{p} \in \Delta^{K-1}$ via Softmax:

$$\mathcal{L}_{CE} = -\sum_{c=0}^{K-1} y_c \log p_c$$

This formulation makes the **orthogonality assumption**: every class is equally distant from every other class. 
Predicting Grade 0 when the ground truth is Grade 4 incurs the exact same loss penalty as predicting Grade 3 when ground truth is Grade 4. Clinically, confusing healthy with proliferative DR is a catastrophic diagnostic failure, whereas confusing Grade 3 with Grade 4 is a subtle boundary adjudication.

---

## The CORAL Formulation

**Consistent Rank Logits (CORAL)** addresses this by reformulating a $K$-class ordinal regression problem into $K - 1$ binary classification sub-tasks sharing feature weights:

$$\text{Task } k: \quad \mathbb{P}(\text{grade} \ge k) \quad \text{for } k \in \{1, \dots, K-1\}$$

### 1. Architectural Weight Sharing
Let $\mathbf{x} \in \mathbb{R}^D$ be the pooled feature embedding from the convolutional backbone. Instead of having $K$ separate linear projection vectors, CORAL uses a **single shared weight vector** $\mathbf{w} \in \mathbb{R}^{D \times 1}$ and $K-1$ independent bias terms $b_k \in \mathbb{R}$:

$$z_k = \mathbf{w}^T \mathbf{x} + b_k \quad \text{for } k \in \{1, \dots, K-1\}$$

Because $\mathbf{w}$ is identical across all $K-1$ tasks, the decision hyperplanes for all thresholds are strictly **parallel** in feature space.

### 2. Loss Function
Binary targets $y_k \in \{0, 1\}$ are assigned as:

$$y_k = \mathbb{1}[\text{true\_grade} \geq k]$$

The overall CORAL loss is the sum of binary cross-entropies:

$$\mathcal{L}_{CORAL}(\mathbf{x}, y) = -\sum_{k=1}^{K-1} \left[ y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$

### 3. Prediction Rule
At inference time, the discrete grade is decoded by counting how many threshold boundaries exceed probability 0.5:

$$\hat{y} = \sum_{k=1}^{K-1} \mathbb{1}[\sigma(z_k) > 0.5]$$

---

## Why CORAL Guarantees High QWK

Quadratic Weighted Kappa penalizes misclassifications quadratically with respect to index distance:

$$w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$

Because CORAL learns parallel threshold hyperplanes along a single scalar projection $\mathbf{w}^T \mathbf{x}$, samples naturally fall along an ordered 1D continuum. This structural inductive bias virtually eliminates catastrophic off-diagonal errors (e.g. predicting Grade 4 for Grade 0), directly driving higher QWK scores compared to unconstrained Cross-Entropy.
