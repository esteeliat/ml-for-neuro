# Homework 1 - Probability

## Question 1: 

### a: Compute $P(X_1 = H)$
$P(X_1 = H) = (0.9 * 0.5) + (0.1 * 0.5) = 0.5$

### b: Compute $P(X_1 = H,\, X_2 = H)$. Are $X_1$ and $X_2$ independent?
$P(X_1 = H,\, X_2 = H) = [(0.9*0.9)*0.5] + [(0.1*0.1)*0.5] = 0.41$ <br>
They are not indepent - independence occurs when $P(X_1 = H,\, X_2 = H) = P(X_1= H) * P(X_2 = H)$ which does not hold here since the probabilities of heads on coin A and B differ

### c: Compute $P(X_2 = H \mid X_1 = H)$
$P(X_2 = H \mid X_1 = H) = \frac{P(X_2 = H, X1 = H)}{P(X_1 = H)} = \frac{0.41}{0.5} = 0.82$

### d: Using Bayes' rule, compute $P(Z = A \mid X_1 = H)$
$P(Z = A \mid X_1 = H) = \frac{P(X_1 = H \mid Z = A) * P(Z = A)}{ P(X_1 = H)} =\frac{ 0.9 * 0.5}{0.5} = 0.9$

## Question 2:
### a. Compute the marginal distribution of X
$P(X = 1) = 0.24 + 0.6 = 0.3$ <br>
$P(X = 2) = 0.18 + 0.12 = 0.3$ <br>
$P(X = 3) = 0.08 + 0.32 = 0.4$
 
### b. compute the conditional probabilities $P(Y \mid X)$ for each x in \{1, 2, 3\}
X=1: <br>
- $P(Y=0 \mid X=1) = \frac{0.24}{0.30}=0.8$ <br>
- $P(Y=1 \mid X=1) = 1 - 0.8 =0.2$ <br>

X=2: <br>
- $P(Y=0 \mid X=2) = \frac{0.18}{0.30}=0.6$ <br>
- $P(Y=1 \mid X=2) = 1 - 0.6 =0.4$ <br>

X=3: <br>
- $P(Y=0 \mid X=3) = \frac{0.08}{0.40}=0.2$ <br>
- $P(Y=1 \mid X=3) = 1 - 0.2 =0.8$ <br>

### c. Determine $g^{*}(1),\, g^{*}(2),\, \text{and}\, g^{*}(3)$

### d. visualize the decision boundary:

$$$ e. Estimate the Bayes classifier error by simulation 
