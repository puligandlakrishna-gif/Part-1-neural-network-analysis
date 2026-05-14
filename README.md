# Neural Network Analysis for Customer Churn Prediction

## Overview
This project analyzes customer churn using neural networks, implementing a feed-forward neural network to predict whether customers will churn based on various features. The dataset contains 2000 customer records with 17 features, and the target variable is binary (churn: 1 = churned, 0 = retained).

**Data Source**: [BITSoM BA - Module 5 - Dataset](https://drive.google.com/drive/folders/1Aihn49cUYMjCgeCTFBTyprjrgZO3UY6r?usp=drive_link)

## Approach
The approach involves building and training a neural network classifier using TensorFlow/Keras framework. Key components include:

- **Model Architecture**: Feed-forward neural network with configurable hidden layers, ReLU activation for hidden layers, and sigmoid for binary output.
- **Data Handling**: One-hot encoding for categorical features, standard scaling for numerical features, stratified train/test split (80/20).
- **Imbalance Handling**: Class weights computed to balance the 63:1 class imbalance during training.
- **Regularization**: Dropout and batch normalization to prevent overfitting.
- **Evaluation**: ROC-AUC as primary metric due to class imbalance, supplemented by confusion matrix and classification report.
- **Optimization**: Adam optimizer with early stopping based on validation loss.

## Steps
1. **Dataset Understanding**
   - Load and explore the customer_churn_nn.csv dataset
   - Analyze feature types (categorical, numerical, binary)
   - Examine target variable distribution and class imbalance
   - Visualize churn rates by categorical features

2. **Data Preprocessing**
   - Remove identifier column (customer_id)
   - One-hot encode categorical variables (region, plan_type, contract_type, payment_method)
   - Separate features and target
   - Perform stratified train/test split
   - Apply StandardScaler to numerical features

3. **Neural Network Model Building**
   - Define a flexible model builder function with configurable parameters
   - Architecture: Input → Dense layers with BatchNorm and Dropout → Sigmoid output
   - Compile with binary cross-entropy loss and Adam optimizer

4. **Training and Evaluation**
   - Train baseline model with early stopping
   - Plot training curves (accuracy and loss)
   - Evaluate on train and test sets
   - Generate classification report, confusion matrix, and ROC curve

5. **Hyperparameter Experimentation**
   - Test 6 different configurations varying:
     - Network depth (2 vs 3 layers)
     - Learning rate (0.01, 0.001, 0.0001)
     - Batch size (32 vs 128)
     - Activation function (ReLU vs tanh)
   - Compare performance across experiments

6. **Final Reflection**
   - Analyze role of weights and biases
   - Explain necessity of activation functions
   - Discuss learning rate effects
   - Assess overfitting/underfitting signs

## Results
The hyperparameter experiments yielded the following key results:

| Experiment | Architecture | Activation | LR | Batch | ROC-AUC | Test Acc |
|------------|--------------|------------|----|-------|---------|----------|
| Exp 1 – Baseline | [64, 32] | relu | 0.001 | 32 | 0.8291 | 0.9675 |
| Exp 2 – Deeper Network | [128, 64, 32] | relu | 0.001 | 32 | 0.8219 | 0.9600 |
| Exp 3 – Higher LR (0.01) | [64, 32] | relu | 0.01 | 32 | 0.7686 | 0.9600 |
| Exp 4 – Lower LR (0.0001) | [64, 32] | relu | 0.0001 | 32 | 0.4353 | 0.6250 |
| Exp 5 – Larger Batch (128) | [64, 32] | relu | 0.001 | 128 | 0.7098 | 0.9000 |
| Exp 6 – tanh Activation | [64, 32] | tanh | 0.001 | 32 | 0.8790 | 0.8750 |

**Best Performing Model**: Exp 6 (tanh activation) achieved the highest ROC-AUC of 0.8790, demonstrating superior ability to distinguish churners from non-churners.

**Key Findings**:
- Accuracy metrics are misleading due to extreme class imbalance (98.45% retained)
- ROC-AUC provides a more reliable performance measure
- Tanh activation outperformed ReLU for this dataset
- Lower learning rates (0.0001) led to poor convergence
- Deeper networks showed marginal improvement over baseline

## Observations

### Role of Weights and Biases
Weights control the strength of connections between neurons, learning which features are most predictive of churn. Biases provide flexibility by shifting activation thresholds, enabling the network to model complex non-linear relationships in the data.

### Importance of Activation Functions
Without activation functions, the network would only learn linear relationships regardless of depth. ReLU introduces non-linearity, enabling complex decision boundaries. Sigmoid in the output layer provides interpretable churn probabilities.

### Learning Rate Effects
- **Too high (0.01)**: Causes oscillations and divergence, as seen in Exp 3 with higher validation loss variance
- **Too low (0.0001)**: Leads to slow convergence and suboptimal performance, as in Exp 4 with lowest ROC-AUC
- **Optimal (0.001)**: Enables steady convergence and good performance

### Overfitting/Underfitting Assessment
- No severe overfitting observed - training and validation losses tracked closely
- Mild underfitting on minority class - low recall for churners at default threshold
- Model learned some signal (ROC-AUC > 0.8 for best models) but struggles with extreme imbalance

### Recommendations for Improvement
- Use SMOTE for oversampling minority class
- Lower prediction threshold (e.g., 0.3) to increase churn recall
- Consider deeper networks or feature engineering
- Explore ensemble methods or different architectures

## Files
- `notebook.ipynb`: Complete analysis notebook
- `customer_churn_nn.csv`: Dataset
- `requirement.txt`: Python dependencies
- `Results/`: Generated plots and comparison table</content>
<parameter name="filePath">f:\BITS\Assignments\5\Part-1-neural-network-analysis\README.md