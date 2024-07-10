# 🧠 Neural Network Training with Neurolab

A mini project exploring the use of Neurolab, a simple and powerful neural network library for Python, to train various neural network architectures. The project demonstrates different network configurations and training strategies using synthetic data.

## 📄 Files
- **Neural Network.py**: Python script containing all the code for data generation, network creation, training, and evaluation.
- **Neural Network Report.pdf**: Report explaining the key result of the experiments.

## 🎥 Explanation Video: [Click to watch!](https://drive.google.com/file/d/10cDQ6Ps8o796JD1mjc9s3RMyM3JDE2m8/view?usp=sharing)

## 🔍 Project Overview
This project involves the following steps:
1. Generating synthetic data.
2. Creating and training different neural network configurations.
3. Evaluating the performance of the trained models.
4. Visualizing the training process.

## 🧪 Experiments and Findings

### Experiment 1: Basic Network with Two Inputs
- **Input Data**: 10 data points with 2 features each.
- **Network Structure**: [6, 1]
- **Training Method**: Standard backpropagation.
- **Key Finding**: Achieved a final error of 0.0014 after 495 epochs.

### Experiment 2: Multi-Layer Network with Two Inputs
- **Input Data**: 10 data points with 2 features each.
- **Network Structure**: [5, 3, 1]
- **Training Method**: Gradient descent.
- **Key Finding**: Stabilized at an error of 0.8282 after 500 epochs.

### Experiment 3: Basic Network with Increased Data Points
- **Input Data**: 100 data points with 2 features each.
- **Network Structure**: [6, 1]
- **Training Method**: Standard backpropagation.
- **Key Finding**: Achieved a final error of 0.0134 after 465 epochs.

### Experiment 4: Multi-Layer Network with Increased Data Points
- **Input Data**: 100 data points with 2 features each.
- **Network Structure**: [5, 3, 1]
- **Training Method**: Gradient descent.
- **Key Finding**: Stabilized at an error of 0.5981 after 1000 epochs.
- **Visulization of training**:

   ![ex4](https://github.com/maharsh3133/neural-network/assets/35959045/91bfe568-685c-44c3-8bd5-ba4c475a7490)

### Experiment 5: Network with Three Inputs
- **Input Data**: 10 data points with 3 features each.
- **Network Structure**: [6, 1] and [5, 3, 1]
- **Training Method**: Standard backpropagation and gradient descent.
- **Key Finding**: Achieved a final error of 0.0113 after 105 epochs for [6, 1] Network.

## 🛠️ How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/maharsh3133/neural-network.git
   ```
2. Navigate to the project directory:
   ```bash
   cd neural-network
   ```
3. Run the script:
   ```bash
   python Neural Network.py
   ```
4. Observe the results printed in the console and visualizations displayed.

## 👤 Author
- [Maharsh](https://www.linkedin.com/in/maharsh-patel-641777168/)

## 🚀 Conclusion
- This project provides a hands-on approach to understanding neural networks using the Neurolab library. By experimenting with different network architectures and training methods, you can gain insights into the impact of network design on model performance.

---

This README outlines the steps to replicate the experiments and provides detailed information about each experiment's configuration and results.
