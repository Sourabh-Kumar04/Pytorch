# 🧠 PyTorch Deep Learning Journey

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> My personal journey learning PyTorch — from basic tensor operations to designing, training, and optimizing neural networks. Every notebook here is a step in my hands-on deep learning education.

---

## 🎯 About This Repository

This is a structured, practical archive of my PyTorch learning path. It includes conceptual breakdowns, working notebooks, mini-projects, and optimization experiments — all built to reinforce deep learning foundations and explore advanced applications.

> **Why I built this:** To reinforce my learning with real code, and to share tested, beginner-friendly examples that actually work.

---

## 📚 Learning Roadmap

### 🔰 Foundation Phase
| Notebook | Key Learnings |
|----------|---------------|
| `00_Introduction to PyTorch.pdf` | Overview of PyTorch and its ecosystem. |
| `01_Tensors_in_pytorch.ipynb` | Tensor creation, indexing, broadcasting, and operations. |
| `02_pytorch_autograd.ipynb` | Gradient computation and backpropagation via `autograd`. |
| `03_pytorch_training_pipeline.ipynb` | Manual training loops, loss calculation, and optimization basics. |

### 🏗️ Building Better Models
| Notebook | Focus Area |
|----------|------------|
| `04_pytorch_nn_module.ipynb` | Writing modular, reusable neural networks using `nn.Module`. |
| `05_dataset_and_dataloader.ipynb` | Clean data handling with `Dataset` and `DataLoader`. |
| `06_ANN_using_pytorch.ipynb` | My first complete multi-layer perceptron (MLP) network. |

### ⚡ Performance & Optimization
| Notebook | What I Explored |
|----------|-----------------|
| `07_neural_network_training_on_GPU.ipynb` | CUDA-powered training on GPU — major performance gains. |
| `08_Optimize_neural_network.ipynb` | Manual optimization techniques — adjusting learning rates, etc. |
| `09_Optimize_neural_network_using_optuna.ipynb` | Automated tuning with Optuna — model selection made efficient. |

### 🧠 Advanced Architectures
| Notebook | Project |
|----------|---------|
| `10_a_CNN_on_fashion_mnist.ipynb` | Basic convolutional neural network for image classification. |
| `11_CNN_on_fashion_mnist.ipynb` | Improved CNN with batch norm, dropout, and deeper layers. |
| `12_RNN_using_pytorch.ipynb` | Intro to RNNs for handling sequential data. |
| `13_next_word_predictor.ipynb` | RNN-based language model for next-word prediction. |

---

## 📊 Datasets

| File | Usage | Notes |
|------|-------|-------|
| `06_fmnist_small.csv` | MLP training | Lightweight dataset for quick tests |
| `07_fashion-mnist_test.csv` | CNN testing | Test set for benchmarking |
| `07_fashion-mnist_train.zip` | CNN training | Full training data |
| `12_100_Unique_QA_Dataset.csv` | NLP with RNNs | Custom Q&A dataset |

---

## ⚙️ Getting Started

### 🧪 Setup Instructions
```bash
# 1. Clone the repo
git clone https://github.com/Sourabh-Kumar04/Pytorch.git
cd Pytorch

# 2. Create environment
conda create -n pytorch-learning python=3.9
conda activate pytorch-learning

# 3. Install dependencies
pip install torch torchvision matplotlib numpy optuna pandas jupyter

# 4. Run the notebooks
jupyter notebook
````

---

## 💡 Key Learnings & Insights

### 💥 What Stood Out

* **Simple models first:** My early ANN helped me understand more than any advanced CNN tutorial.
* **GPU = Game changer:** 10× training speed-up with CUDA.
* **Pipeline precision:** Clean data handling saved me hours of pain.
* **Optuna:** Discovered hyperparameters I wouldn't have considered.

### ✅ What Helped Me Most

* Following a week-by-week structure
* Running experiments on the same dataset to compare architectures
* Keeping visual checkpoints to understand overfitting/underfitting

### 🚧 Pitfalls I Hit

* Forgetting `.to(device)` calls → silent bugs on GPU
* Skipping normalization → exploding losses
* Wrong loss functions → poor convergence
* Overfitting small datasets → false sense of success

---

## 📈 Results Snapshot

| Task                  | Initial Accuracy | Final Accuracy | Improvements                              |
| --------------------- | ---------------- | -------------- | ----------------------------------------- |
| CNN (Fashion MNIST)   | 78%              | 91%            | Model tuning & augmentation               |
| Advanced CNN          | 91%              | 94%            | Dropout, LR scheduler                     |
| Word Prediction (RNN) | 65%              | 82%            | Larger hidden layer, better preprocessing |

---

## 🧰 Tools & Libraries

```bash
torch>=2.0.0          # Core deep learning framework
torchvision>=0.15.0   # Datasets, transforms, pretrained models
numpy>=1.21.0         # Numerical operations
pandas>=1.3.0         # Data wrangling
matplotlib>=3.5.0     # Visualizations
seaborn>=0.11.0       # Advanced plots
optuna>=3.0.0         # Hyperparameter optimization
jupyter>=1.0.0        # Notebook execution
tqdm>=4.62.0          # Loop progress bars
```

---

## 📅 Learning Schedule (What Worked for Me)

| Week   | Topics                                |
| ------ | ------------------------------------- |
| Week 1 | Tensors + Autograd (`01`, `02`)       |
| Week 2 | Training loops & modules (`03`, `04`) |
| Week 3 | Data & MLP models (`05`, `06`)        |
| Week 4 | GPU & Optimization (`07`, `08`, `09`) |
| Week 5 | CNNs for CV (`10`, `11`)              |
| Week 6 | RNNs for NLP (`12`, `13`)             |

> ✅ Tip: Don’t rush the early modules — they’re the real foundation!

---

## ❓ Common Questions

<details>
<summary><strong>Do I need a GPU?</strong></summary>
Not at the start. You can run all notebooks up to `06_ANN` comfortably on CPU. For CNN and RNN, a GPU is highly recommended for speed.
</details>

<details>
<summary><strong>Where should I start?</strong></summary>
Begin with `00_Introduction to PyTorch.pdf`. Even if you know NumPy, PyTorch has its own quirks.
</details>

<details>
<summary><strong>Can I use this for my own learning/projects?</strong></summary>
Absolutely! It's MIT licensed — fork it, adapt it, build on it.
</details>

---

## 📚 Resources That Helped Me

### Official Docs & Courses

* 📘 [PyTorch Tutorials](https://pytorch.org/tutorials/)
* 📗 [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch)
* 🎓 [Fast.ai Course](https://course.fast.ai/)

### When Stuck

* 🧠 [PyTorch Forums](https://discuss.pytorch.org/)
* 🧩 [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)
* 📊 [Papers with Code](https://paperswithcode.com/)

---

## 👨‍💻 About Me

**Sourabh Kumar**
🎓 *AI Programming with Python - Udacity Graduate*

🎖️ AWS AI ML Scholar'24 | AI | ML | GenAI Explorer*

🌐 [GitHub](https://github.com/Sourabh-Kumar04) | [LinkedIn](https://linkedin.com/in/sourabh-kumar04)

> *"Always learning, always building — one notebook at a time."*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ⭐️ Support the Project

If you found this useful:

* Give a ⭐️ to the repository
* Fork it and make it your own
* Share feedback or suggestions
* Connect with me for collabs or questions

---

<div align="center">

### Happy Learning! 🚀

*"The best way to learn deep learning is by building with it."*

</div>



