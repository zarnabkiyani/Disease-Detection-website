# Disease-Detection-based-on-symptoms-using-Machine-learning

## Dataset Overview

The project utilizes a comprehensive medical dataset (`dataset_kaggle.csv`) consisting of multi-symptom columns mapped to target health conditions.

* **Target Column**: `Disease` (Includes 40 distinct clinical conditions like *Heart Disease*, *Bronchitis*, *Gout*, *Leukemia*, and *Melanoma*).
* **Feature Columns**: `Symptom_1` through `Symptom_17` representing the variable combinations of structural and symptomatic manifestations reported by the patient.

---

## Model Architecture

The core classification predictive model is a **Multi-Layer Perceptron (MLP)** deep neural network implemented via Keras and TensorFlow:

* **Input Layer**: Tailored to match the dimension of the processed one-hot encoded or label-encoded distinct symptoms.
* **Hidden Layers**:
* `Dense` Layer 1: 128 Neurons, `ReLU` Activation Function.
* `Dense` Layer 2: 64 Neurons, `ReLU` Activation Function.


* **Output Layer**: `Dense` layer with `Softmax` Activation mapping to the corresponding probability distribution of the 40 distinct target disease classes.
* **Compilation**: Configured with the `Adam` optimizer and optimized using `categorical_crossentropy` loss for robust multi-class metrics.

---

## Setup & Installation

### 1. Prerequisites

Ensure you have Python 3.8+ installed along with standard data science environments.

### 2. Clone the Repository

```bash
git clone [https://github.com/Alveeza-Ilyas/your-repository-name.git](https://github.com/Alveeza-Ilyas/your-repository-name.git)
cd your-repository-name

```

### 3. Install Required Packages

Install the core deep learning, visualization, and application delivery libraries:

```bash
pip install tensorflow pandas scikit-learn plotly streamlit pyngrok

```

---

## Technical Workflow & Execution

### Step 1: Exploratory Data Analysis & Preprocessing

The Jupyter notebook handles raw token cleanup, shuffles categorical combinations securely, and provides structural insights via interactive visualization tools:

* Generates an interactive **Plotly Pie Chart** displaying the global distribution of distinct symptoms.
* Creates a cross-disease **Correlation Matrix** to observe clinical symptoms clusters.

### Step 2: Running the Training Framework

Open `Disease_Detection.ipynb` inside your preferred environment (Jupyter Lab or Google Colab). Execute the pipeline blocks sequentially to:

1. Load and transform feature matrices directly from `dataset_kaggle.csv`.
2. Compile and train the MLP model network over target epochs.
3. Serialize the trained configuration directly to `mlp_model.h5`.

### Step 3: Launching the Web Interface (Streamlit + Ngrok)

The notebook features a terminal injection utility that spins up a background Streamlit server instance (`disease_prediction.py`) and exposes a secure public endpoint using `pyngrok`:

```python
import os
from pyngrok import ngrok

# Run the Streamlit application in background
os.system("nohup streamlit run disease_prediction.py --server.port 8501 &")

# Establish an external tunnel connection via Ngrok
public_url = ngrok.connect(8501, "http")
print("Live Streamlit App Address:", public_url)

```

Simply click the generated `*.ngrok-free.dev` output link to test diagnostic feature inputs instantly via a web browser.

---

## Built With

* **TensorFlow / Keras**: Deep learning engine for MLP compilation and classification tasks.
* **Pandas & Scikit-Learn**: Dataset preprocessing, multi-label encoding matrix manipulation, and shuffle operations.
* **Plotly**: Dynamic exploratory graphical data tracking.
* **Streamlit**: Web interactive presentation layout for clinical diagnosis simulations.
* **Ngrok**: Secure tunnel deployment infrastructure.

```

```
