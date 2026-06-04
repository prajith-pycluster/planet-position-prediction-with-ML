# Planetary Position Prediction using LSTM

This project utilizes a Long Short-Term Memory (LSTM) neural network model to predict the future orbital positions of four planets. 

While the current scope of the project is limited to four planets, the architecture is scalable, and additional planets can be integrated into the dataset and model.

> ⚡ **Hardware Note:** Utilizing a GPU is highly recommended and will significantly reduce training and processing times.

---

## Technical Prerequisites & Libraries

To run this project, you need Python installed along with the specific versions of the libraries listed below. 

### Core Dependencies
* **TensorFlow** (`2.14.0`) - Used for building and training the LSTM model.
* **NumPy** (`1.24.3`) - Handled for numerical operations (strictly version-matched for TensorFlow compatibility).
* **Pandas** - Used for dataset manipulation and preprocessing.
* **Matplotlib** - Used for plotting and visualizing the predicted planetary orbits.
* **Scikit-Learn** - Used for data scaling and preprocessing metrics.
* **JupyterLab** - The recommended environment to run the project notebooks.

### Quick Installation

You can install the required environment using pip:

```bash
pip install numpy==1.24.3 tensorflow==2.14.0 pandas matplotlib jupyterlab scikit-learn
