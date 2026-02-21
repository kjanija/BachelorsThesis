# ECG Signal Denoising with EEMD and Genetic Algorithms

This repository provides a Python-based pipeline for removing noise from Electrocardiogram (ECG) signals. It utilizes Empirical Mode Decomposition (EMD) or Complete Ensemble EMD (CEEMDAN) to separate signal frequencies, combined with a Genetic Algorithm (GA) to adaptively find the optimal thresholds for noise removal.

---

## Methodology

The cleaning process follows an automated pipeline implemented in the `SignalCleaner` class:

1. **Decomposition:** The input noisy ECG signals are decomposed into Intrinsic Mode Functions (IMFs) using either standard EMD or Ensemble EMD.
2. **Boundary Selection:** The probability density functions (PDFs) of the IMFs are estimated. By calculating the Kullback-Leibler divergence, the pipeline automatically separates the IMFs into a "signal-dominant" group and a "noise-dominant" group.
3. **Genetic Algorithm Optimization:** A Genetic Algorithm (powered by `pygad`) explores the parameter space (`C`, `BETA`, `RHO`) to find the optimal thresholding values. The fitness function aims to maximize the Signal-to-Noise Ratio (SNR) improvement.
4. **Adaptive Thresholding:** The optimized thresholds are applied to the noise-dominant IMFs (supporting both Hard and Soft thresholding).
5. **Reconstruction:** The thresholded (cleaned) IMFs are added back to the signal-dominant IMFs and the decomposition residual to reconstruct the clean ECG signal.

---

## Repository Structure

* `utils.py`: Contains the core mathematical utilities (MSE, PRD, MAE metrics), PDF estimation functions, and the `SignalCleaner` class which manages the entire denoising pipeline.
* `requirements`: Lists the required Python packages for the environment.

---

## Requirements

Ensure you have the following dependencies installed. You can install them using the provided `requirements` file:

* `EMD-signal==1.6.0`
* `numpy==1.26.4`
* `scipy==1.13.0`
* `pygad==3.3.1`

---

## Usage

The primary interface for this toolkit is the `SignalCleaner` class. You can pass a list of signals, configure the Genetic Algorithm, and execute the full pipeline using the `run()` method.

### Example

```python
from utils import SignalCleaner

# 1. Prepare your list of original ECG signals
my_ecg_signals = [signal_1, signal_2] 

# 2. Instantiate the cleaner
cleaner = SignalCleaner(
    signal_list=my_ecg_signals,
    ensemble=True,                  # Use CEEMDAN instead of standard EMD
    generations_per_signal=50,      # GA parameter: number of generations
    parents_per_signal=5,           # GA parameter: mating parents
    mutation_percent=10.0,          # GA parameter: mutation probability
    SNR_input=5.0,                  # Input Signal-to-Noise ratio for AWGN
    hard_threshold=True             # Use hard thresholding (False for soft)
)

# 3. Execute the denoising pipeline
cleaner.run(hard_thresholding=True)

# 4. Access the cleaned signals
cleaned_signals = cleaner.y_pred 
```

## Evaluation Metrics

The `utils.py` file also provides several standalone functions to evaluate the quality of your denoised signals against the ground truth:

| Metric | Function | Description |
| :--- | :--- | :--- |
| **SNR Improvement** | `SNR_improvement(noisy_signal, signal, predicted)` | Logarithmic measure of noise reduction (in dB). |
| **MSE** | `signal_MSE(signal, predicted)` | Mean Square Error between the original and reconstructed signal. |
| **PRD** | `PRD(y, y_pred)` | Percent Root Mean Square Difference. |
| **MAE** | `MAE(y, y_pred)` | Maximum Absolute Error across the signal array. |

## References

* Phuong Nguyen and Jong-Myon Kim. Adaptive ecg denoising using genetic algorithm-based thresholding and ensemble empirical mode decomposition. *Information Sciences*, 373:499–511, 2016.