# Low-Light Image Enhancement (LLIE) with ZERO_DEC and Denoising Autoencoder (DAE)

## Overview
This project focuses on enhancing low-light images using a combination of the **ZERO_DEC** method for low-light enhancement and a **Denoising Autoencoder (DAE)**. The pipeline is designed to improve image quality by enhancing brightness and reducing noise, evaluated through key image quality metrics.

## Features
- **ZERO_DEC** for low-light image enhancement.
- **Denoising Autoencoder (DAE)** for noise reduction.
- Evaluation metrics:
    - **Peak Signal-to-Noise Ratio (PSNR)**
    - **Structural Similarity Index Measure (SSIM)**
    - **Mean Squared Error (MSE)**

## Installation

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/gdtan02/CV_Assignment_LLIE.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Results

### Example Metrics

#### Enhanced Images (ZERO_DEC):
| Metric       | Average |
|--------------|---------|
| **PSNR**     | 6.1293  |
| **SSIM**     | 0.1149  |
| **MSE**      | 0.2533  |

#### Final Denoised Images (ZERO_DEC + DAE):
| Metric       | Average |
|--------------|---------|
| **PSNR**     | 6.2181  |
| **SSIM**     | 0.1208  |
| **MSE**      | 0.2488  |

### Visualization
Below are sample outputs from the pipeline:
1. **Original Image**
2. **Enhanced Image (ZERO_DEC)**
3. **Denoised Image (ZERO_DEC + DAE)**

![Example Output](assets/example_output.png)

## Project Structure
```
low-light-enhancement/
├── data/
│   ├── low_light_images/       # Input images
├── output/
│   ├── enhanced/               # Enhanced images
│   ├── denoised/               # Final denoised images
│   ├── metrics/                # Evaluation metrics
├── zero_dce/
│   ├── evaluate_metrics.py     # Evaluation script
├── assets/                     # Images for README
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments
This project utilizes:
- **ZERO_DEC** for low-light enhancement.
- PyTorch for implementing the **DAE**.
- scikit-image for image quality metrics.
