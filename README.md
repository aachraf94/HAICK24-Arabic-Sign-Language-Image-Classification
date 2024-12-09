# HAICK24-Arabic Sign Language Image Classification

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue.svg)  
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

This repository contains my solution for the **[HAICK24 - Arabic Sign Language Image Classification](https://www.kaggle.com/competitions/haik-24-arabic-sign-language-image-classification/overview)** Kaggle competition. The competition challenges participants to develop a model that can classify images of Arabic sign language into their corresponding letters using deep learning techniques.

## 🏆 Competition Objective  
The goal is to classify images that represent Arabic sign language letters. The dataset contains images of Arabic letters shown in sign language, and the model must predict the corresponding letter for each image.

---

### Dataset

- **Size**: 60.56 MB
- **File Types**: JPG, CSV
- **Files**:
  - `train`: Contains images for training.
  - `test`: Contains images for testing.
  - `sample_submission.csv`: A sample submission file in the correct format.

- **Image Format**: Each folder inside the `train` directory corresponds to a label (Arabic letter), and the images in that folder are examples of that letter in sign language.


---

## 🛠️ Solution Overview  

### Key Steps:
1. **Data Preprocessing**:
   - The dataset is split into training and validation sets with an 90%-10% ratio.
   - The images are organized into subdirectories by label (Arabic letter).
   - Each directory contains images of the respective letter.

2. **Modeling**:
   - Used **YOLOv8** (You Only Look Once) model for image classification.
   - Trained the model for 50 epochs with a batch size of 64.
   - The model is fine-tuned on the training dataset and evaluated on the validation set.

3. **Prediction**:
   - The trained model is used to predict labels for the images in the test dataset.
   - The predictions are saved in a CSV file for submission to Kaggle.

4. **Model Evaluation**:
   - The model's accuracy is evaluated by checking the predictions against the test set.

### Implementation Details:
- The dataset is divided into training and validation directories.
- The images are processed and shuffled into the appropriate folders for training.
- After training, the model generates predictions for unseen test images.
- The final results are saved in the required format for submission to Kaggle.

## Dependencies

Install the required Python packages using pip:

```bash
pip install pandas scikit-learn ultralytics pillow matplotlib
