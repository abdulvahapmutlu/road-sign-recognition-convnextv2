# Road Sign Recognition using ConvNeXt v2 Tiny

## Project Overview
This project aims to classify road signs into 9 different categories, such as speed limit signs (20, 30, 50, etc.) and stop signs. The dataset was obtained from [Kaggle Road Sign Recognition Dataset](https://www.kaggle.com/datasets/fhabibimoghaddam/road-sign-recognition).

The model used is ConvNeXt v2 Tiny, a modern deep learning model for image classification tasks, pre-trained on large datasets and fine-tuned for this specific task.

## Dataset
The dataset contains 9 different classes of road signs:
- Stop, 20, 30, 50, 60, 70, 80, 100, 120

## Model Architecture
The model is based on ConvNeXt v2 Tiny, a deep learning architecture that uses convolutional layers for image classification. I've fine-tuned the model for 9 classes.

## Training
The model was trained using the Adam optimizer with a learning rate of 1e-4. A learning rate scheduler (ReduceLROnPlateau) was used to dynamically adjust the learning rate.

## Evaluation
After training, the model achieved **100% accuracy** on the test set.

## How to Run

1. Clone the repository:
    ```
    git clone https://github.com/abdulvahapmutlu/road-sign-recognition-convnextv2.git
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Train the model:
    ```
    python src/train.py
    ```

4. Evaluate the model:
    ```
    python src/evaluate.py
    ```

## License
This project is licensed under the MIT License.
