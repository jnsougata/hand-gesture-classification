# Hand Gesture Recognition for Enhanced Communication in Hospital Environments: A Deep Learning Approach
-- - -
### 📌Abstract
In this project, we present a deep learning-based approach for hand gesture recognition to enhance communication in hospital environments. The system is designed to assist healthcare professionals in conveying important information to patients and colleagues through hand gestures, especially in situations where verbal communication may be challenging. We utilize a convolutional neural network (CNN) architecture to classify hand gestures captured from video streams. Our approach demonstrates the potential of deep learning techniques in improving communication and interaction within healthcare settings.

### 📋Table of Contents
- [Hand Gesture Recognition for Enhanced Communication in Hospital Environments: A Deep Learning Approach](#hand-gesture-recognition-for-enhanced-communication-in-hospital-environments-a-deep-learning-approach)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Results](#results)
  - [Conclusion](#conclusion)

### 📒Introduction
Effective communication is crucial in hospital environments, where healthcare professionals need to convey important information quickly and accurately. However, there are situations where verbal communication may be hindered due to noise, language barriers, or the need for non-verbal cues. Hand gestures can serve as a powerful means of communication in such scenarios. In this project, we aim to develop a hand gesture recognition system using deep learning techniques to facilitate communication in hospital settings.

### 📊Dataset
An augmented dataset of hand gestures was created to train the model. The dataset consists of various hand gestures commonly used in healthcare settings, such as "stop," "come here," "thank you," and "help." Each gesture is represented by a series of images captured from different angles and lighting conditions to ensure robustness. The dataset is divided into training, validation, and testing sets to evaluate the model's performance effectively.
- [Download the dataset here](/dataset/csv/hand_landmark_dataset.csv)
- [Find data augmentation code here](/augment.py)

### 🏛️Model Architecture
The model architecture is based on a convolutional neural network (CNN) designed to extract features from hand gesture images. The architecture consists of several convolutional layers followed by pooling layers, fully connected layers, and a softmax output layer for classification. The model is trained using the Adam optimizer and categorical cross-entropy loss function to optimize performance.

### ✳️Training and Evaluation
The model is trained on the augmented dataset using a batch size of 32 and a learning rate of 0.001. The training process involves data augmentation techniques such as rotation, scaling, and flipping to enhance the model's generalization capabilities. The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics on the validation and testing sets.

### 🔢Results
| Class                    | Precision | Recall   | F1-Score | Support  |
|--------------------------|-----------|----------|----------|----------|
| call_my_family           | 1.00      | 0.99     | 0.99     | 100      |
| clean_the_room           | 0.99      | 0.99     | 0.99     | 100      |
| close_the_door           | 1.00      | 1.00     | 1.00     | 100      |
| do_not_disturb           | 0.98      | 1.00     | 0.99     | 100      |
| go_to_washroom           | 1.00      | 0.99     | 0.99     | 100      |
| good_luck                | 0.98      | 0.98     | 0.98     | 100      |
| i_am_hungry              | 0.99      | 0.99     | 0.99     | 100      |
| i_am_in_emergency        | 0.99      | 0.95     | 0.97     | 100      |
| i_am_okay                | 1.00      | 1.00     | 1.00     | 100      |
| i_am_thirsty             | 0.99      | 0.99     | 0.99     | 100      |
| i_need_help              | 1.00      | 1.00     | 1.00     | 100      |
| please_call_doctor       | 0.97      | 0.99     | 0.98     | 100      |
| please_call_nurse        | 0.99      | 0.98     | 0.98     | 100      |
| please_take_me_to_walk   | 0.99      | 0.99     | 0.99     | 100      |
| power_to                 | 0.98      | 0.99     | 0.99     | 100      |
| reject_or_cancel         | 0.99      | 1.00     | 1.00     | 100      |
| stop                     | 0.99      | 0.99     | 0.99     | 100      |
| switch_off_the_light_fan | 1.00      | 1.00     | 1.00     | 100      |
| thank_you                | 1.00      | 0.99     | 0.99     | 100      |
| there_is_a_fire_breakout | 0.98      | 1.00     | 0.99     | 100      |
| **Accuracy**             |           |          | **0.99** | **2000** |
| **Macro Avg**            | **0.99**  | **0.99** | **0.99** | **2000** |
| **Weighted Avg**         | **0.99**  | **0.99** | **0.99** | **2000** |

![Confusion Matrix](/figures/confusion.png)
![Accuracy](/figures/training.png)

The model achieved an accuracy of 95% on the testing set, demonstrating its effectiveness in recognizing hand gestures. The confusion matrix indicates that the model performs well across different gesture classes, with minimal misclassifications. The training and validation loss curves show a steady decrease, indicating successful convergence during training.

### 🫥Conclusion
In this project, we developed a hand gesture recognition system using deep learning techniques to enhance communication in hospital environments. The model demonstrated high accuracy and robustness in recognizing various hand gestures, making it a valuable tool for healthcare professionals. Future work will focus on real-time implementation and integration with hospital communication systems to further improve patient care and interaction.

### ⚠️Caution
This model is trained on a limited dataset and may not perform well in all scenarios. It is recommended to validate the model's performance in real-world settings before deployment. Additionally, ethical considerations regarding patient privacy and data security should be taken into account when implementing such systems in healthcare environments.
Moreover, the model's performance may vary based on factors such as lighting conditions, camera angles, and individual differences in hand gestures. Continuous monitoring and updates to the model will be necessary to ensure its effectiveness and reliability in diverse hospital settings.
To further enhance the model's performance, we recommend exploring transfer learning techniques by leveraging pre-trained models on larger datasets. This approach can help improve the model's ability to generalize to unseen data and reduce the risk of overfitting.

### ⛓️‍💥 Why the image-based dataset is not used?
The provided 50x50 images are not used here in the model training because they are not suitable for deep learning applications. The images are too small and lack sufficient detail for the model to learn meaningful features. Deep learning models, especially convolutional neural networks (CNNs), require larger and more complex datasets to effectively learn and generalize from the data. Therefore, we opted for a landmark-based approach, which captures the key points of the hand gestures and allows for better feature extraction and classification.