# style-tranfer
Project Overview
This project investigates the integration of style transfer techniques into self-supervised learning for computer vision tasks. The goal is to enhance feature representation learning by reducing the dependency on labeled datasets, leveraging style transfer to generate rich visual representations.

Team
Lina Ben Haj Yahya: Project Lead, Research, Implementation
Synopsis
The project aims to mitigate the reliance on large annotated datasets in self-supervised learning by incorporating style transfer techniques. The hypothesis is that style transfer can help extract semantically meaningful features from unlabeled data, improving the performance of downstream tasks. The project utilizes cutting-edge style transfer models, contrastive learning methods, and datasets such as ImageNet.

Methodology
Style Transfer Model: Utilized a pre-trained style transfer model.
Base Model: Employed ResNet18 for feature extraction.
Dataset: Used a subset of ImageNet for training and testing.
Contrastive Learning: Implemented contrastive learning with positive and negative samples based on style and content.
Evaluation: Trained a linear classifier on frozen features and compared performance metrics.
Results
The approach yielded superior feature representations, demonstrated by improved classification accuracy and qualitative analyses of learned embeddings.

Installation
To set up the environment and run the code, follow these steps:

Clone the Repository:

sh

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Create and Activate a Virtual Environment:

sh

conda create --name style_transfer_venv python=3.8
conda activate style_transfer_venv
Install Dependencies:

sh

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install matplotlib seaborn scikit-learn
Usage
Run Training Script:

sh

jupyter notebook
# Open the notebook file and execute the cells step-by-step
Evaluate Model:
The evaluation results will be saved in the results/ directory as PNG files.

Visualizations
Sample visualizations of training progress and results:

Training Loss Plot: results/training_loss_plot.png
Validation Accuracy Plot: results/validation_accuracy_plot.png
Confusion Matrix: results/confusion_matrix.png
Sample Predictions: results/sample_predictions.png

Contact
For any questions or feedback, please contact me at yahya23@itu.edu.tr.
