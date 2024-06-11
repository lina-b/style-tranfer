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


Contact
For any questions or feedback, please contact me at yahya23@itu.edu.tr.
