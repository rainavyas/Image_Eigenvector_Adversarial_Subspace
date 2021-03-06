# Is there a subspace to describe all adversarial perturbation

This work explores a subspace defined by the eigenvectors of the input data covariance matrix to characterise individual and universal adversarial
perturbations on image classifiers.

# Requirements

python3.4 or above

pip install numpy

pip install torch, torchvision

pip install cnn_finetune

pip install matplotlib


# Experiments

Experiments are performed for three datasets: CIFAR-10, CIFAR-100 and Tiny ImageNet. For each dataset, five different classifier architectures are trained by finetuning the
pre-trained models in the torchvision module. The classifier architectures are: Resnet-18, Resnet-50, VGG-16, DenseNet-121 and GoogLeNet.

The following experiments are carried out:

- Very Small perturbations in each eigenvector directions, where impact is measured using KL-Divergence between original and attacked logits
- Large, undetectable perturbations in each eigenvector direction, measured using fooling rate
- Projected Gradient Descent Attacks (individual and universal) are performed and the distribution of these attacks in the eigenvector basis is analyzed
- Transferability of PGD attacks explained in the eigenvector basis space
- Model robustness to attacks analyzed by re-training where the subspace associated with adversarial attacks is eliminated

# Results

TBF
