# MODUL

Learning Student Network under Universal Label Noise
# Toolbox for MODUL

This repository aims to provide a compact and easy-to-use implementation of our proposed MODUL. 

- Computing Infrastructure:
  - We use one NVIDIA V100 GPU for Office-Home experiments and use one NVIDIA A100 GPU for DomainNet experiments. The PyTorch version is 1.12.

- Please put the datasets (e.g. CIFAR1O) in the `./data/`.
## Get the pretrained teacher models

```bash
## Train the teacher models
python train_teacher.py

## Train the student models
python train_student.py

```
## Cite this repository
```bash
If you use this software or the associated data in your research, please cite it as follows:
@article{tang2024learning,
  title={Learning Student Network under Universal Label Noise},
  author={Tang, Jialiang and Jiang, Ning and Zhu, Hongyuan and Zhou, Joey Tianyi and Gong, Chen},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```
