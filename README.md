# Enhancing Adversarial Transferability by Resolving Gradient Conflicts in Multi-Input Transformations
Adversarial examples mislead deep neural networks by introducing imperceptible perturbations to images, and their transferability enables examples crafted on surrogate model to successfully attack unseen models. Studying this transferability can help to improve defense mechanisms in deep learning systems. Transformation-based input attacks are among the leading strategies for improving transferability. These methods apply various transformations to the original image to generate multiple augmented copies and aggregate their gradients in the surrogate model to iteratively produce more universally effective perturbations. Recent work has further boosted transferability by combining multiple transformations in parallel at each iteration. However, aggregating gradients from these transformed copies introduces gradient conflicts, weakening the dominant gradient and disrupting the perturbation optimization update trajectory. This causes the trajectory to deviate from the direction that maximizes the transferability of adversarial perturbations, leading to overfitting to the surrogate model and reducing its transferability to other models. Although diversity of multi-transformation combinations preserves adversarial efficacy between different views and scales, gradient conflicts remains a major bottleneck. To preserve diversity while resolving such conflicts, we propose a two‑phase framework:
Phase I involves precomputation in which gradients from multiple transformed copies are aggregated to build a composite momentum vector directed toward regions of high generalizability.
Phase II initializes at the clean example by inheriting the composite momentum accumulated in Phase I. Starting from this momentum and reinforcing the dominant gradient direction, the perturbation is iteratively refined to achieve higher cross-model attack success rates.
Extensive experiments demonstrate that our method outperforms existing approaches in both convolutional neural network and transformer architectures, maintaining attack performance under common defense mechanisms and exhibiting strong generalization on real‑world multimodal platforms.
![Overview](./figs.png)
## Usage

### Installation
- Python >= 3.6
- PyTorch >= 1.21
- Torchvision >= 0.13.1
- timm >= 0.6.12

```bash
pip install -r requirement.txt
```
### Import clean examples
Following from previous works, we randomly selecte 1,000 images from ImageNet validation set to run our experiments. All images can be classified properly. You can also construct you own example sets by contrust your folder in the following format. You can also download our prepared dataset from [Google Drive](https://drive.google.com/file/d/1C-XZioCi32VOwpTuaOkLzTbr_YIOA4vP/view?usp=sharing).
```
data
├─images
│  ├─ILSVR2012_val_00000051.JPEG
│  ├─...
│  └─ILSVR2012_val_00001501.JPEG
└─labels.csv
```
labels.csv is a two colums table folloing the (filename,label) format as below
```
filename,label
ILSVRC2012_val_00013716.JPEG,0
```

### Generate adversarial examples
You can run the following command to generate the adversarial examples.
```
python main.py --input_dir /path/to/data --output_dir path/to/advsample --attack l2t --model=resnet18
```
or simply 
```
python main.py 
```
if you put the data under the same directory and use the default settings

### Run for Evaluation
We run the evaluation across 10 models to verify the adversarial transferability of the generated examples by following command.
```
python main.py --eval
```
You can modify the **utils.py** to add or delete models.
