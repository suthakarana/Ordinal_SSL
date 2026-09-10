OrdDist SSL is a semi-supervised learning framework for ordinal diabetic retinopathy classification from retinal fundus images. The method is designed to exploit the ordered nature of diabetic retinopathy severity while learning from both labeled and unlabeled data.

Unlike conventional semi-supervised classification methods that treat disease grades as independent categorical classes, OrdDist incorporates ordinal information into the learning process. The framework combines regression-based ordinal prediction with standard and class-weighted classification branches and uses their predictive distributions to estimate reliable confidence for unlabeled samples.

The model contains three complementary prediction branches:

* **MSE branch:** performs ordinal regression by modeling diabetic retinopathy severity as an ordered numerical target.
* **CE branch:** performs conventional multi-class classification using cross-entropy loss.
* **CE-W branch:** uses class-weighted cross-entropy to improve learning from minority and difficult diabetic retinopathy grades.

The probability distributions produced by the CE and CE-W branches are aggregated and temporally updated using an exponential moving average. Ordinal-aware confidence estimation is then used to evaluate the reliability and concentration of predictions while considering the distance between neighboring disease grades. High-confidence unlabeled samples are selected for pseudo-label-based semi-supervised training.

The framework is designed for five-stage diabetic retinopathy grading:

**No DR → Mild → Moderate → Severe → Proliferative DR**

OrdDist SSL supports multiple CNN and transformer backbones, including ResNet, MobileNetV2, EfficientNet, DenseNet, ShuffleNetV2, and Vision Transformer architectures.

The implementation supports fully supervised and semi-supervised experiments, multi-seed evaluation, class-wise analysis, model selection using validation QWK, and comprehensive evaluation using Accuracy, Balanced Accuracy, Quadratic Weighted Kappa, F1-score, AUC, Precision, Recall, Sensitivity, Specificity, and Matthews Correlation Coefficient.

OrdDist is intended for research on ordinal, imbalanced, and label-efficient medical image classification, particularly diabetic retinopathy grading.
