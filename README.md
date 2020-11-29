# Targeted-Adversarial-Attacks

## Getting Started

### Prerequisites

- Python 3 (Tested on 3.6.8)
- numpy (Tested on 1.19.3)
- opencv-python (Tested on 4.2.0)
- matplotlib (Tested on 3.0.3)
- TensorFlow 2 (Tested on 2.2.0)

### Installation

```
pip install -r requirements.txt
```

### Usage

```
python generate_adversarial_example_targeted.py <input_file_name> <target_class_name>
```
For example:
> python generate_adversarial_example_targeted.py input.jpg airliner

In this code sample, ResNet50 model pre-trained on ImageNet is used. You may refer to `imagenet_index.json` file for a mapping of ImageNet 1000 class labels to their corresponding index.

## Authors

* **Aaron Chong** - *Initial work* - [aaronchong888](https://github.com/aaronchong888)

See also the list of [contributors](https://github.com/aaronchong888/Targeted-Adversarial-Attacks/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This project is built referencing the tutorial at
[pyimagesearch - Targeted adversarial attacks with Keras and TensorFlow](https://www.pyimagesearch.com/2020/10/26/targeted-adversarial-attacks-with-keras-and-tensorflow/), and using the following packages and libraries as listed [here](https://github.com/aaronchong888/Targeted-Adversarial-Attacks/network/dependencies)