# MO-PaDGAN

Experiment code associated with the paper: [MO-PaDGAN: Generating Diverse Designs with Multivariate Performance Enhancement](https://arxiv.org/pdf/2007.04790.pdf)

![Alt text](/architecture.svg)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Chen, W., & Ahmed, F. (2020, July). MO-PaDGAN: Generating Diverse Designswith Multivariate Performance Enhancement. In: Workshop on Negative Dependenceand Submodularity: Theory and Applications in Machine Learning, 37th International Conference on Machine Learning (ICML).

    @inproceedings{chen2020mopadgan,
	  title={MO-PaDGAN: Generating Diverse Designswith Multivariate Performance Enhancement},
	  author={Chen, Wei and Ahmed, Faez},
	  booktitle={Workshop on Negative Dependenceand Submodularity: Theory and Applications in Machine Learning, 37th International Conference on Machine Learning (ICML)},
	  year={2020},
	  month={July}
        }

## Required packages

- tensorflow < 2.0.0
- sklearn
- numpy
- matplotlib
- seaborn

## Usage

1. Download the airfoil dataset [here](https://drive.google.com/file/d/1OZfF4Zl31jzJmucBIlSqO4OKq9CKHh4r/view?usp=sharing) and extract the NPY files into `airfoil/data/`.

2. Train a surrogate model to predict airfoil performances:

   ```bash
   python train_surrogate.py train
   ```

   positional arguments:
    
   ```
   mode	train or evaluate
   ```

   optional arguments:

   ```
   -h, --help            	show this help message and exit
   --save_interval		interval for saving checkpoints
   ```

3. Go to example directory:

   ```bash
   cd airfoil
   ```

4. Run the experiment:

   ```bash
   python run_experiment.py train
   ```

   positional arguments:
    
   ```
   mode	train or evaluate
   ```

   optional arguments:

   ```
   -h, --help            	show this help message and exit
   --naive			use naive loss for quality
   --lambda0		coefficient controlling the weight of quality in the DPP kernel
   --lambda1		coefficient controlling the weight of the performance augmented DPP loss in the PaDGAN loss
   --id			experiment ID
   ```

   The default values of the optional arguments will be read from the file `airfoil/config.ini`.

   The trained model and the result plots will be saved under the directory `airfoil/trained_gan/<lambda0>_<lambda1>/<id>`, where `<lambda0>`, `<lambda1>`, and `<id>` are specified in the arguments or in `airfoil/config.ini`.

