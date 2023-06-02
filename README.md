# Keras Machine Learning for contrast-enhanced mammography 
Keras implementation of RetinaNet object detection as described in [Combining deep learning and handcrafted radiomics for classification of suspicious lesions on contrast-enhanced mammograms](link)
by Manon P.L. Beuque, Marc B.I. Lobbes, Yvonka van Wijk, Yousif Widaatalla, Sergey Primakov, Michael Majer, Corinne Balleyguier, Henry C. Woodruff, Philippe Lambin.

The repository consists in two parts: 
- The first part allows the user to train/test a Mask-RCNN model which predicts the location of a candidate suspicious lesion and predicts a label (benign/malignant).
- The second part allows the user to train/test a machine learning model using radiomics on either the ground truth suspicious lesions or the candidate lesion predicted in the first part. It returns a predicted label benign or malignant.

## Installation
This repository is using Python 3.7 and works with keras version 2.2.4.
Install the packages in requirements.txt file to start using the repository.

## Delineations and predictions with Mask-RCNN
### Training and annotations format
The format of the annotations used for training is described in [Keras MaskRCNN](https://github.com/fizyr/keras-maskrcnn) repository.
Provide 0 for 'benign' and 1 for 'malignant'.
The original dataset used was provided as .mha but DICOM images could also be used after small modification of the preprocessing functions.
The data needs to be preprocessed with preprocessing.utils.preprocessing before usage.
The model can be trained with train_delineation_model.py. 

### Predictions
The delineation predictions are obtained with inference_delineation_and_dl_predictions.py.
Use a csv file to indicate the location of the low energy and corresponding recombined images.
The file should contain "low_energy_paths" and "recombined_paths" headers.


## Predictions using radiomic features
To train and test a machine learning model using radiomic features, the features needs to be extracted from the data.
You will need a csv file containing the following headers: "path_mask","path_low_energy", "path_recombined", "outcome" (which needs to be 0 or 1) to generate a csv file containing the features to use.

## Disclaimer and license
The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
The tool is provided for internal non-commercial academic research purposes only. See license for more information.

[link]: https://github.com/fizyr/keras-maskrcnn