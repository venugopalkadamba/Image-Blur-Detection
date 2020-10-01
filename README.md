<div align="center">

# Image Blur Detection using Machine Learning
</div>

## Question
Image quality detection has always been an arduous problem to solve in
computer vision.<br>
In this assignment you have to come up with features and models to build a classifier which will predict whether a given image is blurred.

## Steps for processing the images in dataset, generating features and model creation

<b>STEP-1:</b> Install all the dependencies mentioned in requirements.txt file by running following command in command prompt<br>

```python
pip install -r requirements.txt
```

<b>STEP-2:</b> After installing all the dependencies, make sure that the dataset contains images in below mentioned paths<br>
<b>TRAIN IMAGES DIRECTORY</b><br>
CERTH_ImageBlurDataset/TrainingSet/Undistorted<br>
CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred<br>
CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred<br>
<b>EVALUATION IMAGES DIRECTORY</b><br>
"CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet<br>
"CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet<br>

<b>STEP-3:</b> After completion of above two steps, for processing the images and generation of features, execute "image_processing_and_feature_generation.py" file in command prompt<br>

```python
python image_processing_and_feature_generation.py
```
After executing the above file, you can notice two csv files in current working directory "train.csv" and "validation.csv".

<b>STEP-4:</b> After executing the above command, execute "train_model.py" in command prompt<br>

```python
python train_model.py
```
After executing the above file, you can notice two pickle files in current working directory "XGBoost.pkl" and "voting_model.pkl".

<b>STEP-5:</b> After executing the above command, execute "validation.py" in command prompt to get the scores on evaluation dataset<br>

```python
python validation.py
```
After executing the above file, you can notice a pickle file "Final_Model.pkl" which got the highest validation accuracy.
