<div align="center">

# Image Blur Detection using Machine Learning
</div>

## QUESTION
Image quality detection has always been an arduous problem to solve in
computer vision.<br>
In this assignment you have to come up with features and models to build a classifier which will predict whether a given image is blurred.

## APPROACH
We can classify whether a image is blurred or not by observing the edges in the image. One of the most commonly used method for detecting edges from the image is Laplacian method. The reason for using laplacian operator is its definition itself, laplacian operator is used to get the second derivatives of the images. The laplacian oprator highlights the regions where their is rapid change of intensity in the image. After applying the laplacian operator to the image we calculate the <b>variance</b> and <b>maximum</b> of the image pixels. The image with high variance and high maximum are expected to have sharp edges i.e.it's a clear image, whereas the image with less variance and less maximum are expected to be a blur image.<br>
Deciding the threshold, below threshold the image is said to be blurred and above threshold the image is said to be clear, is a major task. We can tackle this problem by using machine learning algorithms. We can make use of variance and maximum as features for our machine learning model to predict whether the given image is a blur image or clear image.

**NOTE:** In all the python files <b>Label 1 = 'Blur Image'</b> and <b>Label 0 = 'Clear Image'</b>

## STEPS FOR EXECUTING THE FILES

<b>STEP-1:</b> Install all the dependencies mentioned in requirements.txt file by running following command in command prompt<br>

```python
pip install -r requirements.txt
```

<b>STEP-2:</b> After installing all the dependencies, make sure that the dataset contains images in below mentioned paths<br>
<ul>
  <li>CERTH_ImageBlurDataset/TrainingSet/<br>
    <ul>
      <li>Undistorted/</li>
      <li>Naturally-Blurred/</li>
      <li>Artificially-Blurred/</li>
    </ul>
  </li>
  <li>CERTH_ImageBlurDataset/EvaluationSet/ <br>
    <ul>
      <li>DigitalBlurSet/</li>
      <li>NaturalBlurSet/</li>
      <li>DigitalBlurSet.xlsx</li>
      <li>NaturalBlurSet.xlsx</li>
    </ul
  </li>
</ul>

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
