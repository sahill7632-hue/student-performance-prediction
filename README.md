# Student Performance Prediction

This project predicts a student's final marks (G3, out of 20) using their earlier
exam marks, study time, absences and a few other details. It was made for my
5th semester Machine Learning course.

## Dataset
Kaggle "Math Students" dataset (originally from the UCI Machine Learning
Repository, Cortez and Silva, 2008). It has 395 students and 33 columns.
The file is in the Dataset folder.

## What I did
1. Loaded the data and checked for missing values and duplicates
2. Explored it with a histogram, a correlation heatmap and boxplots
3. Converted text columns into numbers
4. Split the data 80% for training and 20% for testing
5. Trained Linear Regression and Random Forest
6. Compared them using MAE and R2
7. Checked which features mattered most
8. Added a small predictor that asks 6 questions and estimates a student's marks

## Results
| Model | MAE | R2 |
|---|---|---|
| Linear Regression | 1.65 | 0.72 |
| Random Forest | 1.16 | 0.81 |

Random Forest performed better. The most important feature was G2 (second exam
marks), followed by absences.

## How to run
1. Install the libraries: `pip install -r requirements.txt`
2. Keep the Dataset folder in the same folder as the notebook
3. Open Student_Performance_Prediction.ipynb and run the cells from top to bottom

## Files
- Student_Performance_Prediction.ipynb: the full project notebook
- Dataset/student-mat.csv: the data
- Notebook.pdf: a PDF copy of the notebook with outputs
- requirements.txt: the libraries needed