---
title: "Academic Performance"
date: 2018-11-19
tags: [machine learning, data science, neural network]
header:
  image: "/images/Performance.jpg"
excerpt: "Looking at what contributes to student performance"
mathjax: "true"
---

# Project Objective

The purpose behind this project was to attempt to recreate results published by Hussain S, Dahan N.A, Ba-Alwi F.M, and Ribata N. titled "Educational Data Mining and Analysis of Student's Academic Performance Using WEKA". In this paper, the authors take data compiled regarding Indian school students and attempt to determine likelyhood of success using a variety of machine learning techniques. This type of analysis can be very important in determining what factors need to be addressed to improve student performance in school systems and prevent drop outs. It also allows a window into what kind of soceital issues play a part in academic performance.

In the paper the authors compare several types of algorithsm such as Neural Networks, SVM, K nearest neighbors, and Random forest. These are used in classification tasks to determine whether a student will perform well, poorly, etc. Before running these algorithms, the authors use feature selection techniques to determine which features play the largest roles in the labels. Metrics used to evaluate performance were Recall, Precision, F-scores, MSE, MAE, and accuracy. The maximum accuracy attained was 99$$\%$$. 

<img src="{{ site.url }}{{ site.baseurl }}/images/indian_school_attributes.png" alt="Table listing attribute qualities. Taken from Hussain S. et al.">

The label in this study is ESP, end of semester percentage. The dataset used for this study is freely available at [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Student+Academics+Performance). The authors were able to narrow down the feature dimensions to 11, using a variety of feature reduction methods. 

This project will attempt to recreate the results systematically while learning about the effect of feature dimensions on outcomes. First, I will look at using all feature dimensions to predict outcome using KNN, Meanshift (for exploratoy analysis), and SVM. Then, I will narrow features to find the most important ones, and finally will compare the metric of the algorithms to determine which is most effective. 

# Predictions using all given features

## Reading in the Data

The data will be read in using pandas with the following code:

'''

    import pandas as pd
        performance_data = pd.read_csv('C:/Users/Aaron/Desktop/Python Files/academic_performance.txt', 
                               names=['Gender','Caste','Class_X','Class_XII',
                               'Int_Asses_Per','End_Sem_Per','Arrears','Marital','Town_City',
                               'Admission','Family_Income','Family_Size','Father_Edu','Mother_Edu',
                               'Father_Occ','Mother_Occ','Friends','Study_Hours',
                               'School_Type','Language','Travel_Time','Attendence'])

    print(performance_data.head(1), '\n the types of data present are \n', performance_data.dtypes)

'''

From this we can see that all of our data types are of type "object", which means we will need to map this to numeric values before using the data. To do this we will write a function that maps non-numeric values to numeric ones:

'''
    
    import numpy as np
    def dataframe_converter(data):
      #first will create all of the column names in a array
      column_names = data.columns.values
      #now will loop over each column
         for column in column_names:
           value_dic = {}
           if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            #now if a column isnt ints or floats we look into it
            #want to find all unique values in the column
            all_vals = data[column].values.tolist()
            #now find all the unique values
            uniques = set(all_vals)
            #now a value to map to 
            val = 0
            for unique in uniques:
              if unique not in value_dic:
                value_dic[unique] = val
                val +=1 
          #now map unique values to data
          data[column] = data[column].map(lambda x: value_dic.get(x))
        return data

'''

and now can take a look at the data after its passed through. 

<img src="{{ site.url }}{{ site.baseurl }}/images/indian_data_converted.png" alt="Data after running through the converter function">

We can see that the data types are now all int64. Perfect! We can run a quick check for missing or NaN values using '''performance_data.isnull().values.any()''', but see that none are present. This is expected, as the data shown on the UCI website is complete. 
Also, I will drop three columns, IAP, TWP, and ESP, as they all provide some form of final assesment. The metric I will attempt to predict will be TNP. Separating into features and labels can be done using a simple df.drop command. Additionally, outliers will not be a problem since all possible values are categorical. So now we are done with data cleaning and are ready to move on.

## A Niave Approach

As mentioned before, first we will run the complete data set through 3 algorithms and see how they perform before optimizing the data set to remove unnecesary features. The data will be split into training and test sets using cross_validation's train_test_split. Now, I will compare the results of the KNN (K=6 because we have 5 classes available) and SVM. First, I'll use KNN. Accuracy comes out to be $$59.18\%$$ (yikes). Below are pie charts of both the predictions on the test set, and the actual test set. 

<img src="{{ site.url }}{{ site.baseurl }}/images/KNN_all_features_actual.png" alt="Prediction of the KNN algorithm using all features">
<img src="{{ site.url }}{{ site.baseurl }}/images/KNN_all_features_predicted.png" alt="Actual pie chart of the test data set">

The KNN predicted no best scores, and was slightly inaccurate at best on other categories. Hopefully the SVM will do a bit better. 