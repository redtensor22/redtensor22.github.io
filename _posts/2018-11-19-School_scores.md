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
The metric I will attempt to predict will be end of semester percentage, like the paper evaluates. Separating into features and labels can be done using a simple df.drop command. Additionally, outliers will not be a problem since all possible values are categorical. So now we are done with data cleaning and are ready to move on.

## A Niave Approach

As mentioned before, first we will run the complete data set through 3 algorithms and see how they perform before optimizing the data set to remove unnecesary features. The data will be split into training and test sets using cross_validation's train_test_split. Now, I will compare the results of the KNN (K=6 because we have 5 classes available) and SVM. First, I'll use KNN. Accuracy comes out to be $$59.18\%$$ (yikes). Below are pie charts of both the predictions on the test set, and the actual test set. 

<img src="{{ site.url }}{{ site.baseurl }}/images/KNN_all_features_actual.png" alt="Prediction of the KNN algorithm using all features">
<img src="{{ site.url }}{{ site.baseurl }}/images/KNN_all_features_predicted.png" alt="Actual pie chart of the test data set">

The KNN predicted no best scores, and was slightly inaccurate at best on other categories. Hopefully the SVM will do a bit better. 

Running the SVM from scikit learn, I get an accuracy of $$36.6\%$$...really bad. The chart is shown below.

<img src="{{ site.url }}{{ site.baseurl }}/images/SVM_all_features_predicted.png" alt="SVM prediction using entire featureset">

It's obvious that the features need to be narrowed down. The shape of the dataframe is 131 rows and 22 columns, so the number of rows is not too much higher than the number of columns.

# Dimensional Reduction

## Univariate Selection
To reduce the dimensionality of the dataset to its most important factors, I will use a couple of different forms of dimensional reduction and see which one provides the best accuracy. The first method I will use is the SelectKBest method in Python. In this method, python can use a variety of statistical tests to find the best features. In this case, I will use the $$\Chi^2$$ test. In this test, we reject or confirm a null hypothesis based on the sum of the squared errors divided by their expected value. If this sum when charted with a critical value called alpha is greater than the critical value, we can reject the null hypothesis. The code to find the new features is:

    
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    test = SelectKBest(score_func=chi2, k=9)
    fit = test.fit(x_train, x_test)

    features_reduced = fit.transform(features)
    #print('new reduced featureset shape is', features_reduced.shape)

    #now will find which ones are best
    feature_names = list(features.columns.values)

    mask = test.get_support() #booleans of the original set
    new_names = [] #list for new best features

    for bool, feature in zip(mask, feature_names):
      if bool:
        new_names.append(feature)
    print('The features found to be most important are:',new_names)

Using this procedure, I find that the optimal number of features is 9, and the most important features are the internal assesment percentage  IAP (part of a continuous semester evaluation), arrears ARR (whether the student had failed any papers in the past semester), LS (whether the student lived in the city or country), admission category AS (whether the student had a free or paid tuition), the educational level of the father FQ, the occupation of the father FO, the number of friends NF, the native language ME, and the attendence percentage ATD. The factors line up with the factors found by the author, however, the author found 15 important categories whereas I find the optimal number to be 9. The 9 found in my optimization are a subset of the set found in the paper. 

## Decision Trees

Next I will use a bagged decision tree method, ExtraTreesClassifier, to determine feature importance. In this method we will use a series of decision trees to give a importance score to each attribute. First I will unpack and apply the sorting model

    from sklearn.ensemble import ExtraTreesClassifier
    #extract the features
    model = ExtraTreesClassifier()
    model.fit(x_train, x_test)
    print('for the extra trees the importance scores are', model.feature_importances_)

The authors proceed by finding the 15 most important attributes, so now I will make a dictionary of the 15 most important according to the ExtraTrees model.

    #create dictionary to map and rank features
    feature_importance_dic = {}
    values = (model.feature_importances_)

    for i in range(len(feature_names)):

  
    sorted_by_value = sorted(feature_importance_dic.items(), key=lambda kv: kv[1])

    print('from feature extra trees the 15 most important values are:', sorted_by_value[-15:])

Next, I will compare my 15 most important features to the ones found by the author. I will do this by repeating the following code for each possible feature reduction method used by the author

      my_list_trees = sorted_by_value[-15:]

      #get just the string labels from 15 most important
      my_list_trees_names = []
      for i in range(len(my_list_trees)):
        my_list_trees_names.append(my_list_trees[i][0])
      my_list_trees_names = sorted(my_list_trees_names)

      for i in range(len(my_list_trees_names)):
        if my_list_trees_names[i] == correlation[i]:
          corr_vals.append(1)
      corr_percentage = (len(corr_vals)/len(my_list_trees_names))

In the above code, I have sorted the 15 most important elements by value. The list my_list_trees includes names of features as well as their score, so next I remove the scores to leave me with a list of only the 15 most important names. Then I go about appending an empty list with an irrelevant value for each match with the author's list. Finally, I divide the length of the matches list by the total length of the original list. 

After comparing accuracies, I get that the list of important attributes that I find is a $$100\%$$ match with the author's symmetrical uncertainty attribute. The sorted list of least to most important features I find is 

<img src="{{ site.url }}{{ site.baseurl }}/images/my_scores_output_indian.png" alt="Least to most important attributes found using ExtraTreesClassifier">
