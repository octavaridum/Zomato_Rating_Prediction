<h1 align="center">Zomato Rating Class Prediction</h1>

![Intro Img](Images/RR.png)

## Background ❓

In the age of digitalization, online platforms have become essential for users to explore and select dining options. Restaurant ratings play a crucial role in helping users make informed decisions about where to dine. These ratings can have huge ramifications on the restaurant business. Therefore, developing an automated system that is able to accurately predicting restaurant ratings based on various features can significantly improve both customer and business experoience as businesses.

## Objective 🎯
The objective of this project is to develop a machine learning model that accurately predicts restaurant ratings based on a set of features. Specifically, the model will perform multi-class classification to assign restaurants into predefined rating classes, such "Class 1" = 0 to 1.9, Class 2 = 2 to 2.9, Class 3 = 3 to 3.9, Class 4 = 4 to 5.

## Solution 💡

Our process will start with Exploratory Data Analysis (EDA) with a goal to gain better insight about our data set and eventually develop a model using machine learning methodology. I will be deploying Ensmeble learning techniques like stacking and blending that might help improve model performance.

## 1. Exploratory Data Analysis 💾

I have acquired the dataset, the next step involves conducting Exploratory Data Analysis (EDA) to glean insights and refine the dataset in preparation for modeling endeavors.

**1.1 Dataset**

**Zomato.csv** dataset represents real-world Zomato ratings for restaurants in Bangalore, encompassing ratings for approximately 51,717 restaurants across 17 distinct features. Among these features, one stands out as the target variable: the rating assigned to each restaurant.

**1.2 Initial Observation**

- The dataset contains 1296675 observation
- The Data set contains a total of 361,199 null values. The target feature "rate" has 7775 null values while "dish_liked" has the most null value 28078  
- The dataset contains a total of 17 columns out of which we have a single dependent variable labeled "rate"
- Out of these 17 columns 'url','address','name','phone','reviews_list','menu_item','listed_in(city)','rest_type','location','cuisines' were dropped for various reasons   
- The 'rate' feature is an object type data that contains values of rating in the format "4.1/5" which must be cleaned and converetd into teh 4 duffrent classes as thsi is a multiclass classification problem
- Nominal Features such as "online_order" ,"book_table" need to be encoded using one hot encoding
- The data has a lot of noise that must be cleaned 

**1.3 Feature Engineering and Data Cleaning**

- Null values were dealt with by sperating the observations with null values into a different .csv file for Inferencing data
- This sepeartion was done before major feature engineering to avoid **data leakage**   
- "Rate" Featured was renamed to "Rating" and cleaned and type cast as categorical data type with values "1","2"."3","4"
- Rare features of "listed_in(type)" were grouped
- One hot encoding was applied to "online_order", "book_table"

**1.3 Univariate and Bivariate**

The following diagram shows the Countplot, percentage distribution of Online_Order in the dataset and we can observe that the data is imbalanced but it is acceptable for our purpose. Furthermore, the figure also contains a countplot of "rating" 



   
