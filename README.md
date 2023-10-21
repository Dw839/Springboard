![image](https://github.com/Dw839/Springboard/assets/121418737/ac4ee5da-4cfc-4e7f-845e-13b00273f080)

# **Flight Delay Prediction Using Machine Learning**
#### *Springboard/UMGC*
#### *Data Science Bootcamp*
#### *October, 2023*


### **Introduction**
Flight delays in the United States have become a ubiquitous and often frustrating aspect of air travel. Whether you're a frequent flyer or an occasional traveler, the experience of waiting for a delayed flight has likely touched your journey at some point. The U.S. aviation industry is one of the busiest and most complex in the world, serving millions of passengers each day, but it also faces a multitude of challenges that can disrupt flight schedules. From adverse weather conditions and air traffic congestion to technical issues and staffing shortages, a myriad of factors contribute to flight delays, impacting passengers, airlines, and the economy. In this exploration, we will delve into the underlying causes, consequences, and potential solutions surrounding flight delays in the United States, shedding light on the complexities of an issue that affects travelers and the aviation industry as a whole.
The aim of the machine learning engine is to create a reliable method to predict flight delays for better management and or prevention.

### **Problem Statement**
The ability to predict flight delays holds value for various stakeholders, encompassing airlines and passengers alike. In this investigation, we delve into the approach of forecasting flight delays through the classification of individual flights as either delayed or punctual. Upon initial examination, it becomes evident that the flight delay dataset exhibits an inherent skew. This outcome is unsurprising, given that airlines frequently operate a higher number of on-time flights compared to delayed ones. As a result, this study undertakes a comparative analysis of various techniques for addressing the challenges posed by an imbalanced dataset while training a flight delay prediction model.

### **Objectives**
The objectives of this study are:
+	To identify the attributes that affect flight delay
+	To develop machine learning models that classify outcomes (either delayed or nor delayed) with selected features.
+	To evaluate performance of different machine learning models.

### **Data Sources**
The data was obtained from the "Airline Delay and Cancellation Data, 2009 – 2018" at Kaggle page. The dataset consisting of flight information in the United States from 2009 to 2018 was obtained from the source of the U.S. Department of Transportation's Bureau of Transportation Statistics. In this study, the only data utilized was from the year 2016. It consisted of 28 columns and 5, 617, 658 data points.
Here's 2016.csv 
Airport data: airports.csv

### **Data Preprocessing**
* This process was executed by Python
To facilitate the modeling process, the only flight data that was considered and included was the data from the busiest airports since they contained the most significant number of schedules for arrival flights in the U.S. Data cleansing was performed on the name of flight carrier, origin airport and destination airport as the abbreviation of IATA code was used. Attributes with more than 50% of missing values that did not provide helpful information to this analysis were dropped—unrelated attributes such as attributes that recorded the outcome of canceled flights and diverted flights were also removed. Since the main objective was to predict flight delay, attributes relating to canceled flights were eliminated. Instances with missing values were removed as the number of missing values was less than 1%, which was relatively small.
For classification purposes, a binary attribute, namely "flight delay," was added to the record status of the flight. The duration between the flights taking off and the wheels off the ground, as well as flight on land and wheels on land, were derived as this provided information about the actual duration of these activities. Information about a month, day, and day of the week was transformed from the actual flight date. Before modeling, all categorical attributes such as destination airports, day of the week, flight carrier, and flight delay factors were converted to numerical variables via one hot encoding method. One dummy variable would be created for every object in the categorical variable. If the category is presented, the value would be denoted as one (1). Otherwise, the value would be denoted as zero (0).

### **Feature Selection**
This process was executed by Python
The constant variable was removed as it did not provide helpful information to the model. Attributes highly correlated to each other were examined to avoid the multicollinearity effect on the model by selecting the most predictive one. Planned elapsed time, Wheels Off elapse, and actual elapsed time correlate higher than 0.9. In this group, several attributes were highly correlated. To select which attributes to remove, a random forest algorithm was utilized to determine their feature importance. Thus, the wheel off elapsed time was not removed as it gave the greatest importance compared to other attributes (shown in Table below).

| **Attributes** | **Importance Scores** |
| ---------- | -------------    |
| *Wheels_Off_elapse* | **0.96844** |
| *Actual_elapse_time* | **0.003170** |
| *Planned_elapse_time* | **0.3156** | 

Figure below shows the features the random forest classifier reported along with their importance score, arranged in descending order. It is interesting to note that scheduled arrival day, month, and destination airport did not contribute much to a flight's arrival delay. Attributes with low importance scores were eliminated as keeping all of them did not yield better results for training models. Thus, only the first nine attributes were used to train the remaining models.

![image](https://github.com/Dw839/Springboard/assets/121418737/04300308-c41d-431c-9c3c-5ed090a708bb)

### *SUMMARY REPORT*
##### **RANDOM FOREST**
|    | **Precision** | **Recall** | **F1Score** |
| --- | --- | --- | --- |
| Class 0 | 0.19 | 0.99 | 0.96 |
| Class 1 | 0.99 | 0.34 | 0.51 | 
| Weighted Avg | 0.92 | 0.92 | 0.90 | 

##### **LOGISTIC REGRESSION**
| --- | **Precision** | **Recall** | **F1Score** |
| --- | --- | --- | --- |
| Class 0 | 0.98 | 0.99 | 0.98 |
| Class 1 | 0.91 | 0.83 | 0.86 |
| Weighted Avg | 0.97 | 0.97 | 0.97 |

##### **DECISION TREE**
| --- | **Precision** | **Recall** | **F1Score** |
| --- | --- | --- | --- |
| Class 0 | 0.98 | 0.98 | 0.98 |
| Class 1 | 0.86 | 0.85 | 0.86 |
| Weighted Avg | 0.96 | 0.96 | 0.96 |

 + Precision is the frequency with which a model was correct for a particular class. Recall can be defined as, for a particular class, how many did the model correctly identify? 
+	CLASS 0: FLIGHT NOT DELAYED 
+	CLASS 1: FLIGHT IS DELAYED 
+	P( for class 1)= TN(TN + FN) 
+	P( for class 0)= TP/(TP + FP) 
+	R( for class 1)= TN/(TN + FP) 
+	R( for class 0)= TP/(TP + FN) 
+	Where P is precision, TP is true positive, FP is false positive, FN is false negative and R is recall. There are two cases in which our model can give a wrong result. One is FLIGHT NOT 
+	DELAYED, BUT PREDICTED TO BE DELAYED(FN). The second is FLIGHT IS DELAYED, BUT PREDICTED TO BE NOT DELAYED(FP). The second case is far more dangerous for the passenger. Hence we want the recall of the class: flights delayed (CLASS 1) to be high.  

### **Modeling and Performance Evaluation**
In the provided test set results for three different machine learning models, here are the key performance metrics:

##### **Random Forest**
+ ROC-AUC: 0.9785
+ Accuracy: 0.9175
+ Precision: 0.9893
+ Recall: 0.3412
+ F-measure: 0.5074

##### **Logistic Regression**
+ ROC-AUC: 0.9909
+ Accuracy: 0.9677
+ Precision: 0.9052
+ Recall: 0.8270
+	F-measure: 0.8644

##### **Decision Tree**
+	ROC-AUC: 0.9168
+	Accuracy: 0.9646
+	Precision: 0.8608
+	Recall: 0.8533
+	F-measure: 0.8570
Each model’s performance is summarized based on precision, recall, F1-score, and support for two classes (0 and 1) in the dataset. The logistic regression model has the highest ROC-AUC and accuracy, while the random forest model has the highest precision. The decision three model also exhibits strong performance, with a balance between precision and recall.

### **Conclusion**
In conclusion, all three objectives were achieved in this project. Valuable attributes for modeling were discovered, such as Departure Delay, Wheels On/Off Elapse, Taxi In/Out, Distance, and others. The performance of three base algorithms, namely Random Forest, Logistic Regression, and Decision Tree were evaluated on a test dataset 
The   Random Forest model demonstrated a high level of precision, indicating its ability to accurately classify positive instances. However, it had a relatively low recall, suggesting it may miss some positive cases. The Logistics Regression model exhibited an excellent ROC-AUC score and a good balance between precision and recall, making it a strong performer overall. Finally, the Decision Tree model showed competitive results with a reasonable balance between precision and recall.
This machine learning model shows the Logistic Regression or Decision Tree model is suitable for a balance between precision and recall.


