Human Activity Recognition - Weight Lifting
===========================================

Introduction
------------

The purpose of this research is to design a method to recognize the way
an exercise is performed. The focus is therefore on the quality of a
physical activity. Six young participants were asked to perform one set
of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five
different fashions: exactly according to the specification (Class A),
throwing the elbows to the front (Class B), lifting the dumbbell only
halfway (Class C), lowering the dumbbell only halfway (Class D) and
throwing the hips to the front (Class E). Class A corresponds to the
specified execution of the exercise, while the other 4 classes
correspond to common mistakes. Participants were supervised by an
experienced weight lifter to make sure the execution complied to the
manner they were supposed to simulate. The exercises were performed by
six male participants aged between 20-28 years, with little weight
lifting experience, who could easily simulate the mistakes in a safe and
controlled manner by using a relatively light dumbbell (1.25kg). Sensors
were mounted in the users' glove, armband, lumbar belt and dumbbell.
Each sensor recorded features on the Euler angles (roll, pitch and yaw),
as well as the raw accelerometer, gyroscope and magnetometer readings.
Additional feature were calculated: mean, variance, standard deviation,
max, min, amplitude, kurtosis and skewness, generating in total 96
derived feature sets.

The data used in this research has been downloaded from:
<http://groupware.les.inf.puc-rio.br/har#ixzz4hIdtJOTy>

Data Splitting
--------------

The training file has been divided in a training dataset and a
validation dataset with a 70-30 split. Since the dataset is ordered by
Class, in order to avoid imbalances in the outcome variable, the split
can't be done with K-fold splitting but it's done with random sampling.

    library("caret", lib.loc="~/R/win-library/3.3")

    ## Warning: package 'caret' was built under R version 3.3.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library("lattice", lib.loc="~/R/win-library/3.3")
    library("ggplot2", lib.loc="~/R/win-library/3.3")
    library("randomForest", lib.loc="~/R/win-library/3.3")

    ## Warning: package 'randomForest' was built under R version 3.3.3

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library("gbm", lib.loc="~/R/win-library/3.3")

    ## Warning: package 'gbm' was built under R version 3.3.3

    ## Loading required package: survival

    ## Warning: package 'survival' was built under R version 3.3.3

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.3

    traindata <- read.csv("pml-training.csv")
    testdata <- read.csv("pml-testing.csv")
    set.seed(666)
    valindex <- createDataPartition(traindata$classe, p = 0.70,list=FALSE)
    trainingdata <- traindata[valindex,]
    validationdata <- traindata[-valindex,]

Exploratory Analysis and Data Cleaning
--------------------------------------

To confirm that the training dataset is not unbalanced regarding Class,
the total number of records in each group is calculated. More records
are present for Class A but the difference is tolerable and will not
jeopardize the results of the model.

    table(trainingdata$classe)

    ## 
    ##    A    B    C    D    E 
    ## 3906 2658 2396 2252 2525

Some columns have NAs or "" for the vast majority of records. 13457 is
identified as the threshold so every variable that has more than 13457
NAs or "" is eliminated from the dataset. Then the first 7 columns are
eliminated since they refer to features like name of the subject or
timestamp, that have no explanatory power. Finally the columns whose
values are stored as integer are converted to numeric values. In order
to keep the datasets consistent, the same columns are then eliminated or
converted in the validation and testing datasets.

    #eliminate columns that are mostly NAs
    apply(is.na(trainingdata),2,sum)

    ##                        X                user_name     raw_timestamp_part_1 
    ##                        0                        0                        0 
    ##     raw_timestamp_part_2           cvtd_timestamp               new_window 
    ##                        0                        0                        0 
    ##               num_window                roll_belt               pitch_belt 
    ##                        0                        0                        0 
    ##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
    ##                        0                        0                        0 
    ##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
    ##                        0                        0                        0 
    ##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
    ##                        0                        0                    13458 
    ##           max_picth_belt             max_yaw_belt            min_roll_belt 
    ##                    13458                        0                    13458 
    ##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
    ##                    13458                        0                    13458 
    ##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
    ##                    13458                        0                    13458 
    ##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
    ##                    13458                    13458                    13458 
    ##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
    ##                    13458                    13458                    13458 
    ##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
    ##                    13458                    13458                    13458 
    ##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
    ##                        0                        0                        0 
    ##             accel_belt_x             accel_belt_y             accel_belt_z 
    ##                        0                        0                        0 
    ##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
    ##                        0                        0                        0 
    ##                 roll_arm                pitch_arm                  yaw_arm 
    ##                        0                        0                        0 
    ##          total_accel_arm            var_accel_arm             avg_roll_arm 
    ##                        0                    13458                    13458 
    ##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
    ##                    13458                    13458                    13458 
    ##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
    ##                    13458                    13458                    13458 
    ##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
    ##                    13458                    13458                        0 
    ##              gyros_arm_y              gyros_arm_z              accel_arm_x 
    ##                        0                        0                        0 
    ##              accel_arm_y              accel_arm_z             magnet_arm_x 
    ##                        0                        0                        0 
    ##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
    ##                        0                        0                        0 
    ##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
    ##                        0                        0                        0 
    ##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
    ##                        0                        0                    13458 
    ##            max_picth_arm              max_yaw_arm             min_roll_arm 
    ##                    13458                    13458                    13458 
    ##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
    ##                    13458                    13458                    13458 
    ##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
    ##                    13458                    13458                        0 
    ##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
    ##                        0                        0                        0 
    ##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
    ##                        0                        0                        0 
    ##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
    ##                        0                        0                    13458 
    ##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
    ##                    13458                        0                    13458 
    ##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
    ##                    13458                        0                    13458 
    ## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
    ##                    13458                        0                        0 
    ##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
    ##                    13458                    13458                    13458 
    ##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
    ##                    13458                    13458                    13458 
    ##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
    ##                    13458                    13458                    13458 
    ##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
    ##                    13458                        0                        0 
    ##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
    ##                        0                        0                        0 
    ##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
    ##                        0                        0                        0 
    ##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
    ##                        0                        0                        0 
    ##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
    ##                        0                        0                        0 
    ##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
    ##                        0                        0                        0 
    ##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
    ##                        0                    13458                    13458 
    ##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
    ##                        0                    13458                    13458 
    ##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
    ##                        0                    13458                    13458 
    ##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
    ##                        0                        0                    13458 
    ##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
    ##                    13458                    13458                    13458 
    ##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
    ##                    13458                    13458                    13458 
    ##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
    ##                    13458                    13458                    13458 
    ##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
    ##                        0                        0                        0 
    ##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
    ##                        0                        0                        0 
    ##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
    ##                        0                        0                        0 
    ##                   classe 
    ##                        0

    notNA <- apply(is.na(trainingdata),2,sum)>13457
    trainingdata1 <- trainingdata[,!notNA]

    #eliminate columns that are mostly ""
    apply(trainingdata1=="",2,sum)
    notNA2 <- apply(trainingdata1=="",2,sum)>13457
    trainingdata2 <- trainingdata1[,!notNA2]
    #eliminate columns that refer to user or time measurement and have no explanatory power
    trainingdata3 <- trainingdata2[,8:60]
    str(trainingdata3)
    #Convert integer variables to numeric
    trainingdata3[,c(4,8:13,17,21:26,30,34:38,43,47:50)] <- sapply(trainingdata3[,
                  c(4,8:13,17,21:26,30,34:38,43,47:50)],as.numeric)
    #Repeat same steps for validation and testing data sets
    validationdata1 <- validationdata[,!notNA]
    testdata1 <- testdata[,!notNA]
    validationdata2 <- validationdata1[,!notNA2]
    testdata2 <- testdata1[,!notNA2]
    validationdata3 <- validationdata2[,8:60]
    testdata3 <- testdata2[,8:60]
    validationdata3[,c(4,8:13,17,21:26,30,34:38,43,47:50)] <- sapply(validationdata3[,
                  c(4,8:13,17,21:26,30,34:38,43,47:50)],as.numeric)
    testdata3[,c(4,8:13,17,21:26,30,34:38,43,47:50)] <- sapply(testdata3[,
                  c(4,8:13,17,21:26,30,34:38,43,47:50)],as.numeric)

The selected explanatory variables have been compared to each other and
to the outcome Class in different plots. Due to the high number of
variables, it was impossible to plot them all in the same graph, so they
were divided in groups of closely related variables. For example, the
first plot included the roll, pitch and yaw of the belt sensor mounted
at the hips. This is the most interesting plot since it shows the
different pattern for Class E, which is the class of exercises performed
by throwing the hips to the front. No other plot showed any particular
pattern between an explanatory variable and Class so they have been
omitted from this report.

    pairs(trainingdata3[,c(1:3,53)])

![](Practical-Machine-Learning/blob/master/plot1-1.png)

    pairs(trainingdata3[,c(4:10,53)])

    pairs(trainingdata3[,c(11:13,53)])

    pairs(trainingdata3[,c(14:16,53)])

    pairs(trainingdata3[,c(17:23,53)])

    pairs(trainingdata3[,c(24:26,53)])

    pairs(trainingdata3[,c(27:29,53)])

    pairs(trainingdata3[,c(30:36,53)])

    pairs(trainingdata3[,c(37:39,53)])

    pairs(trainingdata3[,c(40:42,53)])

    pairs(trainingdata3[,c(43:49,53)])

    pairs(trainingdata3[,c(50:53)])

Model Building
--------------

### 1. Random Forest

The first model built is with random forest, due to the noise in the
sensor data. Random forest is a typical choice for such problems.
10-fold cross validation is used to fine tune the model, which means
determining the ideal value for mtry, which is the number of variable
that are randomly selected at every step of the algorithm. A typical
starting value for mtry is floor(sqrt(ncol(x))), where ncol is the
number of explanatory variables, in this case 52. The formula gives mtry
= 7, so the search will be done with mtry in the range \[1; 15\]. From
the results below, the best performance is achieved with mtry = 9,
although most of the values are very close to each other so the
differences might be due more to noise than the effect of mtry.

    tunegrid <- expand.grid(.mtry=c(1:15))
    set.seed(999)
    RFmodel <- train(classe~.,data=trainingdata3,method="rf",verbose = FALSE,
                     trControl=trainControl(method="cv",number=10),tuneGrid=tunegrid)
    print(RFmodel)

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 12363, 12364, 12364, 12363, 12362, 12363, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    1    0.9884259  0.9853573
    ##    2    0.9921386  0.9900552
    ##    3    0.9929390  0.9910676
    ##    4    0.9937400  0.9920807
    ##    5    0.9940309  0.9924491
    ##    6    0.9942495  0.9927254
    ##    7    0.9938128  0.9921731
    ##    8    0.9943222  0.9928177
    ##    9    0.9946860  0.9932778
    ##   10    0.9941038  0.9925416
    ##   11    0.9946133  0.9931859
    ##   12    0.9941037  0.9925412
    ##   13    0.9944676  0.9930016
    ##   14    0.9942491  0.9927252
    ##   15    0.9944676  0.9930018
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 9.

The accuracy of the model, calculated on the validation data is 99.47%.
The most important variable is by far the roll measured by the sensor at
the belt, followed by the yaw at the belt, the pitch of the forearm and
the z value of the dumbell. We can say that the greatest indication of
how the exercise was done is the position of the hips: if the hips are
rotated too much or too little it is likely a sign that a mistake is
being made. Also it is an expected result that the pitch of the forearm
is another strong indicator since the dumbell curl requires to move the
forearm up and down, which is exactly the type of rotation measured by
the pitch.

    RFpred <- predict(RFmodel,validationdata3)
    confusionMatrix(RFpred, validationdata3$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    8    0    0    0
    ##          B    1 1128    2    0    0
    ##          C    0    3 1024    9    0
    ##          D    0    0    0  954    7
    ##          E    0    0    0    1 1075
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9947          
    ##                  95% CI : (0.9925, 0.9964)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9933          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9903   0.9981   0.9896   0.9935
    ## Specificity            0.9981   0.9994   0.9975   0.9986   0.9998
    ## Pos Pred Value         0.9952   0.9973   0.9884   0.9927   0.9991
    ## Neg Pred Value         0.9998   0.9977   0.9996   0.9980   0.9985
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1917   0.1740   0.1621   0.1827
    ## Detection Prevalence   0.2856   0.1922   0.1760   0.1633   0.1828
    ## Balanced Accuracy      0.9988   0.9949   0.9978   0.9941   0.9967

    varImp(RFmodel)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt             100.00
    ## yaw_belt               64.51
    ## pitch_forearm          61.40
    ## magnet_dumbbell_z      57.98
    ## magnet_dumbbell_y      51.51
    ## pitch_belt             51.04
    ## roll_forearm           44.35
    ## magnet_dumbbell_x      30.29
    ## accel_dumbbell_y       28.53
    ## roll_dumbbell          28.37
    ## magnet_belt_z          26.04
    ## magnet_belt_y          25.92
    ## accel_belt_z           23.60
    ## accel_forearm_x        20.91
    ## accel_dumbbell_z       19.94
    ## roll_arm               19.21
    ## magnet_forearm_z       17.81
    ## gyros_belt_z           17.30
    ## total_accel_dumbbell   15.90
    ## magnet_belt_x          14.80

### 2. Boosting

Another model is created using Generalized Boosting Method with 10-fold
cross validation. In this case the accuracy is 96.14%. The most
important variables are similar as for the random forest model but the
roll at the belt is even more important compared to the rest. Also the
yaw at the belt is less important than the pitch of the forearm.

    set.seed(999)
    Boostmodel <- train(classe~.,data=trainingdata3,method="gbm",verbose = FALSE,
                     trControl=trainControl(method="cv",number=10))

    ## Loading required package: plyr

    print(Boostmodel)

    ## Stochastic Gradient Boosting 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 12363, 12364, 12364, 12363, 12362, 12363, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7549695  0.6892473
    ##   1                  100      0.8186647  0.7704376
    ##   1                  150      0.8533890  0.8144969
    ##   2                   50      0.8546997  0.8158389
    ##   2                  100      0.9092959  0.8852102
    ##   2                  150      0.9329560  0.9151747
    ##   3                   50      0.8944456  0.8664014
    ##   3                  100      0.9419099  0.9265075
    ##   3                  150      0.9602543  0.9497150
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final values used for the model were n.trees = 150,
    ##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

    Boostpred <- predict(Boostmodel,validationdata3)
    confusionMatrix(Boostpred, validationdata3$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1637   29    0    0    1
    ##          B   29 1077   27    3   12
    ##          C    6   29  989   42    9
    ##          D    1    2    7  912   17
    ##          E    1    2    3    7 1043
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9614          
    ##                  95% CI : (0.9562, 0.9662)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9512          
    ##  Mcnemar's Test P-Value : 1.125e-06       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9779   0.9456   0.9639   0.9461   0.9640
    ## Specificity            0.9929   0.9850   0.9823   0.9945   0.9973
    ## Pos Pred Value         0.9820   0.9382   0.9200   0.9712   0.9877
    ## Neg Pred Value         0.9912   0.9869   0.9923   0.9895   0.9919
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2782   0.1830   0.1681   0.1550   0.1772
    ## Detection Prevalence   0.2833   0.1951   0.1827   0.1596   0.1794
    ## Balanced Accuracy      0.9854   0.9653   0.9731   0.9703   0.9806

    varImp(Boostmodel)

    ## gbm variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                   Overall
    ## roll_belt         100.000
    ## pitch_forearm      48.654
    ## yaw_belt           41.641
    ## magnet_dumbbell_z  33.526
    ## magnet_dumbbell_y  26.812
    ## roll_forearm       23.471
    ## magnet_belt_z      22.932
    ## accel_forearm_x    16.095
    ## roll_dumbbell      14.721
    ## gyros_belt_z       14.714
    ## magnet_forearm_z   11.666
    ## accel_dumbbell_y   11.433
    ## pitch_belt         10.119
    ## accel_dumbbell_x    7.549
    ## gyros_dumbbell_y    7.152
    ## yaw_arm             6.977
    ## magnet_belt_y       6.213
    ## accel_forearm_z     6.181
    ## magnet_arm_z        5.367
    ## magnet_belt_x       4.860

Final Predictions
-----------------

The random forest model is used to perform prediction on the testing
data given the better performances obtained on the validation dataset
and all the predicted outcomes turned out to be correct.

    finalpred <- predict(RFmodel,testdata3)
    print(finalpred)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
