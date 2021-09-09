#1. many faces of regression: varieties of regression analysis
#simple linear: predicting a quantitative response variable from a quantitative explanatory variable

#polynomial: predicting a quantitative response variable from a quantitative explanatory variable, the relationship is modelef as an nth order polynomial

#multiple linear: predicting a quantitative response variable from two or more explanatory variables

#multilevel: predicting a response variable from data that have a hierachical structure(for example, students within classrooms within schools). Also called hieraichical, nested or mixed models

#multivariate: predicting more than one response variable from one or more explanatory variables

#logistic: predicting a categorical response variable from one or more explanatory variables

#poisson: predicting a response variable representing counts from one or more explanatory variables

#cox proportional hazards: predicting time to an event(death, failure, relapse) from one or more explanatory variables

#time-series: modeling time series data with correlated errors

#nonlinear: predicting a quantitative repsonse variable from one or more explanatory variables where the form of the model is nonlinear

#nonparametric: predicting a quantitative response variable from one or more explanatory variables where the form of the model is derived from the data and not specified a priori

#robust: predicting a quantitative response variable from one or more explanatory variables using an approach that's resistent to the effect of influential observations 


#2. Ordinary least squares regression including simple linear regression, polynomial regression and multiple linear regression.

#in OLS regression, a quantitatvie dependent variable is predicted from a weighted sum of predictor variables


#3. some assumptions 

#normality is for fixed values of the independent variables, the dependent variables are normally distributed

#independece is that the Yi values are independent of each other

#linearity is that the dependent variable is linearly related to the independent variables

#homoscedasticity is that the variance of the dependence variable doesnt vary with the levels of the independent variables

#4. fitting regression models with lm()
myfit <-lm(formula, data) #where formula describes the model to be fit, myfit is a list that contains extensive information about the fitted model
#the formula is tipically written as Y~ X1+X2+..XK

#5. Symbols commonly used in R formulas
# ~: separates response variables on the left from the explanatory variables on the right
# + separates predictor variables
# : denotes an interaction between predictor variables: for example, y ~x+z+x:z
# * a shortcut for edenoting all possible interactions. the code y ~x*z*w expands to y ~ x+z +w + x:z+x:w+x:z:w
# ^ denotes interactions up to a specified degree. The code y ~(x+z+w)^2 
# - a minus sign removes a variable from the equation, y~ (x+z+w)^2 - x:w expands to y ~ x+z+w+x:z+z:w
# -1 suppresses the intercept, for example y ~ x -1 fits a regression of y on x, and forces the line through the origin at x=0
# I() elements within the parentheses are interpreted arithmetically. For example, y ~ x+(z+w)^2 would expand to y ~ x+z +w+z:w. In contrast, the code y ~ x+I((z+w)^2) would expand to y ~x+h, where h is a new variable created by squaring the sum of z and w
# function: mathematical functions can be used in formaulas for example, log(y) ~ x+z+w would predict log(y) from x,z,and w

#6. other functions that are useful when fitting linear models
#summary(): displays detailed results for the fitted model
#coefficients(): lists the model parameters(intercept and slopes) for the fitted model
#fitted(): lists the predictd values in a fitted model
#residuals(): lists the residual values in a fitted model
#anova(): generates an ANOVA table for a fitted model, or an ANOVA table comparing two or more fitted models
#vcov(): list the covariance matrix for the model parameters
#AIC(): prints Akaike's information criterion
#plot(): generates diagnostic plots for evaluating the fit of a model
#predict(): uses a fitted model to predict response values for a new dataset


#7. simple linear regression
fit <-lm(weight ~ height, data=women)
summary(fit)

women$weight

fitted(fit)

residuals(fit)

plot(women$height, women$weight, xlab="Height (in inches)", ylab="Weight (in pounds)")

abline(fit)
#multiple R squared 0.991 indicates that the model accounts for 99.1% of the variance in weights
#the residual satndard error 1.53 can be thought of as the average error in predicting weight from height using the model
#multiple R squared also the squared correlation between the actual and predicted value
#since there is only one predictor variable in simple regression, F test is equivalent to the t test for the regression coefficient for height

#8. polynomial regression
fit2 <- lm(weight ~ height + I(height^2), data=women)#try to improve the prediction using a regression with a quadratic term(that is, X^2)
summary(fit2)

plot(women$height, women$weight, xlab="Heigt (in inches)", ylab="Weight (in lbs)")

lines(women$height, fitted(fit2))

#9. scatterplot() funtion in the car package provides a simple and convenient method of plotting a bivariate relationship

library(car)
scatterplot(weight ~ height, data= women,
            spread=F, smoother.args=list(lty=2), pch=19,
            main="Women Age 30-39",
            xlab="Height (inches)",
            ylab="Weight(lbs.)")
#spread=F option suppress spread and asymmetry informtaion
#smoother.args=list(lty=2) option specifies that loess fit be rendered as a dashed line

#10. Multiple linear regression
states <-as.data.frame(state.x77[,c("Murder", "Population","Illiteracy","Income","Frost")])

cor(states)

library(car)

scatterplotMatrix(states, spread=F, smmother.args=list(lty=2),
                  main="scatter Plot Matrix")
#scatterplotMatrix() function provides scatter plots of the variables with each other in the off-diagonals and superimposes smoothes(loess) and linear fit lines on these plots
#the principal diagonal contains density and rug plots for each variable 

states <-as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income", "Frost")])

fit <-lm(Murder ~Population +Illiteracy+Income +Frost, data=states)

summary(fit)


#11. multiple linear regression with interactions
fit <-lm(mpg ~ hp +wt+hp:wt, data=mtcars)
summary(fit)

#we can visulize the interaction through using the effect()function in the effects package 
plot(effect(term, mod, xlevels), multiline=T) #term is the quoted model term to plot, mod is the fitted model returned by lm(), xlevels is a list specifying the variables to be set to constant values and the values to employ, multiline=T option superimposes the lines being plotted

install.packages("effects")
library(effects)
plot(effect("hp:wt", fit,, list(wt=c(2.2,3.2,4.2))), multiline=T)


#12. regression diagnostics: evaluate whether meeting the statistical assumptions underlying the approach before we can have confidence in the inferences to draw

confint()

fit <-lm(Murder ~ Population +Illiteracy +Income + Frost, data=states)

confint(fit)
#the results suggest that you can be 95% confident that
#the interval [2.38,5.90] contains the true change in murder rate for a 1% change in illiteracy rate
#the confidence interval for frost contains 0, we can conclude that a change in temperature is unrelated to murder rate, holding the other variables consttant

#13. a typical approach on regression diagnostics
fit <- lm(weight~ height, data=women)
par(mfrow=c(2,2))
plot(fit)

#consider four parts of the assumptions of OLS regression: 1. normality, 2. independence, 3. linearity, 4. Homoscedasticity
#1. normality: if meet the normality assumption(the dependent variable is normally distrbuted thus the residual values are normally idstributed with the mean of 0), the points on Normal Q-Q plot should fall on the straight 45-degree line
#2. independence: the independence of the data of the predictor values
#3. linearity: if the dependent variable is linearly related to the independent variables, there should be no systematic relationship
#between the residuals and the predicted(that is, fitted)values. In the Residual s vs. fitted graph, there is a curved relationship which suggests that we may add aquadratic term to the regression
#homoscedasticity: if we meet the constand variance assumption, the points in the Scale-location graph should be a random band around a horizontal line.

#finally, the residuals vs leverage graph provides information about individual observations that we may wish to attend to
#the graph identifies outliers, high-leverage points and influential observations

#an outlier is an observation that is not predicted well by the fitted regression model
#an observation with a high leverage value has an unusual combination of predictior values, 
#an influential observation is an observation that has a disproportionate impact on the determination of the model parameters

fit2 <-lm(weight ~height +I(height^2), data=women)
par(mfrow=c(2,2))
plot(fit2)
#the second graph suggests the polynomial regression provides a better fit with regard to the linearity assumption, normality of residuals(except for observation 13) and homoscedasticity(constant residual variance)

#observation 15 appears to be influential(based on a large Cook's D value)
newfit <-lm(weight ~height +I(height^2), data=women[-c(13,15),])
par(mfrow=c(2,2))
plot(newfit)

#14. enhanced approach
#car package provides a number of functions that significantly enhave the ability to fit and evaluate rergression models
qqPlot() #Quantile comparisons plot
durbinWatsonTest()#durbin-watson test for autocorrelated errors
crPlots()# component plus residual plots
ncvTest()# score test for nonconstant error variance
spreadLevelPlot()#spread-level plots
outlierTest()#bonferroni outlier test
avPlots()# added variable plots
influencePlot()# regression influence plots
scatterplot()#enhaced scatter plot 
scatterplotMatrix()#emjaced scatter plot matrixes
vif()#variance inflation factors

#the gvlma package provides a global test for linear model assumptions as well

#ggplt() function provides a more accurate method of assessing the normality assumption than that provided by the plot() funtion in the base package
#it plots the studentized redisuals(also called studentized deleted residuals or jackknifed residuals) against a t distribution with n-p-1 d.o.f

library(car)
states <- as.data.frame(state.x77[,c("Murder","Population", "Illiteracy","Income","Frost")])
fit <-lm(Murder ~Population+Illiteracy +Income+Frost, data=states)
qqPlot(fit, labels=row.names(states), id.method="identify", simulate=T, main="Q-Q Plot")
#id.method="identify" makes the plot interactive after the graph is drawn, mouse clicks on points in the graph will label them with values specifed in the labels option of the function
#simulate=T, a 95% confidence envelope is produced using a parametric bootstrap

#Nevade has a large positive residual
states["Nevada",]

fitted(fit)["Nevada"]

residuals(fit)["Nevada"]

rstudent(fit)["Nevada"]

#the murder rate is 11.5 but the model predict 3.9, residplot() function generates a histogram of the studentized residuals and superimposes a normal curve, kernel-density curve and rug plot
# it is for visualing errors further

residplot <-function(fit, nbreaks=10){
  z<-rstudent(fit)
  hist(z, breaks=nbreaks, freq=F, 
       xlab="Studentized Residual",
       main="Distribution of Errors")
  rug(jitter(z), col="brown")
  curve(dnorm(x, mean=mean(z), sd=sd(z)),
        add=T, col="blue", lwd=2)
  lines(density(z)$x, density(z)$y, col="red", lwd=2, lty=2)
  legend("topright",
         legend = c("Normal Curve", "Kernel Density Curve"),
         lty=1:2, col=c("blue", "red"), cex=.7)
  residplot(fit)
}
residplot(fit)
#the errors follow a normal distribution quite well with the exception of a large outlier
#although the Q-Q plot is probably more informative, it may be easier to gauge the skew of a distribution from a histogram or density plot than from a probability plot. Why not use both?

#15. independence of errors
#the best way to assess whether the dependent variable values(and thus the residuals) are independent is from your knowledge of how the data were collected

#using Durbin-Watson test to detect serially correlated errors related to the multiple-regression problem
durbinWatsonTest(fit)

#the nonsignificant p value 0.282 suggests a lack of autocorrelation and coversely, an independence of errors
# the lag value =1 indicates that each observation is being compared with the one next to it in the dataset

#notice that durbinWatsonTest() function uses bootstrapping to derive p values. We will get a slightly different value each time we run unless we add the option simulate=F

#16. linearity: check nonlinearity in the relationship bewteen dependent variable and the independent variables by using component plus redisual plots(partial residual plots)
crPlots(fit)
#this component plus residual plots confirm that we have met the linearity assumption


#17. Homoscedasticity(nonconstant error variance)
ncvTest() #produce a score test of the hypothesis of constant error variance against the alternative that the error variance changes with the level of the fitted values

spreadLevelPlot() #generate a scatter plot of the absolute standardized residuals versus teh fitted values and superimposes a line of best fit

ncvTest(fit)
#the score test is nonsignificant(p=.19), suggest that we meet the constant variance assumption

spreadLevelPlot(fit)#the points from a random horizontal band around a horizontal line of best fit

#suggested power p, if p=.5, then using y^0.5 rather than Y in the regression. If the power suggested is 0, we would use a log transformation



#18. global validation of linear model assumption
#gvlma() function performs a global validation of linear model assumptions as well as separate evaluations of skewness, kurtosis and heteroscedasticity
install.packages("gvlma")
library("gvlma")
gvmodel <-gvlma(fit)
summary(gvmodel)

#if yhe decision line indicatedthat the assumptions were violated(like p<0.05), we should have to explore the data using the previous methods before


#19. Multicollinmearity
#a case: when conducting a study of grip strength, the independent variables include date of birth and age. We find that the grip strength on
#DOB and age and find a significant overall F test at p<.001. But when we look at the individual regression coefficient 
#for DOB and age, we find that they are both nonsignificant
#the problem is that DOB and age are perfectly correlated within rounding error. a regression coefficient means the 
#impact of one predictor variable on the response variable, holding all other predictor variables constant
#this amounts to looking at the relationship of grip strength and age, holding age constant


#multicollinearity can be detected using a statistic called variance inflation factor VIF


#the square root of the VIF indicates the degree to which the confidence interval for that variabl's regression parameter is expanded relative to a model with uncorrelated predictors

#vif^0.5 > 2 indicate a multicollinearity problem generally
library(car)
vif(fit)
sqrt(vif
(fit)) >2



#20. unusual observations

#outliers: unusually large positive or negative residuals
#positive means the model underestimate the response value
#negative means the model overestimate the response value

#In Q-Q plot, a rough rule of thumb is that standardized residuals that are larger than 2 or less than -2 are worth attention


outlierTest() #in car package, reports the Bonferroni adjusted p value for the largest absolute studentizd resudual

outlierTest(fit)
#this function tests the single largest residual for significance as an outlier 
#if it is not significant, there are no outliers in the dataset, if it is significant, you must delete it and rerun the test

#21. high leverage points

#observations with high leverage are identified through the hat statistic

#for a given dataset, the average hat value is p/n, where p is the number of parameters estimated in the model, n is the sample size

#an observation with a hat value greater than 2 or 3 times the average hat value should be examined

hat.plot <- function(fit){
  p<-length(coefficients(fit))
  n<-length(fitted(fit))
  plot(hatvalues(fit),main="Index Plot of Hat Values")
  abline(h=c(2,3)*p/n,col="red", lty=2)
  identify(1:n, hatvalues(fit), names(hatvalues(fit)))#locator function
}
hat.plot(fit)

#the horizontal lines are drawn at 2 and 3 times the average hat value



#22.influential observations have a disproportionate impact on the values of the model parameters

#two methods for identifying inflential observations: Cook's distance(or D statistic) and added variable plots

#Cook's D values greater than 4/(n-k-1) where n is the sample size and k is the number of predictor variables, indicates the influential observations

cutoff <-4/(nrow(states)-length(fit$coefficients)-2)
plot(fit, which=4, cook.levels=cutoff)
abline(h=cutoff, lty=2, col="red")

#the graph identifies Alaska, Hawii and Nevada as influential observations, deleting them will have a notable impact on the values of the intercvept and slopes in the regression model
#cook D plots can help identify influential observations but they dont provide information about how these observations affect the model
 
#for one response variable and k predictor variables, we would create k added-variable plots 
#for each predictor Xk, plot the residuals from regressing the response variable on the other k-1 predictors versus the residuals from regressing xk on the other k-1 predictors
avPlots(fit, ask=F, id.method="identify")


#23. we can combine the information from outlier, leverage and influence plots into one highly informative plot using 
influencePlot()

influencePlot(fit, id.method="identify", main="Influence Plot",
              sub="Circle size is proportional to Cook's distance")
#the resulting plot shows that Nevada and Rhode Island are outliers, new york and california and washington and hawaii have high leverage; nevada, alska and hawai are influential observations

#influence plot, horizontally above +2 or below -2 on the vertical axis are considered outliers
#above 0.2 or 0.3 on the horizontal axis have high leverage
#circle size is proportional to influence, those large circles may have disproportionate influence on the parameter estimates of the model


#24. corrective measures to dealing with the violations of regression assumptions

#deleting observations
#transforming variables
#adding or deleting variables
#using another regression approach

#25. deleting observations

#be careful about uncovering why an observation differs from the rest
#it can contribute greate insight to the topic at hand and to other topics we might no have thought of

#26. transforming variables when cannot meet the normality

#when models don't meet the normality, linearity or homoscedasticity assumptions, transforming one or more variables can often improve or correct the situation

#transorming typically involve replacing a variable Y with Y^lambda. 
#common value of lamda and their interpretations are listed below, if Y is a proportion, a logit transformation [ln (Y/1-Y)] is often used

lambda           -2       -1  -.5         0        .5      1    2
Tranformation    1/Y^2   1/Y   1/Y^.5    log(Y)   Y^.5   None   Y^2

#when the model violates the normality assumption, we typically attempt a transofrmation of the response variable
#we can use the powerTransform() function in the car package 
#to generate a maximum-likelihood estimation of the power lamda most likely to normalize the variable x^lamda

library(car)
summary(powerTransform(states$Murder))

#we can normalize the variable Murder by replacing it with Murder^.6, because .6 is close to .5, we could try a square-root transormation to improve the model's fit to normality


#27. transofrming variables when cannot meet the linearity
boxTidwell()# generate maximum-likelihood estimates of predictor powers that can improve linearity

boxTidwell(Murder~Population+Illiteracy, data=states)

#the result suggest typing the transformations Population^.86 and Population^1.36 to achieve greater linearity
#but the score tests for population(p=.75) and illiteracy(p=.54) suggest that neither variable needs to be transformed

#28. transforming variables when cannot meet the heteroscedasticity
spreadLevelPlot() #offers a power transformation for improving homoscedasticity

#29. adding or deleting variables
#if the goal is only to make predictions then multicollinearity is not a problem
#if wanna make interpretaions about individual predictor variables, we must deal with the problem of multicollinearity
#the most common approach is to delete one of the variables involved in the multicollinearity(that is, one of the variables with a vif^.5>2)
#another solution is to use ridege regression, a variant of multiple regression designed to deal with multicollinearity situations

#30. trying a different approach

#if there are outliers and/or influential observations, we can fit a robust regression model rather than an OLS regression 

#if w eviolate the assumption of independence of errors, we can fit a model that specifically takes the error structure into account, such as time-series models to fit a wide range of models in situations where the assumptions of OLS regression dont hold

#31. selecting the "best" regression model

#the selection of a final regression model always involves a compromise between predictive accuracy(a model that fits the data asa well as possible) 
#and parisimony(a simple and replicable model)

#32. comparing models
anova() #compared the fit of two nested models, nested means that one whose terms are completely included in the other model


states <-as.data.frame(state.x77[,c("Murder", "Population",
                                    "Illiteracy","Income","Frost")])
fit1 <- lm(Murder ~ Population +Illiteracy +Income +Frost, data=states)
fit2 <- lm(Murder ~ Population +Illiteracy, data=states)
anova(fit2, fit1)
#here model1 is nested within model2
#because the test is nonsignificant(p=.994) we conclude that they dont add to the linear prediction and we are justified in dropping them from the model

#Akaike Information Criterion AIC provides another method for comparing models, the index takes into account a model's statistical fit and the number of parameters needed to achive  this fit
#models with Smaller AIC values indicating adequate fit with fewer parameters are prefererd

AIC(fit1, fit2)

#AIC values suggest that the model without Income and Frost is the better model
#note that althoug the ANOVA approach requires nested models, the AIC approach does not


#33. variable selection

#two popular appriaches to selecting a final set of predictor variables from a larger pool of candidate variables are stepwise methods and all-subsets regression

#stepwise regression
#variables are added to or deleted from a model one at a time until some stopping criterion is reached
#in forward stepwise regression, we add predictor variables to the model and stpping when the addition would no longer improve the model
#in the backward stepwise regression, we start with a model that includes all predictor variable and then delete them until removing would degrad the quality of the moddel

#in stepwise steowise regression(usually called stepwise to avoid souding silly), combine the forward and backward stepwise approachs,the variables are entered one at a time, but at each step, the variables in the model are reevaluatedm and those that dont contribute to the model are deleted 
#the predictor variabe may be added to and deleted from a model several times before a final solution is reached

stepAIC()  #in MASS package performs stepwise model selection(forward, backward or stepwise) using an exact AIC cirterion
install.packages("MAss")
library(MASS)
states <-as.data.frame(state.x77[,c("Murder", "Population",
                                    "Illiteracy","Income","Frost")])
fit <- lm(Murder ~Population +Illiteracy +Income +Frost, data=states)
stepAIC(fit, direction="backward")
#start with all four predictors in the model, for each stepp the AIC column provides the model AIC resulting from the deltion of the variable listed in that row
#the AIC  value for <none> is the model AIC if no variables are removed. In the first step Frost is removed, decreasing the AIC from 97.75 to 95.75
#In the second step, income is removed, decreasing the AIC to 93.76. deleting anymore variabels would increase the AIC, so the process stops

#stepwise regression is vontroversial, although it may find a good model, there is no guarantee that it will find the best model
#that is because not every possible model is evaluated. an approach that attempts to overcome this limitation is all subsets regression

#All aubsets regression
#evfery possible model is inspected
#we can choose to have all possible results displayed or ask for the nbest models of each subset size(one predictorm two predictor and so on)

regsubsets() # from leaps package. we can choose the R-squared, adjusted R-squared or Mallows Cp statistic as the criterion for reporting best models
#R-squared is the amount of variance accounted for in the response variable by the predictors variables
#Adjusted R-square is more that can take into account the number of parameters in the model
#R sqaure alwasy increase with the addition of predictors
#when the number or predictors is large compared to the sample size, this can lead to significant overfitting
#the adjusted r squared is an attempt to provide a more honest estimate of the population R-squared-- one that is less likely to take advantage of chance variation in the data
#the mallows Cp statistic is also used as a stopping rule in stepwise regression, it has been widely suggested that a good mdoel is one in which the Cp statistic is close to the number of model parameters(including the inercept)

install.packages("leaps")
library(leaps)
leaps <- regsubsets(Murder ~ Population +Illiteracy + Income + Frost, data=states, nbest=4)
plot(leaps, scale="adjr2")
library(car)
subsets(leaps, statistic="cp", main="Cp Plot For All Subsets Regression")
abline(1,1,lty=2, col="red")

#the plot can be confusing to read. looking at the first row(starting at the bottom) we can se that a model with the intercept and income has an adjusted r square of .33
#a model with the intercept and population has an adjusted r square of .1
#jumping to the 12th row, a model with the intercept, population, illteracy, and income has anadjusted r squared of .54
#one with intercept, popultion and illteracy alone has an adjusted r square of .55
#here we see that a model with fewer predictors has a larger adjusted r square(something tha cant happen with an unadjusted  r square)
# so that graph suggests that the two predictor model is the best: Population and illiteracy

#in the leaps model, it shows the best four models for each subset size based on the Mallows Cp statistic
#better models will fall close to a line with intercept 1 and slope 1
#the plot suggest that we consider a two predictor model with population and illiteracyl a three predictor model with population, illitercay and frost or population, illiteracym and income; or a four predictor model iwth population, illiteracy, income and frost. We can reject the other possibe models

#mostly all subsets regression is preferable to stepwise regression because more models are considered, but when the number of predictors is large, the procedure can require significant computing time
#in general automated variable selection methods should be seen as an aid rather than a directing force in model selection
#*** a well-fitting model that doesnt make sense doesnt help you. It is your knowledge of the subject matter that should guide you


#34. taking the analysis further

#cross validation
#the description is the primary goal, the selection and interpretation of a regression model signals the end of our labor
#but when the goal is prediction, we should justifiably aks "How well will the equation perform in the real world"
#cross-validation is a useful method for evaulating the generalizabiilty of a regression equation

#in cross validation, a portion of the data is as the training sample and a portion is selected as the hold-out sample
#the regression equation is developed on the training sample and them applied to the hold-out sample
#because the hold-out sample wasnt involved in the selection of the model parameters, the performance on this sample is a more accurate estmate of the operating characteristic of the model with new data

#in K-fold vross validation, the sample is divided into k subsamples, each of the k subsamples serves as a hold-out group
#the combined observations from theremaining k-1 subsamples serve as the training group
#the performance for the k prediction equations applierd to the k hold out sampes is recorded and then averaged(when k equals n, the total number of observations, this approach is called jackknifing)
#we can perform k-fold cross-validation using the crossval() function in the bootstrap package

install.packages("bootstrap")
library(bootstrap)
shrinkage <- function(fit, k=10){
  require(bootstrap)
  theta.fit <-function(x,y){lsfit(x,y)}
  theta.predict <-function(fit, x) {cbind(1,x)%*%fit$coef}
  
  x<-fit$model[,2:ncol(fit$model)]
  y <-fit$model[,1]
  results<-crossval(x,y, theta.fit, theta.predict, ngroup=k)
  r2 <-cor(y, fit$fitted.values)^2
  r2cv <- cor(y, results$cv.fit)^2
  cat("Original R-Square =", r2, "\n")
  cat(k, "Fold Cross-Validated R_Square =", r2cv, "\n")
  cat("Change =", r2-r2cv, "\n")
}
#using this, we define the function that create a matrix of predictor and predicted values, get the raw R-square and get the cross validated R squared
#the shrinkage() funtion is then used to perform a 10-fold cross-validation with the states data, using a model with all four predictor variables

states <- as.data.frame(state.x77[, c("Murder","Population", "Illiteracy", "Income", "Frost")])
fit <- lm(Murder ~ Population +Income+ Illiteracy +Frost, data=states)
shrinkage(fit)
# the r square based on the sample (.567) is overly optimistic
#the better estimate of the amount of variance in murder rates that this model will account for with new data is the cross validated R -square (.448)
#the observations are assigned to the k groups randomly, so the result will be slightly difference each time executed

#we can use cross-validation in variable selection by choosing a model that demonstrates better generalizability
#for example, a model with two predictors shows less R-square shrinkage than the full model

fit2 <- lm(Murder ~ Population +Illiteracy, data=states)
shrinkage(fit2)

#this may make the two predictor model a more attractive alternative

#a regression equation that is based on a larger training sample and one that is mopre representative of the population of intertest will cross validate better
#get less R squared shrinkage and make more accurate predictions


#35. relative importance
#rank-order leadership practices by their relative importance of predictor variables
#if variables were uncorrelated, this would be a simple task
#in most case, the predictors are correlated with each other, and this complicates the task significantly


#the simplest to relative importance of predictors is to compare standardized regression coefficient which describe the expected change in the response variable for a standard deviation changing in a predictor variable and holding the other predictor variables constant

#we can use scale() function first to standardize each of the variables in the dataset to a mean of 0 and a sd of 1
zstates <-as.data.frame(scale(states))
zfit <-lm(Murder~Population +Income+ Illiteracy + Frost, data= zstates)
coef(zfit)

#we can see that a one standard-deviation increase in illiteracy rate yields a .68 sd increase in murder rate
#illiteracy is the most important predictor and frost is the least



#there have been many other attempts at quantifying relative importance. 
#relatvie importance can be though of as the contribution each predictor makes to R square both alone and in combbination with other predictors

#several possible approaches to relative importance are captured in the relaimpo package written by ULrike Gromping(http://mng.bz/KDYF)

#relative weight
#a new method called relative weights shows significant promise
#it approximates the average increase in R square obtanined by adding a predictor variable across all possible submodels

relweights <- function(fit,...){
  R <-cor(fit$model)
  nvar <- ncol(R)
  rxx <-R[2:nvar, 2:nvar]
  rxy <-R[2:nvar,1]
  svd <- eigen(rxx)
  evec <-svd$vectors
  ev<- svd$values
  delta <-diag(sqrt(ev))
  lambda <- evec %*% delta %*% t(evec)
  lambdasq <-(lambda ^ 2)
  beta <- solve(lambda) %*% rxy
  rsquare<- colSums(beta ^2)
  rawwgt <- lambdasq %*% beta^2
  import <-(rawwgt/rsquare)*100
  import <-as.data.frame(import)
  row.names(import)<-names(fit$model[2:nvar])
  names(import)<-"Weights"
  import <-import[order(import), 1 , drop=F]
  dotchart(import$Weights, labels=row.names(import),
  xlab="% of R Square", pch=19,
  main="Relative Importance of Predictor Variables",
  sub=paste("Total R Square=", round(rsquare, digits=3)),
  ...)
return(import)
  }
#the code is adaped from an SPSS program generously provided by Dr.Jognson. For an explanation of how the relative weights are derived

#the relweights() function is applied to the states data with murder rate predicted by the population, illiteract, income and tempreture

fit <-lm(Murder~Population +Illiteracy+Income +Frost, data=states)
relweights(fit, col="blue")

#see that the total amount of variance accounted for by the model(R square=.567) has been divided among the predictor variables
#illiteracy accounts for 59% of the R square, frost 20.79% and so forth based on the method ofrelative weights, illiteracy has the greates relative importance, followed by Frost, population and income in that order

#relatvie importance measure have wide applicability, they come much closer to our intuitive conception of relative importance than standardized regression coefficients do
