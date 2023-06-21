import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('potato_data.csv')

# Specify the variables for the path analysis
predictors = ['Land Size', 'Irrigation', 'Fertilizers']
outcome = 'Potato Yield'

# Fit the path model
model = sm.OLS(data[outcome], sm.add_constant(data[predictors])).fit()

# Print the path coefficients
print(model.params)

# Plot the path diagram
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, model.params['Land Size']], 'r-', linewidth=2, label='Land Size')
plt.plot([0, 1], [0, model.params['Irrigation']], 'g-', linewidth=2, label='Irrigation')
plt.plot([0, 1], [0, model.params['Fertilizers']], 'b-', linewidth=2, label='Fertilizers')
plt.xlabel('Predictors')
plt.ylabel('Outcome')
plt.title('Path Diagram')
plt.legend()
plt.show()


#const: The constant term in the path model. In this case, it is 9.130281. It represents the intercept or the expected value of the outcome variable when all predictor variables are set to zero.
#Land Size: The path coefficient for the predictor variable "Land Size". In this case, it is 2.000662. It represents the effect of a one-unit increase in "Land Size" on the outcome variable, holding other predictors constant.
#Irrigation: The path coefficient for the predictor variable "Irrigation". In this case, it is 2.999923. It represents the effect of a one-unit increase in "Irrigation" on the outcome variable, holding other predictors constant.
#Fertilizers: The path coefficient for the predictor variable "Fertilizers". In this case, it is 3.999051. It represents the effect of a one-unit increase in "Fertilizers" on the outcome variable, holding other predictors constant.
