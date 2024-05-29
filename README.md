name : Devatha Jayanth Machineni
id:CT08PP682
domain: Data Science
duration: May 20 - June 20
mentor: Sravani Gouni
description: 
The provided code performs Exploratory Data Analysis (EDA) on a food demand forecasting dataset using Python, Flask, and various data visualization libraries. The analysis includes loading datasets, generating visualizations, and displaying results in a web application.

 Data Loading
Three CSV files are loaded:
- `train.csv`: Contains weekly food demand data.
- `fulfilment_center_info.csv`: Information about fulfillment centers.
- `meal_info.csv`: Details about different meals.

 Visualizations and Forecasting
The application creates several visualizations to explore and understand the data:
1. Histogram of Number of Orders: Displays the distribution of the number of orders.
2. Bar Plot of Center Types: Shows the distribution of different types of fulfillment centers.
3. Pie Chart of Cuisines: Illustrates the proportion of different cuisines.
4. Line Plot of Total Orders Over Time: Depicts the trend of total orders over the weeks.
5. Seasonal Decomposition Plot: Decomposes the time series data into trend, seasonal, and residual components.
6. Heatmap of Correlation Matrix: Shows the correlation between numerical features like checkout price, base price, and number of orders.
7. Forecasting using Holt-Winters Exponential Smoothing: Forecasts future demand and calculates the Root Mean Squared Error (RMSE) to evaluate the model.

 Data Insights
The code extracts and displays key insights:
- Top 3 Cuisines: The most ordered cuisines.
- Top 3 Weeks with High Orders: Weeks with the highest total orders.
- Top 3 Predicted Cuisines: Predicted demand for cuisines based on the forecast.

 Additional EDA
Additional exploratory visualizations include:
1. Scatter Plot of Checkout Price vs. Number of Orders.
2. Bar Plot of Demand by Category.
3. Scatter Plot of Base Price vs. Checkout Price.
4. Bar Plot of Demand based on Promotional Activities.
5. Distribution of Checkout Prices: Histogram showing the spread of checkout prices.
6. Box Plot of Number of Orders by Cuisine: Displays the distribution of orders across different cuisines.

 Web Application
The results are rendered in a Flask web application, with two main routes:
- The index page (`/`) displays primary visualizations and key insights.
- The EDA page (`/eda`) provides more detailed exploratory visualizations.

This comprehensive analysis and visualization approach helps understand the food demand patterns and aids in accurate forecasting.
