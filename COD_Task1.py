from flask import Flask, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import io
import base64
import numpy as np

app = Flask(__name__)

# Load the datasets
train_data = pd.read_csv('D:/food_demand/train.csv')
fulfilment_center_data = pd.read_csv('D:/food_demand/fulfilment_center_info.csv')
meal_data = pd.read_csv('D:/food_demand/meal_info.csv')

def plot_to_img(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    # Generate visualizations
    # Histogram of num_orders
    plt.figure()
    plt.hist(train_data['num_orders'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Orders')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Orders')
    hist_img = plot_to_img(plt)

    # Bar plot of center_type
    plt.figure()
    sns.countplot(x='center_type', data=fulfilment_center_data, palette='viridis')
    plt.title('Distribution of Center Types')
    bar_center_type_img = plot_to_img(plt)

    # Pie chart of cuisine
    plt.figure(figsize=(8, 8))
    meal_data['cuisine'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('viridis'))
    plt.title('Distribution of Cuisines')
    pie_cuisine_img = plot_to_img(plt)

    # Line plot of num_orders over time
    plt.figure()
    plt.plot(train_data.groupby('week')['num_orders'].sum(), color='purple')
    plt.xlabel('Week')
    plt.ylabel('Total Orders')
    plt.title('Total Orders Over Time')
    line_orders_img = plot_to_img(plt)

    # Seasonal decomposition plot
    plt.figure()
    result = seasonal_decompose(train_data.groupby('week')['num_orders'].sum(), model='additive', period=52)
    result.plot()
    seasonal_decompose_img = plot_to_img(plt)

    # Heatmap of correlation
    plt.figure()
    corr = train_data[['checkout_price', 'base_price', 'num_orders']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    heatmap_corr_img = plot_to_img(plt)

    # Forecasting using Holt-Winters
    train, test = train_test_split(train_data, test_size=0.2, shuffle=False)
    model = ExponentialSmoothing(train['num_orders'], trend='add', seasonal='add', seasonal_periods=12)
    hw_model = model.fit()
    forecast = hw_model.forecast(len(test))
    rmse = np.sqrt(mean_squared_error(test['num_orders'], forecast))

    plt.figure(figsize=(12, 6))
    plt.plot(train['week'], train['num_orders'], label='Training Data', color='blue')
    plt.plot(test['week'], test['num_orders'], label='Actual Demand', color='green')
    plt.plot(test['week'], forecast, label='Predicted Demand', color='red')
    plt.legend()
    plt.title('Demand Forecasting using Holt-Winters Exponential Smoothing')
    plt.xlabel('Week')
    plt.ylabel('Demand')
    forecast_img = plot_to_img(plt)

    # Top 3 cuisines
    top_3_cuisines = meal_data['cuisine'].value_counts().head(3).reset_index()
    top_3_cuisines.columns = ['Cuisine', 'Count']

    # Top 3 weeks with high orders
    top_3_weeks = train_data.groupby('week')['num_orders'].sum().sort_values(ascending=False).head(3).reset_index()
    top_3_weeks.columns = ['Week', 'Total Orders']

    # Predicted orders by cuisine
    predicted_orders = pd.DataFrame({'week': test['week'], 'num_orders': forecast})
    predicted_data = pd.merge(test[['meal_id']], predicted_orders, left_index=True, right_index=True)
    predicted_data = pd.merge(predicted_data, meal_data[['meal_id', 'cuisine']], on='meal_id')
    top_3_predicted_cuisines = predicted_data.groupby('cuisine')['num_orders'].sum().sort_values(ascending=False).head(3).reset_index()
    top_3_predicted_cuisines['num_orders'] = top_3_predicted_cuisines['num_orders'].astype(int)
    top_3_predicted_cuisines.columns = ['Cuisine', 'Predicted Orders']

    # Data insights
    data_insights = {
        'Total Orders': train_data['num_orders'].sum(),
        'Total Weeks': train_data['week'].nunique(),
        'Total Meals': meal_data['meal_id'].nunique(),
        'Total Centers': fulfilment_center_data['center_id'].nunique(),
    }
    data_insights_df = pd.DataFrame(list(data_insights.items()), columns=['Metric', 'Value'])

    return render_template('index.html', hist_img=hist_img, bar_center_type_img=bar_center_type_img,
                           pie_cuisine_img=pie_cuisine_img, line_orders_img=line_orders_img,
                           seasonal_decompose_img=seasonal_decompose_img, heatmap_corr_img=heatmap_corr_img,
                           forecast_img=forecast_img, rmse=rmse,
                           top_3_cuisines=top_3_cuisines.to_html(classes='table table-striped', index=False),
                           top_3_weeks=top_3_weeks.to_html(classes='table table-striped', index=False),
                           top_3_predicted_cuisines=top_3_predicted_cuisines.to_html(classes='table table-striped', index=False),
                           data_insights=data_insights_df.to_html(classes='table table-striped', index=False))

@app.route('/eda')
def eda():
    # EDA visualizations
    # Scatter plot of checkout_price vs. num_orders
    plt.figure()
    plt.scatter(train_data['checkout_price'], train_data['num_orders'], c='blue', alpha=0.5)
    plt.xlabel('Checkout Price')
    plt.ylabel('Number of Orders')
    plt.title('Checkout Price vs. Number of Orders')
    scatter_checkout_num_orders_img = plot_to_img(plt)

    # Bar plot of num_orders by category
    plt.figure(figsize=(10, 6))
    sns.barplot(x='category', y='num_orders', data=pd.merge(train_data, meal_data, on='meal_id'), palette='viridis')
    plt.title('Demand by Category')
    bar_category_img = plot_to_img(plt)

    # Scatter plot of base_price vs. checkout_price
    plt.figure()
    plt.scatter(train_data['base_price'], train_data['checkout_price'], c='green', alpha=0.5)
    plt.xlabel('Base Price')
    plt.ylabel('Checkout Price')
    plt.title('Base Price vs. Checkout Price')
    scatter_base_checkout_price_img = plot_to_img(plt)

    # Bar plot of num_orders by emailer_for_promotion and homepage_featured
    plt.figure(figsize=(10, 6))
    sns.barplot(x='emailer_for_promotion', y='num_orders', hue='homepage_featured', data=train_data, palette='viridis')
    plt.title('Demand based on Promotional Activities')
    bar_promo_img = plot_to_img(plt)

    # Additional EDA: Distribution of Checkout Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['checkout_price'], bins=30, color='blue', kde=True)
    plt.title('Distribution of Checkout Prices')
    dist_checkout_price_img = plot_to_img(plt)

    # Additional EDA: Box Plot of Number of Orders by Cuisine
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='cuisine', y='num_orders', data=pd.merge(train_data, meal_data, on='meal_id'), palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Number of Orders by Cuisine')
    box_orders_cuisine_img = plot_to_img(plt)

    return render_template('eda.html', scatter_checkout_num_orders_img=scatter_checkout_num_orders_img,
                           bar_category_img=bar_category_img, scatter_base_checkout_price_img=scatter_base_checkout_price_img,
                           bar_promo_img=bar_promo_img, dist_checkout_price_img=dist_checkout_price_img,
                           box_orders_cuisine_img=box_orders_cuisine_img)

if __name__ == '__main__':
    app.run(debug=True)
