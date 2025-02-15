# Video Game Sales Prediction

## Project Overview

This project aims to estimate the total sales of a video game based on several descriptive variables, including:

- Country of origin
- Development studio
- Publisher
- Game description
- Platform released on
- Game genre

The dataset was sourced from **VGChartz.com** (2016) and enriched with missing values and additional features through **web scraping** from various websites, including **Metacritic.com**, where we gathered user reviews, critic scores, and user comments.

## Objective

The main goal of this project was to develop a predictive model that estimates the sales of video games using the descriptive data we collected. 
By analyzing the relationships between game attributes and sales performance, we aimed to create a robust tool that can provide game publishers 
and developers with insights into sales forecasting.

## Tools and Technologies

- **Python** for data processing and model development
- **Streamlit** for creating the interactive dashboard
- **BeautifulSoup**, **Selenium** for web scraping
- **Pandas**, **Numpy** for data manipulation
- **Matplotlib**, **Seaborn**, **Plotly** for data visualization
- **Scikit-learn** for building the predictive model

## Project Steps

1. **Data Collection**:  
   We scraped data from **VGChartz.com** (2016) and **Metacritic.com** to gather missing data and enrich the dataset with user and critic scores, comments, and additional features.

2. **Data Preprocessing**:  
   The collected data was cleaned, and missing values were imputed using the data from web scraping. 
   Features such as user ratings, game descriptions, game rate, and developers were integrated into the dataset.

3. **Model Development**:  
   We built a predictive model using **machine learning algorithms** to estimate video game sales based on the 'raw' / enriched dataset. 
   The model was trained to identify key factors influencing sales performance.

4. **Dashboard Creation**:  
   Using **Streamlit**, we created an interactive dashboard to present our insights findings, tested models and data enhancing explanations. (Text in french)

## Results

The final model successfully predicts video game sales with a 62% level of accuracy. The data collected for dataset enhancement helps us to upgrade our model precision by 20%.

### Key Findings

- Games with **higher critic scores** and positive user reviews tend to have better sales.
- The **platform** (e.g., PC, PlayStation) significantly affects sales predictions, with certain platforms showing higher sales potentials.
- **Game genre** was another important feature influencing sales, with action and adventure genres generally performing better.

## Usage

To run the Streamlit app and explore the results interactively, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Jaysidi/DataScientestProject.git
   ```


2. Install the required dependencies

	```bash
    pip install -r requirements.txt
	```

3. Run the Streamlit app:
    
	```bash
	streamlit run Video_Game_Project.py
	```

4. Open the app in your browser and start exploring!

Possible Improvements
* Adding more features such as game release dates, marketing budget, and developer reputation for more accurate predictions.


Contributors
* François Dumont
* Olivier Steinbauer
* Thomas Bouffay