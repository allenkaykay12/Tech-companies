import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

### The st.title() function sets the title of the Streamlit application
st.title("Tech Companies Salaries Data Analysis👩🏽‍💻🧑🏽‍💻:")

### menu bar

selected = option_menu(
    menu_title=None,
    options=["Overview", "Visualisation", "Regression", "Improvements"],
    icons=["menu-up", "pie-chart-fill", "graph-up-arrow", "recycle"],
    default_index=0,
    orientation="horizontal",

)

if selected == "Overview":
    st.markdown("##### 🎯Objectives")
    st.write(
        "The objective of this app is for current employees and job seekers to have a better understanding of the "
        "tech industry based on different job categories (software, science, management etc.) and how annual base "
        "salaries are different within different companies, across different departments to help them make better "
        "decisions. In short it is an interactive and user-friendly platform for job seekers, career counselors, "
        "and HR professionals to gain valuable insights into the job market.")
    st.markdown("##### 🌟Data Summary with Looker")
    ### Embedding Looker into Streamlit
    components.iframe("https://lookerstudio.google.com/embed/reporting/08b12163-5ae8-4d9b-acdb-e41514f998e5/page/paWrD",
                      height=520)
### The pd.read_csv() function loads a CSV file into a Pandas DataFrame called "df".
df = pd.read_csv('CleanedDataset.csv')

# data cleaning
# Calculate the mean of years of experience and employer_years of experience
mean_years_experience = df['total_experience_years'].mean()
mean_years_employer_experience = df['employer_experience_years'].mean()

# Replace missing values with the mean
df['total_experience_years'].fillna(mean_years_experience, inplace=True)
df['employer_experience_years'].fillna(mean_years_employer_experience, inplace=True)

df['total_experience_years'] = df['total_experience_years'].round(1)
df['employer_experience_years'] = df['employer_experience_years'].round(1)

# replacing the signing bonus and annual bonus missing values by 0
df['signing_bonus'] = df['signing_bonus'].fillna(0)
df['annual_bonus'] = df['annual_bonus'].fillna(0)
df['stock_value_bonus'] = df['stock_value_bonus'].fillna(0)

df.drop(['salary_id', 'job_title_rank', 'comments', 'location_state', 'location_country'], axis=1, inplace=True)
df = df.sort_values(by='annual_base_pay', ascending=False, ignore_index=True)

# First, identify the top 10 employers by the number of entries
top_employers = df['employer_name'].value_counts().head(10).index
#############################################################
df_calculator = df.copy()  # to be used later

# if visualisation change the page
if selected == "Visualisation":

    # Filter the dataset for these employers
    filtered_df = df[df['employer_name'].isin(top_employers)]
    filtered_df = filtered_df.sort_values(by=['annual_base_pay'], ascending=False)
    df_filtered = filtered_df.copy()
    tab1, tab2, tab3 = st.tabs(["Introduction", "Bar Chart", "Box Plot"])
    with tab1:
        st.subheader("Introduction")
        st.write(
            "In this section our goal is to visualize the relationship between the Annual Base Pay and various "
            "factors such as job categories, years of experience, or the companies. It will provide a more complete "
            "chart/plot with the top 10 companies in our dataset for a broader understanding of the trends. To "
            "achieve our objectives, box plots, and bar charts are more suitable. Therefore we will use them and each "
            "one will give access to the user to enter some information suiting their needs.")
    with tab2:
        st.subheader("Bar Chart Visualisations")
        # Dynamic Selections for Visualizations
        employer_options = st.multiselect('Select Employers', top_employers, ["google", "amazon"])
        # Apply dynamic selections for filtering only if selections are made
        # creating data frame
        df_filtered_employers = df_filtered[df_filtered['employer_name'].isin(employer_options)]
        # Check if there is any data to display
        if not df_filtered_employers.empty:
            # Plotting the bar chart
            sns.barplot(x='employer_name', y='annual_base_pay', data=df_filtered_employers, hue='employer_name')
            plt.xticks(rotation=45)
            plt.title('Annual Salary based on Company')
            plt.xlabel('Employer Name')
            plt.ylabel('Annual Salary (USD)')
            st.pyplot()
            st.write(
                "The bar chart above provides a visual comparison of the annual base pay between employees at the top "
                "10 tech companies in our dataset. By top 10, we looked at the companies with a higher number of "
                "employees represented. This can quickly indicate which company, on average, offers higher base pay, "
                "which is a critical factor for job seekers and those comparing employment offers. In addition, "
                "for current and prospective employees, understanding where their compensation falls in relation to "
                "the broader market can inform their career decisions. For instance, if someone at Amazon observes "
                "that Google employees, on average, have higher base pay, they might consider seeking opportunities "
                "at Google or use this information to negotiate a higher salary.")
        else:
            st.warning("No data available, please select variables.")

        ### Displaying a Bar Chart for the Annual Base Pay of different job categories

        st.subheader(' Barchart Representing the Annual Base Pay')
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df, y='annual_base_pay', x='job_title_category')
        plt.title('Barchart Representing the Annual Base Pay')
        plt.xlabel('Job Title Category')
        plt.ylabel('Annual Salary (USD)')
        st.pyplot()
        st.write(
            "In this chart, we are looking at the annual base pay depending on the job category. This is important "
            "since it allows individuals to make informed decisions about their career path and potential earnings. "
            "This would be helpful for people who are unsure about which field they want to pursue or for those who "
            "are considering switching careers. By comparing their current compensation to industry standards, "
            "users can better assess their value and make strategic decisions about their future career development. ")

        # First, identify the top 10 employers by the number of entries
        top_5_employers = filtered_df['employer_name'].value_counts().head(5).index
        filtered_df_top_5 = filtered_df[filtered_df['employer_name'].isin(top_5_employers)]
        # Comparison Features
        plt.figure(figsize=(10, 8))
        plt.title('Salary Comparison for Different Job in the Top 5 Tech Companies')
        sns.barplot(data=filtered_df_top_5, x='employer_name', y='annual_base_pay', hue='job_title_category')
        plt.legend(title='Job Category')
        plt.xlabel('Employer Name')
        plt.ylabel('Annual Salary (USD)')
        st.pyplot()
        st.write(
            "With this chart, our users can get insights into the average salaries across major companies for "
            "different job categories. This is essential for understanding compensation trends and making informed "
            "career decisions. Additionally, by breaking down salaries by job categories, the chart helps individuals "
            "compare not just the salaries at these companies but how the salary varies across different categories. "
            "Users can also use this chart to see which companies tend to pay more for certain roles. For example, "
            "a software engineer might see that Facebook pays more on average for their role compared to others. This "
            "information can be valuable for negotiating salaries or deciding which companies to target for job "
            "applications. Furthermore, this information would help the users in planning their career trajectory; "
            "understanding which job categories command higher salaries can inform decisions on specialisation and "
            "skill development.")

        # Identify top 10 employers
        top_employers = df['employer_name'].value_counts().head(10).index

        # create a new dataframe with the ten employers
        df_top_10_employers = df[df['employer_name'].isin(top_employers)]
        # sort the employers based on their experience
        df_sorted_by_experience = df_top_10_employers.sort_values(by='total_experience_years', ascending=False)
        top_10_most_experience_employees = df_sorted_by_experience['total_experience_years'].value_counts().head(
            5).index
        df_most_experienced = df_sorted_by_experience[
            df_sorted_by_experience['total_experience_years'].isin(top_10_most_experience_employees)]

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_most_experienced, x='total_experience_years', y='annual_base_pay', hue='employer_name')
        plt.title('Relationship Between Total Experience Years and Annual Base Pay Among Top 10 Employers')
        plt.xlabel('Total Experience Years')
        plt.ylabel('Annual Base Pay')
        plt.legend(title='Employer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Total Years of Experience')
        plt.ylabel('Annual Salary (USD)')
        plt.xticks(rotation=45)
        st.pyplot()
        st.write(
            "The chart above illustrates how annual base pay relates to the total years of experience within top tech companies, which can help tech professionals evaluate their career progress and set expectations. In addition, the chart allows for comparison not only across experience levels but also across companies. This can guide users in understanding which companies might offer better compensation growth with increased experience.For instance, if one company's pay does not scale as much with experience as another's, this could influence career decisions.")

    with tab3:
        st.subheader("Box Plot Visualisations")
        employer_options = st.multiselect('Select Employers', top_employers, ["google", "facebook"])
        job_category_options = st.multiselect('Select Job Categories', df['job_title_category'].unique(), ["Software"])
        # df_filtered_categories = df_filtered[df_filtered['job_title_category'].isin(job_category_options)]
        df_filtered_categories = df_filtered[(df_filtered['employer_name'].isin(employer_options)) & (
            df_filtered['job_title_category'].isin(job_category_options))]

        if not df_filtered_categories.empty:
            # Creating the box plot
            sns.boxplot(data=df_filtered_categories, x='job_title_category', y='annual_base_pay', hue='employer_name')
            plt.legend(title='Employer Name')
            plt.title('Box Plot based on Annual Salary and Job Category')
            plt.xlabel('Job Title Category')
            plt.ylabel('Annual Salary (USD)')
            plt.xticks(rotation=45)
            st.pyplot()
            st.write(
                "The box plot shows the range of salaries (from the lower to the upper quartile). This would give the user a sense of what salary they can expect for a specific role and company. Additionally, by comparing the pay distributions for the same job category across different companies, employees can assess how comparable the pay is. If one company's pay distribution for a role is significantly higher than another's, it may suggest issues with pay equity. Employees can use this data to negotiate salaries by demonstrating their knowledge of the typical pay at competing companies. It provides a benchmark to ensure they are within or above the typical pay range for their role. For job seekers, seeing which company tends to offer higher pay or a wider salary range for their role can influence their decision on where to apply or accept a job offer.")
        else:
            st.warning("No data available, please select variables.")

        ### A graph representing the relationship between the Job Category and total years of experience
        plt.figure(figsize=(15, 8))  # Set the figure size for better readability
        sns.boxplot(data=df, x='job_title_category', y='annual_base_pay')
        plt.title('Relationship Between Job Category and Annual Base Pay')
        plt.xlabel('Job Title Categories')
        plt.ylabel('Annual Base Pay(USD)')
        st.pyplot()

if selected == "Regression":
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf with NaN
    df.dropna(inplace=True)  # Drop any rows with NaN values
    newVariables = LabelEncoder()

    # Select columns of object data type for encoding
    object_columns = df.select_dtypes(include=['object']).columns

    # Additionally, ensure all columns intended for LabelEncoder are converted to string
    for column in object_columns:
        df[column] = df[column].astype(str)  # Convert to string to ensure uniformity
        df[column] = newVariables.fit_transform(df[column])

    # Convert all integer columns to float for consistency

    df = df.astype(float)

    plt.figure(figsize=(15, 8))
    fig1 = sns.displot(df['annual_base_pay'], kde=True)
    plt.title("The density Distribution Chart for the Annual Base Salary!", loc='left', pad=20)
    st.pyplot(fig1)

    prediction_type = st.sidebar.selectbox('Select Type of Prediction', ['Linear Regression'])

    list_variables = df.columns
    select_variable = st.sidebar.selectbox('🎯 Select Variable to Predict', list_variables)
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    new_df = df.drop(labels=select_variable, axis=1)
    list_var = new_df.columns

    output_multi = st.multiselect("Select Explanatory Variables", list_var,
                                  default=['employer_experience_years', 'total_experience_years'])

    new_df2 = new_df[output_multi]
    x = new_df2
    y = df[select_variable]

    scaler = StandardScaler()
    scaler.fit(df.drop(select_variable, axis=1))
    scaled_features = scaler.transform(df.drop(select_variable, axis=1))

    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    x = df_feat[output_multi]
    y = df[select_variable]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=train_size)

    if prediction_type == 'Linear Regression':
        lr_start_time = time.time()
        lr_model = LinearRegression()

        lr_model.fit(x_train, y_train)
        lr_pred = lr_model.predict(x_test)
        lr_end_time = time.time()
        lr_execution_time = lr_end_time - lr_start_time

        st.write("Execution time:", lr_execution_time, "seconds")

        st.subheader('📈 The Importance of the Explanatory Variables!')
        # Ensure you have the correct feature names from the original DataFrame
        feature_names = x.columns  # Directly use the column names

        # Ensure 'coefficients' is correctly retrieved from the model
        coefficients = lr_model.coef_

        # Making the coefficients positive to compare magnitude
        importance = np.abs(coefficients)

        # Plotting feature importance with feature names
        plt.figure(figsize=(10, 8))
        plt.barh(feature_names, importance)
        plt.xlabel('Absolute Coefficient Value')
        plt.title('Feature Importance (Linear Regression)')
        st.pyplot(plt)

        plt.clf()  # Clear the figure to avoid conflicts with further plots

        st.subheader('🎯 Results')
        # Note: You may want to adjust or remove this part since Linear Regression
        # predictions are continuous and may not directly apply to classification metrics.
        from sklearn.metrics import mean_squared_error, r2_score

        mse = mean_squared_error(y_test, lr_pred)
        r2 = r2_score(y_test, lr_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R^2 Score: {r2}")

    st.subheader('Interactive Salary Calculator')

    # Assuming 'job_title' and 'location_name' are columns in your dataframe
    job_titles = df_calculator['job_title_category'].unique()
    locations = df_calculator['location_name'].unique()

    user_job_title = st.selectbox('Select Your Job Title', options=job_titles)
    user_location = st.selectbox('Select Your Location', options=locations)
    user_experience = st.number_input('Enter Your Total Years of Experience', min_value=0.0,
                                      max_value=df_calculator['total_experience_years'].max(), value=0.0, step=0.5)

    # Filter the dataset based on the user's selected job title and location
    filtered_df = df_calculator[
        (df_calculator['job_title_category'] == user_job_title) & (df_calculator['location_name'] == user_location)]

    if not filtered_df.empty:
        # Calculate statistics, e.g., median salary for the selected job title and location
        median_salary = filtered_df['annual_base_pay'].median()

        # Display comparative information
        st.write(f"The median annual base pay for {user_job_title} in {user_location} is ${median_salary:,.2f}.")

        # Additional insights based on user's experience could be added here
        # For example, comparing the user's experience with the dataset distribution
    else:
        st.write("No data available for the selected job title and location.")

    # Visual representation
    if not filtered_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_df['annual_base_pay'], kde=True)
        plt.axvline(x=median_salary, color='r', linestyle='--', label='Median Salary')
        plt.xlabel('Annual Base Pay')
        plt.ylabel('Frequency')
        plt.title(f'Salary Distribution for {user_job_title} in {user_location}')
        plt.legend()
        st.pyplot()

if selected == "Improvements":
    st.write(
        f"The R square from our Linear Regression Model is low , indicating that the explanatory variables do not have a strong linear relationship with the annual base pay that we are trying to predict. To improve our prediction, we would need to consider adding more relevant variables.")
    st.caption(
        "For instance, adding factors such as education level, specific technical skills, or certifications could potentially improve our model's accuracy and capture more of the variation in annual base pay.")
    st.write(
        "Lack of Linear relationships between the current explanatory variables and annual base pay could reduce the effectiveness of our model in accurately predicting salaries.")
    st.caption(
        "For example, the effect of total years of experience on annual base salary could plateau after a certain point, making it necessary to include additional variables to account for this.")
    st.markdown(
        "**Therefore, we need to explore other models that incorporate nonlinear relationships and interactions between variables to better capture the complexity of factors influencing annual base pay.**")
    st.divider()
    st.write(
        f"The size of the dataset {df.shape} and the quality of the data collected are crucial for the accuracy of our model predictions.")
    st.write("If the dataset is too small or contains inaccuracies, the model's predictions will be lower.")
    st.markdown(
        "For our dataset, **we did some data manipulation due to missing values and outliers**, which could impact the accuracy of our model. With this data manipulation, we could have potentially introduced bias into our model, so it is important to carefully assess the impact of these changes on the accuracy of our predictions.")
    st.markdown(
        "**To address this, we could merge our current dataset with external datasets that may provide additional insights. By incorporating external datasets, we can enhance the quality and accuracy of our model predictions.**")
