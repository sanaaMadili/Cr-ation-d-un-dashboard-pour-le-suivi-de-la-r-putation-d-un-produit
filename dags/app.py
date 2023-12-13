import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the DataFrame
df = pd.read_csv('df.csv')  # Replace 'your_file.csv' with your actual file path

# Set page title
st.title('Interactive Dashboard')

# Visualization 1: Number of reviews per month (Bar Chart)
st.subheader('Number of Reviews per Month')
reviews_per_month = df.groupby('Month').size().reset_index()  # Reset index
st.bar_chart(reviews_per_month.set_index('Month'))

# Visualization 2: Number of positive reviews per month (Bar Chart)
st.subheader('Number of Positive Reviews per Month')
positive_reviews = df[df['Positive'] == 1].groupby('Month').size().reset_index()  # Reset index

fig_positive, ax_positive = plt.subplots()
ax_positive.bar(positive_reviews['Month'], positive_reviews[0], color='green')
ax_positive.set_xlabel('Month')
ax_positive.set_ylabel('Number of Positive Reviews')
st.pyplot(fig_positive)

# Visualization 3: Number of negative reviews per month (Bar Chart)
st.subheader('Number of Negative Reviews per Month')
negative_reviews = df[df['Positive'] == 0].groupby('Month').size().reset_index()  # Reset index

fig_negative, ax_negative = plt.subplots()
ax_negative.bar(negative_reviews['Month'], negative_reviews[0], color='red')
ax_negative.set_xlabel('Month')
ax_negative.set_ylabel('Number of Negative Reviews')
st.pyplot(fig_negative)

# Visualization 4: Number of users per positive review (Histogram)
st.subheader('Number of Users per Positive Review')
users_per_positive_review = df[df['Positive'] == 1].groupby('comments.comment')['comments.comment'].count()

fig_users_positive, ax_users_positive = plt.subplots()
ax_users_positive.hist(users_per_positive_review, bins=20, color='green')
ax_users_positive.set_title('Users per Positive Review')
ax_users_positive.set_xlabel('Number of Reviews per User')
ax_users_positive.set_ylabel('Number of Users')
st.pyplot(fig_users_positive)

# Visualization 5: Number of users per negative review (Histogram)
st.subheader('Number of Users per Negative Review')
users_per_negative_review = df[df['Positive'] == 0].groupby('comments.comment')['comments.comment'].count()

fig_users_negative, ax_users_negative = plt.subplots()
ax_users_negative.hist(users_per_negative_review, bins=20, color='red')
ax_users_negative.set_title('Users per Negative Review')
ax_users_negative.set_xlabel('Number of Reviews per User')
ax_users_negative.set_ylabel('Number of Users')
st.pyplot(fig_users_negative)
