# Python
import streamlit as st
import pandas as pd
import random
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AzureOpenAI
def generate_response(input):
    client = AzureOpenAI(api_version='2024-06-01', azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai', api_key='4ceeaa9071277c5b')
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': input}],
        temperature=0.7,
        max_tokens=2560,
        top_p=0.6,
        frequency_penalty=0.7
    )
    return res.choices[0].message.content

# Generate dummy feedback data
def generate_feedback_data():
    feedback_list = [
        "The application process was seamless and easy.",
        "The system was slow and caused frustration.",
        "The customer support was very helpful and resolved my issue quickly.",
        "The UI/UX design could be more intuitive.",
        "I had trouble uploading documents; please improve the file upload system.",
        "Excellent overall experience. Thank you!",
        "It was difficult to navigate through the website.",
        "More detailed instructions would be helpful for first-time users.",
        "The chatbot provided incorrect information.",
        "Great platform with minimal technical issues."
    ]

    data = {
        "Candidate_ID": [f"C{random.randint(1, 100)}" for _ in range(50)],
        "Feedback_Text": [random.choice(feedback_list) for _ in range(50)],
        "Feedback_Date": pd.date_range(start="2024-01-01", periods=50).to_list(),
    }
    feedback_df = pd.DataFrame(data)
    return feedback_df


# Perform sentiment analysis
def analyze_sentiment(feedback_text):
    sentiment = TextBlob(feedback_text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"


# Generate actionable insights
def generate_insights(df):
    insights = {
        "Positive Feedback": len(df[df["Sentiment"] == "Positive"]),
        "Negative Feedback": len(df[df["Sentiment"] == "Negative"]),
        "Neutral Feedback": len(df[df["Sentiment"] == "Neutral"]),
        "Top Suggestions": df[df["Sentiment"] == "Negative"]["Feedback_Text"].tolist()[:3]
    }
    return insights


# Sentiment distribution plot
def plot_sentiment_distribution(df):
    sentiment_counts = df["Sentiment"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(plt)


# Main Streamlit application
def main():
    st.title("Feedback Analysis Application")

    # Generate and display feedback data
    df = generate_feedback_data()
    st.write("Sample Feedback Data:")
    st.dataframe(df.head(10))

    # Button to analyze sentiment
    if st.button("Analyze Feedback"):
        df["Sentiment"] = df["Feedback_Text"].apply(analyze_sentiment)
        st.write("Sentiment Analysis Completed")

        # Display actionable insights
        insights = generate_insights(df)
        st.write("Actionable Insights:")
        st.write(f"Positive Feedback: {insights['Positive Feedback']}")
        st.write(f"Negative Feedback: {insights['Negative Feedback']}")
        st.write(f"Neutral Feedback: {insights['Neutral Feedback']}")
        st.write(f"Top Suggestions: {insights['Top Suggestions']}")

        # Plot sentiment distribution
        plot_sentiment_distribution(df)
        st.write("GenAI suggestions")
        prompt = 'you are a feedback revieerw suggestor below is the feedback we got for this trainer Positive Feedback: 30 egative Feedback: 12 Neutral Feedback: 8 Top Suggestions: [It was difficult to navigate through the website.I had trouble uploading documents; please improve the file upload system.The system was slow and caused frustration. suggest some improvemnt plan for the trainer'
        st.write(generate_response(prompt))


if __name__ == "__main__":
    main()
