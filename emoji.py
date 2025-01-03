import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the pre-trained emotion detection model
@st.cache_resource  # Cache the model to avoid reloading it each time the app runs
def load_model():
    try:
        model = joblib.load(open("model/text_emotion.pkl", "rb"))
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None

pipe_lr = load_model()

# Dictionary of emotions and their corresponding emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Function to predict emotions based on the pre-trained model
def predict_emotions(docx):
    try:
        results = pipe_lr.predict([docx])
        return results[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Function to get prediction probabilities
def get_prediction_proba(docx):
    try:
        results = pipe_lr.predict_proba([docx])
        return results
    except Exception as e:
        st.error(f"Probability prediction failed: {e}")
        return None

# Main function to run the Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")
    
    if pipe_lr is None:
        st.stop()  # Stop execution if the model fails to load

    # Input form for user to type text
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here", help="Enter the text you want to analyze for emotion")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        if not raw_text:
            st.error("Please enter some text for emotion detection.")
        else:
            # Split the screen into two columns
            col1, col2 = st.columns(2)

            # Predict emotion and probability
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            if prediction and probability is not None:
                with col1:
                    st.success("Original Text")
                    st.write(raw_text)

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict.get(prediction, "❓")  # Fallback emoji if prediction is unknown
                    st.write(f"{prediction}: {emoji_icon}")
                    st.write(f"Confidence: {np.max(probability):.2f}")

                with col2:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]

                    # Create a bar chart using Altair
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    )
                    st.altair_chart(fig, use_container_width=True)
            else:
                st.error("An error occurred during the prediction process.")

# Run the app
if __name__ == '__main__':
    main()
