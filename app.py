import streamlit as st
import joblib
import pandas as pd
import numpy as np
model=joblib.load("Model.pkl")
st.set_page_config(page_title="Fitness Checker App",
                   page_icon="icon.jpg")
st.title("Fitness Checker App")
st.logo("icon.jpg",size="large")
st.text("""This app will tell you your fitness level given all the inputs.""")
st.divider()

st.header("Age")
age=st.number_input(label="How old are you?",min_value=18,step=1,label_visibility="hidden")

st.header("Weight (kg)")
weight=st.number_input(label="weight", label_visibility="hidden", min_value=0.0)

st.header("Height (cm)")
height=st.number_input(label="height", label_visibility="hidden", min_value=0.0)


st.header("Workout Duration (min)")
workout=st.number_input(label="workout", label_visibility="hidden", min_value=0.0)


st.header("Heart Rate (bpm)")
heart_rate=st.number_input(label="heart", label_visibility="hidden", min_value=0.0)


arr=np.array([age, weight, height, workout, heart_rate])
df_x=pd.DataFrame([arr], columns=["Age", "Weight (kg)", "Height (cm)", "Workout Duration (min)", "Heart Rate (bpm)"])
st.dataframe(df_x,)

if st.button(label="Submit",type="primary", icon="📤"):
    predictions=model.predict(df_x)
    if predictions==0:
        st.info("Your fitness level is Beginner!")
    elif predictions==1:
        st.info("Your fitness level is Intermediate!")
    else:
        st.info("Your fitness level is Advanced!")

sb=st.sidebar
with sb:
    st.title("""
Hi, I'm Nazzal.
An AI Engineer and the developer of
this app. If you like it, provide your 
feedback.
""")
