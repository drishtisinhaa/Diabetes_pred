import numpy as np
import joblib # To access pickle file
import streamlit as st

# Load the trained model
model=joblib.load('model1.pkl')
scaler=joblib.load('scaler.pkl')

# streamlit app title
st.title('Machine learning model Deployment')
st.write('Enter your medical details to know about your diabetic status')

#define the input fields
st.sidebar.header('Your medical records')
preg=st.sidebar.number_input('preg',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
plas=st.sidebar.number_input('plas',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
pres=st.sidebar.number_input('pres',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
skin=st.sidebar.number_input('skin',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
test=st.sidebar.number_input('test',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
mass=st.sidebar.number_input('mass',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
pedi=st.sidebar.number_input('pedi',min_value=0.0,max_value=100.0,value=50.0,step=0.1)
age=st.sidebar.number_input('age',min_value=0.0,max_value=100.0,value=50.0,step=0.1)

input_data=np.array([[preg,plas,pres,skin,test,mass,pedi,age]])
scaled_input=scaler.transform(input_data)

if st.sidebar.button('Predict'):
    prediction=model.predict(scaled_input) # streamlit we dont have to write the backend code , when using aws we'll need it to write frontend and backend
    st.success(f'Prediction : {prediction[0]}') # you can also write a condition to show message 


# if we run this directly the  it will throw error we have to write the comman streamlit run stream.py