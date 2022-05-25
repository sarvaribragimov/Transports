from pyexpat import model
import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
plt = platform.system()
if plt =='linux':pathlib.windowsPath = pathlib.PosixPath
#title
st.title("Transportni  klassifikatsiya qiluvchi model (Boat,Airplane,Car)  Muallif: Ibragimov Sarvar ")
# Rasmni yuklash
file = st.file_uploader("Rasmni yuklash",type=['png','jpg','jpeg',"gif",'svg'])
if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('image_predict.pkl')
# Bashorat

    pred, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
