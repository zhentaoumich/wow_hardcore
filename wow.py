import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde


#set title
st.title('Death Statistics in the WOW')
st.text('Webapp by Team 12 Hardcore')

#displaying the image on streamlit app

placeholder=st.image("https://boosting.pro/wp-content/uploads/2019/08/WoW-Classic-Vanilla-Map.jpg")













