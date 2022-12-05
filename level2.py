import streamlit as st
import pandas as pd
st.set_page_config(page_title='NoCodeAI App',layout='wide')

st.markdown('# NoCodeAI App')
st.markdown("")
st.markdown("---")

# st.sidebar.header('Upload your CSV data')
# uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv","xlsx"])
# df=pd.read_csv(uploaded_file)

st.sidebar.header('Upload your CSV data')
data = st.sidebar.file_uploader('Upload Dataset in .CSV', type=['CSV'])
df = pd.read_csv(data)
st.markdown("---")

if st.checkbox('Show  Dataset'):
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

st.markdown("---")

