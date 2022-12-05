import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score

st.set_page_config(page_title='NoCodeAI App',layout='wide')

st.markdown('# NoCodeAI App')
st.markdown("")
st.markdown("---")

st.sidebar.header('Upload your CSV data')
data = st.sidebar.file_uploader('Upload your file', type=['CSV','xlsx'])
df = pd.read_csv(data)
# st.write(df.shape)
# st.markdown("---")

if st.checkbox('Show  Dataset'):
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

    if head == 'Head':
        st.table(df.head(num))
    else:
        st.table(df.tail(num))

    st.markdown("---")
    st.markdown("##### Number of rows and columns in the dataset")
    st.write("##")
    st.write("Rows , Column :",df.shape)

    st.markdown("---")

    df_null=(df.isnull().sum()/len(df))*100
    null_percent = df_null.sum()
    if null_percent==0:
        st.write("##")
        st.markdown('No missing values')

st.markdown("---")

st.markdown("##### Model Building")
option = st.selectbox('Choose target variable',(df.columns))

st.write('Target variable:', option)

Y=df[option]
X=df.drop(option,axis=1)

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)

lr_mod=LogisticRegression()
lr_mod.fit(xtrain,ytrain)
train_pred=lr_mod.predict(xtrain)
test_pred=lr_mod.predict(xtest)

accuracy = accuracy_score(ytest,test_pred)
f1score = f1_score(ytest,test_pred)

option = st.selectbox('evaluation metrics',(accuracy,f1score))

st.write('score:', option)






# if st.checkbox('Check for Missing values'):








