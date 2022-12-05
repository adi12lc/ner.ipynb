import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score

st.set_page_config(page_title='NoCodeAI App',layout='wide')

st.markdown('# NoCodeAI App')
st.markdown("")
st.markdown("---")

data1 = st.radio('View from top (head) or bottom (tail)', ('File', 'Database'))
if data1 == 'File':
    st.sidebar.header('Upload your file')
    dataset = st.sidebar.file_uploader('', type=['CSV', 'xlsx'])
    df = pd.read_csv(dataset)


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

if st.checkbox('Check for null values'):
    df_null = (df.isnull().sum() / len(df)) * 100
    null_percent = df_null.sum()
    if null_percent == 0:
        st.write("##")
        st.markdown('No missing values')
st.markdown("---")

st.markdown("##### Model Building")
option = st.sidebar.selectbox('Choose target variable', (df.columns), help="Select dependent variable")

# st.write('Target variable:', option)

Y = df[option]
X = df.drop(option, axis=1)

x_train, xtest, y_train, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

option = st.sidebar.selectbox('Choose evaluation metric',('Logistic Regression','RandomForestClassifier'))


if option == 'Logistic Regression':
    st.write("Algorithm:",option)
    mod = LogisticRegression()
    mod.fit(x_train, y_train)

elif option == 'RandomForestClassifier':
    st.write("Algorithm:", option)
    values = st.slider('n_estimators',0, 100, (0))
    mod = RandomForestClassifier(n_estimators=values)
    mod.fit(x_train, y_train)

train_pred = mod.predict(x_train)
test_pred = mod.predict(xtest)

st.markdown("---")

st.markdown("##### Model performance")

option = st.sidebar.selectbox('Choose evaluation metric',('accuracy','f1_score'))

if option == 'accuracy':
    accuracy_var = accuracy_score(ytest, test_pred)
    accuracy_var = round(accuracy_var,2)
    st.write(option, 'score:', accuracy_var)
else:
    f1score_var = f1_score(ytest, test_pred)
    f1score_var = round(f1score_var,2)
    st.write(option, 'score:', f1score_var)

st.markdown("---")









