import streamlit as st
import numpy as np
import pickle
import time
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
train=pd.read_csv(r"C:\Users\shash\Downloads\archive (16)\train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv(r"C:\Users\shash\Downloads\archive (16)\test_Y3wMUE5_7gLdaTN.csv")
train["Loan_Status"].replace({"Y":1,"N":0},inplace=True)
train.Dependents.replace({"3+":3},inplace=True)
test.Dependents.replace({"3+":3},inplace=True)
one=train[train["Loan_Status"]==1]
zero=train[train["Loan_Status"]==0]
one=one.sample(192)
train=pd.concat([one,zero])
train=train.sample(384)
x=train.drop(columns=["Loan_Status","Credit_History","Loan_ID"])
y=train["Loan_Status"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=42)
print(xtrain.head())
trf1=ColumnTransformer([
    ("impute_const",SimpleImputer(fill_value=0),[2]),
    ("impute_cat",SimpleImputer(strategy='most_frequent'),[0,1,3,4,9]),
    ("impute_num",SimpleImputer(),[5,6,7,8])
],remainder="passthrough")

trf2=ColumnTransformer([
    ("ohe",OneHotEncoder(sparse=False,handle_unknown="ignore"),[0,1,3,4,9])
],remainder="passthrough")

trf3=ColumnTransformer([
    ("scaling",MinMaxScaler(),slice(0,10))
])

trf4=DecisionTreeClassifier(criterion="entropy")
trf5=GaussianNB()
trf6=LogisticRegression()

pipe=make_pipeline(trf1,trf2,trf3,trf6)
pipe.fit(xtrain,ytrain)

ypred=pipe.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))

t=pd.DataFrame(np.array([["Male","Yes",1,"Graduate","No",3076,1500,10000,360,"Urban"]]),columns=xtrain.columns)

st.title("Github Architects")

st.sidebar.title("Team Github Architects")
choice=st.sidebar.radio("Select the option",["Project Overview","Model Summary","Predict Your Loan Status","About the Team"])
if choice=="Project Overview":
    st.header("Project Overview")
    st.write('''In our loan prediction endeavor, we delved into a comprehensive dataset 
encompassing various applicant attributes crucial for determining loan approval. The 
primary focus was on essential factors such as Credit_History, Self_Employed, and 
Loan Amount, all of which play a significant role in financial institutions' lending 
decisions. The dataset, while anonymized, provided rich insights into applicants' 
financial backgrounds, allowing us to derive valuable patterns essential for our 
predictive models.''')
    st.header("Data Exploration and Preprocessing")
    st.write('''Our initial exploration revealed intriguing correlations between features. Notably, 
Credit_History surfaced as the most influential factor affecting loan outcomes. We 
meticulously handled missing values and employed advanced imputation techniques 
to maintain data integrity. Categorical features like Dependents and 
Loan_Amount_Term underwent careful encoding to make them compatible with 
machine learning algorithms. Exploratory data analysis aided in understanding the 
distribution and characteristics of the data, providing crucial insights for feature 
engineering''')
    st.header("Machine Learning Algorithms:")
    st.write('''Our approach involved leveraging a diverse set of machine learning algorithms. 
Decision Tree Classifier proved invaluable in capturing complex nonlinear 
relationships within the data. Additionally, the implementation of ensemble 
methods Random Forest, allowed us to harness the 
collective predictive strength of multiple decision trees. These algorithms, trained 
on our carefully preprocessed dataset, underwent rigorous cross-validation to 
optimize hyperparameters, ensuring their robustness in real-world scenarios.
''')
    st.header("Conclusion")
    st.write('''In conclusion, our machine learning-based loan prediction project underscores the 
transformative potential of data-driven decision-making in the financial sector. By 
employing sophisticated techniques and powerful algorithms, we not only crafted 
accurate predictive models but also unearthed nuanced insights into the dynamics of 
loan approval. This knowledge equips institutions with the tools needed to make 
informed decisions, fostering a more inclusive and sustainable lending ecosystem.
''')
if choice=="Model Summary":
    st.header("Model Summary")
    st.subheader("Model PipeLine")
    img=Image.open(r"C:\Users\shash\Downloads\pipeline.png")
    st.image(img,width=750)
    st.subheader("Accuracy and Confusion Matrix")
    mg = Image.open(r"C:\Users\shash\Downloads\Confusion matrix.png")
    st.image(mg, width=750)
if choice=="Predict Your Loan Status":
    st.subheader("Check your Loan Status in 2 minutes")
    st.subheader("Fill in the Details...")
    gender=st.selectbox("Enter your Gender",["Male","Female"])
    one,two=st.columns(2)
    marriage=one.selectbox("Marital Status",["Yes","No"])
    education=two.selectbox("Education",["Graduate","Not Graduate"])
    dependents=st.number_input("Number of Dependents",min_value=0)
    three,four=st.columns(2)
    area=three.selectbox("Property Area",["Urban","Rural","Semiurban"])
    employed=four.selectbox("Self Employed",["Yes","No"])
    income=st.slider("Enter Applicant Annual Income",100,1000000)
    family=st.slider("Enter Family Gross Income",100,1000000)
    loan=st.slider("Enter Loan Amount Needed",50,1000000)
    time=st.slider("Loan Amount Term _in months_ ",1,240)
    t = pd.DataFrame(np.array([[gender,marriage,dependents,education,employed,income,family,loan,time,area]]),
                     columns=xtrain.columns)
    if st.button("Check your Status"):
        import time
        progress=st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)
        k=pipe.predict(t)[0]
        if k==1:
            st.success("Loan will be Approved")
        else:
            st.error("Loan Not Approved")
if choice=="About the Team":
    st.subheader("Team Details")
    st.markdown("### B1 Parnika Jain    21103012")
    st.markdown("### B1 Sritama Ray     21103014")
    st.markdown("### B1 Shashvat Ahuja  21103026")
    st.markdown("### B1 Harsh Dhariwal  21103267")
    st.subheader("\nSubmitted to Dr Alka Singhal.")