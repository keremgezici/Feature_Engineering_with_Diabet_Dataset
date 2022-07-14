import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("diabetes.csv")

##### Exploratory Data Analysis

df.head()
df.info()
df.shape
df.isnull().sum()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, head=3)

# Numerical And Categorical Variable Analysis
def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)

cat_cols
num_cols
cat_but_car

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col,True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col,)

#Target Variable Analysis

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#Correlation Analysis

df.corr()

# Korelasyon Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,check_outlier(df,col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    print(col,grab_outliers(df,col,True))


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
     replace_with_thresholds(df, col)

missing_zero=[col for col in df.columns if (df[col].min()==0 and col not in ["Pregnancies","Outcome"])]
missing_zero

for col in missing_zero:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.head()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_columns=missing_values_table(df,True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

df["SkinThickness"].fillna(df.groupby("BMI")["SkinThickness"].transform("mean"),inplace=True)

df["Insulin"].fillna(df.groupby("Glucose")["Insulin"].transform("mean"),inplace=True)

for col in missing_zero:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

# Feature Extraction
#Age
df.loc[(df["Age"]<=30),"NEW_AGE_CAT"]="Young"
df.loc[(df["Age"]>30) & (df["Age"]<=50),"NEW_AGE_CAT"]="Mature"
df.loc[(df["Age"]>50),"NEW_AGE_CAT"]="Old"

df.head()

#BMI
df.loc[(df["BMI"]<18),"NEW_BMI"]="Underweight"
df.loc[(df["BMI"]>=18)&(df["BMI"]<=25),"NEW_BMI"]="Idealweight"
df.loc[(df["BMI"]>=26)&(df["BMI"]<=30),"NEW_BMI"]="Overweight"
df.loc[(df["BMI"]>=31),"NEW_BMI"]="Obese"

df.head()

#Glucose
df.loc[(df["Glucose"]<70),"NEW_GLUCOSE"]="Hipoglisemi"
df.loc[(df["Glucose"]>=70)&(df["Glucose"]<=120),"NEW_GLUCOSE"]="Normal"
df.loc[(df["Glucose"]>=121),"NEW_GLUCOSE"]="Diabete"

df.head()

#Age&BMI
df.loc[(df["Age"]<=30) &(df["BMI"]<18), "NEW_AGE_BMI"]="Young_Underweight"
df.loc[(df["Age"]<=30) & (df["BMI"]>=18)&(df["BMI"]<=25), "NEW_AGE_BMI"]="Young_Idealweight"
df.loc[(df["Age"]<=30) & (df["BMI"]>=26)&(df["BMI"]<=30), "NEW_AGE_BMI"]="Young_Overweight"
df.loc[(df["Age"]<=30) &(df["BMI"]>=31), "NEW_AGE_BMI"]="Young_Obese"
df.loc[((df["Age"]>30) & (df["Age"]<=50)) &(df["BMI"]<18), "NEW_AGE_BMI"]="Mature_Underweight"
df.loc[((df["Age"]>30) & (df["Age"]<=50)) &(df["BMI"]>=18)&(df["BMI"]<=25), "NEW_AGE_BMI"]="Mature_Idealweight"
df.loc[((df["Age"]>30) & (df["Age"]<=50)) &(df["BMI"]>=26)&(df["BMI"]<=30), "NEW_AGE_BMI"]="Mature_Overweight"
df.loc[((df["Age"]>30) & (df["Age"]<=50)) &(df["BMI"]>=31), "NEW_AGE_BMI"]="Mature_Obese"
df.loc[(df["Age"]>50) &(df["BMI"]<18), "NEW_AGE_BMI"]="Old_Underweight"
df.loc[(df["Age"]>50) &(df["BMI"]>=18)&(df["BMI"]<=25), "NEW_AGE_BMI"]="Old_Idealweight"
df.loc[(df["Age"]>50) &(df["BMI"]>=26)&(df["BMI"]<=30), "NEW_AGE_BMI"]="Old_Overweight"
df.loc[(df["Age"]>50) &(df["BMI"]>=31), "NEW_AGE_BMI"]="Old_Obese"

df.head()

#Age&Glucose
df.loc[(df["Age"]<=30) &(df["Glucose"]<70), "NEW_AGE_GLUCOSE"]="Young_Hipoglisemi"
df.loc[(df["Age"]<=30) &((df["Glucose"]>=70)&(df["Glucose"]<=120)), "NEW_AGE_GLUCOSE"]="Young_Normal"
df.loc[(df["Age"]<=30) &(df["Glucose"]>=121), "NEW_AGE_GLUCOSE"]="Young_Diabete"
df.loc[((df["Age"]>30) & (df["Age"]<=50)) & (df["Glucose"]<70), "NEW_AGE_GLUCOSE"]="Mature_Hipoglisemi"
df.loc[((df["Age"]>30) & (df["Age"]<=50) & (df["Glucose"]>=70)&(df["Glucose"]<=120)), "NEW_AGE_GLUCOSE"]="Mature_Normal"
df.loc[((df["Age"]>30) & (df["Age"]<=50) & (df["Glucose"]>=70)& (df["Glucose"]>=121)), "NEW_AGE_GLUCOSE"]="Mature_Diabete"
df.loc[((df["Age"]>50) & (df["Glucose"]<70)), "NEW_AGE_GLUCOSE"]="Old_Hipoglisemi"
df.loc[((df["Age"]>50) & (df["Glucose"]>=70)&(df["Glucose"]<=120)), "NEW_AGE_GLUCOSE"]="Old_Normal"
df.loc[((df["Age"]>50) & (df["Glucose"]>=121)), "NEW_AGE_GLUCOSE"]="Old_Diabete"

df.head()

#BMI&Glucose
df.loc[(df["BMI"]<18) &(df["Glucose"]<70), "NEW_BMI_GLUCOSE"]="Underweight_Hipoglisemi"
df.loc[(df["BMI"]<18) &(df["Glucose"]>=70)&(df["Glucose"]<=120), "NEW_BMI_GLUCOSE"]="Underweight_Normal"
df.loc[(df["BMI"]<18) &(df["Glucose"]>=121), "NEW_BMI_GLUCOSE"]="Underweight_Diabete"
df.loc[(df["BMI"]>=18)&(df["BMI"]<=25) &(df["Glucose"]<70), "NEW_BMI_GLUCOSE"]="Idealweight_Hipoglisemi"
df.loc[(df["BMI"]>=18)&(df["BMI"]<=25) &(df["Glucose"]>=70)&(df["Glucose"]<=120), "NEW_BMI_GLUCOSE"]="Idealweight_Normal"
df.loc[(df["BMI"]>=18)&(df["BMI"]<=25) &(df["Glucose"]>=121), "NEW_BMI_GLUCOSE"]="Idealweight_Diabete"
df.loc[(df["BMI"]>=26)&(df["BMI"]<=30) &((df["Glucose"]<70)), "NEW_BMI_GLUCOSE"]="Overweight_Hipoglisemi"
df.loc[(df["BMI"]>=26)&(df["BMI"]<=30) &(df["Glucose"]>=70)&(df["Glucose"]<=120), "NEW_BMI_GLUCOSE"]="Overweight_Normal"
df.loc[(df["BMI"]>=26)&(df["BMI"]<=30) &(df["Glucose"]>=121), "NEW_BMI_GLUCOSE"]="Overweight_Diabete"
df.loc[(df["BMI"]>=31) &(df["Glucose"]<70), "NEW_BMI_GLUCOSE"]="Obese_Hipoglsemi"
df.loc[(df["BMI"]>=31) &(df["Glucose"]>=70)&(df["Glucose"]<=120), "NEW_BMI_GLUCOSE"]="Obese_Normal"
df.loc[(df["BMI"]>=31) &(df["Glucose"]>=121), "NEW_BMI_GLUCOSE"]="Obese_Diabete"

# Derive Categorical Variable with Insulin Value

def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df.head()

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

df.columns = [col.upper() for col in df.columns]


#Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols

df=one_hot_encoder(df, ohe_cols)
df.head()

# Scaling
cat_cols, num_cols, cat_but_car = grab_col_names(df)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

# Modeling

y = df["OUTCOME"]

X = df.drop(["OUTCOME"], axis=1)

log_model = LogisticRegression().fit(X, y)

y_pred = log_model.predict(X)

y_pred[0:10]

y[0:10]

######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()


cv_results['test_precision'].mean()


cv_results['test_recall'].mean()


cv_results['test_f1'].mean()


cv_results['test_roc_auc'].mean()
