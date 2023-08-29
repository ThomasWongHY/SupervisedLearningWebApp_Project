import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from pickle import dump
from pickle import load

class SupervisedLearning:
    
    def __init__(self):
        self.x_train_transformed = 0
        self.df = pd.read_csv('./KSI_train.csv')
        self.best_model = []
        self.template = {}
        self.result = []
        
    def exploration(self):
        
        # Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. 
        print('Descriptions: ')
        print(self.df.describe(), '\n')
        print('Types: ')
        print(self.df.info(), '\n')
        df_stats = self.df.describe()
        df_stats.loc['Range'] = df_stats.loc['max'] - df_stats.loc['min']
        print('Range:')
        print(df_stats.loc['Range'], '\n')
        
        # Statistical assessments including means, averages, correlations
        print('Means:')
        print(df_stats.loc['mean'], '\n')
        print('Median:')
        print(df_stats.loc['50%'], '\n')
        print('Correlation:')
        print(self.df.corr(), '\n')

        # Missing data evaluations – use pandas, numpy and any other python packages
        print(self.df.shape)

        print('Missing Values')
        print(self.df.isnull().sum(), '\n')

        print('Unknown Values of INVAGE column')
        print(self.df['INVAGE'].isin(["unknown"]).sum())

        print(self.df.shape)
        
    def preprocessing(self, df):
        if df is None:
            df = pd.read_csv('./KSI_train.csv')
            
        df_clean = df.drop_duplicates('ACCNUM')
        df_clean.replace(' ', np.nan, regex=False, inplace=True)
        
        for col in df_clean:
            df_clean = df_clean[df_clean[col] != 'unknown']
            if df_clean[col].dtype == object and len(df_clean[col].unique()) == 2:
                df_clean[col].fillna('No', inplace=True)

        # Convert integer column to time column
        df_clean = self.convertTime(df_clean)
        
        df_clean['ACCLASS'] = df_clean['ACCLASS'].replace({'Non-Fatal Injury':0, 'Property Damage Only': 0, 'Fatal':1})
        df_clean = df_clean[df_clean['ACCLASS'].notna()]

        return df_clean
    
    def convertTime(self, df_clean):
        # Convert integer column to time column
        df_clean['TIME'] = df_clean['TIME'].apply(lambda x: datetime.time(x // 100, x % 100))

        # Define time intervals with a given frequency
        freq = datetime.timedelta(hours=3)
        start_time = datetime.time(0, 0)
        end_time = datetime.time(23, 59)
        today = datetime.date.today()
        period = (datetime.datetime.combine(today, end_time) - datetime.datetime.combine(today, start_time)) // freq + 1
        intervals = [(datetime.datetime.combine(today, start_time) + i*freq).time() for i in range(period)]

        # Classify time into a specific time interval
        def classify_time(time):
            for i, interval in enumerate(intervals):
                if time < interval:
                    return f"{intervals[i-1].strftime('%H%M')}-{interval.strftime('%H%M')}"
            return f"{intervals[-2].strftime('%H%M')}-{intervals[-1].strftime('%H%M')}"

        df_clean['INTERVAL'] = df_clean['TIME'].apply(classify_time)
        df_clean.drop(['TIME','DATE'], axis=1, inplace=True)

        return df_clean
        
    def visualization(self, df):
        df_clean = self.preprocessing(df)
        for col in df_clean.select_dtypes(include=object):
            if len(df_clean[col].unique()) > 2 and len(df_clean[col].unique()) < 20:
                plt.figure()
                df_clean[col].value_counts().plot(kind='bar', color=list('rgbkmc'))
                plt.xlabel(col)
                plt.ylabel('Count')
                # plt.show()
            
            if len(df_clean[col].unique()) == 2:
                combo_counts = df_clean.groupby([col]).size().reset_index(name='Count')
                combo_counts.plot(kind='bar', x=col, y='Count', stacked=True, title=col)
                # plt.show()
        
        return df_clean
    
    def resampling(self, df_selected):
        # Managing imbalanced classes
        df_majority = df_selected[df_selected.ACCLASS==0]
        df_minority = df_selected[df_selected.ACCLASS==1]

        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=44)

        df_resampled = pd.concat([df_majority, df_minority_upsampled])
        
        return df_resampled
    
    def modelling(self, df=None):
        # select the columns that is related to 'ACCLASS' and do not have too much unique value
        df_clean = self.preprocessing(df)
        df_selected = df_clean[[
            'INTERVAL', 'DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'PEDESTRIAN', 'CYCLIST', 
            'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
            'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'ACCLASS'
        ]]
        df_final = self.resampling(df_selected)

        return df_final
        
    def pipelines(self, input):
        # df_final = self.modelling()
        cat_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy="constant",fill_value='No')),
                ('one_hot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        # num_pipeline = Pipeline(
        #     [
        #         ('imputer', SimpleImputer(strategy="median")),
        #         ('scaler', MinMaxScaler())
        #     ]
        # )

        # num_attribs = input.select_dtypes(exclude=object).columns
        cat_attribs = input.select_dtypes(include=object).columns

        transformer = ColumnTransformer(
            [
                # ("num", num_pipeline, num_attribs),
                ("cat", cat_pipeline, cat_attribs)
            ]
        )

        return transformer
    
    def data_split(self, X, y):
        # Train, Test data splitting

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=44)

        # Split the data into training and testing sets
        for train_index, test_index in split.split(X, y):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
        
        return X_train, X_test, y_train, y_test

    def build_model(self):
        lr = LogisticRegression(random_state=42)
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        # svc = SVC(probability=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        nn = MLPClassifier(hidden_layer_sizes=(5, 2), solver='lbfgs', random_state=42)

        models = [lr, dt, rf, nn]
    
        return models
    
    def build_pipe(self):
        piplines = []
        models = self.build_model()
        for mod in models:
            full_pipeline = Pipeline(
                [
                    ('clf', mod)
                ]
            )
            piplines.append(full_pipeline)
        
        return piplines
    
    def build_param(self):
        # Fine tune the models using Grid search and randomized grid search. 
        param_grid_lr = {
            'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'clf__C': [0.1, 1, 10, 100],
            'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }

        param_grid_dt = {
            'clf__min_samples_split': range(10, 300, 30),
            'clf__max_depth': range(1, 30, 3),
            'clf__min_samples_leaf': range(1, 15, 3),
            'clf__criterion': ['gini', 'entropy']
        }

        param_grid_rf = {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [3, 5, 7, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }

        param_grid_nn = {
            'clf__hidden_layer_sizes': [(10,), (20,), (10, 5), (20, 10)],
            'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'clf__solver': ['lbfgs', 'sgd', 'adam'],
            'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1]
        }
        
        param_list = [param_grid_lr, param_grid_dt, param_grid_rf, param_grid_nn]
        
        return param_list
        
    def evaluation(self):
        # Predictive model building
        # Use logistic regression, decision trees, SVM, Random forest and neural networks algorithms as a minimum– use scikit learn
        df_final = self.modelling()
        transformer = self.pipelines(df_final)
        
        X = df_final.drop('ACCLASS', axis=1)
        y = df_final['ACCLASS']
        
        # Train, Test data splitting
        X_train, X_test, y_train, y_test = self.data_split(X, y)
        transformer.fit(X_train, y_train)
        X_train_prepared = transformer.transform(X_train)
        X_test_prepared = transformer.transform(X_test)
        dump(transformer, open('transformer.pkl', 'wb'))
        
        X_train_df = pd.DataFrame(X_train_prepared.toarray())
        X_test_df = pd.DataFrame(X_test_prepared.toarray())

        piplines = self.build_pipe()
        param_list = self.build_param()
        
        best_param = []
        modeltitle = ['LogisticRegression','DecisionTree','RandomForest','NeuralNetwork']
        
        for i in range(len(param_list)):
            rand = RandomizedSearchCV(
                estimator=piplines[i], 
                param_distributions=param_list[i], 
                scoring='accuracy', cv=5,
                n_iter=7, refit=True, 
                verbose=3)
            
            search = rand.fit(X_train_df, y_train)
            self.best_model.append(search.best_estimator_)
            best_param.append(search.best_params_)
            dump(search.best_estimator_, open(f'best_model_{modeltitle[i]}.pkl', 'wb'))
            # print("Best Params:", search.best_params_)
            # print("Best Score:",search.best_score_)
            # print("Best Estimator:",best_model)
        
        models = self.build_model()
        
        for i in range(len(self.best_model)):
            self.best_model[i].fit(X_train_df, y_train)
            y_test_pred = self.best_model[i].predict(X_test_df)

            # create ROC curve
            y_pred_proba = self.best_model[i].predict_proba(X_test_df)[::,1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(12,9))
            plt.plot(fpr,tpr)
            
            plt.title(f'ROC - {models[i]}')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

            plt.savefig(f'./static/roc_{i}.png')

            self.template = {}
            self.template["a_name"] = modeltitle[i]
            self.template["accuracy"] = accuracy_score(y_test, y_test_pred)
            self.template["precision"] = precision_score(y_test, y_test_pred)
            self.template["recall"] = recall_score(y_test, y_test_pred)
            self.template["f1_score"] = f1_score(y_test, y_test_pred)
            
            con_mat = ''
            for i in confusion_matrix(y_test, y_test_pred):
                # for j in i:
                #     con_mat = con_mat + str(j) + ' '
                con_mat = con_mat + str(i) + '\n'
            con_mat = con_mat
            self.template["confusion_matrix"] = con_mat
            
            self.result.append(self.template)

        dump(self.result, open(f'score_result.pkl', 'wb'))
            
    def final_data(self):
        df_final = self.modelling()
    
        X = df_final.drop('ACCLASS', axis=1)
        y = df_final['ACCLASS']
        
        # Train, Test data splitting
        X_train, X_test, y_train, y_test = self.data_split(X, y)
        
        return X_train, X_test, y_train, y_test

    def prediction(self, input, model):

        transformer = load(open('transformer.pkl', 'rb'))
        input_prepared = transformer.transform(input)
        input = pd.DataFrame(input_prepared.toarray())
        print(input)
        pred = model.predict(input)

        return pred
    
sl = SupervisedLearning()
sl.evaluation()

# data = ['0600-0900', 'Toronto and East York', 'Clear', 'Daylight', 'Dry', 'Yes', '', 'Yes', '', '', '', '', '', '', 'Yes', '', '', '']

# cols = [
#             'INTERVAL', 'DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'PEDESTRIAN', 'CYCLIST', 
#             'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
#             'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY'
#         ]

# df = pd.DataFrame([data], columns=cols)
# pred = sl.prediction(df, load(open('best_model_LogisticRegression.pkl', 'rb')))
# print(pred)



