import streamlit as st
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
from sklearn.externals import joblib

import random
random.seed(1234)

plt.style.use('ggplot')

def get_data():
    dat = pd.read_csv('/Users/ianigbogu001/Downloads/PS_20174392719_1491204439457_log.csv/PaySimTransactions.csv')
    dat = dat.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig','oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
    return dat

def dataCleanAddFeatures(dat):
    #Data Cleansing

    X = dat.loc[(dat.type == 'TRANSFER') | (dat.type == 'CASH_OUT')] #Fraud only occurs in these two cases
    Y = X['isFraud']
    del X['isFraud']

    # Eliminate columns shown to be irrelevant for analysis in the EDA
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

    with st.echo():
        # Binary-encoding of labelled data in 'type'
        X.loc[X.type == 'TRANSFER', 'type'] = 0
        X.loc[X.type == 'CASH_OUT', 'type'] = 1
        X.type = X.type.astype(int) # convert dtype('O') to dtype(int)

        X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), \
            ['oldBalanceDest', 'newBalanceDest']] = - 1
        X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), \
            ['oldBalanceOrig', 'newBalanceOrig']] = np.nan

        #Feature Engineering
        X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
        X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest

    return X, Y

def plotStrip(x, y, hue, figsize = (10, 5)):
        
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y,hue = hue, jitter = 0.4, marker = '.', size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1),loc=2, borderaxespad=0, fontsize = 10);
    return ax

def XgboostModel(X,Y):
    with st.echo():
        #Data Partioning
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

        #Model Creation
        clf = XGBClassifier(max_depth = 20, subsample= 0.80000000000000004, scale_pos_weight = 40, \
            n_jobs = 4,colsample_bytree = 0.8, eta = 0.1, gamma = 0.1)

        probabilities = clf.fit(trainX, trainY).predict_proba(testX)
        auprc = average_precision_score(testY, probabilities[:, 1])
        
    return auprc, clf


def main():
    data = get_data()
    st.header('Anomaly Detection in Financial Services')
    st.subheader('A demo on Fraud Detection using PaySim Synthetic Dataset')
    st.image('https://miro.medium.com/max/640/0*_6WEDnZubsQfTMlY.png', width=600)

    if st.checkbox('Show first rows of the data & shape of the data'):
        st.write(data.head(20))
        st.write(data.shape)

    st.write('\nThe distribution of the target variable:\n')
    st.write(data['isFraud'].value_counts())

    st.subheader('Dataset Entity Describtion')
    st.write(pd.DataFrame({
    'Columns': data.columns,
     'Describtion': ['a unit of time in the real world. steps 744 - 30 days simulation.', 'types of transactions: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER', 
     'amount of the transaction in local currency', 'customer who started the transaction', ' initial balance before the transaction', 'new balance after the transaction', 
     'customer who is the recipient of the transaction','initial balance recipient before the transaction.', 'new balance recipient after the transaction.', 
     'This is the transactions made by the fraudulent agents inside the simulation.', 
     'Aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.'] }))

    st.subheader('Descriptive Analysis of Dataset')

    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    data.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))
    ax.set_xlabel("Type")   
    ax.set_ylabel("Count of transaction")
    plt.show()
    st.pyplot()

    ax = data.groupby(['type', 'isFraud']).size().plot(kind='bar', figsize=(8,8))
    ax.set_title("Number of transaction which are the actual fraud per transaction type")
    ax.set_xlabel("(Type,isFraud)")
    ax.set_ylabel("Count of transaction")
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.00))
    st.pyplot()

    st.markdown('From the above, Fraud only occurs in Transfer and CASH_OUT Transactions. This highlights a proabble methodology of fraud transactions \
        where money is transfered between accounts and then cashed out at another accout')

    st.empty()

    st.markdown('A charaterizing feature for fradulent transaction is the presence of **_zero balances_** in the destination balance columns: _oldbalanceDest_ and _newbalanceDest_')

    st.markdown('Removing all other transactions types, and adding two new features ')

    cleanDataX, cleanDataY = dataCleanAddFeatures(data)

    if st.checkbox('Show first rows of the new dataset & shape of the data'):
        st.write(cleanDataX.head(20))
        st.write ('The above dataset is the used for training the ML Model')
        st.write(cleanDataX.shape)

    limit = len(cleanDataX)

    ax = plotStrip(cleanDataY[:limit], cleanDataX.step[:limit], cleanDataX.type[:limit])
    ax.set_ylabel('time [hour]', size = 16)
    ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size = 14);
    st.pyplot()

    ax = plotStrip(cleanDataY[:limit], cleanDataX.amount[:limit], cleanDataX.type[:limit], figsize = (10, 5))
    ax.set_ylabel('amount', size = 16)
    ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 14);
    st.pyplot()

    ax = plotStrip(cleanDataY[:limit], - cleanDataX.errorBalanceDest[:limit], cleanDataX.type[:limit], figsize = (10, 5))
    ax.set_ylabel('- errorBalanceDest', size = 16)
    ax.set_title('Opposite polarity fingerprints over the error in destination account balances', size = 14);
    st.pyplot()

    st.subheader('Model Selection')
    st.markdown('A usual characteristic of anomaly dataset in financial services is the very little of amount of the anomaly class present in the dataset.\
        Techniques used to handle this challenge include Minortity Oversampling Techniques, Majority Undersampling techniques, Tree ensemble Techniques and Neural Networks.\
            In this demo, a Tree ensemble method - eXtreme Gradient Boosted Tree (XGBoost) is used.')
    
    metric,model = XgboostModel(cleanDataX,cleanDataY)

    st.write('Using the area under the precision curve (AUPRC) metric:', metric)

    st.markdown('The important features used by the model to predict fraudulent transaction is below')
        
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(111)
    colours = plt.cm.Set1(np.linspace(0, 1, 9))
    ax = plot_importance(model, height = 1, color = colours, grid = False, show_values = False, importance_type = 'cover', ax = ax);
    for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)          
    ax.set_xlabel('importance score', size = 16);
    ax.set_ylabel('features', size = 16);
    ax.set_yticklabels(ax.get_yticklabels(), size = 12);
    ax.set_title('Ordering of features by importance to the model learnt', size = 20);
    st.pyplot()

    st.subheader('Model decision tree')
    st.write(to_graphviz(model))

    st.markdown('Deployed Model using Azure Machine Learning Service and this is the scoring API')
    st.empty()

    st.subheader('Test Input for API Test ')
    step = st.number_input('step')
    type = st.text_input('type')
    amount = st.number_input('amount')
    nameOrig = st.text_input('nameOrig')
    oldBalanceOrig = st.number_input('oldBalanceOrig')
    newBalanceOrig = st.number_input('newBalanceOrig')
    nameDest = st.text_input('nameDest')
    oldBalanceDest = st.number_input('oldBalanceDest')
    newBalanceDest = st.number_input('newBalanceDest')
    isFlaggedFraud = st.number_input('isFlaggedFraud')

    st.subheader('Result')
    st.empty()

if __name__ == "__main__":
    main()
