from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
rfm_data = None  
cluster_mapping = {}

def load_and_clean_data(file_path):
    #Load data
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    #Convert CustomerID to string and create Amount  column
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']

    #Compute RFM metrics
    rfm_m = retail.groupby('CustomerID')['Amount'].sum()
    rfm_m = rfm_m.reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f = rfm_f.reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min()
    rfm_p = rfm_p.reset_index()
    rfm_p.columns = ['CustomerID', 'Recency']
    rfm_p['Recency'] = rfm_p['Recency'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount','Frequency','Recency']
    
    #Remove Outliers
    # Amount
    Q1 = rfm.Amount.quantile(0.05)
    Q3 = rfm.Amount.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

    # Recency
    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

    # Frequency
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

    return rfm


def preprocess_data(file_path):
    rfm= load_and_clean_data(file_path)
    rfm_df = rfm[['Amount','Frequency','Recency']]
    #Instantiate
    scalar=StandardScaler()
    #fit_transform
    rfm_df_scaled = scalar.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    #rfm_df_scaled
    rfm_df_scaled.columns=['Amount', 'Frequency', 'Recency']

    return rfm, rfm_df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df= preprocess_data(file_path)[1]
    results_df=model.predict(df)
    #ressult_df = pd.DataFrame(results_df)
    df_with_id = preprocess_data(file_path)[0]
    df_with_id['Cluster_Id'] = results_df

    # ---------------- DYNAMIC CLUSTER MAPPING ----------------
    cluster_summary = df_with_id.groupby("Cluster_Id").agg({
        "Amount": "mean",
        "Frequency": "mean",
        "Recency": "mean"
    }).reset_index()

    # High Value → high Amount + Frequency + low Recency
    high_cluster = cluster_summary.sort_values(
        by=["Amount", "Frequency", "Recency"],
        ascending=[False, False, True]
    ).iloc[0]["Cluster_Id"]

    # Low Value → low Amount + Frequency + high Recency
    low_cluster = cluster_summary.sort_values(
        by=["Amount", "Frequency", "Recency"],
        ascending=[True, True, False]
    ).iloc[0]["Cluster_Id"]

    # Remaining → Regular
    all_clusters = set(cluster_summary["Cluster_Id"])
    regular_cluster = list(all_clusters - {high_cluster, low_cluster})[0]

    # Mapping
    cluster_map = {
        high_cluster: "High Value",
        regular_cluster: "Regular",
        low_cluster: "Low Value"
    }

    # Save globally for reuse
    global cluster_mapping
    cluster_mapping = cluster_map

    # Add segment column
    df_with_id["Segment"] = df_with_id["Cluster_Id"].map(cluster_map)

    global rfm_data
    rfm_data = df_with_id
    total_customers = df_with_id.shape[0]

    #Generate the images and save them
    plt.figure(figsize=(10,6))
    order = ["Low Value", "Regular", "High Value"]
    sns.stripplot(x='Segment', y='Amount', data=df_with_id, hue='Segment', order=order)
    plt.title("Customer Segments vs Amount")
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()


    plt.figure(figsize=(10,6))
    order = ["Low Value", "Regular", "High Value"]
    sns.stripplot(x='Segment', y='Frequency', data=df_with_id, hue='Segment', order=order)
    plt.title("Customer Segments vs Frequency")
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()

    plt.figure(figsize=(10,6))
    order = ["Low Value", "Regular", "High Value"]
    sns.stripplot(x='Segment', y='Recency', data=df_with_id, hue='Segment', order=order)
    plt.title("Customer Segments vs Recency")
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    cluster_counts = df_with_id['Cluster_Id'].value_counts()

    #labels = ['Low Value', 'Regular', 'High Value']
    #sizes = [cluster_counts.get(i,0) for i in range(3)]

    segment_counts = df_with_id["Segment"].value_counts()

    labels = segment_counts.index.tolist()
    sizes = segment_counts.values.tolist()

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    # donut effect
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    donut_path = 'static/cluster_donut.png'
    plt.title("Customer Segmentation Distribution")
    plt.savefig(donut_path)
    plt.clf()
    
    #Return the filennames of the generated images as a JSON response
    response = {
    'amount_img': amount_img_path,
    'freq_img': freq_img_path,
    'recency_img': recency_img_path,
    'donut_img': donut_path,
    'total_customers': int(total_customers)
}

    
    return json.dumps(response)

@app.route('/customer', methods=['POST'])
def customer():
    global rfm_data

    if rfm_data is None:
        return "Please upload CSV first "

    cust_id = request.form['customer_id']
    cust_id = cust_id.strip()   # remove spaces
    rfm_data['CustomerID'] = rfm_data['CustomerID'].astype(float).astype(int).astype(str)
    cust_id = cust_id.strip()

    customer = rfm_data[rfm_data['CustomerID'] == cust_id]

    if len(customer) == 0:
        return "Customer not found "

    cluster = customer['Cluster_Id'].values[0]

    #segment_map = {
    #0: "Low Value Customers ",
    #1: "Regular Customers ",
    #2: "High Value Customers "
    #}
#
    #insight_map = {
    #    0: "This customer is at risk. Offer discounts or re-engagement campaigns.",
    #    1: "This is a regular customer. Maintain engagement with occasional offers.",
    #    2: "This is a high-value customer. Provide premium offers and loyalty rewards."
    #}

    global cluster_mapping

    segment = cluster_mapping[cluster]

    def get_insight(segment):
        if segment == "High Value":
            return "This is a high-value customer. Provide premium offers and loyalty rewards."
        elif segment == "Regular":
            return "This is a regular customer. Maintain engagement with occasional offers."
        else:
            return "This customer is at risk. Offer discounts or re-engagement campaigns."

    insight = get_insight(segment)

    #segment = segment_map[cluster]
    #insight = insight_map[cluster]

    return f"""
    <h3>Customer ID: {cust_id}</h3>
    <p><b>Segment:</b> {segment}</p>
    <p><b>Insight:</b> {insight}</p>
    """

#return render_template('index.html', prediction='This user will click on social network ad {}'.format(output))

#for local
if __name__=="__main__":
    app.run(debug= True)

#for cloud
#if __name__="__mani__":
    # app.run(host = '0.0.0.0'  ,port=8080)