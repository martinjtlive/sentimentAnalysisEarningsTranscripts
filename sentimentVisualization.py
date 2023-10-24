# to reference envinronment variables
from dotenv import load_dotenv
import os
# to search input file folder
import glob
# for the UX
import streamlit as st
# working with pandas
import pandas as pd
# working with numpy
import numpy as np
# to convert selected date to string
import datetime


def main():

    load_dotenv()

    # load dataframes of the sentiment data
    agg_metrics, agg_det = get_inputs()
    
    # creating company list for selection later on
    # start here need to think about whether all this is best in a function
    company_array = agg_metrics['company'].unique()
    company_list_w_none = np.append('None', company_array)


    st.set_page_config(page_title='Sentiment analysis on transcripts', layout='wide')
    st.header('Sentiment Metrics on transcripts')




    # 2 tabs - agg - aggregate data, det - details data
    agg, det = st.tabs(['Aggregate', 'Details'])

    with agg:
        st.header('Aggregate Info.')        
        selected_company = st.selectbox('Company', company_list_w_none, key=1)

        if selected_company != 'None':
            p_df, q_df, qa_df = df_per_company_category(agg_metrics, selected_company)

            # 3 columns layout
            p, q, qa = st.columns(3) # p, q, qa = presentation, question, questionAndanswer

            with p:
                st.header('Presentation')
                st.line_chart(p_df, x='publish_date', y = ['avg_neutral','avg_positive','avg_negative'], color = ["#E53935", "#CFD8DC", "#004D40", ])

            
            with q:
                st.header('Questions')
                st.line_chart(q_df, x='publish_date', y = ['avg_neutral','avg_positive','avg_negative'], color = ["#E53935", "#CFD8DC", "#004D40", ])
            
            with qa:
                st.header('Question and Answers')
                st.line_chart(qa_df, x='publish_date', y = ['avg_neutral','avg_positive','avg_negative'], color = ["#E53935", "#CFD8DC", "#004D40", ])
                

            
    
    with det:
        st.header('Detailed Info.')
        selected_company = st.selectbox('Company', company_list_w_none, key=2)
        if selected_company != 'None':
            publish_date_array = get_publish_date_list(agg_metrics, selected_company)
            selected_publish_date = st.selectbox('Publish Date', publish_date_array, key=3)
            pd_df, qd_df, qad_df = get_df_per_cat_per_file(selected_company, selected_publish_date, agg_det)

            with st.container():
                st.header('Presentation')
                st.bar_chart(pd_df, x='flow_of_call', y = 'sentiment')
            
            with st.container():
                st.header('Questions')
                st.bar_chart(qd_df, x='flow_of_call', y = 'sentiment')
            
            with st.container():
                st.header('Question and Answers')
                st.bar_chart(qad_df, x='flow_of_call', y = 'sentiment')


############################################################################################################################
############################################################################################################################

def get_inputs():
    '''
    Looks through the input folder and returns the latest aggregate metrics and aggregate details file
    '''
    input_folder = os.getenv('save_df_location')
    metrics_files = glob.glob(input_folder + 'AggSentMetrics*.csv')
    latest_metrics_file = max(metrics_files, key = os.path.getctime)
    
    details_files = glob.glob(input_folder + 'AggSentDetails*.csv')
    latest_details_file = max(details_files, key = os.path.getctime)
    
    # load the 2 files into dataframes
    metrics_df = pd.read_csv(latest_metrics_file)
    details_df = pd.read_csv(latest_details_file)
    
    return metrics_df, details_df

def df_per_company_category(df, comp):
    '''
    Arg:
    df - the aggregagte metrics dataframe
    comp - company to filter by
    
    Returns
    3 dataframes filtered by company and the three category values - Presentation, Question, QuestionAndAnswer
    
    '''
    
    dic_df = {} # dic_df - dictionary of dataframes
    
    category = ['Presentation', 'Question', 'QuestionAndAnswer']
    
    for count,c in enumerate(category):
        filtered_df = df[(df['company'] == comp) & (df['category'] == c)]

        # make publish_date a datetime datatype and sort by publish_date
        filtered_df2 = filtered_df.astype({'publish_date':'datetime64'})        
        filtered_df3 = filtered_df2.sort_values(by = 'publish_date', ascending=True)

        # create dictionary item with category as key and dataframe as value
        dic_df[c] = filtered_df3
    
    return dic_df.values() 

def get_publish_date_list (met_df, comp):
    '''
     The met_df is filtered by company. The publish_date column is changed to datetime. T
     he unique list of publish_date is returned
    
    Arg:
    met_df - dataframe with aggregate senitment metrics
    comp - selected company 
    
    
    Returns:
    The publish date list
    
    '''
    met_df_by_comp = met_df[(met_df['company'] == comp)]
    met_df_by_comp_1 = met_df_by_comp.astype({'publish_date': 'datetime64'})
    met_df_by_comp_2 = met_df_by_comp_1.sort_values(by = 'publish_date', ascending=True)
    
    pub_date = met_df_by_comp_2['publish_date'].unique()
    
    # changing datetime to show YYYY-MM-DD
    pub_date_formatted = pub_date.astype('datetime64[D]')
    
    return pub_date_formatted

def get_df_per_cat_per_file (comp, pub_date, sent_det):
    '''
    function returns 3 dataframes. 1 per category.
    Arg:
    comp - company to filter sentiment details df on
    pub_date - publish date to filter sentiment details df on
    sent_det - dataframe that has details sentiment data for all the processed files
    
    Returns:
    3 dataframes - it is sent_def filtered by pub_date, company and category. the 3 categories are:
    presnetation, question, questionAndAnswer
    
    '''
    
    # category list
    category = ['Presentation', 'Question', 'QuestionAndAnswer']
    
    # formatted date for filtering
    date_selection = datetime.datetime.strptime(str(pub_date), '%Y-%m-%d') # making a date object
    formatted_pub_date = date_selection.strftime('%b %d, %Y')
    
    # sent details df filtered by company and publish date
    sent_det_filtered =  sent_det[(sent_det['company'] == comp) & (sent_det['publish_date'] == formatted_pub_date)]
    
    # dictionary to hold the category key and the corresponding df as value. The df is filtered by category
    dic_df = {}
    
    for cat in category:
        sent_det_filtered_by_cat = sent_det_filtered[(sent_det_filtered['category'] == cat)]
        dic_df[cat] = sent_det_filtered_by_cat
    
    return dic_df.values()


if __name__ == '__main__': main()