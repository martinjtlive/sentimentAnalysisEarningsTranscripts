# to reference env. variables
from dotenv import load_dotenv
import os
# to process files in a zipped file
import zipfile
# to work with json files
import json
# to work with regular expressions. 
# use cases for this script include: extract date from text
import re
# to use chat model
from langchain.chat_models import AzureChatOpenAI
# to define roles in chat
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# to create prompt template
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# set up langchain
from langchain.chains import LLMChain
# working with dataframes
import pandas as pd
# to write datetime to filename
import time
# to reference S&P list for processing
from s_and_p_500 import SPCIQ_ids
# to run as cli
import typer
# to show logs
from loguru import logger
# to show progres
from tqdm import tqdm


def main():


    load_dotenv()  

    # get input zip file location
    zip_file_location = os.getenv('in_file_location')
    filename = 'json 2.zip'
    
    processZipFile(zip_file_location+filename)




def processZipFile(filepath):
    '''
    Takes a zip file and processes each file within it
    processing involves  extracting content of 3 categories from it

    Arg:
    filepath = absolute path to zip file

    Returns:

    '''

    # list to hold dataframes of sentiment details for the files being processed. 1 list item corresponds to 1 processed file
    sentiment_detail_list_df = []

    # save df location 
    save_df_location = os.getenv('save_df_location')

    logger.info("Opening zip file...")

    with zipfile.ZipFile(filepath, 'r') as myzip:
        file_member_list = myzip.namelist()

        file_count = len(file_member_list)


        for count, member in tqdm(enumerate(file_member_list), desc='File processing', total=file_count):
            try:
                
                with myzip.open(member) as file:
                    file_content = json.load(file)
                    filename = member

                    # extract company and file details
                    company, ciq_id, event_type, publish_date, month, year = getCompanyDetails(file_content)

                    if ciq_id not in SPCIQ_ids:
                        # print(f'{ciq_id} not present in curated s_and_p_500 list')
                        logger.info(f"{ciq_id} not present in curated s_and_p_500 list")
                        continue

                    # get list of dictionaries of transcript content
                    content_list = file_content['content']
                    
                    # plist - presentation list, qlist - question list, qalist -question and answer(s) list
                    plist, qlist, qalist = extractCategories(content_list)

                    # pass content to LLM
                    p_sentiment_list = extractSentiment(plist, 'Presentation Speech at an Investor relations call') # p_sentiment_list - presentation sentiment list
                    q_sentiment_list = extractSentiment(qlist, 'Questions raised by Analysts at an Investor relations call') # q_sentiment_list - question sentiment list
                    # q_sentiment_list - question and answer(s) sentiment list
                    qa_sentiment_list = extractqaSentiment(qalist, 'Conversation between Analyst and Executives at an Investor relations call') 


                    # write data to file
                    sentiment_df_for_file = createSentimentDFForAFile(plist, qlist, qalist, member
                                                                      , company, ciq_id, event_type, publish_date, month, year)
                    
                    sentiment_detail_list_df.append(sentiment_df_for_file)

                    logger.info(f"Sentiment extracted and saved for file {count+1} from list of {file_count} files ")



                    

                    
            except Exception as e:
                logger.warning(f'File:{member} - failed with exception: {str(e)}')
                
        
    
    # concatenate all the dfs in list and write aggregate df to file
    aggregate_sentiment_details_df = pd.concat(sentiment_detail_list_df, axis = 0, ignore_index = True)
    
    # get datetime
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    # save df to local
    aggregate_sentiment_details_df.to_csv(save_df_location + 'AggSentDetails-' +  timestr +  '.csv', index=False)
    
    # write sentiment aggregate metrics to file
    agg_sent_met = aggregateSentimentMetricsDFCreation(aggregate_sentiment_details_df)
    agg_sent_met.to_csv(save_df_location + 'AggSentMetrics-' +  timestr +  '.csv', index=False)

    
                
                    



               


                



    

def getCompanyDetails(fileData):
    '''
    From the json content extract values of certain keys
    Arg:
    fileData - json from file
    Returns:
    Company details       
    '''

    company = fileData['company_name']
    event_date = extractDateFromText(fileData['title'])
    event = extract_event_type(fileData['title'])
    company_id, year, month = fileData['company_id'], fileData['year'], fileData['month']

    return company, company_id, event, event_date, month, year


def extractDateFromText(text):
    '''
    Arg:
    text - title that has the date within in it
    Returns: 
    date as string
    '''
    
    date_pattern = r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}\b"
    date_string = re.search(date_pattern, text)
    
    return date_string.group()

def extract_event_type(text):
    '''
    Arg:
    text - text from which the event type is to be extracted
    
    Returns:
    str of earnings call nature
    '''
    listOfParts = text.split(', ', 2)
    
    
    event_type = str(listOfParts[1])
    
    return event_type


def extractCategories(ld):
    '''
    From a list of dictionaries, extract applicable values based on another key value within the dictionary
    This is to be done for applicable list items of type (dict)

    This function will call 2 other functions because extraction of presentation and question content follow 
    the same pattern. But the extraction of question and subsequent answers to it into a list of dictionaries is slightly different.
    
    Arg:
    ld - list of dictionaries

    Returns:
    3 lists of dictionaries. 1 for each category - presentation, questions, questions and answer(s)
    
    '''

    presentation_list = listBasedOfTranscriptType (ld, 'Presenter Speech')
    question_list = listBasedOfTranscriptType (ld, 'Question')
    question_answer_list = get_list_qa(ld)

    return presentation_list, question_list, question_answer_list



def listBasedOfTranscriptType (contentList, componentType):
    '''
    Creates a list of values from list of dictionaries based on specified dictionary key
    
    Args:
    contentList = list containing dictionaries from which component Type values are to be extracted
    
    Returns:
    list of values of specified key for all the dictionary items in list
    '''
    
    valueList = []
    
    valueList = [{'flow_of_call': c.get('flow_of_call'), 'componenttext': c.get('componenttext')}  for c in contentList if c['transcriptcomponenttypename'] == componentType]
    
    return valueList

def get_list_qa(contentListData):
    '''
    From the list of dictionaries, create another list of dictionaries. Each list items dictionary has 3 keys - 
    id, question, answers. Answers is a list of all the answers by the different executives
    
    Arg:
    contentListData - list of dictionaries
    
    Returns:
    list of dictionaries of fewer keys. A unit is a dictionary of 3 keys - id for question, question, answer. 
    Value for answer is a list of all the answers to a particular question
    
    '''
    
    list_qa = []
    i = 0
    
    for i in range(len(contentListData)):
        if contentListData[i]['transcriptcomponenttypename'] == 'Question':
            qa_dict = {'flow_of_call': '', 'Question':'', 'Answer(s)': []}
            qa_dict['flow_of_call'] = contentListData[i]['flow_of_call']
            qa_dict['Question'] = contentListData[i]['componenttext']
            j=i
            while contentListData[j+1]['transcriptcomponenttypename'] == 'Answer':
                qa_dict['Answer(s)'] += [contentListData[j+1]['componenttext']]
                j+=1
            list_qa.append(qa_dict)
    
    return list_qa

def extractSentiment(l_2keydict, variable_value_for_system_prompt):
    '''
    Determines sentiment for value of specific key of dictionary that is list item. 
    This is done for each item of list.
    
    Arg:
    1_2keydict - list of 2 key dictionary items
    varaible_value_for_system_prompt - variable to be passed to system prompt for chat model

    Returns:
    list of dicionary with 2 keys. 1 key is call id, other key is sentiment
    '''
    # set chatModel
    chat_model = AzureChatOpenAI(deployment_name = 'gpt-4')
    
    # create prompt
    prompt = PromptTemplate(
    template = 'You are very good at determinining sentiment based on transcripts \
    of human conversations of the following type of speech - {speechType}. \
    For a given piece of text you can respond with 1 word describing \
    the sentiment of the text. The word choice is 1 of these three: Positive, Negative, Neutral',
    input_variables = ['speechType']
    )
    
    human_template = '{text}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    
    # create chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # set up chain
    chain = LLMChain(llm=chat_model, prompt= chat_prompt)

    sentimentList = []

    for d in l_2keydict:
        response = chain.run({
            'speechType' : variable_value_for_system_prompt,
            'text' : d['componenttext'] 
        })
        sent_dict = {'flow_of_call': d.get('flow_of_call'), 'sentiment' : response}
        sentimentList.append(sent_dict)
    
    return sentimentList


def extractqaSentiment(dataList, variable_value_for_system_prompt):
    '''
    Extract sentiment on a conversation. The conversation has 2 parties. 1 party is the Analyst. The second party is the collective response by executives to the 
    question raised by the analyst.

    Arg:
    dataList - list of dictionaries. Each dictionary has 3 keys. 1 key is call id corresponding to 2nd key - Question. 3rd key is answer. Answer value is 
    list of responses to question.
    variable_value_for_system_prompt - context input to pass to system prompt

    Returns:
    list of dicionary with 2 keys. 1 key is call id, other key is sentiment
    '''

    # set chatModel
    chat_model = AzureChatOpenAI(deployment_name = 'gpt-4')

    prompt_for_qa = PromptTemplate(
        template = 'You are very good at determining the sentiment of human conversation based on \
        transcripts of these conversations in the following context - {speechContext}. \
        For a given conversation, can you please respond with 1 word that indicates the overall sentiment of the conversation. \
        The response is to be in 1 of the 3 following words: Positive, Negative, Neutral',
        input_variables = ['speechContext']
        )
    
    human_qa_template = 'Question by the analyst: {question}. Responses by Company executives - {answers}'
    
    system_message_qa_prompt = SystemMessagePromptTemplate(prompt = prompt_for_qa)
    
    human_message_qa_prompt = HumanMessagePromptTemplate.from_template(human_qa_template)
    
    qa_chat_prompt = ChatPromptTemplate.from_messages([system_message_qa_prompt, human_message_qa_prompt])

    qa_chain = LLMChain(llm = chat_model, prompt = qa_chat_prompt)

    sentiment_for_list_qa = []

    for d in dataList:
        q = d['Question']
        answers_from_list = '\n'.join(d['Answer(s)'])

        response = qa_chain.run(
            {
                'speechContext': variable_value_for_system_prompt,
                'question': q,
                'answers': answers_from_list
            }
        )

        qa_sent_dict = {'flow_of_call': d.get('flow_of_call'), 'sentiment' : response}
        sentiment_for_list_qa.append(qa_sent_dict)

    return sentiment_for_list_qa




def createSentimentDFForAFile(pld, qld, qald, filename, company, ciq_id, event_type, publish_date, month, year):
    '''
    creates a dataframe from a 3 lists of dictionaries and other attributes - 
    company, ciq_id, event_type, publish_date, month, year 
    
    Arg:
    pld - presentation list of dictionaries
    qd - question list of dictionaries
    qald - question and answers list of dictionaries
    
    ...
    
    Returns:
    a dataframe that aggregates all this data
    
    '''
    
    dfl = [] # dfl - list of dataframes
    
    lld = [pld, qld, qald] # lld - list of list of dictionaries
    
    categoryList = ['Presentation','Question','QuestionAndAnswer']
    
    for index, ld in enumerate(lld):
        dfc = pd.DataFrame(ld )# dfc - dr
        dfc['category'] = categoryList[index]
        
        dfc = dfc.assign (filename = filename, company= company, ciq_id = ciq_id, event_type = event_type
                          , publish_date = publish_date, month = month, year = year)
        
        dfl.append(dfc)
    
    df_for_file = pd.concat(dfl, axis=0, ignore_index=True)
    
    return df_for_file

def aggregateSentimentMetricsDFCreation(sentDetailsDF):
    '''
    Arg:
    sentDetailsDF - dataframe that has the sentiment per flow id per category per processed file
    
    Returns:
    
    df_sentiment_metrics_all_processed_files - dataframe that holds the average neutral, positive, negative occurences 
    per category per processed file
    
    '''
    
    # list that will have each df of sentiment metrics for a file as an item
    df_sentiment_metrics_list = []
    
    # get unique filename
    processedfiles = sentDetailsDF['filename'].unique()
    
    
    categoryList = ['Presentation','Question','QuestionAndAnswer']
    
    for f in processedfiles:
        # print(f) - consider a logru
        
        sentimentDetailsForf = sentDetailsDF.query(f'filename == "{f}"')
        
        # new df to save metrics for a file
        sentimentMetricsForAFile = pd.DataFrame(columns = ['avg_neutral','avg_positive','avg_negative','category'])
        
        
        for c in categoryList:
            # filter dataframe by filename and category
            sentimentDetailsForfForSpecificCategory = sentimentDetailsForf.query(f'category == "{c}"')
            # extract sentiment per flow call id for a category
            sent_list = sentimentDetailsForfForSpecificCategory['sentiment'].tolist() 
            
            avg_neutral, avg_positive, avg_negative = sentimentMetrics(sent_list)
            
            # add metrics for category into df
            sentimentMetricsForAFile.loc[len(sentimentMetricsForAFile)] = [avg_neutral, avg_positive, avg_negative, c]
        
        # top row data from sentimentDetails dataframe filtered only by filename to add 
        # to sentimentMetrics dataframe for a file
        top_row_data = sentimentDetailsForf.iloc[0, 3:]
        top_row_columns = sentimentDetailsForf.columns[3:]
        
        # add reference top row data to each metrics row
        sentimentMetricsForAFile[top_row_columns] = top_row_data
        
        df_sentiment_metrics_list.append(sentimentMetricsForAFile)
    
    
    df_sentiment_metrics_all_processed_files = pd.concat(df_sentiment_metrics_list, axis=0, ignore_index=True)
    
    return df_sentiment_metrics_all_processed_files     

def sentimentMetrics(sentimentList):
    '''
    Arg:
    sentimentList - list where items are 1 of 3 values - Positive, Negative, Neutral

    Returns:
    avg. +, -, neutral of document
    '''

    positive_count = neutral_count = negative_count = 0.0
    
    for i in range(len(sentimentList)):
        if sentimentList[i] == 'Positive':
            positive_count += 1
        elif sentimentList[i] == 'Negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    avg_neutral = neutral_count/len(sentimentList)
    avg_positive = positive_count/len(sentimentList)
    avg_negative = negative_count/len(sentimentList)    
   
    return avg_neutral, avg_positive, avg_negative


if __name__=='__main__': typer.run(main)


