# to load pdf
from PyPDF2 import PdfReader
# for chunking the text
from langchain.text_splitter import CharacterTextSplitter
# remove URL references from text and extracting date from the filename
import re
# Natural Language tool kit
import nltk
# tokenize words
from nltk.tokenize import word_tokenize
#  remove stop words
from nltk.corpus import stopwords
#nltk.download('stopwords')
# to perform lemmatization
from nltk.stem import WordNetLemmatizer
#nltk.download("wordnet")
#nltk.download("omw-1.4")
# create prompt
# create prompt template
from langchain import PromptTemplate
# get response from OpenAI
from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
# to reference envinronment variables
from dotenv import load_dotenv
import os
# to determine overall sentiment
import statistics
from statistics import mode 
# for working with dataframes
import pandas as pd
# for UX
import streamlit as st
# convert date entry as string to date
from datetime import datetime



def main():

    load_dotenv()

    
    st.set_page_config(page_title='Sentiment analysis on transcripts')
    st.header('Determine sentiment on transcripts')
    st.text('''Load the transcript. Text is chunked and preprocessed. 
For each chunk, sentiment is determined. Finally for the whole document 
the following are determined: avg +, - , neutral sentiment and the 
overall sentiment of the document''')
    
    

    # initialize a dataframe
    df_docresult = pd.DataFrame(columns=['company','publicationDate', 'avg.Postive','avg.Negative'
                                    ,'avg.Neutral','overallSentiment'])

    # inputs
    company, pdf_files = getinputs()

    # save df location 
    save_df_location = os.getenv('save_df_location')

    # button to process through llm
    button1 = st.button('Process')

    # button to save df to file
    # utton2 = st.button('Save Dataframe to file')

    # for messages
    global placeholder  
    placeholder = st.empty()

      
    if button1:
        
        for file in pdf_files:

            # publication date from filename
            publishDate = extractDateFromFilename(file.name)


            chunks_for_llm = processPDF(file)
            avg_positive, avg_negative, avg_neutral, doc_sent = processingThroughLLM(chunks_for_llm)
            
            # add row to df
            new_row = [company, publishDate, avg_positive, avg_negative, avg_neutral, doc_sent]
            
            df_docresult.loc[len(df_docresult)] = new_row
            writeMessages(f'Processed file: {file.name}')
            
        st.dataframe(df_docresult)    
        df_docresult.to_csv(save_df_location + company + '-sentMetrics.csv')
            
                
        
        

def getinputs():
    
    cmpny = st.text_input('Company')
    pdfs = st.file_uploader("Upload your PDFs", type = "pdf", accept_multiple_files = True)

    return cmpny, pdfs

def writeMessages(message):
	
	placeholder.empty()
	with placeholder.container():
            st.info(message)

def extractDateFromFilename(text):
    '''
    Arg:
    text - file name that has the date within in
    Returns 
    date as string
    '''
    date_string = re.search(r'\d{4}-\d{2}-\d{2}', text)

    return date_string.group(0)


    




def lowercaseRemoveURLPunc(input_text):
    '''
    Arg:
    input_text - input text

    Returns:
    cleaned_text with no url and punctuations and in lower case
    '''
    # regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    noURLText = re.sub(url_pattern, '', input_text)

    # Define a regular expression pattern to match punctuation characters
    pattern = r"[^\w\s]"
    # Substitute the punctuation characters with an empty string
    no_punct_string = re.sub(pattern, " ", input_text)

    # make input text lower case
    lowerCaseTextWithNoURLNoPunc = no_punct_string.lower()

    return lowerCaseTextWithNoURLNoPunc

def remove_stopwords(input_stringlist):
    filtered_words = [word for word in input_stringlist if word not in stopwords.words('english')]
    # filtered_string = ' '.join(filtered_words)
    return filtered_words

def lemmatizeText(input_stringlist):
    # Initialize wordnet lemmatizer
    wnl = WordNetLemmatizer()
    
    lemmatized_words = []
    for word in input_stringlist:
        lemmatized_words.append(wnl.lemmatize(word, pos="v"))
    
    lemmatized_string = ' '.join(lemmatized_words)
    return lemmatized_string #, lemmatized_words


def processPDF(pdfFile):
    '''
    Arg:
    pdfFile - pdf File

    Return:
    chunks of text from pdfFile ready for LLM 

    '''

    text = ''

    pdf_reader = PdfReader(pdfFile)
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    textSpliter = CharacterTextSplitter(separator='\n'
                                        , chunk_size = 1000
                                        , length_function = len)

    chunks = textSpliter.split_text(text)

    # remove url, punctuations, lower case
    level_1_cleaned_chunks = []
    #level_1_cleaned_text = ''

    for ch in chunks:
        level_1_cleaned_text = lowercaseRemoveURLPunc(ch)
        level_1_cleaned_chunks.append(level_1_cleaned_text)
    
    level_2_cleaned_chunks = []
    #level_2_cleaned_text = ''

    for ch1 in level_1_cleaned_chunks:
        tokenized_chunk_textList = word_tokenize(ch1)
        # remove stop words from tokenized chunk
        no_stop_in_chunk = remove_stopwords(tokenized_chunk_textList)
        lemmatized_chunk = lemmatizeText(no_stop_in_chunk)

        level_2_cleaned_chunks.append(lemmatized_chunk)
    
    writeMessages('Chunking Done!')
    

    return level_2_cleaned_chunks

def processingThroughLLM(chunkList):
    '''
    Arg:
    chunkList - processed text as chunks in a list

    Function loops through each chunk and returns sentiment
    Then it aggregates to determine avg. +, -, neutrality of document
    Finally it give the overall sentiment for the document based on mode.
    These sentiment metrics is done in another function call

    
    Returns:
    avg. +, -, neutrality of document, statistic mode of document

    '''

    # define prompt
    template_for_prompt = '''For the following text - {chunk}, determine sentiment. Respond with 1 word: Negative, Positive, Neutral. '''

    prompt = PromptTemplate(
    input_variables=["chunk"],
    template = template_for_prompt
    )

    # set llm
    llm = AzureOpenAI(deployment_name='gpt-3')
    chain = LLMChain(llm = llm, prompt = prompt)
    
    # to collect the sentiment for the chunks
    sentiment_for_chunks = []

    for c in chunkList:
        response = chain.run(c)
        sentiment_for_chunks.append(response)

    # start here: function that does sentiment metrics
    postiveAvg, negativeAvg, neutralAvg, docSentiment = sentimentMetrics(sentiment_for_chunks)

    writeMessages('LLM processing done')

    

    return postiveAvg, negativeAvg, neutralAvg, docSentiment


def sentimentMetrics(sentimentList):
    '''
    Arg:
    sentimentList - list where items are 1 of 3 values - Positive, Negative, Neutral

    Returns:
    avg. +, -, neutrality of document
    Finally it give the overall sentiment for the document based on mode.
    
    '''

    # some items may not be 1 word, it may have few other words or suffixed with a fullstop
    for i in range(len(sentimentList)):
        if 'Positive'in sentimentList[i]:
            sentimentList[i] = 'Positive'
        elif 'Negative' in sentimentList[i]:
            sentimentList[i] = 'Negative'
        else:
            sentimentList[i] = 'Neutral'
    
    positive_count = neutral_count = negative_count = 0.0
    
    for i in range(len(sentimentList)):
        if sentimentList[i] == 'Positive':
            positive_count += 1
        elif sentimentList[i] == 'Negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    avg_positive = positive_count/len(sentimentList)
    avg_negative = negative_count/len(sentimentList)
    avg_neutral = neutral_count/len(sentimentList)

    document_sentiment = mode(sentimentList)

    return avg_positive, avg_negative, avg_neutral, document_sentiment










    





        
        


    


    

if __name__=='__main__': main()

