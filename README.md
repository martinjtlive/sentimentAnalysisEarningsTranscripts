# Summary
The project ingests earnings transcripts for a few CIQ companies. The earnings transcripts are in json format. Content is categorized in 1 of 3 types: Presentation, Questions, Question and Answer(s) units.
These categorized content is passed to OpenAI gpt-4 using the langchain framework to determine sentiment. 
Sentiment determination was on transcripts of 3 categories within an earnings call:
1.	Presentation – speeches by company executives
2.	Questions – raised by investors/analysts
3.	Question and Answer units – Each unit is a full dialogue of a question by an analyst followed by responses/rebuttals by company executive(s) to the question.

The UX is done via streamlit.

Input JSON source files are not in repository.
SSL certification of end point is not in repository. Issue relating to it can be solved via https://community.openai.com/t/ssl-certificate-verify-failed/32442/37?page=2
