# NLP-sentiment-analysis-using-distilBERT-
Designing my own architecture for distilBERT  with LSTM layer and fine tuned on reddit dataset.
#### I have taken the data from “https://files.pushshift.io/” website a couple of months ago. As push-shift is under violation, so they turned off Pushshift’s access to Reddit’s Data API from the previous month. Before it was open to access and then I downloaded my dataset.
![image](https://github.com/sharath-yanamandra/NLP-sentiment-analysis-using-distilBERT-/assets/30403425/0bde82bf-1707-420e-8296-fec818d8941e)

#### STEP 1: Here, I will explain about the dataset, after downloading from the website I got Zip files containing JSON object files. {key: value} format.
The structure of JSON is as follows: this is only for a single subreddit comment. 
{
“date”      : “02/12/2022 “
“Clean_text”: “if she doesn't thaw after this it's on her I hope she will “
“link_id”     : t3_z9x304 
“Parent_id”  :  t1_iylz06o 
“id”         : iym31cf The comment’s identifier e.g. “dbumnq8” (String). 
“author”    : “trishsf”
“Created_utc” : 1669983305
“Num_comments” : None
“Over_18”   :  None 
“Is_self”     :  None 
“Score”     : 1 
“Selftext”     :  None 
“Stickied”    :  FALSE 
“Subreddit”   : MomForAMinute 
“Subreddit_id” : t5_3g7sw 
“Title”         : None
}

#### STEP 2: On this JSON file, we extracted only the useful information we needed and created a new dataset containing: “author”, “date”, and “clean_text”.I did a group by clear_text on the date column and sorted according to the month of year.
#### STEP 3: Saved this as CSV file. With “month”, and “processed_text” as columns. Also did basic preprocessing of text before saving the file.
#### STEP 4: On this new dataset, I implemented “SentimentIntensityAnalyser” from nltk.sentiment and calculated polarity score of each row containing the text, and appended it to the original data frame. 
#### STEP 5: Now the dataset is have:  “month”, “processed_text”, “positive_score”, “negative_score”, “negative_score”. Later I considered this as my dataset and split into training, test and validation set for my distilBERT model. 

