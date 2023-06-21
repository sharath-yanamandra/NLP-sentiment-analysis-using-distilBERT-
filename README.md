# NLP-sentiment-analysis-using-distilBERT-
Designing my own architecture for distilBERT  with LSTM layer and fine-tuned on Reddit dataset.
#### I have taken the data from “https://files.pushshift.io/” website a couple of months ago. As push-shift is under violation, so they turned off Pushshift’s access to Reddit’s Data API from the previous month. Before it was open to access and then I downloaded my dataset.
![image](https://github.com/sharath-yanamandra/NLP-sentiment-analysis-using-distilBERT-/assets/30403425/0bde82bf-1707-420e-8296-fec818d8941e)

## STEP 1: Here, I will explain about the dataset, after downloading from the website I got Zip files containing JSON object files. {key: value} format.
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

## STEP 2: 
#### On this JSON file, we extracted only the useful information we needed and created a new dataset containing: “author”, “date”, and “clean_text”.I did a group by clear_text on the date column and sorted according to the month of the year.

## STEP 3: 
#### Saved this as a CSV file. With “month”, and “processed_text” as columns. Also did basic preprocessing of text before saving the file.

## STEP 4: 
#### On this new dataset, I implemented “SentimentIntensityAnalyser” from nltk.sentiment and calculated the polarity score of each row containing the text, and appended it to the original data frame. 

## STEP 5: 
#### Now the dataset has:  “month”, “processed_text”, “positive_score”, “negative_score”, “negative_score”. Later I considered this as my dataset and split into training, test and validation set for my distilBERT model. 

### BUT, distilBERT needs a dataset in the form of tensors, so I used distilBERT tokenizer to convert the input text into tokens containing {input_ids, attention_masks} and appended to the original dataset. all these steps I have written in the report. 
### Finally, I have the dataset as I wanted, now I split the data into train, val, test [70%, train] with the shape of [n rows, 3 columns]

## STEP 6: Model Architecture. 
### I created SentimentClassifier class with "Transformer layer" + "LSTM layer"+ "dropout layer" and "flatten layer".
pretrained_model_name = 'distilbert-base-uncased'
hidden_size = 768  # DistilBERT uses a hidden size of 768
num_classes = 3  # positive, negative, neutral
dropout_rate = 0.2

# *important : move the model and weights to GPU (if CUDA is available) it will save you time. 
model.to(device)
inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        labels = batch['label']

        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)

## STEP 7: After the model architecture, define dataloaders.
### Since, distilBERT will accept the data in the form of tensors, We need to define our dataloaders.

## STEP 8: training loop
### Define the optimizer and learning rate (hyper parameters)
learning_rate = 0.001
batch_size = 16
num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## STEP 9: Evaluation 
training and val sets
#### Epoch 1/5: Loss: 0.4033636321297714, Val F1-Score: 0.8859443882534721
#### Epoch 2/5: Loss: 0.2049401729239949, Val F1-Score: 0.8947587134433282
#### Epoch 3/5: Loss: 0.11445843069488183, Val F1-Score: 0.9057909193843998
#### Epoch 4/5: Loss: 0.08149987924917201, Val F1-Score: 0.8820809024397902
#### Epoch 5/5: Loss: 0.07761695404560305, Val F1-Score: 0.8950398208619635

Test set score: 
## Test F1-Score: 0.8850140739617232
![image](https://github.com/sharath-yanamandra/NLP-sentiment-analysis-using-distilBERT-/assets/30403425/07177cff-9df6-44b5-ae9b-a03b28aa2738)

# conclusion : I was able to achieve 88% F1 score and 90% accuracy. 
feel free to ask me if you have any questions. 
contact: sharathchandra172@gmail.com
