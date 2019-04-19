Pre-built usage:

1. Replace paths in the <mark>Project_main.py</mark> with your own local machine path.

2. Load video files by using local video path or mp4 urls in <mark>Project_main.py</mark>.

3. Run the Project_main.py and generate the srt file.

Note : Run the video file along with the srt file (VLC recommended).



Current Progress results:

Check the <mark>Test</mark> folder.




In Progress :

1. Prebuilt models for code codensation.

2. API support.

3. CNN for extracting emotions through images from videos.

4. Markov models for combining results.

Dependencies:

ffpmpeg

scikit-learn

keras



I. INTRODUCTION

Many companies use sentiment analysis to
understand their customers’ reactions to their
products. Primarily, these analyses have been
carried out on social media and text information.
However, recently companies have started using
multimodal analyses to understand their
customers’ reactions in much more details. This
project is based on bimodal sentimental analysis,
where we utilize textual as well as speech signals
to detect emotions. There are three parts to this
project: sentiment analysis on text generated from
speech, sentiment analysis on speech signal and
detection of emotions based on both.

II. RELATED WORK

There have been several studies related to
the extraction of important features from signals
to analyze sentiments. Babak Basharirad et al.
study of feature extraction involves extracting
features such Pitch, Energy, MFCC to analyze
sentiments [1]. Fakhri Karray et al. study on
feature extraction also involves using MFCC
features to identify sentiments in speech [2].

III. RESEARCH QUESTIONS

There are three primary questions we are looking
to address in this project:
1. We want to understand which features play an
important role in speech signals with regards to
emotions. For this, we want to look into the
theory to understand how human emotions and
vocal responses are related. We also want to investigate through experimentation with feature
selection.

2. We want to understand whether through textual
sentiment analysis, we can detect contextual
emotions associated with certain words, for
example, positive words used in a negative
sentence to convey sarcasm.

3. We also want to see if both textual and speech
sentiment analysis result can be combined to
provide better result than each used
independently.

IV. METHODOLOGY

This project identifies the emotion of users
based on speech. Currently, it is limited to four
emotions namely happiness, sadness, fear and
anger. The project is divided into three parts: The
first part involves converting the speech signals
to text. The emotion in the text is then identified
using a Support Vector Machine (SVM) classifier.
The second part involves extracting features
from signals. The emotions are then extracted
from the dataset of features using Support
Vector Machines (SVM) and Random Forest as
base models. The data is then trained using
Convolution Neural Network. Each speech signal
is broken into frames. Each frame will produce a
particular emotion from part 1 and part 2. A
voting algorithm will pick the highest occurring
sentiment and the sentiment is analyzed.

A. DATA

The training dataset which will be used for
emotion detection of audio is taken from
multiple sources. The datasets are as
follows:
“The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS) [6].”
The dataset contains 24 professional actors
(12 female and 12 male), vocalizing two
lexically matched statements in a neutral
North American accent. Speech includes 8
expressions - calm, happy, sad, angry,
fearful, surprise and disgust. It consists of 60
speeches (3 seconds each) per actor x 24
actors = 1440 audio speech files (16-bit, 48
kHz, .wav file) with total size of 215 MB. Each
file in the dataset has a unique name.
“Toronto Emotional Speech Set (TESS)”
collection [3] is a set of 200 target words
which were spoken in the carrier phrase "Say
the word _____' by two actresses (aged 26
and 64 years) and recordings were made of
the set portraying each of seven emotions
(anger, disgust, fear, happiness, pleasant
surprise, sadness, and neutral). There are
2800 stimuli in total.

“Berlin Database of Emotional Speech” [4]
contains about 500 utterances spoken by 10
actors in a happy, angry, anxious, fearful,
bored and disgusted way as well as in a
neutral version.
“CREMA-D (Crowd-sourced Emotional
Multimodal Actors Dataset)” [5] is a data set
of 7,442 original clips from 91 actors. These
clips were from 48 male and 43 female
actors between the ages of 20 and 74
coming from a variety of races and
ethnicities. Actors spoke from a selection of
12 sentences. The sentences were presented
using one of six different emotions (Anger,
Disgust, Fear, Happy, Neutral, and Sad) and
four different emotion levels (Low, Medium,
High, and Unspecified).
On the other hand, the training dataset
which will be used for emotion detection of
text is explained below:

“International Survey on Emotion
Antecedents and Reactions (ISEAR)” [7]
dataset contains survey data in which
student respondents, both psychologists
and non-psychologists, were asked to report
situations in which they had experienced all
of 7 major emotions (joy, fear, anger,
sadness, disgust, shame, and guilt). In each
case, the questions covered the way they
had appraised the situation and how they
reacted. The final dataset thus contained
reports on seven emotions each by close to
3000 respondents in 37 countries on all 5
continents.
For validation and testing, we are using a
part of the above dataset, randomly
selected. We are also using other audio and
video speeches to test this model. Though
we do not have any specific labels for that
data. We are evaluating our model on a
subjective understanding on the emotion of
that data.

B. MODELS

Given an audio file we first break it into 30
second frames using ffmpeg package. Each
frame is then converted to text using IBM
Watson speech to text API.
For detecting emotions through text, we first
cleaned the data (removing uppercase, stop
words, punctuation, etc.) and tokenized the
sentences into words. After this, we
performed stemming using Porter Stemmer
and vectorized the words which is required
for SVM model. Finally, these vectorized
words are fed to the SVM classifier for
classifying the corresponding emotions.

For emotion recognition from speech data,
we are using Librosa library in python to
extract audio features which are basically
categorized into spectral (related to
spectrum of audio), prosodic (stress and
intonation patterns of speech) and
qualitative (representing characteristics of
voice). Based on theory referred from
various papers, we have found that spectral
features such as Mel-frequency cepstral
coefficient (MFCCs), Pitch Chroma, Spectral
Centroid and Spectral Skewness/Contrast,
prosodic features such as Zero Crossing Rate
and Root Mean Square and Energy (RMSE),
and qualitative feature Mel-frequency play
an important role in recognition of
emotions. Using these and through
experimentations, we reduced the features
to hundred to be used in final training.

For feature selection, we have analyzed
correlation between the features and
removed highly correlated features. We
have also used Recursive Feature
Elimination (RFE) to select top 100 most
relevant features. This reduced the time to
train the model and provided slightly better
results in some models.
For training the model, we have used
Support Vector Classifier (SVC) and Random Forest to give us an idea of the accuracy.
Finally, the model was trained using
Convolution Neural Network with four
hidden layers. The test dataset was used as
validation set in the neural network. We
have used Keras library for the same.

The results from both text and speech
models are then combined using a priority
algorithm. Longer audio files are broken into
bits of 5-10 seconds and for each bit, we
predict emotion using text and speech. The
frequencies of emotions present in both the
arrays are calculated and priorities are
assigned to each of them. For the final result,
for each bit, emotions predicted through
text and speech are weighed and if found
conflicting, the emotion with higher priority
is selected, otherwise, it is taken as the final
emotion for that bit.

V. RESULTS

For extracting emotion from text, the model was
trained using 80% of the training data. We used
the rest and randomly selected 20% of the data
for testing the model. The same approach was
used for training the model which extracted
emotions from speech.
An accuracy of 69% was achieved using SVM
multiclass classifier while extracting emotions
from text.
For the model, which was used to extract
emotions from speech, the base model accuracy
for SVM multiclass classifier was 44% for
unscaled data and 61% for scaled data using all
the features except highly correlated. For
Random Forest, the highest accuracy was
achieved at 71% for data with reduced features.
The final training model gave us the highest
accuracy of 72% using Convolution Neural
Network.
For the final results, we used longer videos and
divided into 10 seconds bits. The result from both these were combined using the priority
algorithm, which produced satisfactory results.

VI. DISCUSSION

Although, we achieved an accuracy of around
70% in both the cases, this can be further
improved a lot. One of the major challenges in
this for us was the lack of availability of data
which had labelled emotions for text as well as
speech. For text, there are many datasets
available for sentiment analysis, but the number
is very limited for datasets linked with emotional
analysis. Similarly, there are very few data sets
which are available for speech and emotions,
and most of them have actors trained in very
controlled conditions.
In reality, the sound data can contain lot of
noises and text data can contain lot of slangs.
This will affect the accuracy of the model a lot.
Another major challenge was to find any dataset
which provided combined analysis of text and
speech in terms of emotion which would have
helped us to produce final result with better
accuracy.
Research into features which are most
significant for emotions in speech and isolation
of those can play a major role in improving the
accuracy of emotion recognition.

VII. CONCLUSION

In conclusion, we have made a successful
attempt in predicting emotions using both text
and speech and combining them to give a more
relevant result. While we limited the number of
emotions to four, a lot of different emotions can
be covered by analyzing relevant datasets. We
believe that a real time emotion detection
system, using this method which we attempted
can be very useful for lot of companies to analyze
the responses of their customers over calls or
other feedbacks.

VIII. REFERENCES

[1] https://aip.scitation.org/doi/pdf/10.1063/
1.5005438

[2] https://www.semanticscholar.org/paper/
Survey-on-speech-emotion-recognition%3AFeatures%
2C-and-Ayadi-Kamel/5eb2a1e6bcd22
f97373c38afda9556f27670d680

[3] https://tspace.library.utoronto.ca/handle/
1807/24487

[4] http://emodb.bilderbar.info/index-1024.ht
ml

[5] Cao H, Cooper DG, Keutmann MK, Gur RC,
Nenkova A, Verma R. CREMA-D: Crowdsourced
Emotional Multimodal Actors Dataset.
IEEE transactions on affective computing.
2014;5(4):377-390.
doi:10.1109/TAFFC.2014.2336244.

[6] Livingstone SR, Russo FA (2018) The Ryerson
Audio-Visual Database of Emotional Speech and
Song (RAVDESS): A dynamic, multimodal set of
facial and vocal expressions in North American
English. PLoS ONE 13(5): e0196391.
https://doi.org/10.1371/journal.pone.0196391.

[7] https://www.unige.ch/cisa/research/
materials-and-online-research/researchmaterial/

