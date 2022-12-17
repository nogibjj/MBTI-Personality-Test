# MBTI Personality Test


This project 

![imgs](/imgs/MBTI_Test.svg)
## Introduction
### Run from github
create a seperate environment,clone this repo to your local computer,and run this in your terminal 
3
​To create a virtual environment, run the following command in the terminal:
```
python3 -m venv venv
```
To activate the virtual environment, run the following command in the terminal:
```
source venv/bin/activate
```
To install the dependencies, run the following command in the terminal:
```
pip install -r requirements.txt
```
more information about the dependencies can be found in the [requirements.txt](https://github.com/main/requirements.txt) file.
To run the app server, run the following command in the terminal:
```
python3 app.py
```
then you can input sentences describing yourself or your feelings and then the app will generate your most possible persenality as an output.
### Run from docker
```
docker run --rm -it -p 3000:3000 shunianchen/ids706:latest
```

## Model

We fine-tuned two models for this project, which are the [Bert-base](https://huggingface.co/Shunian/mbti-classification-bert-base-uncased) model and the [Roberta-base](https://huggingface.co/Shunian/mbti-classification-roberta-base) model. The corresponding model can be found using the link on Huggingface. 


## Background Infomation
The MBTI personality test, also known as the Myers-Briggs Type Indicator (MBTI), is a psychological assessment tool designed to measure an individual's psychological preferences in how they perceive the world and make decisions. It is based on the theory of psychological type developed by Carl Jung and the work of Isabel Briggs Myers and her mother, Katherine Briggs.

The MBTI assessment consists of a series of questions that ask about your preferences in a number of areas, including how you get your energy, how you process information, and how you make decisions. Based on your responses, the assessment assigns you a four-letter type that represents your personality type.

The four dimensions measured by the MBTI are:

- Extraversion (E) vs. Introversion (I): This dimension measures where you get your energy from. Extraverts tend to be more outgoing and enjoy being around people, while introverts are more reserved and get their energy from being alone or with a small group of people.

- Sensing (S) vs. Intuition (N): This dimension measures how you process information. Sensors tend to focus on the details and facts of a situation, while intuitives tend to focus on the big picture and the possibilities of a situation.

- Thinking (T) vs. Feeling (F): This dimension measures how you make decisions. Thinkers tend to be more logical and analytical in their decision-making, while feelers tend to be more emotion-driven and consider the impact of their decisions on others.

- Judging (J) vs. Perceiving (P): This dimension measures how you approach the outside world. Judgers tend to be more organized and structured in their approach, while perceivers tend to be more flexible and adaptable.

Once you have determined which style you prefer for each of the four dichotomies, you can figure out your four-letter type code. In Myers and Briggs’ system, the four letters of a personality type are the first initials of each of your prefererences. For example, someone with a preference for Extraversion, Intuition, Feeling and Judging would have the type “ENTJ.” A preference for Intuition is signified with the letter “N,” to avoid confusion with Introversion.
There are sixteen possible combinations of preferences, making up [16 total personality types.](https://www.truity.com/myers-briggs/4-letters-myers-briggs-personality-types)

## Dataset
![image](https://user-images.githubusercontent.com/90811429/208213986-7e6740ac-9c5a-4860-aedd-65c0c1a774f9.png)



### Data Augmentation



## Continuous Integration/Continuous Deployment

