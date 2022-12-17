![Python application test with Github Actions](https://github.com/nogibjj/MBTI-Personality-Test/tree/main/.github/workflows/main.yml/badge.svg)

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

This (MBTI) Myers-Briggs Personality Type Dataset dataset is collected by MITCHELL J, can be found in [kaggle for more detail ](https://www.kaggle.com/datasets/datasnaek/mbti-type)


### Data Augmentation

Because of the uneven data distribution, we use two types of methods to expansion our original dataset.
- **[nlpaug](https://github.com/makcedward/nlpaug)**
- **[Paraphraser](https://github.com/vsuthichai/paraphraser)**
Data augmentation is a technique that can be used to improve the accuracy of natural language processing (NLP) models by increasing the size and diversity of the training data. This can be particularly useful when there is limited annotated data available, as it allows the model to learn from a larger and more diverse set of examples.

There are a number of different techniques that can be used for data augmentation in NLP. Some common approaches include:

- Synonym replacement: This involves replacing certain words in a sentence with their synonyms, which can help the model learn the meaning of words in different contexts.

- Random insertion: This involves randomly inserting words into a sentence, which can help the model learn to handle variations in word order and sentence structure.

- Random swap: This involves randomly swapping the positions of two words in a sentence, which can help the model learn to handle variations in word order.

- Random deletion: This involves randomly deleting words from a sentence, which can help the model learn to handle missing or incomplete information.

By using data augmentation techniques like these, it is possible to increase the size and diversity of the training data, which can lead to improved accuracy for NLP models.

## Continuous Integration/Continuous Deployment

to set up CI/CD using GitHub and AWS Elastic Container Registry (ECR) is as follows:

- Set up a repository on GitHub that contains your codebase.

- Create a build pipeline in AWS CodePipeline that is triggered whenever a change is made to the repository on GitHub.

- Set up a build stage in the pipeline that uses AWS CodeBuild to build and test the code.

- Set up a deploy stage in the pipeline that uses AWS CodeDeploy to deploy the code to a staging environment.

- Set up a release process that allows you to manually promote the code from the staging environment to production, or to automatically promote the code based on certain conditions.

By following these steps, you can set up a CI/CD pipeline that automatically builds, tests, and deploys your code whenever changes are made to the repository on GitHub. This can help you to deliver code changes more frequently and with fewer errors, and can save you time by automating many of the manual tasks involved in the software development process.
