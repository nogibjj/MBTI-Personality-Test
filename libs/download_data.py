import os
import kaggle

def download():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('datasnaek/mbti-type', path='/workspaces/MBTI-Personality-Test/Data', unzip=True)

if __name__ == '__main__':
    #if file does not exist, download it
    if not os.path.exists('../Data/mbti_1.csv'):
        download()
        