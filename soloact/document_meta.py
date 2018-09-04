import xml.etree.ElementTree as ET 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import fnmatch

# Your path here
cc = '/Users/BhavishDaswani/Downloads/IDMT-SMT-GUITAR_V2/dataset1/'

def IDMT_annotation_reader(DOWNLOAD_PATH, overwrite = False):

    """
    READING ANNOTATION FILES FOR META DATA
    WRITING TRAIN / TEST SPLIT TO DATA.JSON FOR CONTROL

    https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html

    """
    # Operates at a subdirectory level

    GUITAR_TYPES = os.listdir(DOWNLOAD_PATH) # subdirectories  

    df_head=['audioFileName', 'instrument', 'recordingDate', 'pitch', 'onsetSec', 'offsetSec', 'fretNumber', 'stringNumber', 'excitationStyle', 'expressionStyle']
    columns=df_head
    meta=[]
    for model in GUITAR_TYPES:
        annotation_files = DOWNLOAD_PATH+"/"+ model +'/annotation' #tells where the annotation files are
        for path, dirs, all_files in os.walk(annotation_files):            
            for file in all_files:
                if file.endswith('.csv'):
                    continue
                tree = ET.parse(annotation_files+"/"+file)
                rows = []
                for child in tree.find('.//globalParameter'):
                    if not rows:
                        fullpath = DOWNLOAD_PATH + model + '/audio/' + child.text
                        rows.append(fullpath)
                        continue
                    rows.append(child.text) 
                rows2 = []
                for child in tree.find('.//event'):
                    rows2.append(child.text) 
                all_rows = rows+rows2
                meta.append(all_rows)

    df = pd.DataFrame(meta, columns=columns)
    
    print ('Omitting chords! Mono only!')
    df_mono = df[df['audioFileName'].str.contains('Major|Minor')==False]
    # reset index for an even 312 
    df_mono.reset_index(drop = True, inplace = True)
 
    print ('{} soundfiles available'.format(df_mono.index.max()))

    print ('Splitting data!')
    train_ix, test_ix = train_test_split(df_mono.index.values, random_state = 666)

    # These are full paths! dd
    training_audio = df_mono.loc[train_ix,:]['audioFileName']
    test_audio = df_mono.loc[test_ix,:]['audioFileName']

    print ('TRAIN: {} files | TEST: {} files'.format(len(training_audio), len(test_audio)))

    filedate = datetime.strftime(datetime.now(), '%Y-%m-%d')
    filename = 'data_' + filedate + '.json'

    with open(filename, 'w') as datafile:

        records = {'train' : training_audio.tolist(), 'test' : test_audio.tolist()}

        print ('Wrote to {}'.format(filename))

        json.dump(obj = records, fp = datafile, indent = 4)

    return df_mono

print (IDMT_annotation_reader(DOWNLOAD_PATH = cc).head())
