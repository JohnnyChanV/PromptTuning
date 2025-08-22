import pandas as pd
import json
from sklearn.model_selection import train_test_split

data = pd.read_excel("data_proc/highschoolAPEng_Comment_data.xlsx",sheet_name="coded for explanation")

data['input'] = data['Comment']
data['label'] = data['Explanation (human code)']
data['Dimension.Name'] = data['Dimension.Name'].apply(lambda x:x.lower())

records = data.to_dict('records')

dev,test = train_test_split(records,test_size=14228,random_state=42)

from collections import Counter
Counter([item['Dimension.Name'] for item in test])
Counter([item['Dimension.Name'] for item in dev])

from collections import Counter

Counter([item['Dimension.Name'] for item in records])

json.dump(test,open("data_proc/proc_test_data.json",'w'),ensure_ascii=False,default=str)
json.dump(dev,open("data_proc/proc_dev_data.json",'w'),ensure_ascii=False,default=str)