import nlpaug.augmenter.word as naw
import pandas as pd
import csv
import argparse
import numpy as np

aug = naw.SynonymAug(aug_src='wordnet')


traindata = pd.read_csv("knowBERT.csv")
savefile='knowBERT_syn.csv'
final_prompt,final_emotion = [],[]
headers = ['text', 'label']
rows = []
words=[]
labels=[]
texts=[]
textlabel=[]
confusion=[]
for j,text in enumerate(traindata["text"]):
	texts.append(text)
	textlabel.append(traindata["label"][j])
for i,val in enumerate(traindata["label"]):
	st = []
	cause=traindata["text"][i].replace(f"á\r\n","")
	augmented_cause = aug.augment(cause)
	if augmented_cause not in words and augmented_cause not in confusion:
		words.append(augmented_cause)
		labels.append(val)
		st.append(augmented_cause)
		st.append(val)
		rows.append(st)
	elif augmented_cause not in texts and augmented_cause in words and augmented_cause not in confusion:
		texts.append(augmented_cause)
		textlabel.append(val)
	elif augmented_cause in texts and augmented_cause in words and augmented_cause not in confusion:
		if textlabel[texts.index(augmented_cause)] !=labels[words.index(augmented_cause)]:
			print(augmented_cause,textlabel[texts.index(augmented_cause)], 'but has in: ', labels[words.index(augmented_cause)])
			rows.pop(rows.index([augmented_cause,labels[words.index(augmented_cause)]]))
			confusion.append(augmented_cause)

	else:
		print('else: ',augmented_cause, val)
print(confusion)

with open(savefile, 'w', encoding='utf-8') as f:
    f_csv = csv.writer(f)
    # f_csv.writerow(headers)
    f_csv.writerows(rows)

	# final_prompt.append(traindata["text"][i].replace(f"á\r\n",""))
	# final_emotion.append(aman_label_dict[val])
