for l in tqdm(range(len(test))):
    try:
      i=test.iloc[l]
      k=pd.DataFrame(pred(i['utterance'],j)).iloc[0:1]
      k['utterance']=i['utterance']
      k.rename(columns={'intent':'intent_predicted'},inplace=True)
      k['intent']=i['intent']
      k['compare']=k['intent']==k['intent_predicted']
      out=pd.concat([out,k])
    except:
      pass