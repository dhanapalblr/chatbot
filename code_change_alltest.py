for j in ['data','easre','ibgpdts','riskgov','iris','hiri','lcs']:
  out=pd.DataFrame()
  print("starting prediction for {}".format(j))
  for l in tqdm(range(len(test_data))):
    try:
      i=test_data.iloc[l]
      k=pd.DataFrame(pred(i['utterance'],j))
      k['utterance']=i['utterance']
      k['Order of confidence scores']=[1,2,3]
      k.rename(columns={'intent':'intent_predicted'},inplace=True)
      k['intent']=i['intent']
      k['intent1']=i['intent1']
      out=pd.concat([out,k])
    except:
      pass

  print("completed prediction for {}. Saving the file..".format(j))
  out.to_csv('All_models_test_output_{}.csv'.format(j),index=None)
