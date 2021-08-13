import json
import requests
import pandas as pd

api_url = 'https://10.91.135:8543/speechToTextService/ws/hkbot/processUserIntent'

df=pd.read_csv('All_models_test_output.csv')

res1=pd.DataFrame()

for i in df['utterance']:
    create_row_data = {'requestId': '11082021_testing',
                        'conversation_id':'11082021_testing',
                        'inputText':"test_common_intent_preocesson "+str(i),
                        'channelName':'TEAMS',
                        'regTemp':'R',
                        'inputLanguage':'EN',
                        'containsChinese':'false',
                        'country':'SG',
                        'notificationId':124,
                        'oneBankId':'testing'
                        }

    print("querying for utterance {}".format(i))
    r = requests.post(url=api_url, data=create_row_data)
    res={}
    res['status']=[r.status]
    res['code']=[r.code]
    res['requestId']=[r.requestId]
    res['intent']=[r.intent]
    res['translatedResponse']=[r.translatedResponse]
    res['tokens']=[r.tokens]
    res['channelName']=[r.channelName]
    res['conversation_id']=[r.conversation_id]
    res1=pd.concat([res1,pd.DataFrame(res)])

res1.to_excel('output_api.xlsx',index=None)

