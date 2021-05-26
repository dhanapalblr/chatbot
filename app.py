from flask import Flask
from string import Template
from flask import Flask, render_template, request
from flask import jsonify

from utils import *
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/utterance',methods=['POST'])
def utterance():
    if request.method=='POST':
        m = request.get_json()
        text=m['utterance']

    output=[]
    for j in ['iris','lcs','hiri']:
        k=pred(text,j)
        [output.append(h) for h in k]

    res={'intentList':sorted(output,key=lambda x: x['confidence'],reverse=True)}

    return jsonify(res)

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port,debug=False)
	#optional if we want to run in debugging mode
	#app.run(debug=True)