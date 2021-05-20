from flask import Flask
from string import Template
from flask import Flask, render_template, request
from flask import jsonify

from utils import *
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/utterance',methods=['POST'])
def utterance():
  if request.method=='POST':
    m = request.get_json()
    message=m['utterance']

    p=pred(message,tokenizer,MODEL)  
    res = {"intentList":[ {"intent":p["pred_values"],"confidence":p["top_p1"]},{"intent":p["predictions1"],"confidence":p["top_p2"]},{"intent":p["predictions2"],"confidence":p["top_p3"]}]}

  return jsonify(res)

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)