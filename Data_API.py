# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 01:45:06 2023

@author: Alex
"""

from flask import Flask, request, jsonify
import json, os

app = Flask(__name__)

if os.path.exists("Data.json"):
    with open("Data.json") as jin:
        Data = json.load(jin)
else:
    Data = {"7Z0X": {}, "6M0J": {}}

@app.route('/ProteinDesign', methods=['GET'])
def events():
   if request.method == 'GET':
       return jsonify(Data)

@app.route('/ProteinDesign/<string:RCSBCODE>', methods=['GET'])
def rcsbcode(RCSBCODE):
   if request.method == 'GET':
       return jsonify(Data.get(RCSBCODE))

           
@app.route('/ProteinDesign/<string:RCSBCODE>/<string:seq>', methods=['GET', 'POST'])
def individual_seq(RCSBCODE, seq):
   if request.method == 'GET':
       return jsonify(Data.get(RCSBCODE).get(seq))

   if request.method == 'POST':
       Data[RCSBCODE][seq] = request.json
       with open("Data.json", 'w') as jout: jout.write(json.dumps(Data, indent=4))
       return jsonify({"status": "success", "src": request.json["Source"]}), 201
           
if __name__ == '__main__':
   app.run(debug=True, use_reloader=False)