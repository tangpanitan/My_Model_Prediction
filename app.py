from flask import Flask,request,jsonify
import json
import keras
import cv2
import base64
import numpy as np
app = Flask(__name__)
#CORS(app)
model = keras.models.load_model('1-MV2-weight-fold 4-37-0.99.h5')
rimg = []

#response = json.loads(open('./result.json', 'r').read())
response = json.loads(open('string-to-json-online.json',encoding='utf-8').read())

@app.route('/post', methods =['POST'])
def hello_post():

    value = request.form['value']

    jpg_original = base64.b64decode(value)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    img = cv2.resize(img, (224, 224))

    rimg = np.array(img)
    rimg = rimg.astype('float32')
    rimg /= 255

    rimg = np.reshape(rimg, (1, 224, 224, 3))

    predict = model.predict(rimg)

    label = ['งูทับสมิงคลา', 'งูสามเหลี่ยม', 'งูเกอะลอหัวศร', 'งูเขียวพระอินทร์', 'งูลายสาบคอแดง', 'งูเขียวหางไหม้']
    result = label[np.argmax(predict)]

    num_predict = np.argmax(predict)

    obj = 'snake_' + str(num_predict+1)
    #name = response[obj]['name']
    detail_1 = response[obj]['detail_1']
    detail_2 = response[obj]['detail_2']
    detail_3 = response[obj]['detail_3']


    data = [{"id": str(num_predict),
             "name": result,
             "detail_1": detail_1,
             "detail_2": detail_2,
             "detail_3": detail_3
             }
            ]

    print(result)

    return jsonify(data)
# route http posts to this method

if __name__ == "__main__":
    app.run(host='0.0.0.0')