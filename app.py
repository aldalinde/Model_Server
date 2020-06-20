from flask import Flask, request, jsonify
import xgboost as xgb


from proccess_data import get_dmatrix

# for logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

app = Flask(__name__)

handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# decorator pointing to root route указывает на корневой маршрут
@app.route("/")
# маршрут будет обрабатываться функцией:
def index():
    return "API for predict service"



# передача методом POST на страницу with route /predict запрос JSON
@app.route("/predict", methods=['POST'])
#sending a JSON file for proccessing
def predict():
    json_input = request.json


    model = xgb.Booster()
    model.load_model('D:/AI/Machine learning/models/xgb_claimcount.model')

    # request logging
    current_datatime = strftime('[%Y-%b-%d %h:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()


    id = json_input['ID']
    d_matrix = get_dmatrix(json_input)

    xgb_prediction = model.predict(d_matrix)

    result = {
        'ID': id,
        'result': int(xgb_prediction)
    }

    # response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %h:%M:%S]')
    logger.info(f'{current_datatime}, predicted for {duration} msec: {result}\n')

    return jsonify(result)


@app.errorhandler(Exception)
def exeptions(e):
    current_datatime = strftime('[%Y-%b-%d %h:%M:%S]')
    error_message = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                 current_datatime,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 error_message)
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)