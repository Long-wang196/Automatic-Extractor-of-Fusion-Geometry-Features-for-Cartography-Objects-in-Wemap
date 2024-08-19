import joblib, json, os, ujson
import pandas as pd
import numpy as np


def ReaddataFromJSON(file_path_json):
    with open(file_path_json, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data

def determin_layer(saliency, S1, S2, S3, S4):
    if saliency > S1:
        return 'Layer 1'
    elif saliency > S2:
        return 'Layer 2'
    elif saliency > S3:
        return 'Layer 3'
    elif saliency > S4:
        return 'Layer 4'
    elif saliency > 0:
        return 'Layer 5'
    else:
        return 'None'

def update_threshold_layer(row, S1, S2, S3, S4):
    # 这里实现基于新的 S1 至 S4 值重新计算阈值层
    # 类似于训练阶段的 determin_layer 函数
    return determin_layer(row['Siginificance'], S1, S2, S3, S4)

# 2. 加载模型
def load_model(filename):
    saved_data = joblib.load(filename)
    return saved_data['model'], saved_data['label_encoder']

def predict_with_json_data(model, le, new_data):
    # 假设 new_data 已经预处理好，和训练数据格式一致
    predictions = model.predict(new_data)
    # 将数值标签解码为原始标签
    decoded_predictions = le.inverse_transform(predictions)
    return decoded_predictions

def save_via_layers(intial_data, predictions):
    pred_structure = {}
    threshold_structure = {}
    for i, pred in enumerate(predictions):
        predicted_layer = f'{pred}'  # 预测的层级
        threshold_layer = intial_data[i]['Threshold_Layer']  # 计算的 Threshold_Layer
        id_value = intial_data[i]['ID']
        coordinate = intial_data[i]['Coordinates']

        # 如果当前层不存在于输出结构中，则初始化该层
        if predicted_layer not in pred_structure:
            pred_structure[predicted_layer] = {}

        if threshold_layer not in threshold_structure:
            threshold_structure[threshold_layer] = {}

        # 将 ID 和 Coordinates 加入到该层中
        pred_structure[predicted_layer][f'ID{id_value}'] = coordinate
        # 将 ID 和 Coordinates 加入到 Threshold_Layer 中
        threshold_structure[threshold_layer][f'ID{id_value}'] = coordinate
    return pred_structure, threshold_structure

def write_json_file(json_data, file_path):

    try:
        with open(file_path, 'w') as f:
            ujson.dump(json_data, f, indent=4)
        print(f'存储数据为 {file_path} 完成')
    except (IOError, OSError) as e:
        print(f"File operation error: {e}")
    except (TypeError, ValueError) as e:
        print(f"Data serialization error: {e}")

def calculation_S_value(data):
    Siginicicance = [value['Siginificance'] for value in data.values()]

    # 从小到大排序Siginicicance值
    sorted_siginicicance = np.sort(Siginicicance)

    # 计算分割点
    quantiles = np.percentile(sorted_siginicicance, [40, 70, 90, 99])

    return quantiles[3], quantiles[2], quantiles[1], quantiles[0]

def processing_data(data, S1, S2, S3, S4):
    records = []
    for key, value in data.items():
        records.append({
            'ID': int(key[2:]),
            'Coordinates': value['coordinates'],
            'Siginificance': value['Siginificance'],
            'Visibility': value['Visibility'],
            'Geometry': value['Geometry'],
            'Size': value['Geometry'],
            'Threshold_Layer': update_threshold_layer(value, S1, S2, S3, S4)
        })
    return records

def main(class_file_needed, write_file_prediction, write_file_threshold):

    new_data = ReaddataFromJSON(class_file_needed)
    S1, S2, S3, S4 = calculation_S_value(new_data)
    records = processing_data(new_data, S1, S2, S3, S4)

    # 将提取出的特征传入模型进行预测
    X_new = pd.DataFrame(records)[['Siginificance']]

    # 确保输入数据是 2D 数组
    X_new = X_new.values.reshape(-1, X_new.shape[1])  # reshape 成 2D 数组
    model, le = load_model('Automatic_extractor.pkl')
    prediction = predict_with_json_data(model, le, X_new)

    prediction_layer_coor, threshold_layer_coor = save_via_layers(records, prediction)

    return (write_json_file(prediction_layer_coor, write_file_prediction),
            write_json_file(threshold_layer_coor, write_file_threshold))

if __name__ == "__main__":
    class_needed_file = 'Including_salienceAttribite.json'
    write_file_prediction = 'class_prediction.json'
    write_file_threshold = 'class_threshold.json'
    main(class_needed_file, write_file_prediction, write_file_threshold)