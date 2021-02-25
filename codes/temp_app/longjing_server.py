# -*- coding: utf-8 -*-
from flask import request, Flask
import json
import cv2
import numpy as np
import os
import threading
from datetime import datetime
from shutil import copyfile
import time
from lib.utils.status_code import Status
from lib.utils.except_err import err_log, try_except_form, pose_log
import base64
app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


##############################
def mark_dic_1(req):
    err = try_except_form(request, ["file", "filename"], [])
    if len(err) == 0:
        all_order = []
        try:
            # 接收图片
            upload_file = req.files['file']
            # 获取imageName
            filename = req.form['filename']
            file_name = "image/" + filename
            # 保存接收的图片到桌面
            upload_file.save(file_name)
            img = cv2.imread(file_name)
            dst = img

        except Exception as e:
            err = "mark_dic_1：", str(e)
            err_log("mark_dic_1", err)
            all_order = []
        if len(all_order) == 0 and False:
            info = {"code": Status.LOCAL_ERROR.get_code(),
                    "info": {"boxs": [{'box': [409, 159, 977, 1009], 'name': '1_0_8_21_31_0', 'display_name': '综保_2',
                                       'parameter': '1', 'angle': 0, 'scores': '0.9885338', 'id': 1, 'code': '0',
                                       'msg': '成功', 'error': 'success', 'display': '正常'}],
                             "max_map_tactics": {},
                             "min_name_dict": {}, "locals": []},
                    "error": err,
                    "msg": Status.LOCAL_ERROR.get_msg()}
        else:
            info = {"code": Status.OK.get_code(),
                    "info": {"boxs": [{'box': [409, 159, 977, 1009], 'name': '1_0_8_21_31_0', 'display_name': '综保_2',
                                       'parameter': '1', 'angle': 0, 'scores': '0.9885338', 'id': 1, 'code': '0',
                                       'msg': '成功', 'error': 'success', 'display': '正常'}],
                             "max_map_tactics": {},
                             "min_name_dict": {}, "locals": []},
                    "error": "success",
                    "msg": Status.OK.get_msg()}
    else:
        info = {"code": Status.PARAM_ILLEGAL.get_code(),
                "info": {},
                "error": err,
                "msg": Status.PARAM_ILLEGAL.get_msg()}
    return info


def mark_dic_2(req):
    global recognition
    err = try_except_form(request, ["file", "filename"], ["boxs"])
    errs = ""
    if len(err) == 0:
        all_order = []
        try:
            # 接收图片
            upload_file = req.files['file']
            # 获取imageName
            filename = req.form['filename']
            file_name = "image/" + filename
            # 保存接收的图片到桌面
            upload_file.save(file_name)
            img = cv2.imread(file_name)
            dst = img
            # 相机参数矫正
            info_dic = eval(req.form['info'])
            if len(all_order) == 0 and False:
                info = {"code": Status.PARAM_IS_NULL.get_code(),
                        "info": {"boxs": [], "max_map_tactics": "",
                                 "min_name_dict": "", "locals": []},
                        "error": err,
                        "msg": Status.PARAM_IS_NULL.get_msg()}
            else:

                info = {"code": Status.OK.get_code(),
                        "info": {
                            "boxs": [{'box': [409, 159, 977, 1009], 'name': '1_0_8_21_31_0', 'display_name': '综保_2',
                                      'parameter': '1', 'angle': 0, 'scores': '0.9885338', 'id': 1, 'code': '0',
                                      'msg': '成功', 'error': 'success', 'display': '正常'}],
                            "max_map_tactics": {},
                            "min_name_dict": {}, "locals": []},
                        "error": Status.OK.get_msg(),
                        "msg": "success"}
        except Exception as e:
            err = "info：", str(e)
            err_log("mark_dic_2", err)
            info = {"code": Status.PARAM_ILLEGAL.get_code(),
                    "info": {},
                    "error": err,
                    "msg": Status.PARAM_ILLEGAL.get_msg()}
            return info
    else:
        info = {"code": Status.LOCAL_ERROR.get_code(),
                "info": {},
                "error": err,
                "msg": Status.LOCAL_ERROR.get_msg()}
    return info


def mark_dic_3(req):
    info = {"code": Status.WAY_IS_NULL.get_code(),
            "info": {},
            "error": "error",
            "msg": Status.WAY_IS_NULL.get_msg()}
    return info


def mark_dic_4(req):
    global recognition
    errs = ""
    boxs_result = []
    err = try_except_form(request, ["file", "filename"],
                          ["max_map_tactics", "locals", "boxs"])
    if len(err) == 0:

        try:
            # 接收图片
            upload_file = req.files['file']
            # 获取imageName
            filename = req.form['filename']
            file_name = "image/" + filename
            # file_paths = os.path.join(file_path, file_name)
            # 保存接收的图片到桌面
            upload_file.save(file_name)
            img = cv2.imread(file_name)
            img_copy = img.copy()
            dst = img
            # 相机参数矫正
            info_dic = eval(req.form['info'])
        except Exception as e:
            err = str(e), "imageName:", file_name
            err_log("mark_dic_4", err)
            info = {"code": Status.PARAM_ILLEGAL.get_code(),
                    "info": {},
                    "error": err,
                    "msg": Status.PARAM_ILLEGAL.get_msg()}
            return info

        if len(boxs_result) == 0 and False:

            info = {"code": Status.LOCAL_ERROR.get_code(),
                    "info": {},
                    "error": "",
                    "msg": errs}
        else:
            info = {"code": Status.OK.get_code(),
                    "info": {"boxs": [{'box': [409, 159, 977, 1009], 'name': '1_0_8_21_31_0', 'display_name': '综保_2',
                                       'parameter': '1', 'angle': 0, 'scores': '0.9885338', 'id': 1, 'code': '0',
                                       'msg': '成功', 'error': 'success', 'display': '正常'}]},
                    "error": "success",
                    "msg": Status.OK.get_msg()}
    else:
        info = {"code": Status.LOCAL_ERROR.get_code(),
                "info": {},
                "error": err,
                "msg": Status.LOCAL_ERROR.get_msg()}
    return info

class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        """
        只要检查到了是bytes类型的数据就把它转为str类型
        :param obj:
        :return:
        """
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)
def mark_dic_5(req):
    err = try_except_form(request, ["file", "filename"],
                          ["max_map_tactics", "locals", "boxs"])
    errs = ""
    if len(err) == 0:
        pts = []
        file_name = ""
        max_map_tactics = []
        min_map_tactics = {}
        all_order = []
        boxs_result = []

        try:
            # 接收视频
            upload_file = req.files['file']
            # 获取视频名
            filename = req.form['filename']
            file_name = "image/" + filename
            # 保存接收的图片到桌面
            upload_file.save(file_name)
            # img = cv2.imread(file_name)
            # dst = img
            # 相机参数矫正
            info_dic = eval(req.form['info'])


        except Exception as e:
            err = str(e), "imageName:", file_name
            err_log("mark_dic_5", err)
            info = {"code": Status.PARAM_ILLEGAL.get_code(),
                    "info": {},
                    "error": err,
                    "msg": Status.PARAM_ILLEGAL.get_msg()}
            return info

        if len(boxs_result) == 0 and False:

            info = {"code": Status.LOCAL_ERROR.get_code(),
                    "info": {"boxs": [], "max_map_tactics": "",
                             "min_name_dict": "", "locals": []},
                    "error": err,
                    "msg": Status.LOCAL_ERROR.get_msg()}
        else:

            info = {"code": Status.OK.get_code(),
                    "info": {"boxs": [{'box': [409, 159, 977, 1009], 'name': '1_0_8_21_31_0', 'display_name': '综保_2',
                                       'parameter': '1', 'angle': 0, 'scores': '0.9885338', 'id': 1, 'code': '0',
                                       'msg': '成功', 'error': 'success', 'display': '正常'}]},
                    "error": "success",
                    "msg": Status.OK.get_msg()}
    else:
        info = {"code": Status.LOCAL_ERROR.get_code(),
                "info": {},
                "error": err,
                "msg": Status.LOCAL_ERROR.get_msg()}
    return info


### 构造一个用户判断的字典  4、5机器人、吊柜使用，14、15固定摄像头使用#####
ysf = {'1': mark_dic_1, '2': mark_dic_2, '3': mark_dic_3, '4': mark_dic_4, '5': mark_dic_5}


def func(fun, req):
    return (ysf.get(fun)(req))


@app.route("/upload_tactics", methods=['POST'])
def upload_tactics():
    '''
    code:0为成功，其他失败，
    info：返回信息
    {
        type_id:吊轨机器人1，巡检机器人0
    }
    '''
    json_dic = {}
    global post_time
    post_time = time.time()
    # get_data = json.loads(request.get_data(as_text=True))
    # time.sleep(60*10)
    err = try_except_form(request, ["mark_id", "info"], [])
    if len(err) == 0:
        try:
            type_err = try_except_form(request, ["type_id"], [])
            if len(type_err) == 0:
                type_id = request.form['type_id']
            else:
                type_id = 0
            mark_id = request.form['mark_id']
            if str(type_id) == "1" and mark_id != "1" and mark_id != "2":
                is_manual = check_way(request)
                if is_manual:
                    mark_id = "1" + mark_id
            # 返回排序后一维给前段：all_order，策略算法：all_rank，表盘中包含指示灯策略算法:min_name_dict,裁减坐标local_list
            info_dic = func(str(mark_id), request)
            print(str(mark_id), request.form['filename'], "接口：", request.form['info'])
            print("接口：", request.form['info'])
            # print(str(mark_id), "接口：", request.form['filename'])
            json_dic = json.dumps(info_dic, ensure_ascii=False)
        except Exception as e:
            err = "Method：", str(e)
            err_log("upload_tactics", err)
            info_dic = {"code": Status.PARAM_ILLEGAL.get_code(),
                        "info": {},
                        "error": err,
                        "msg": Status.PARAM_ILLEGAL.get_msg()}
            json_dic = json.dumps(info_dic, ensure_ascii=False)
    else:
        info_dic = {"code": Status.PARAM_ILLEGAL.get_code(),
                    "info": {},
                    "error": err,
                    "msg": Status.PARAM_ILLEGAL.get_msg()}
        json_dic = json.dumps(info_dic, ensure_ascii=False)
    # print("info_dic:", str(info_dic), time.time() - post_time)
    return json_dic

@app.route("/upload_file", methods=['POST'])
def upload_file():
    '''
    code:0为成功，其他失败，
    info：返回信息
    {
        type_id:吊轨机器人1，巡检机器人0
    }
    '''
    json_dic = {}
    global post_time
    post_time = time.time()
    # get_data = json.loads(request.get_data(as_text=True))
    # time.sleep(60*10)
    err = ""
    if len(err) == 0:
        try:
            # 接收图片
            upload_file = request.files['file']
            # 获取imageName
            filename = request.form['filename']
            file_name = filename
            # file_paths = os.path.join(file_path, file_name)
            # 保存接收的图片到桌面
            upload_file.save(file_name)
            os.system(f'python3 -m demucs.separate --dl -n demucs {file_name}')
            pre = file_name.split('.')[0]
            f1 = f'separete/{pre}/bass.wav'
            with open(f1, 'rb') as f:
                res1 = base64.b64encode(f.read())
            f2 = f'separete/{pre}/other.wav'
            with open(f2, 'rb') as f:
                res2 = base64.b64encode(f.read())
            f3 = f'separete/{pre}/drums.wav'
            with open(f3, 'rb') as f:
                res3 = base64.b64encode(f.read())
            f4 = f'separete/{pre}/vocal.wav'
            with open(f4, 'rb') as f:
                res4 = base64.b64encode(f.read())

            info_dic = {"code": Status.OK.get_code(),
                        "info": {"mp3":"网址","":"","":"","":"",},
                        "audio1":res1,
                        "audio2":res2,
                        "audio3":res3,
                        "audio4":res4,
                        "error": err,
                        "msg": Status.PARAM_ILLEGAL.get_msg()}
            # print(str(mark_id), "接口：", request.form['filename'])
            json_dic = json.dumps(info_dic, ensure_ascii=False,cls=MyEncoder,indent=4)
        except Exception as e:
            err = "Method：", str(e)
            err_log("upload_tactics", err)
            info_dic = {"code": Status.PARAM_ILLEGAL.get_code(),
                        "info": {},
                        "error": err,
                        "msg": Status.PARAM_ILLEGAL.get_msg()}
            json_dic = json.dumps(info_dic, ensure_ascii=False)
    else:
        info_dic = {"code": Status.PARAM_ILLEGAL.get_code(),
                    "info": {},
                    "error": err,
                    "msg": Status.PARAM_ILLEGAL.get_msg()}
        json_dic = json.dumps(info_dic, ensure_ascii=False)
    # print("info_dic:", str(info_dic), time.time() - post_time)
    return json_dic
@app.route("/panel_list", methods=['POST'])
def panel_list():
    info = {"code": "0", "info": [{"name": "1_0_0_1_53_0", "display_name": "电流表_小_100A"},
                                 {"name": "1_0_0_1_50_0", "display_name": "电流表_小_200A"},
                                 {"name": "1_0_0_1_10_0", "display_name": "电流表_小_250A"},
                                 {"name": "1_0_0_1_8_0", "display_name": "电流表_小_400A"},
                                 {"name": "1_0_0_1_51_0", "display_name": "电流表_小_500A"},
                                 {"name": "1_0_0_1_12_0", "display_name": "电流表_小_75A"},
                                 {"name": "1_0_0_1_11_0", "display_name": "电流表_小_800A"},
                                 {"name": "0_0_0_1_17_0", "display_name": "电流表_中_1.5KA"},
                                 {"name": "0_0_0_1_16_1", "display_name": "电流表_中_1200A_特殊"},
                                 {"name": "0_0_0_1_19_0", "display_name": "电流表_中_1KA"},
                                 {"name": "0_0_0_1_18_0", "display_name": "电流表_中_2.5KA"},
                                 {"name": "0_0_0_1_14_1", "display_name": "电流表_中_200A_特殊"},
                                 {"name": "0_0_0_1_9_0", "display_name": "电流表_中_2KA"},
                                 {"name": "0_0_0_1_15_0", "display_name": "电流表_中_600A"},
                                 {"name": "0_0_0_1_52_0", "display_name": "电流表_中_800A_特殊"},
                                 {"name": "1_0_0_1_22_0", "display_name": "电压表_小_450V"},
                                 {"name": "0_0_0_1_23_0", "display_name": "电压表_中_12KV"},
                                 {"name": "0_0_0_1_22_0", "display_name": "电压表_中_450V"},
                                 {"name": "1_0_3_22_44_0", "display_name": "高压带电显示器"},
                                 {"name": "0_0_0_40_1_0", "display_name": "空开_1"},
                                 {"name": "0_0_0_14_0_0", "display_name": "数显_低压无功综合测控装置"},
                                 {"name": "0_0_0_16_0_0", "display_name": "数显_干式变压器电脑温控箱"},
                                 {"name": "0_0_0_15_0_0", "display_name": "数显_直流屏"},
                                 {"name": "0_0_0_13_0_0", "display_name": "数显_ANTHONE"},
                                 {"name": "0_0_0_30_2_1", "display_name": "特殊旋转开关_2_1"},
                                 {"name": "0_0_0_20_1_0", "display_name": "特殊指示灯"},
                                 {"name": "0_0_0_30_2_0", "display_name": "旋转开关_2"},
                                 {"name": "0_0_0_30_3_0", "display_name": "旋转开关_3"},
                                 {"name": "0_0_0_30_4_0", "display_name": "旋转开关_4"},
                                 {"name": "0_0_0_30_7_0", "display_name": "旋转开关_7"},
                                 {"name": "0_0_0_28_0_0", "display_name": "压板"},
                                 {"name": "0_0_0_20_0_0", "display_name": "指示灯"},
                                 {"name": "1_0_6_21_40_0", "display_name": "综保_充电模块"},
                                 {"name": "1_0_6_21_39_0", "display_name": "综保_电池监控"},
                                 {"name": "1_0_4_21_39_0", "display_name": "综保_断路器"},
                                 {"name": "1_0_6_21_41_0", "display_name": "综保_逆变器_b02"},
                                 {"name": "1_0_8_21_37_0", "display_name": "综保_2AE"},
                                 {"name": "1_0_8_21_38_0", "display_name": "综保_3AE"}], "error": "success", "msg": "成功"}
    return json.dumps(info, ensure_ascii=False)
    # global recognition
    # err = try_except_form(request, ["display_name"], [], is_save=False)
    # display_name = ''
    # if len(err) == 0:
    #     display_name = request.form['display_name']
    # try:
    #     category_index = []
    #     for v in recognition.name_list:
    #         if display_name in v["display_name"]:
    #             my_name = v["name"]
    #             my_display_name = v["display_name"]
    #             category_index.append({"name": my_name, "display_name": my_display_name})
    #     info = {"code": Status.OK.get_code(),
    #             "info": category_index,
    #             "error": "success",
    #             "msg": Status.OK.get_msg()}
    # except Exception as e:
    #     err = "panel_list：", str(e)
    #     err_log("panel_list", err)
    #     info = {"code": Status.PARAM_ILLEGAL.get_code(),
    #             "info": [],
    #             "error": err,
    #             "msg": Status.PARAM_ILLEGAL.get_msg()}
    # return json.dumps(info, ensure_ascii=False)


# def clear_image_func():
#     global post_time
#     dt = datetime.now()  # 创建一个datetime类对象
#     if dt.hour == 0 and int(time.time() - post_time) >= 3600:
#         dirPath = "image/"
#         for file in os.listdir(dirPath):
#             os.remove(dirPath + file)
#     # 60*60
#     t = threading.Timer(3600, clear_image_func)
#     t.start()


def main():
    # global recognition
    # recognition = Recognition("models_longyuan")
    # global post_time
    post_time = time.time()
    # clear_image_func()

    app.run("0.0.0.0", port=9898, debug=True)  # 端口为8081


if __name__ == "__main__":
    main()
