# 将检测结果进行形式转换：字符串转换成字典数据；同时，将关键数据b_box坐标数值化
def parse_result(predict_result):
    #result_dict = json.loads(predict_result) #在线模型结果直接是dict类型，不需要转换
    result_dict = predict_result
    b_box_num = len(result_dict)
    result_dict_new = {}
   
    b_boxes = []
    scores = []
    class_name = []

    for i in range(b_box_num):
        
        brand_str = str(result_dict[i][0], encoding = "utf-8")
        class_name.append(brand_str)
        scores.append(result_dict[i][1])
        
       # center_x,center_y,width,height
        x1_min = int(result_dict[i][2][0] - result_dict[i][2][2]/2)
        y1_min = int(result_dict[i][2][1] - result_dict[i][2][3]/2) #left-top 

        x1_max = int(result_dict[i][2][0] + result_dict[i][2][2]/2)
        y1_max = int(result_dict[i][2][1] + result_dict[i][2][3]/2)
        b_boxes.append((x1_min,y1_min,x1_max,y1_max))
        
    result_dict_new["b_box_num"] = b_box_num
    result_dict_new["detection_boxes"] = b_boxes 
    result_dict_new["detection_classes"] = class_name
    result_dict_new["detection_scores"] = scores
    
    return result_dict_new


# 定义一个函数，根据车牌号查询比对车辆信息库（模拟车管所车辆登记信息库），判定是否是套牌车（车牌号和车辆品牌信息对应不上，后续可再增加车辆颜色信息）
def isFakePlate(inputCarInfo, carInfoDatabase):
    carBrandList = []
    isFakePlateCar = False
    trueCarBrand = ''
    plateNo = inputCarInfo[0]
    carBrand = inputCarInfo[1]
    if carBrand == '其他':
        isFakePlateCar = False
        trueCarBrand = 'Null'
    else:
        result = carInfoDatabase[(carInfoDatabase['plateNo']==plateNo)] # 从车管所数据库中拉出车牌号对应的车辆信息，保存到result中
        if len(result) > 0:
            carBrandList = result['carBrand'].values  # list结构 
           
            if (carBrand == carBrandList[0]):
                # print(carBrand, "==", carBrandList[0])
                isFakePlateCar = False
                trueCarBrand = carBrandList[0]
            else:
                # print(carBrand, "!=", carBrandList[0])
                isFakePlateCar = True
                trueCarBrand = carBrandList[0]
        else:
            isFakePlateCar = False
            trueCarBrand = 'Null'
    return isFakePlateCar, trueCarBrand

