
prelabeling_map = {
    "smart_logistics": {               # 模型名称
        "Model_type": "yolov5",   # 模型类型
        "Num_classes": 3,             # 模型类别数
        "Score_thr": 0.3,               # 置信度阈值
        "Box_thr": 0.65,                # iou阈值
        "Fixed_scale": 1,               # 数据预处理方式
        "Color_space": "rgb",       # 输入颜色空间
        "Anchors": [[(9, 3), (6, 16), (26, 8)], [(15, 40), (32, 73), (63, 130)], [(91, 99), (190, 182),(339, 276)]], # 模型anchors
        "Used_classes": ["class1", "class3"],     # 使用了这个模型中哪几个类别
        "Class_index": [0, 2],                            # 使用的类别在classes中的索引
        "Parent": [None, None]                        # 使用的类别的父级
    },
    "car_tail": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(28, 47), (79, 78), (112, 142)], [(124, 309), (332, 128), (238, 211)], [(319, 294), (411, 347), (553, 345)]],
        "Used_classes": ["ContainerCarBackDoor"],
        "Class_index": [0], 
        "Parent": None 
    },
    "person": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(10, 22), (17, 39), (25, 61)], [(35, 96), (47, 79), (55, 135)], [(87, 115), (79, 169), (116, 183)]],
        "Used_classes": ["Passerby"],
        "Class_index": [0], 
        "Parent": None 
    },
    "plate": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(10, 3), (17, 6), (26, 8)], [(37, 10), (28, 15), (44, 13)], [(51, 23), (105, 38), (171, 91)]],
        "Used_classes": ["Plate"],
        "Class_index": [0], 
        "Parent": None 
    },
    "StagnantWater": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(45, 33), (93, 52), (89, 100)], [(157, 71), (162, 115), (222, 96)], [(135, 175), (390, 105), (408, 307)]],
        "Used_classes": ["StagnantWater"],
        "Class_index": [0], 
        "Parent": None
    },
    "Roller": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(12, 12), (16, 17), (21, 25)], [(32, 24), (27, 33), (36, 37)], [(32, 55), (48, 52), (61, 69)]],
        "Used_classes": ["Roller"],
        "Class_index": [0], 
        "Parent": None
    },
    "Bulk": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(11, 9), (17, 11), (14, 15)], [(23, 15), (19, 20), (31, 21)], [(25, 28), (40, 33), (63, 50)]],
        "Used_classes": ["Bulk"],
        "Class_index": [0], 
        "Parent": None
    },
    "Foreignmatter": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(21, 16), (27, 43), (69, 19)], [(70, 41), (48, 78), (143, 35)], [(117, 62), (94, 115), (189, 97)]],
        "Used_classes": ["Foreignmatter"],
        "Class_index": [0], 
        "Parent": None
    },
    "gage_det": {
        "Model_type": "yolov5",
        "Num_classes": 1,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(17, 62), (19, 130), (26, 160)], [(20, 221), (28, 216), (40, 184)], [(35, 248), (27, 322), (55, 344)]],
        "Used_classes": ["WaterGage"],
        "Class_index": [0], 
        "Parent": None
    },
    "gage_rec": {
        "Model_type": "yolov5",
        "Num_classes": 12,
        "Score_thr": 0.3,
        "Box_thr": 0.65,
        "Anchors": [[(14, 6), (19, 7), (22, 8)], [(27, 9), (24, 11), (32, 12)], [(29, 15), (37, 14), (52, 19)]],
        "Used_classes": ["GraduationCharacter_1", "GraduationCharacter_2", "GraduationCharacter_3", "GraduationCharacter_4", "GraduationCharacter_5", "GraduationCharacter_6", "GraduationCharacter_7", "GraduationCharacter_8", "GraduationCharacter_9", "GraduationCharacter_0", "GraduationCharacter_E", "GraduationCharacter_E1"],
        "Class_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        "Parent": ["WaterGage"]*12
    }
}