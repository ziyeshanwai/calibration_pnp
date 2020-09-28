from easydict import EasyDict as edict
import os
from Util.util import write_json
__C = edict()

cfg = __C

"""
Required
"""
# __C.output_path = r"\\192.168.20.63\ai\double_camera_data\2020-08-21\161240\output4"
# __C.config_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\AIFADATA"
# __C.actor_name = 'zhaoshan_nobs'
# __C.auto_tracked_path = r"\\192.168.20.63\ai\double_camera_data\2020-08-21\161240\AT"  # 跟踪点路径
# __C.img_root_path = r"\\192.168.20.63\ai\double_camera_data\2020-08-21\161240"

__C.output_path = r"\\192.168.104.99\double\2020-09-15\151104\output"
__C.config_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\AIFADATA"
__C.actor_name = 'caidonghao'
__C.auto_tracked_path = r"\\192.168.104.99\double\2020-09-15\151104\AT"  # 跟踪点路径
__C.img_root_path = r"\\192.168.104.99\double\2020-09-15\151104"

"""
No modification
"""
__C.distortion_matrix_xml = r"cameraParams.xml"  # 暂时没用到 畸变
__C.camera_instrinsics = r"./camera_matrix"  # 相机标定根目录 暂时没用
__C.config_file_path = os.path.join(__C.config_root_path, __C.actor_name)
__C.weights_list_path = __C.output_path
__C.blend_shape_path = os.path.join(__C.config_file_path, 'face_bs')  # blend shape 存放路径 name is fixed
__C.base_mesh_name = "head_geo.obj"
__C.base_face_mesh_name = "face_geo.obj"
__C.base_mesh_path = os.path.join(__C.config_file_path, 'basemesh')
__C.obj_ind = os.path.join(__C.base_mesh_path, "obj_ind.pkl")  # 人脸3D点路径 name is fixed
__C.weights_list_name = "weights_list.pkl"
__C.proposal_mouth_contour_path = os.path.join(__C.config_file_path, 'contour')  # 下唇轮廓线在面部的索引
__C.eye_left_up_contour_index = os.path.join(__C.proposal_mouth_contour_path, "left_eye_up_contour_0.pkl")  # 左上眼睛轮廓线索引
__C.eye_left_down_contour_index = os.path.join(__C.proposal_mouth_contour_path, "left_eye_down_contour_0.pkl")  # 左下眼睛轮廓线索引
__C.eye_right_up_contour_index = os.path.join(__C.proposal_mouth_contour_path, "right_eye_up_contour_0.pkl")  # 右上眼睛轮廓线索引
__C.eye_right_down_contour_index = os.path.join(__C.proposal_mouth_contour_path, "right_eye_down_contour_0.pkl")  # 右下眼睛轮廓线索引
__C.boundry_index = os.path.join(__C.proposal_mouth_contour_path, "boundry.pkl")
__C.camera_with_Intrinsics = False  # 是否使用提前标定好的相机参数
__C.triangle_mesh_path = os.path.join(__C.base_mesh_path, "face_geo_tri.obj")
__C.sort_mouth_ind_contour = False  # 是否对嘴唇轮廓线的顶点索引进行排序
__C.use_contour_constraint = True  # 是否使用嘴部轮廓线约束
__C.use_corrective_pose = True  # 是否使用修正pose优化
__C.use_key_points_constraint = False
__C.use_eye_sliding_contour = True
__C.use_default_bounds = True  # 是否使用默认边界
__C.log_camera_parameters = True  # 是否缓存相机参数

"""
Optional
"""
__C.corrective_pose_number = 0.999
__C.image_height = 640 / 2  # 图片的高
__C.image_width = 480 / 2  # 图案的宽
__C.log_img_num = 1  # 每隔多少次log一次图片
__C.max_iteration = 5000  # 轮廓线拟合的最大迭代次数
__C.bounds = [0, 1]  # 边界值
__C.pixel_error_thresh_hold = 40  # 光流误差像素 大于该误差不参与拟合
__C.not_include_key = []  # 不参与优化的点与AT工具对应
"""stage2 weights settings """
__C.weight_left_mouth_down_loss = 1/4  # 下嘴唇左侧投影loss权重
__C.weight_right_mouth_down_loss = 1/4  # 下嘴唇右侧投影loss权重
__C.weight_left_mouth_up_loss = 1/4  # 左侧图片上嘴唇内唇线投影loss
__C.weight_right_mouth_up_loss = 1/4  # 右侧图片上嘴唇内唇线投影loss
__C.weight_left_eye_left_up_loss = 1/8  #
__C.weight_left_eye_left_down_loss = 1/8  #
__C.weight_left_eye_right_down_loss = 1/8  #
__C.weight_left_eye_right_up_loss = 1/8  #
__C.weight_right_eye_left_up_loss = 1/8  #
__C.weight_right_eye_left_down_loss = 1/8  #
__C.weight_right_eye_right_down_loss = 1/8  #
__C.weight_right_eye_right_up_loss = 1/8  #
__C.regularization_contour = 30.0  # 轮廓线约束的正则权重

"""
compute corrective pose
"""
__C.weight_3d_constraint = 0.5  # 3d点拟合
__C.weight_projection = 0.0001  # 投影拟合
__C.weight_mouth_down_contour = 0.002
__C.weight_mouth_up_contour = 0.002
__C.weight_eye_up_contour = 0.002
__C.weight_eye_down_contour = 0.002

print("write json")
config_file_path = os.path.join(__C.output_path, 'config')  # config 存储路径
if not os.path.exists(config_file_path):
    os.makedirs(config_file_path)
write_json(os.path.join(config_file_path, "solver.config"), cfg)
