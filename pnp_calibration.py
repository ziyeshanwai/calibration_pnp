from config.config import cfg
from Util.util import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Cache:
    def __init__(self):
        self.left_camera_ind = []
        self.right_camera_ind = []
        self.middle_camera_ind = []
        self.left_camera_array = None
        self.right_camera_array = None
        self.ind = None
        self.obj_index = None
        self.get_index()

    def get_index(self):
        """
        获取索引参与目标约束优化的索引2D坐标点索引以及3D坐标点索引
        :return:
        """
        self.ind = np.array(load_pickle_file(cfg.obj_ind))  # 这里后续要改成三角坐标
        self.obj_index = self.ind  # obj 上添加优化的点序
        self.left_camera_array = np.zeros((len(self.ind), 2), dtype=np.float32)  # 左侧相机value存储
        self.right_camera_array = np.zeros((len(self.ind), 2), dtype=np.float32)  # 右侧相机value存储

    def json2array(self, data):
        """
        将json 数据转换为程序需要的numpy 数据 加载左右相相机检测的关键点
        """
        right_points_kv = data['data']['camera2']['marker']
        left_points_kv = data['data']['camera1']['marker']
        self.left_camera_ind = []
        for key in left_points_kv.keys():

            if key not in cfg.not_include_key:
                self.left_camera_ind.append(int(key))
                self.left_camera_array[int(key), :] = left_points_kv[key]
        # self.right_camera_array = np.zeros((cfg.all_points_number, 2), dtype=np.float32)
        self.right_camera_ind = []
        for key in right_points_kv.keys():
            if key not in cfg.not_include_key:
                self.right_camera_ind.append(int(key))
                self.right_camera_array[int(key), :] = right_points_kv[key]
        self.left_camera_ind = sorted(self.left_camera_ind)
        print("self.left_camera_ind is {}".format(self.left_camera_ind))
        self.right_camera_ind = sorted(self.right_camera_ind)
        print("self.right_camera_ind is {}".format(self.right_camera_ind))
        self.middle_camera_ind = sorted(list(set(self.left_camera_ind).intersection(set(self.right_camera_ind))))


def convert_rotation_tran_to_projection_matrix(K, D, rotation_vector, translation_vector):
    """

    :param K: 相机内参projection
    :param D: 相机畸变系数
    :param rotation_vector: 旋转向量
    :param translation_vector: 平移向量
    :return: 投影矩阵
    """
    rmat, _ = cv2.Rodrigues(rotation_vector)
    cameraPosition = -np.matrix(rmat).T * np.matrix(translation_vector)
    I = np.identity(3)
    I1_extended = np.hstack((I, -cameraPosition))
    P_cam = K.dot(rmat).dot(I1_extended)  # 相机的投影矩阵 建立世界坐标和图像坐标的关系
    return P_cam


if __name__ == '__main__':
    fx = 1.3882728533305058e+03
    fy = 1.3397439280840849e+03
    cx = 6.3287690286970792e+02
    cy = 5.1534858523019784e+02

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([-1.0783564334083487e-01, 2.8610485412572417e-01, 6.6166437051155104e-03, 1.4952919843722928e-03, 0], dtype=np.float32)
    auto_tracked_path = cfg.auto_tracked_path
    json_names = sorted(os.listdir(auto_tracked_path))
    start = 0
    cache = Cache()
    output = cfg.output_path
    face_v, face_f = loadObj(os.path.join(cfg.config_root_path, cfg.actor_name, 'basemesh', 'face_geo.obj'))
    face_v_array = np.array(face_v, dtype=np.float32)
    mesh_homo = np.hstack((face_v_array, np.ones([face_v_array.shape[0], 1])))
    objPoints = face_v_array[cache.obj_index, :]
    output_path = os.path.join(cfg.output_path, 'debug')
    if not os.path.exists(output_path):
        os.makedirs(os.path.join(cfg.output_path, 'debug'))
    for i in range(start, len(json_names)):
        if os.path.exists(os.path.join(auto_tracked_path,
                                       "{}".format(json_names[i]))):
            data = load_json(os.path.join(auto_tracked_path, "{}".format(json_names[i])))
            data = formate_at(data)
            cache.json2array(data)
            left_objPoints = objPoints[cache.left_camera_ind, :]
            left_imagePoints = cache.left_camera_array[cache.left_camera_ind, :]
            right_objPoints = objPoints[cache.right_camera_ind, :]
            right_imagePoints = cache.right_camera_array[cache.right_camera_ind, :]

            (flag, left_rotation_vector, left_translation_vector) = cv2.solvePnP(left_objPoints, left_imagePoints, K, D)
            # (_, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(left_objPoints, left_imagePoints, K, D, cv2.SOLVEPNP_ITERATIVE)
            left_P_cam = convert_rotation_tran_to_projection_matrix(K, D, left_rotation_vector, left_translation_vector)

            (flag, right_rotation_vector, right_translation_vector) = cv2.solvePnP(right_objPoints, right_imagePoints, K, D)
            # (_, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(left_objPoints, left_imagePoints, K, D, cv2.SOLVEPNP_ITERATIVE)
            right_P_cam = convert_rotation_tran_to_projection_matrix(K, D, right_rotation_vector, right_translation_vector)

            # img_pro = P_cam.dot(mesh_homo.T)
            # img_coor = (img_pro/img_pro[2, :]).T
            left_img = cv2.imread(os.path.join(cfg.img_root_path, 'c1_rot', "{}.jpg".format(json_names[i][:-5])))
            right_img = cv2.imread(os.path.join(cfg.img_root_path, 'c2_rot', "{}.jpg".format(json_names[i][:-5])))
            (left_img_coor, jacobian) = cv2.projectPoints(face_v_array, left_rotation_vector, left_translation_vector, K, D)
            (right_img_coor, jacobian) = cv2.projectPoints(face_v_array, right_rotation_vector, right_translation_vector,
                                                          K, D)
            p3d = cv2.triangulatePoints(left_P_cam, right_P_cam, cache.left_camera_array[cache.middle_camera_ind, :].T,
                                        cache.right_camera_array[cache.middle_camera_ind, :].T)
            p3d = p3d.T / (p3d.T[:, -1][:, np.newaxis])
            p3d = p3d[:, :-1]
            print(p3d.shape)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], c='b', s=1, linewidths=3, marker='o', label='3d重建点')
            plt.show()
            for ind in face_f:
                ind = np.array(ind, dtype=np.int32) - 1  # face 是从1 开始的 obj点序
                l_pts = left_img_coor[ind, :2].astype(np.int32)
                r_pts = right_img_coor[ind, :2].astype(np.int32)
                cv2.polylines(left_img, [l_pts], True, (0, 0, 255), thickness=1)
                cv2.polylines(right_img, [r_pts], True, (0, 0, 255), thickness=1)
            cv2.namedWindow('left pnp projection', cv2.WINDOW_NORMAL)
            cv2.imshow('left pnp projection', left_img)
            cv2.namedWindow('right pnp projection', cv2.WINDOW_NORMAL)
            cv2.imshow('right pnp projection', right_img)
            cv2.waitKey(0)
