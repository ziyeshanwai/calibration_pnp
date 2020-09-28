import os
import pickle
import json
import numpy as np
from scipy import optimize
from math import atan2
import cv2


def formate_at(at_new):
    """
    formate at data 将at工具产生的数据转化为snapsolver可用数据
    :param at_new:
    :return: at_old
    """
    old = {
        'data': {
            'camera1': {
                'contour': {
                    'upper_eyelid_left': [],
                    'lower_eyelid_left': [],
                    'upper_eyelid_right': [],
                    'lower_eyelid_right': [],
                    'upper_lip': [],
                    'lower_lip': []
                }
            },
            'landmarks': {
            },
            'marker': {
            },
            'camera2': {
                'contour': {
                    'upper_eyelid_left': [

                    ],
                    'lower_eyelid_left': [

                    ],
                    'upper_eyelid_right': [

                    ],
                    'lower_eyelid_right': [

                    ],
                    'upper_lip': [

                    ],
                    'lower_lip': [

                    ]
                },
                'landmarks': {

                },
                'marker': {

                }
            }
        }
    }
    """
    camera1
    """
    contours = at_new['camera1']['contour']
    for contour in contours:
        p = []
        for point in contour['points']:
            p.append([point['x'], point['y']])
        if len(p) > 0:
            old['data']['camera1']['contour'][contour['tag']] = np.unique(p, axis=0).tolist()
        else:
            old['data']['camera1']['contour'][contour['tag']] = p
    opts = at_new['camera1']['optical_flow']
    tmp = {}
    for opt in opts:
        tmp[str(opt['key'])] = [opt['x'], opt['y']]
    old['data']['camera1']['marker'] = tmp

    """
    camera2
    """
    contours = at_new['camera2']['contour']
    for contour in contours:
        p = []
        for point in contour['points']:
            p.append([point['x'], point['y']])
        if len(p) > 0:
            old['data']['camera2']['contour'][contour['tag']] = np.unique(p, axis=0).tolist()
        else:
            old['data']['camera2']['contour'][contour['tag']] = p
    opts = at_new['camera2']['optical_flow']
    tmp = {}
    for opt in opts:
        tmp[str(opt['key'])] = [opt['x'], opt['y']]
    old['data']['camera2']['marker'] = tmp
    return old


def LoadXML(file_name, node_name):
    """
    :param file_name: 读取的xml文件的路径和名称
    :param node_name: 读取的xml文件的节点名字
    :return: 返回对应节点名称的内容
    """
    # just like before we specify an enum flag, but this time it is
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    # for some reason __getattr__ doesn't work for FileStorage object in python
    # however in the C++ documentation, getNode, which is also available,
    # does the same thing
    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    matrix = cv_file.getNode(node_name).mat()
    cv_file.release()
    return matrix


def load_pts(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    line = lines[0]
    while not line.startswith('{'):
        line = lines.pop(0)

    data = []
    for line in lines:
        if not line.strip().startswith('}'):
            xpos, ypos = line.split()[:2]
            data.append((float(xpos), float(ypos)))

    return data


def Anti_shake_single(index, coord):
    global tmp_coord
    MPT = 3.14159265358979323846
    MINCUTOFF = 10
    BETA = 0.05
    FREQUENCY = 10
    if index == 0:
        tmp_coord = coord
    else:
        tmp2_coord = []
        for indx in range(0, coord.shape[0]):
            dcutoff_x = MINCUTOFF + BETA*abs(coord[indx] - tmp_coord[indx])
            tao_x = 1./(2*MPT*dcutoff_x)
            alpha_x = 1./(1+tao_x*FREQUENCY)
            new_coord_x = alpha_x*coord[indx]+(1-alpha_x)*tmp_coord[indx]
            tmp2_coord.append(new_coord_x)
        tmp_coord = tmp2_coord
    return tmp_coord


def save_pickle_file(filename, file):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)
        print("save {}".format(filename))


def load_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            file = pickle.load(f)
        return file
    else:
        print("{} not exist".format(filename))


def loadObj(path):
    """Load obj file
    读取三角形和四边形的mesh
    返回vertex和face的list
    """
    if path.endswith('.obj'):
        f = open(path, 'r')
        lines = f.readlines()
        vertics = []
        faces = []
        vts = []
        for line in lines:
            if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
                line_split = line.split()
                ver = line_split[1:4]
                ver = [float(v) for v in ver]
                vertics.append(ver)
            else:
                if line.startswith('f'):
                    line_split = line.split()
                    if '/' in line:  # 根据需要这里补充obj的其他格式解析
                        tmp_faces = line_split[1:]
                        f = []
                        for tmp_face in tmp_faces:
                            f.append(int(tmp_face.split('/')[0]))
                        faces.append(f)
                    else:
                        face = line_split[1:]
                        face = [int(fa) for fa in face]
                        faces.append(face)

        return (vertics, faces)

    else:
        print('格式不正确，请检查obj格式')
        return


def writeObj(file_name_path, vertexs, faces, vts=None):
    """write the obj file to the specific path
       file_name_path:保存的文件路径
       vertexs:顶点数组 list
       faces: 面 list
    """
    with open(file_name_path, 'w') as f:
        for v in vertexs:
            # print(v)
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            if len(face) == 4:
                f.write("f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])) # 保存四个顶点
            if len(face) == 3:
                f.write("f {} {} {}\n".format(face[0], face[1], face[2]))  # 保存三个顶点
        if vts != None:
            for vt in vts:
                f.write("vt {} {}\n".format(vt[0], vt[1]))
        print("saved mesh to {}".format(file_name_path))


def loadmarkpoint(txt_path):
    with open(txt_path, 'r') as f:
        data = json.load(f)
    mark_points = np.array(data)
    return mark_points


def load_json(json_file):
    """
    load json file
    :param json_file: json 文件路径
    :return:
    """
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        print("{} is not exits".format(json_file))
        data = None
    return data


def write_json(file_name, file):
    with open(file_name, 'w') as fw:
        json.dump(file, fw)
        print("save {}".format(file_name))


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """
    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]
    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = atan2(r, t - 1)
    # Normalise the axis.
    axis = axis / r
    # Return the data.
    return axis, theta


def R_axis_angle(matrix, axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """
    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca
    # Depack the axis.
    x, y, z = axis
    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC
    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca
    return matrix


def resSimXform(b, A, B):
    t = b[4:7]
    R = np.zeros((3, 3))
    R = R_axis_angle(R, b[0:3], b[3])
    rot_A = b[7]*R.dot(A) + t[:, np.newaxis]  # fix error
    result = np.sqrt(np.sum((B-rot_A)**2, axis=0).mean())
    return result


def caculate_transform(Model, Data):
    """
    使用此方法要求点数大于等于8
    对齐Data 到 Model
    calculate the R t s between PointsA and PointsB
    :param Model: n * 3  ndarray
    :param Data: n * 3  ndarray
    :return: R t s
    """

    model = Model.T  # 3 * n
    data = Data.T  # 3 * n
    cent = np.vstack((np.mean(model, axis=1), np.mean(data, axis=1))).T
    cent_0 = cent[:, 0]
    model_center = cent_0[:, np.newaxis]
    cent_1 = cent[:, 1]
    data_center = cent_1[:, np.newaxis]
    model_zerocentered = model - model_center
    data_zerocentered = data - data_center
    n = model.shape[1]
    Cov_matrix = 1.0/n * model_zerocentered.dot(data_zerocentered.T)
    U, D, V = np.linalg.svd(Cov_matrix)
    V = V.T
    W = np.eye(V.shape[0], V.shape[0])
    if np.linalg.det(V.dot(W).dot(U.T)) == -1:
        print("计算出的旋转矩阵为反射矩阵，纠正中..")
        W[-1, -1] = np.linalg.det(V.dot(U.T))
    R = V.dot(W).dot(U.T)
    sigma2 = (1.0 / n) * np.multiply(data_zerocentered, data_zerocentered).sum()
    s = 1.0 / sigma2 * np.trace(np.dot(np.diag(D), W))
    t = model_center - s*R.dot(data_center)
    b0 = np.zeros((8,))
    if np.isreal(R).all():
        axis, theta = R_to_axis_angle(R)
        b0[0:3] = axis
        b0[3] = theta
        if not np.isreal(b0).all():
            b0 = np.abs(b0)
    else:
        print("R is {}".format(R))
        print("R中存在非实数")
    b0[4:7] = t.T
    b0[7] = s
    # b = least_squares(fun=resSimXform, x0=b0, jac='3-point', method='lm', args=(data, model),
    #                   ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)  # 参数只能是一维向量么
    b = optimize.minimize(fun=resSimXform, x0=b0, jac=False, method='BFGS', args=(data, model),
                          options={"gtol": 0.001, "maxiter": 8000, "disp":False, "eps": 1e-7})
    r = b.x[0:4]
    t = b.x[4:7]
    s = b.x[7]
    R = R_axis_angle(R, r[0:3], r[3])
    rot_A = s*R.dot(data) + t[:, np.newaxis]
    res = np.sum(np.sqrt(np.sum((model-rot_A)**2, axis=1)))/model.shape[1]
    # print("对齐误差是{}".format(res))
    return R, t, s, res


if __name__ == '__main__':
    '''unit test'''

