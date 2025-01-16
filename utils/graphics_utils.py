import torch
import math

def getProjectionMatrixFromFOV(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixFromIntrinsics(znear, zfar, cx, cy, fx, fy, W, H):
    """
    Infinite Perspective Matrix
    Reference: https://www.songho.ca/opengl/gl_projectionmatrix.html
    """
    # Calculate the left, right, top, and bottom bounds of the viewing frustum
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0

    # Scale bounds by near plane distance and focal length
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom

    # Initialize the 4x4 projection matrix
    P = torch.zeros(4, 4)

    # Perspective projection formula components
    # The general perspective projection matrix is:
    # P = [[ 2*znear / (right-left),  0,                  (right+left)/(right-left),  0                 ],
    #      [ 0,                      2*znear / (top-bottom), (top+bottom)/(top-bottom),   0                 ],
    #      [ 0,                      0,                  -(zfar+znear)/(zfar-znear),    -2*zfar*znear/(zfar-znear) ],
    #      [ 0,                      0,                  -1,                           0                 ]]

    z_sign = 1.0  # Determines the sign convention for depth

    # Populate the matrix values based on the formula
    P[0, 0] = 2.0 * znear / (right - left)  # Scaling factor for x
    P[1, 1] = 2.0 * znear / (top - bottom)  # Scaling factor for y
    P[0, 2] = (right + left) / (right - left)  # Translation along x
    P[1, 2] = (top + bottom) / (top - bottom)  # Translation along y
    P[3, 2] = z_sign  # Homogeneous coordinate adjustment
    P[2, 2] = z_sign * zfar / (zfar - znear)  # Depth adjustment
    P[2, 3] = -(zfar * znear) / (zfar - znear)  # Depth translation

    return P


"""
Reference: https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view
"""
def fov2focal(fov, pixels):

    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
