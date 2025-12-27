import numpy as np
import cv2

def get_corners(img_path:str, grid_size:tuple) -> np.ndarray:
    img = cv2.imread(img_path)
    return_value, corners = cv2.findChessboardCorners(img, patternSize=grid_size)
    corners = corners.reshape((grid_size[0] * grid_size[1],2))
    if return_value:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # tuple for specifying the termination criteria of the iterative refinement procedure cornerSubPix()
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
    else:
        raise Exception(f"Corners not found in image {img_path}")
    return corners

def get_homography(img_path:str, grid_size:tuple, square_size:int) -> np.ndarray:
    """
    Computes the homography matrix for a checkerboard pattern in an image.
    This function reads an image containing a checkerboard pattern, detects the corners of the pattern,
    and computes the homography matrix that maps the 2D grid coordinates of the checkerboard to the 
    pixel coordinates in the image.
    Args:
        img_path (str): Path to the image file containing the checkerboard pattern.
        grid_size (tuple, optional): Dimensions of the checkerboard grid as (columns, rows). 
            Default is (8, 11).
        square_size (int, optional): Size of each square in the checkerboard grid, in millimeters. 
            Default is 11.
    Returns:
        H (np.ndarray): The computed 3x3 homography matrix.
    Raises:
        AssertionError: If `img_path` is not a string, `grid_size` is not a tuple, `square_size` 
            is not an integer, or `grid_size` does not have exactly two elements.
        Exception: If the checkerboard pattern is not found in the image.
    Notes:
        - The function assumes that the checkerboard pattern is visible in the image and that the 
            grid dimensions are correctly specified.
        - The homography matrix is computed using Singular Value Decomposition (SVD) of the matrix 
            of linear equations derived from the corner correspondences.
    """
    
    assert isinstance(img_path, str), f"img_path is not a string: type {type(img_path)}."
    assert isinstance(grid_size, tuple), f"grid_size is not a tuple: type {type(grid_size)}."
    assert isinstance(square_size, int), f"square_size is not a int: type {type(square_size)}."
    assert len(grid_size) == 2, f"grid_size has dimension {len(grid_size)}: expected 2."
    
    corners = get_corners(img_path, grid_size)
        
    # CONSTRUCT A
    A = []
    grid_size_cv2 = tuple(reversed(grid_size))  # we want (rows, cols), not (cols, rows)
    for index, corner in enumerate(corners):
        # getting the coordinates in pixels
        u_coord = corner[0]
        v_coord = corner[1]
        
        # defining the grid structure of the checkerboard
        u_index, v_index = np.unravel_index(index, grid_size_cv2)  # the first corner is at position (0,0), the second (0,1)
        
        # finding the (x,y) coordinates wrt the checkerboard
        x_mm = u_index * square_size
        y_mm = v_index * square_size
        
        eq_1 = [x_mm, y_mm, 1, 0, 0, 0, -u_coord*x_mm, -u_coord*y_mm, -u_coord]
        eq_2 = [0, 0, 0, x_mm, y_mm, 1, -v_coord*x_mm, -v_coord*y_mm, -v_coord]
        
        A.append(eq_1)
        A.append(eq_2)

    # evaluating the SVD of A
    A = np.array(A)
    _, _, V = np.linalg.svd(A, full_matrices=False) # full_matrices = False -> no padding and faster

    # V is transposed so the values of H are in the last row
    H = V[-1, :].reshape(3,3)
    return H

def get_v_vector(H:np.ndarray, i:int, j:int) -> np.ndarray:
    """
    Args:
        H (np.ndarray): Homography matrix
        i (int): can be 1 or 2
        j (int): can be 1 or 2

    Returns:
        np.array: returns the vector v transposed (dimension = (6,))
    """
    
    assert isinstance(H, np.ndarray), f"H is not a numpy array: type {type(H)}."
    assert H.shape == (3, 3), f"H does not have shape (3, 3): shape {H.shape}."
    assert isinstance(i, int), f"i is not an integer: type {type(i)}."
    assert isinstance(j, int), f"j is not an integer: type {type(j)}."
    assert i in [1, 2], f"i must be 1 or 2: value {i}."
    assert j in [1, 2], f"j must be 1 or 2: value {j}."

    i -= 1
    j -= 1
    
    return np.array([
        H[0,i] * H[0,j],
        H[0,i] * H[1,j] + H[1,i] * H[0,j],
        H[1,i] * H[1,j],
        H[2,i] * H[0,j] + H[0,i] * H[2,j],
        H[2,i] * H[1,j] + H[1,i] * H[2,j],
        H[2,i] * H[2,j]
    ])
    
def get_intrinsic(V:np.ndarray) -> np.ndarray:
    """
    Computes the intrinsic camera matrix K from the input matrix V.
    This function performs the following steps:
    1. Computes the Singular Value Decomposition (SVD) of the input matrix V.
    2. Extracts the last row of the transposed S matrix (smallest singular vector),
        which corresponds to the solution of the homogeneous system.
    3. Constructs the symmetric matrix B from the extracted vector.
    4. Performs a Cholesky decomposition of B to obtain a lower triangular matrix L.
    5. Computes the intrinsic matrix K as the inverse of the transpose of L.
    6. Normalizes K such that K[2, 2] equals 1.
    Args:
        V (np.ndarray): Input matrix, typically derived from a set of constraints
                        on the camera calibration parameters.
    Returns:
        np.ndarray: The intrinsic camera matrix K, a 3x3 upper triangular matrix.
    Notes:
        - The normalization step ensures that the intrinsic matrix K is scaled
            appropriately. The necessity of this step may depend on the specific
            application.
        - The input matrix V is expected to be structured such that the smallest
            singular vector corresponds to the desired solution.
    """
    
    assert isinstance(V, np.ndarray), f"V is not a numpy array: type {type(V)}."
    assert V.ndim == 2, f"V is not a 2D array: ndim {V.ndim}."
    assert V.shape[1] == 6, f"V does not have 6 columns: shape {V.shape}."
    
    _, _, S = np.linalg.svd(V, full_matrices=False) # full_matrices = False -> no padding and faster
    B = S[-1, :] # S is transposed so the values of B are in the last row
    B /= B[-1]
    
    # __________ CHOLESKY __________
    B = np.array([
        B[0], B[1], B[3],
        B[1], B[2], B[4],
        B[3], B[4], B[5]
    ])
    L = np.linalg.cholesky(B.reshape(3, 3))
    K = np.linalg.inv(L.transpose())
    K = K / K[2,2]
    return K


def get_extrinsic(K:np.ndarray, H:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the extrinsic parameters (rotation matrix R and translation vector t) 
    from the intrinsic matrix K and homography matrix H.
    Parameters:
    -----------
    K : np.ndarray
        The intrinsic camera matrix of shape (3, 3).
    H : np.ndarray
        The homography matrix of shape (3, 3).
    Returns:
    --------
    R : np.ndarray
        The rotation matrix of shape (3, 3).
    t : np.ndarray
        The translation vector of shape (3,).
    """
    
    assert isinstance(K, np.ndarray), f"K is not a numpy array: type {type(K)}."
    assert K.shape == (3, 3), f"K does not have shape (3, 3): shape {K.shape}."
    assert isinstance(H, np.ndarray), f"H is not a numpy array: type {type(H)}."
    assert H.shape == (3, 3), f"H does not have shape (3, 3): shape {H.shape}."
    
    K_inv = np.linalg.inv(K)
    lam = 1 / np.linalg.norm(K_inv @ H[:, 0])

    t = lam * K_inv @ H[:, 2]
    # checking t[2] < 0 in order to understand whether the object is considered behind the camera, in which case coords would be inverted 
    if t[2] < 0:
        t = -t
        lam = -lam

    r1 = lam * K_inv @ H[:, 0]
    r2 = lam * K_inv @ H[:, 1]
    r3 = np.linalg.cross(r1, r2)

    R = np.column_stack((r1, r2, r3))

    # since R might not be orthonormal after the estimation phase, take the closest matrix that satisfies the properties
    U, _, Vt = np.linalg.svd(R) # SVD-based orthonormalization to ensure orthonormal columns and det(R) = +1
    R = U @ Vt

    return R, t

def get_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes the projection matrix for a camera given its intrinsic matrix, 
    rotation matrix, and translation vector.
    Args:
        K (np.ndarray): The intrinsic matrix of the camera (3x3).
        R (np.ndarray): The rotation matrix representing the camera's orientation (3x3).
        t (np.ndarray): The translation vector representing the camera's position (3,).
    Returns:
        np.ndarray: The projection matrix (3x4) obtained by combining the intrinsic 
        and extrinsic parameters.
    """
    
    assert isinstance(K, np.ndarray), f"K is not a numpy array: type {type(K)}."
    assert K.shape == (3, 3), f"K does not have shape (3, 3): shape {K.shape}."
    assert isinstance(R, np.ndarray), f"R is not a numpy array: type {type(R)}."
    assert R.shape == (3, 3), f"R does not have shape (3, 3): shape {R.shape}."
    assert isinstance(t, np.ndarray), f"t is not a numpy array: type {type(t)}."
    assert t.shape in [(3,), (3, 1)], f"t does not have shape (3,) or (3,1): shape {t.shape}."
    
    G = np.column_stack((R, t))
    return  K @ G

def project(points:np.ndarray, P:np.ndarray) -> np.ndarray:
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    
    # directly compute normalised projections
    projected_u = (P[0, :] @ points.transpose()) / (P[2, :] @ points.transpose())
    projected_v = (P[1, :] @ points.transpose()) / (P[2, :] @ points.transpose())
    
    return np.stack([np.array([u, v]) for u, v in zip(projected_u, projected_v)])

def compute_reprojection_error(all_observed_corners, all_projected_corners):
    total_error = 0
    total_points = 0

    for i in range(len(all_observed_corners)):
        observed_corners = all_observed_corners[i]
        for j, (u_proj, v_proj) in enumerate(all_projected_corners[i]):
            u_obs, v_obs = observed_corners[j]
            err = np.sqrt((u_obs - u_proj)**2 + (v_obs - v_proj)**2)
            total_error += err
            total_points += 1

    return total_error, total_error / total_points
    
def superimpose_cylinder(
    img_path: str, 
    P: np.ndarray,
    radius: float, 
    height: float, 
    center_x: float, 
    center_y: float, 
    num_sides: int = 30, 
    num_height_slices: int = 5,
    line_thinkness: int = 2,
) -> np.ndarray:
    """
    Generate the 3D points of a cylinder, calculate the projection matrix and 
    overlay the object onto the 2D image.

    Parameters:
    -----------
    img_path : str
        The path to the image file.
    P : np.ndarray
        Projection matrix
    radius : float
        The radius of the cylinder (in the world reference frame).
    height : float
        The height of the cylinder (in the world reference frame).
    centre_x : float
        The X coordinate of the centre of the base (in the world reference frame).
    centre_y : float
        The Y coordinate of the centre of the base (in the world reference frame).
    num_sides : int 
        Number of sides to approximate the circle.
    num_height_slices : int
        Number of sections along the height.

    Returns:
    --------
    np.ndarray
        The image with the cylinder superimposed.
    """
    assert isinstance(img_path, str), f"img_path is not a string: type {type(img_path)}."
    assert isinstance(P, np.ndarray), f"P is not a numpy array: type {type(P)}."
    assert P.shape == (3, 4), f"P does not have shape (3, 4): shape {P.shape}."
    assert isinstance(radius, (int, float)), f"radius is not a number: type {type(radius)}."
    assert isinstance(height, (int, float)), f"height is not a number: type {type(height)}."
    assert isinstance(center_x, (int, float)), f"center_x is not a number: type {type(center_x)}."
    assert isinstance(center_y, (int, float)), f"center_y is not a number: type {type(center_y)}."
    assert isinstance(num_sides, int), f"num_sides is not an integer: type {type(num_sides)}."
    assert num_sides >= 3, f"num_sides is less than 3: value {num_sides}."  

    
    # 3D Points 
    theta = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    z_slices = np.linspace(0, height, num_height_slices) 
    points_3d = []

    # perimetral points for each height slice
    for z in z_slices:
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        Z = np.full_like(x, z)
        # homogeneous coords [x, y, z, 1]
        homogeneous_slice = np.vstack((x, y, Z, np.ones_like(x))).T
        points_3d.append(homogeneous_slice)

    # bottom slice (Z = 0)
    x_base = center_x + radius * np.cos(theta)
    y_base = center_y + radius * np.sin(theta)
    base_points = np.stack([x_base, y_base, np.zeros_like(x_base), np.ones_like(x_base)], axis=1)
    points_3d.append(base_points)

    # top slice (Z = height)
    x_top = center_x + radius * np.cos(theta)
    y_top = center_y + radius * np.sin(theta)
    top_points = np.stack([x_top, y_top, np.full_like(x_top, height), np.ones_like(x_top)], axis=1)
    points_3d.append(top_points)
    
    object_points = np.concatenate(points_3d, axis=0) # shape (N, 4)
    
    # projection of 3D Points onto the Image Plane
    img = cv2.imread(img_path)
    if img is None:
        raise FileExistsError(f"Cannot find the image file at {img_path}.")

    points_2D_hom = project(object_points, P)
    u = points_2D_hom[:, 0]
    v = points_2D_hom[:, 1]
    
    projected_points = np.stack([u, v], axis=1).astype(np.int32) 

    # base
    base_start_idx = projected_points.shape[0] - 2 * num_sides
    base_points_proj = projected_points[base_start_idx : base_start_idx + num_sides]
    cv2.polylines(img, [base_points_proj], isClosed=True, color=(0, 0, 255), thickness=line_thinkness)
    
    # top
    top_points_proj = projected_points[base_start_idx + num_sides : ]
    cv2.polylines(img, [top_points_proj], isClosed=True, color=(0, 255, 0), thickness=line_thinkness)
    
    center_projection = project(np.array([center_x, center_y, 0, 1]), P)[0]
    cv2.circle(img, (int(center_projection[0].round()), int(center_projection[1].round())), radius=round(2.5*line_thinkness), color=(255, 0, 0), thickness=-1)
    # vertical sides
    side_idxs = np.linspace(0, num_sides - 1, num_sides, dtype=int)
    for side in side_idxs:
        pt_base = base_points_proj[side]
        pt_top = top_points_proj[side]
        cv2.line(img, tuple(pt_base), tuple(pt_top), color=(255, 0, 0), thickness=line_thinkness-1)
        
    return img

def get_rot_axis_from_R(R: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes the rotation axis and angle from a given rotation matrix.
    This function takes a 3x3 rotation matrix `R` and calculates the 
    corresponding rotation axis (a unit vector) and the rotation angle 
    (in radians) using the Rodrigues' rotation formula.
    Args:
        R (np.ndarray): A 3x3 rotation matrix.
    Returns:
        tuple[np.ndarray, float]: A tuple containing:
            - A numpy array representing the rotation axis scaled by the angle.
            - A float representing the rotation angle in radians.
    Raises:
        ValueError: If the input matrix `R` is not a valid 3x3 rotation matrix.
    """
    assert isinstance(R, np.ndarray), f"R is not a numpy array: type {type(R)}."
    assert R.shape == (3, 3), f"R does not have shape (3, 3): shape {R.shape}."
    
    theta = float(np.arccos((np.trace(R) - 1) / 2))
    if abs(theta) < 1e-8:
        return np.zeros(3), 0.0
    _r = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ])
    return (1/(2*np.sin(theta))) * _r * theta, theta

def get_R_from_axis(r: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix from an axis-angle representation.
    This function takes a 3D vector `r` representing the axis of rotation 
    scaled by the rotation angle (in radians) and computes the corresponding 
    3x3 rotation matrix using the Rodrigues' rotation formula.
    Parameters:
    -----------
    r : np.ndarray
        A 3-element numpy array representing the axis of rotation scaled 
        by the rotation angle.
    Returns:
    --------
    np.ndarray
        A 3x3 rotation matrix corresponding to the input axis-angle 
        representation.
    Notes:
    ------
    - The input vector `r` is normalized internally to extract the axis of 
      rotation.
    - The function assumes that the input vector `r` is non-zero.
    """
    assert isinstance(r, np.ndarray), f"r is not a numpy array: type {type(r)}."
    assert r.shape == (3,), f"r does not have shape (3,): shape {r.shape}."
    
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    r = r / theta
    # cross product matrix
    r_x = np.stack([
        [ 0  , -r[2], r[1]  ],
        [r[2] ,   0  , -r[0]],
        [-r[1], r[0],   0   ],
    ])
    
    return np.eye(3) + \
            np.sin(theta) * r_x + \
            (1 - np.cos(theta)) * np.linalg.matrix_power(r_x, 2)
            
def compute_residuals(params: np.ndarray, checkerboard_world_corners: np.ndarray, checkerboard_image_corners: np.ndarray):
    """
    Compute the residuals between the projected checkerboard corners and the observed image corners.
    Args:
        params (np.ndarray): A 1D array containing the camera parameters. The first 5 elements are the intrinsic 
            parameters (alpha_u, skew, u_0, alpha_v, v_0). The remaining elements are the rotation vectors (r) 
            and translation vectors (t) for each image, concatenated in the order [r1, r2, ..., t1, t2, ...].
        checkerboard_world_corners (np.ndarray): A 2D array of shape (n_points, 4) representing the 3D coordinates 
            of the checkerboard corners in the world coordinate system.
        checkerboard_image_corners (np.ndarray): A 2D array of shape (n_images, n_points, 2) representing the 2D 
            coordinates of the checkerboard corners in the image plane for each image.
    Returns:
        np.ndarray: A 1D array of residuals, representing the difference between the projected 2D points and the 
            observed 2D image points. The residuals are flattened into a single array.
    """
    assert isinstance(params, np.ndarray), f"params is not a numpy array: type {type(params)}."
    assert params.ndim == 1, f"params is not a 1D array: ndim {params.ndim}."
    assert isinstance(checkerboard_world_corners, np.ndarray), f"checkerboard_world_corners is not a numpy array: type {type(checkerboard_world_corners)}."
    assert checkerboard_world_corners.ndim == 2 and checkerboard_world_corners.shape[1] == 4, f"checkerboard_world_corners must have shape (n_points, 4): shape {checkerboard_world_corners.shape}."
    assert isinstance(checkerboard_image_corners, np.ndarray), f"checkerboard_image_corners is not a numpy array: type {type(checkerboard_image_corners)}."
    assert checkerboard_image_corners.ndim == 3 and checkerboard_image_corners.shape[2] == 2, f"checkerboard_image_corners must have shape (n_images, n_points, 2): shape {checkerboard_image_corners.shape}."
    
    n_images = (len(params) - 5) // 3 // 2
    alpha_u, gamma, u_0, alpha_v, v_0 = params[:5]
    K = np.stack([
        [alpha_u, gamma, u_0],
        [0, alpha_v, v_0],
        [0, 0, 1],
    ])
        
    r = np.array(params[5:(5 + (n_images * 3))]).reshape(-1, 3)
    t = np.array(params[-(n_images * 3):]).reshape(-1, 3)

    R = np.stack([get_R_from_axis(r[i]) for i in range(n_images)])
    P = np.stack([get_projection_matrix(K, R[i], t[i]) for i in range(n_images)])
    
    projected_points = P @ checkerboard_world_corners.T
    projected_points = np.stack([projected_points[:, 0] / projected_points[:, 2], projected_points[:, 1] / projected_points[:, 2]], axis=2)
    
    residuals = projected_points - checkerboard_image_corners
    return residuals.ravel()

def get_radial_distorsion(images_path, grid_size, square_size, K, all_P):
    alpha_u = K[0,0]
    alpha_v = K[1,1]
    u_0 = K[0, 2]
    v_0 = K[1, 2]

    A = []
    b = []

    grid_size_cv2 = tuple(reversed(grid_size))
    for i, img in enumerate(images_path):
        corners = get_corners(img, grid_size)
        for index, corner in enumerate(corners):
            # Get the projected points
            u_hat = corner[0]
            v_hat = corner[1]
            u_index, v_index = np.unravel_index(index, grid_size_cv2)
            x_mm = (u_index) * square_size
            y_mm = (v_index) * square_size
            point_m = np.array([x_mm, y_mm, 0, 1])
            u_proj, v_proj = project(point_m, all_P[i])[0]

            r2 = ((u_proj - u_0) / alpha_u)**2 + ((v_proj - v_0) / alpha_v)**2

            # Get the equation system for each image
            A.append([(u_proj - u_0) * r2, (u_proj - u_0) * r2*r2])
            A.append([(v_proj - v_0) * r2, (v_proj - v_0) * r2*r2])

            b.append(u_hat - u_proj)
            b.append(v_hat - v_proj)

    A = np.array(A)
    b = np.array(b)

    # solve the system to obtain k1 and k2
    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    k1, k2 = k
    return k1, k2

def project_with_distortion(world_corners, K, rvec, tvec, k1, k2):
    R = get_R_from_axis(rvec)
    Xc = (R @ world_corners.T).T + tvec
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]

    r2 = x**2 + y**2
    radial = 1 + k1*r2 + k2*r2*r2
    x_hat = x*radial
    y_hat = y*radial

    alpha_u = K[0, 0]
    alpha_v = K[1, 1]
    u0 = K[0, 2]
    v0 = K[1, 2]

    u_dist = x_hat*alpha_u + u0
    v_dist = y_hat*alpha_v + v0
    return np.column_stack([u_dist, v_dist])

def pack_params(K, k1, k2, rvecs, tvecs):
    params = [
        K[0, 0],  # alpha_u
        K[1, 1],  # alpha_v
        K[0, 2],  # u0
        K[1, 2],  # v0
        k1,
        k2
    ]

    for rvec, tvec in zip(rvecs, tvecs):
        params.extend(rvec.ravel())
        params.extend(tvec.ravel())

    return np.array(params)

def unpack_params(params, n_images):
    alpha_u, alpha_v, u0, v0, k1, k2 = params[:6]

    K = np.array([
        [alpha_u,  0, u0],
        [ 0, alpha_v, v0],
        [ 0,  0,  1]
    ])

    rvecs = []
    tvecs = []

    idx = 6
    for _ in range(n_images):
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        rvecs.append(rvec)
        tvecs.append(tvec)
        idx += 6

    return K, k1, k2, rvecs, tvecs

def reprojection_residuals(params, all_world_corners, all_observed_corners):
    n_images = len(all_observed_corners)
    K, k1, k2, rvecs, tvecs = unpack_params(params, n_images)

    residuals = []

    for i in range(n_images):
        proj = project_with_distortion(
            all_world_corners[i],
            K,
            rvecs[i],
            tvecs[i],
            k1,
            k2
        )

        obs = all_observed_corners[i]
        residuals.append((proj - obs).ravel())

    return np.concatenate(residuals)