import numpy as np
import cv2

def get_corners(img_path:str, grid_size:tuple) -> np.ndarray:
    img = cv2.imread(img_path)
    return_value, corners = cv2.findChessboardCorners(img, patternSize=grid_size, corners=None) # type: ignore
    if not return_value:
        raise Exception(f"Corners not found in image {img_path}")
    return corners.squeeze(1)

def get_homography(img_path:str, grid_size, square_size) -> tuple[np.ndarray, np.ndarray]:
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
        tuple: A tuple containing:
            - A (np.ndarray): The matrix of linear equations used to compute the homography.
            - H (np.ndarray): The computed 3x3 homography matrix.
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
    for index, corner in enumerate(corners):
        # getting the coordinates in pixels
        u_coord = corner[0]
        v_coord = corner[1]
        
        # defining the grid structure of the checkerboard
        grid_size_cv2 = tuple(reversed(grid_size))  # we want (rows, cols), not (cols, rows)
        u_index, v_index = np.unravel_index(index, grid_size_cv2)  # the first corner is at position (0,0), the second (0,1)
        
        # finding the (x,y) coordinates wrt the checkerboard
        x_mm = (u_index) * square_size
        y_mm = (v_index) * square_size
        
        eq_1 = [x_mm, y_mm, 1, 0, 0, 0, -u_coord*x_mm, -u_coord*y_mm, -u_coord]
        eq_2 = [0, 0, 0, x_mm, y_mm, 1, -v_coord*x_mm, -v_coord*y_mm, -v_coord]
        
        A.append(eq_1)
        A.append(eq_2)

    # evaluating the SVD of A
    A = np.array(A)
    _, _, V = np.linalg.svd(A, full_matrices=False) # fill_matrices = False -> no padding and faster

    # V is transposed so the values of H are in the last row
    H = V[-1, :].reshape(3,3)
    return A, H

def get_v_vector(H:np.ndarray, i:int, j:int) -> np.ndarray:
    """_summary_

    Args:
        H (_type_): Homography matrix
        i (_type_): can be 1 or 2
        j (_type_): can be 1 or 2

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
    B = S[-1, :]  # S is transposed so the values of B are in the last row

    # __________ CHOLESKY __________
    B = np.array([
        B[0], B[1], B[3],
        B[1], B[2], B[4],
        B[3], B[4], B[5]
    ])
    L = np.linalg.cholesky(B.reshape(3, 3))
    K = np.linalg.inv(L.transpose())
    K = K / K[2,2]
    # K[2, 2] = 1.0
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
    if t[2] < 0:
        t = -t
        lam = -lam

    r1 = lam * K_inv @ H[:, 0]
    r2 = lam * K_inv @ H[:, 1]
    r3 = np.linalg.cross(r1, r2)

    R = np.stack([r1,r2,r3]).transpose()
    # R = np.column_stack((r1, r2, r3))

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
    
    G = np.zeros((3, 4))
    G[:, :3] = R
    G[:, 3] = t
    return  K @ G

def project(points:np.ndarray, P:np.ndarray) -> np.ndarray:
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    
    projected_u = (P[0, :] @ points.transpose()) / (P[2, :] @ points.transpose())
    projected_v = (P[1, :] @ points.transpose()) / (P[2, :] @ points.transpose())
    
    return np.stack([np.array([u, v]) for u, v in zip(projected_u, projected_v)])
    
def superimpose_cylinder(
    img_path: str, 
    P: np.ndarray,
    radius: float, 
    height: float, 
    center_x: float, 
    center_y: float, 
    num_sides: int = 30, 
    num_height_slices: int = 5
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

    
    # Generation of 3D Points 
    theta = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    z_slices = np.linspace(0, height, num_height_slices) 
    points_3d = []

    # Perimetral points for each height slice
    for z in z_slices:
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        Z = np.full_like(x, z)
        # Coordinate omogenee [x, y, z, 1]
        homogeneous_slice = np.vstack((x, y, Z, np.ones_like(x))).T
        points_3d.append(homogeneous_slice)

    # Bottom slice (Z = 0)
    x_base = center_x + radius * np.cos(theta)
    y_base = center_y + radius * np.sin(theta)
    base_points = np.stack([x_base, y_base, np.zeros_like(x_base), np.ones_like(x_base)], axis=1)
    points_3d.append(base_points)

    # Top slice (Z = height)
    x_top = center_x + radius * np.cos(theta)
    y_top = center_y + radius * np.sin(theta)
    top_points = np.stack([x_top, y_top, np.full_like(x_top, height), np.ones_like(x_top)], axis=1)    # TODO: Z axis is inverted
    points_3d.append(top_points)
    
    object_points = np.concatenate(points_3d, axis=0) # shape (N, 4)
    
    # Pojection of 3D Points onto the Image Plane
    img = cv2.imread(img_path)
    if img is None:
        raise FileExistsError(f"Cannot find the image file at {img_path}.")

    points_2D_hom = project(object_points, P)
    u = points_2D_hom[:, 0]
    v = points_2D_hom[:, 1]
    
    projected_points = np.stack([u, v], axis=1).astype(np.int32) 

    # Base
    base_start_idx = projected_points.shape[0] - 2 * num_sides
    base_points_proj = projected_points[base_start_idx : base_start_idx + num_sides]
    
    # Top
    top_points_proj = projected_points[base_start_idx + num_sides : ]

    # Drawing base
    cv2.polylines(img, [base_points_proj], isClosed=True, color=(0, 0, 255), thickness=2) 
    
    # Drawing top
    cv2.polylines(img, [top_points_proj], isClosed=True, color=(0, 255, 0), thickness=2) 
    center_projection = project(np.array([center_x, center_y, 0, 1]), P)[0]
    cv2.circle(img, (int(center_projection[0].round()), int(center_projection[1].round())), radius=5, color=(255, 0, 0), thickness=-1)
    # Drawing vertical sides
    side_idxs = np.linspace(0, num_sides - 1, num_sides, dtype=int)
    for side in side_idxs:
        pt_base = base_points_proj[side]
        pt_top = top_points_proj[side]
        cv2.line(img, tuple(pt_base), tuple(pt_top), color=(255, 0, 0), thickness=1)
        
    return img