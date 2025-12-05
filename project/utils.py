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
    
    _, _, S = np.linalg.svd(V, full_matrices=False) # fill_matrices = False -> no padding and faster
    B = S[-1, :]  # S is transposed so the values of B are in the last row
    B = np.array([
        B[0], B[1], B[3],
        B[1], B[2], B[4],
        B[3], B[4], B[5]
    ])

    L = np.linalg.cholesky(B.reshape(3, 3))
    K = np.linalg.inv(L.transpose())
    K = K / K[2,2]  # TODO: do we have to divide?
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
    r1 = lam * K_inv @ H[:, 0]
    r2 = lam * K_inv @ H[:, 1]
    r3 = np.linalg.cross(r1, r2)
    t = lam * K_inv @ H[:, 2]
    R = np.stack([r1,r2,r3]).transpose()
    return R, t