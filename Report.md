# Computer Vision and Pattern Recognition Project

## Lorenzo Cusin – Giacomo Serafini – Pietro Terribile

## Project 1 — Camera Calibration

### Introduction

The **camera calibration problem** consists in estimating the intrinsic and extrinsic parameters of a camera through several measurements.  
The outcome of these calculations is the **Perspective Projection Matrix** \( P \), which can be written as:

\[
P = K [R | t]
\]

Here:

- **\( K \)** is the intrinsic matrix, containing the internal parameters of the camera (specific to the camera itself), like the .  
- **\( R \)** and **\( t \)** are respectively the rotation matrix and the translation vector, describing the camera pose for a specific image relative to the **World Reference Frame**.
Once that these parameters are found, many computer vision tasks can be performed, such as **Triangulation**, **Structure from motion**, **Camera pose**, **Stereoscopy** and many other things that have become more and more popular and usefull nowadays.

### Task 1 - Zhang's Calibration method 
It is required to calibrate (so to find the unique K and a pair [R | t] for each image) using the Zhang's procedure, which is based on a key principle: instead of requiring a single image of many non-coplanar points (as is necessary for Direct Linear Transform, or DLT, methods), Zhang's approach utilizes multiple images (at least three) of a simple planar calibration pattern.
In our case we are provided with 81 images of a checkboard, each image is taken from a different point in the World reference frame. 
The foundation of Zhang's method relies on establishing a mathematical relationship, known as a **homography** (H), between the known 3D plane in the scene and its 2D perspective projection onto the image plane.
First of all, we import **numpy** and **OpenCV** libraries to our code: 
<pre>
  import numpy as np
  import cv2
</pre>
Then we followed the **LabLecture_1** steps to find the keypoints between the given images, here we utilize the function **findChessboardCorners** from OpenCV library, getting the corresponcences we need to estimate the **homography**

```python
def get_corners(img_path:str, grid_size:tuple) -> np.ndarray:
    img = cv2.imread(img_path)
    return_value, corners = cv2.findChessboardCorners(img, patternSize=grid_size, corners=None) # type: ignore
    if not return_value:
        raise Exception(f"Corners not found in image {img_path}")
    return corners.squeeze(1)
```

This function is then called inside another function we wrote to compute the homography matrix H:

```python
  def get_homography(img_path:str, grid_size, square_size) -> tuple[np.ndarray, np.ndarray]:
    
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
```

Another function, called "get_v_vector", is used to compute the constraints vector of six unknowns starting from the homography 

```python
  def get_v_vector(H:np.ndarray, i:int, j:int) -> np.ndarray:

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
```

After that, we wrote other two functions, respectively "get_intrinsic" and "get_extrinsic", which will compute the both K and the pair [R | t].
The first one computes the Singular Value Decomposition (SVD) of the constraints matrix V (in which are stacked 2n x 6 equations, given n planes), then extracts from it the smallest singular vector which will be the solution to the problem. Later on, it performs Cholesky decomposition, finally finding K matrix.

```python
  def get_intrinsic(V:np.ndarray) -> np.ndarray:
    
    assert isinstance(V, np.ndarray), f"V is not a numpy array: type {type(V)}."
    assert V.ndim == 2, f"V is not a 2D array: ndim {V.ndim}."
    assert V.shape[1] == 6, f"V does not have 6 columns: shape {V.shape}."
    
    _, _, S = np.linalg.svd(V, full_matrices=False) # full_matrices = False -> no padding and faster
    B = S[-1, :]  # S is transposed so the values of B are in the last row

    # __________ ZHANG PAPER __________
    B11, B12, B22, B13, B23, B33 = B
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    # focal lengths
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / (B11 * B22 - B12**2))

    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha**2 / lam

    # intrinsic matrix
    K = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1 ]
    ])

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
```

The latter computes column-wise the rotation matrix R and t, starting from the fact that P = [R | t] = K [r1 r2 r3 | t].

```python
def get_extrinsic(K:np.ndarray, H:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    assert isinstance(K, np.ndarray), f"K is not a numpy array: type {type(K)}."
    assert K.shape == (3, 3), f"K does not have shape (3, 3): shape {K.shape}."
    assert isinstance(H, np.ndarray), f"H is not a numpy array: type {type(H)}."
    assert H.shape == (3, 3), f"H does not have shape (3, 3): shape {H.shape}."
    
    K_inv = np.linalg.inv(K)
    lam = 1 / np.linalg.norm(K_inv @ H[:, 0])
    r1 = lam * K_inv @ H[:, 0]
    r2 = lam * K_inv @ H[:, 1]
    r3 = np.linalg.cross(r1, r2)

    R = np.stack([r1,r2,r3]).transpose()
    # R = np.column_stack((r1, r2, r3))

    U, _, Vt = np.linalg.svd(R) # SVD-based orthonormalization to ensure orthonormal columns and det(R) = +1
    R = U @ Vt

    t = lam * K_inv @ H[:, 2]
    return R, t
```

Now that we have everything required, the following execution code is shown. 
Here we process all the checkerboards we are provided with: 
```python
# constants
grid_size = (8,11)
square_size = 11

# getting the images path
images_path = "../images_and_poses_for_project_assignment/"
images_path = [os.path.join(images_path, imagename) for imagename in os.listdir(images_path) if imagename.endswith(".png")]
```

After that we can recall the functions shown upon for all the images provided: 
First we compute the homographies

```python
V = []
all_H = []  # saving the homographies for each image

# getting the homographies
for img in images_path:
    _, H = u.get_homography(img, grid_size, square_size)
    all_H.append(H)
    
    v_12 = u.get_v_vector(H, 1, 2)
    v_11 = u.get_v_vector(H, 1, 1)
    v_22 = u.get_v_vector(H, 2, 2)
    
    V.append(v_12)
    V.append(v_11 - v_22)
    
# computing params
V = np.array(V)
```

Then we compute the matrix of intrinsics parameters: 

```python
K = u.get_intrinsic(V)
```

And finally, we compute the pair [R|t] for each image processed: 

```python
all_R = []
all_t = []

for H in all_H:
    R, t = u.get_extrinsic(K, H)
    all_R.append(R)
    all_t.append(t)
```
