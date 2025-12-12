# Computer Vision and Pattern Recognition Project

## Lorenzo Cusin – Giacomo Serafini – Pietro Terribile

## Project 1 — Camera Calibration

### Introduction

The **camera calibration problem** consists in estimating the intrinsic and extrinsic parameters of a camera through several measurements.  
The outcome of these calculations is the **Perspective Projection Matrix** \( P \), which can be written as:

P = K [ R | t ]

Here:

- **\( K \)** is the intrinsic matrix, containing the internal parameters of the camera (specific to the camera itself), like the .  
- **\( R \)** and **\( t \)** are respectively the rotation matrix and the translation vector, describing the camera pose for a specific image relative to the **World Reference Frame**.
Once that these parameters are found, many computer vision tasks can be performed, such as **Triangulation**, **Structure from motion**, **Camera pose**, **Stereoscopy** and many other things that have become more and more popular and usefull nowadays.

### Task 1 - Zhang's Calibration method 
It is required to calibrate (so to find the unique K and a pair [R | t] for each image) using the Zhang's procedure, which is based on a key principle: instead of requiring a single image of many non-coplanar points (as is necessary for Direct Linear Transform, or DLT, methods), Zhang's approach utilizes multiple images (at least three) of a simple planar calibration pattern.
In our case we are provided with 81 images of a checkerboard, each image is taken from a different point in the World reference frame. 
The foundation of Zhang's method relies on establishing a mathematical relationship, known as a **homography** (H), between the known 3D plane in the scene and its 2D perspective projection onto the image plane.
First of all, we import **numpy** and **OpenCV** libraries to our code: 
<pre>
  import numpy as np
  import cv2
</pre>
Then we followed the **LabLecture_1** steps to find the keypoints between the given images, here we utilize the function **findChessboardCorners** from OpenCV library, getting the corresponcences we need to estimate the **homography**:

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

Another function, called "get_v_vector", is used to compute the constraints vector of six unknowns starting from the homography:

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

```
Here later on the realization of the project, we had to add this portion of code to the function 
```python
    if t[2] < 0:
        t = -t
        lam = -lam
```
This had to be done because there exists two possible solutions to the problem when computing extrinsics, but only one has the right physical meaning: in fact, being the checkerboard in front of the camera, we expect the value of t_z to be positive (since we defined the camera reference frame this way, with Z > 0), but sometimes this was not true, and in the superimposition task we observed that for some images, the value wass negative and the cylinder was entering the frame rather that getting out; this corresponded to the WRF to be considered behind the camera, which is clearly unfeasable. So, we are able to detect the wrong solution by checking this value and correct it taking the opposite, which means taking the opposite scale factor **lambda**. 

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

### Task 2 - Total Reprojection Error 
For this task we are required to choose one of the calibration images and compute the total reprojection
error, which quantifies the projection error, i.e. the distance between the projections (coordinates) of the measured image points and the projections estimated by the geometric model of the camera (perspective projection matrix P). This job is asked to be done for each grid point and to visualize it. 
To realize that, first of all we defined the function **get_projection_matrix** to compute the P matrix for an image given intrinsics and extrinsics parameters: 

```python
def get_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    
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
```

After that, to achieve the requirements we need to project 3D points onto a 2D image plane using the provided projection matrix. So we defined the function **project**:

```python
def project(points:np.ndarray, P:np.ndarray) -> np.ndarray:
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    
    projected_u = (P[0, :] @ points.transpose()) / (P[2, :] @ points.transpose())
    projected_v = (P[1, :] @ points.transpose()) / (P[2, :] @ points.transpose())
    
    return np.stack([np.array([u, v]) for u, v in zip(projected_u, projected_v)])
```

Now that we have the necessary tools, the following code is executed: 

```python
  # getting the image and extrinsics
img_path = images_path[1]
R1 = all_R[1]
t1 = all_t[1]

# combining R and t
P = u.get_projection_matrix(K, R1, t1)

corners = u.get_corners(img_path, grid_size)
projected_corners = []

error = 0
for index, corner in enumerate(corners):
    u_coord = corner[0]
    v_coord = corner[1]

    grid_size_cv2 = tuple(reversed(grid_size))
    u_index, v_index = np.unravel_index(index, grid_size_cv2)

    # the coordinates of the corner w.r.t. the reference corner at position (0,0) of the corners array
    x_mm = (u_index) * square_size
    y_mm = (v_index) * square_size

    point_m = np.array([x_mm, y_mm, 0, 1])

    projected_u, projected_v = u.project(point_m, P)[0]
    projected_corners.append((projected_u, projected_v))
    
    error += (projected_u - u_coord)**2 + (projected_v - v_coord)**2
print(f"Error: {error:.2f}")
print(f"Mean error per corner: {error/len(corners):.2f}")

```
The results we got were the following: 
  -  Error: 30.01
  -  Mean error per corner: 0.34

The second one is the most interesting: a value of 0.34 means that, on average, the points that the geometric model predicts are located on the image are about a third of a pixel away from their actual position in the image. This is considered a good result overall, meaning that the camera model is geometrically accurate.

To show the projected corners, the code below is executed: 

```python
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore

for corner in projected_corners:
    u_coord, v_coord = int(corner[0]), int(corner[1])
    cv2.circle(image_rgb, (u_coord, v_coord), radius=5, color=(255, 0, 0), thickness=-1)

px.imshow(image_rgb)
```

### Task 3 - Superimposing a cylinder 
The next task requires to superimpose an object, in this case a cylinder, on 25 checkerboards and visualize the correctness of the previous computations and results. 
To complete the task, we defined a function called **superimpose_cylinder**, that in the first place generates the 3D points of a cylinder of a given radius and height by approximating it through sides and slices. Then, given matrix P of an image, projects the 3D cylinder on the 2D image. 

```python
   
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
```

The execution code of the task is right below, in which we recalled the **get_projection_matrix** for each of the 25 images before superimposing the cylinders: 

```python
import random

random.seed(0)
NUM_IMAGES_TO_PROCESS = 25

images_indices = random.sample(range(len(images_path)), NUM_IMAGES_TO_PROCESS)

# 3D parameters of the cylinder (remain fixed for all projections)
radius_mm = 22.0
height_mm = 80.0

# Positioning consistent with the origin of the checkerboard (e.g. 4 squares, 4 squares)
center_x_mm = 5 * square_size 
center_y_mm = 4 * square_size
num_sides_cyl = 30 # Cylinder resolution
num_height_slices_cyl = 5

superimposed_image_list = []

for i in images_indices:
    img_path = images_path[i]
    R_i = all_R[i]
    t_i = all_t[i]
    P = u.get_projection_matrix(K, R_i, t_i)
    
    superimposed_image = u.superimpose_cylinder(
        img_path=img_path, 
        P=P,
        radius=radius_mm, 
        height=height_mm, 
        center_x=center_x_mm, 
        center_y=center_y_mm,
        num_sides=num_sides_cyl,
        num_height_slices=num_height_slices_cyl
    )
    
    superimposed_image_list.append(superimposed_image)
    
px.imshow(superimposed_image_list[0])
```

