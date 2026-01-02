# Computer Vision and Pattern Recognition Project

Authors: Lorenzo Cusin – Giacomo Serafini – Pietro Terribile

**AI use**
In this project we used AI tools to:
- write the documentation of the functions in the code;
- ??(altro?)

# Project 1 — Camera Calibration

## Introduction

The **camera calibration problem** consists in estimating the intrinsic and extrinsic parameters of a camera through several measurements.  
The outcome of these calculations is the **Perspective Projection Matrix** $P$, which can be written as:

$$P = K [ R | t ]$$

where:

- $K$ is the intrinsic matrix, containing the internal parameters of the camera (specific to the camera itself);
- $R$ and $t$ are respectively the rotation matrix and the translation vector, describing the camera pose.

## Task 1 - Zhang's Calibration Method

It is required to calibrate the camera (thus finding the unique K and the pair $[R | t]$ for each image) by using the Zhang's procedure, which is based on a key principle: instead of using a single image of many non-coplanar points ??(NON CAPISCO SIGNIFICATO FRASE) -> @@(as is necessary for basic Direct Linear Transform, or DLT, method), Zhang's approach requires multiple images (at least three) of a simple planar calibration pattern.

??(Cercherei di essere meno specifico riguardo al codice e spiegherei più ad alto livello l'algoritmo. Ad esempio, invece di specificare che librerie abbiamo importato e il nome delle funzioni, mi concentrerei di più sui passaggi fatti. Penso che renda più semplice la comprensione del report. Provo a scrivere un esempio della parte qui sotto).



@@(
In our case we are provided with 81 images of a checkerboard, our calibration pattern, where each image is taken from a different point in the World reference frame. The checkerboard is composed by a grid of $(8,11)$ reference corners whose coordinates will be used to estimate the parameters.

The foundation of Zhang's method relies on establishing a mathematical relationship, known as a homography (a matrix $H$), between the known 3D plane in the scene (the checkerboard) and its 2D perspective projection onto the image plane. The corners previously mentioned are usefull in this sense, in fact their world and image coordinates are sufficient to estimate $H$. The first ones are easily derived by fixing the world reference orgin into a point in the checkerboard, in our case the bottom-left corner, and knowing the length of the squares' side; the latter ones, instead, are simply their location in pixels, which can be easily computed using the `findChessboardCorners` OpenCV function (we also used `cornerSubPix` to improve the accuracy of the location). After collecting these data, a system of equation is defined as follows:

$$
A_ih = 0
$$

where:
- $A_i$ are the coefficients of the equations derived by the corner $i$ of the image and whose entries are:
  $$
  \begin{bmatrix}
  x & y & 1 & 0 & 0 & 0 & -ux & -uy & -u \\
  0 & 0 & 0 & x & y & 1 & -vx & -vy & -v
  \end{bmatrix}
  $$
  with $(x,y,z)$ as world coordinates and $(u,v)$ as image coordinates;
- $h$ is a vector of size 9 that contains the entries of the matrix $H$.

All the $A_i$ are stacked together and the overdetermined system solution is solved by means of Singular Value Decomposition (practically, the solution is the last column of the obtained matrix V). 

After the estimation of the homography for each image, to estimate the camera parameters another system of equation must be solved. 
$$[\dots]$$
)
In our case we are provided with 81 images of a checkerboard, where each image is taken from a different point in the World reference frame. 
The foundation of Zhang's method relies on establishing a mathematical relationship, known as a **homography** ($H$), between the known 3D plane in the scene and its 2D perspective projection onto the image plane.
First of all, we needed to import `numpy` and `OpenCV` libraries to our code. Then we followed the **LabLecture_1** steps to find the keypoints between the given images. We defined a function `get_corners` in which we utilized the function `findChessboardCorners` from the OpenCV library, in order to get the corresponcences we needed to estimate the **homography**.

The function `get_corners` is then called inside another function: `get_homography`. This function is used to compute the homography matrix $H$ that allows us to map 3D points on a calibration pattern (the checkerboard) to 2D pixel coordinates in an image. It implements the Direct Linear Transform (DLT) algorithm, which is the first fundamental step in Zhang’s calibration procedure:

- It first calls `get_corners` to detect the $(u, v)$ pixel coordinates of the checkerboard corners in the provided image
- It generates the corresponding world coordinates $(x, y)$ in millimeters by multiplying the grid indices by the `square_size` (11mm in our case). It assumes the board lies on the $Z=0$ plane
- It builds a system of linear equations: $Ah = 0$. Each detected corner contributes two rows to the matrix $A$, ??(NON CAPISCO SIGNIFICATO FRASE)representing the relationship:

$$
\begin{bmatrix}
x & y & 1 & 0 & 0 & 0 & -ux & -uy & -u \\
0 & 0 & 0 & x & y & 1 & -vx & -vy & -v
\end{bmatrix}
$$

- SVD Solver: It uses Singular Value Decomposition to solve the system. The homography parameters are found in the last row of the $V$ matrix (the right-singular vector associated with the smallest singular value)
- Output: The resulting 9-element vector is reshaped into the final 3x3 Homography matrix

Another function, called `get_v_vector`, is used to linearize the constraints that the homography $H$ imposes, ??(NON CAPISCO SOGGETTO)represented by the symmetric matrix $B = K^{-T}K^{-1}$. Since $B$ is symmetric, ??(E' UNA CONSEGUENZA?)it is defined by 6 elements collected in a vector $b$. The function extracts specific products from two columns of the homography matrix ($h_i$ and $h_j$) so that the constraint $h_i^T B h_j$ can be written as the dot product $v_{ij}^T b$. Each homography provides constraints that are stacked into a system of equations. The vector is defined as:

$$
v_{ij} = \begin{bmatrix}
H_{1i}H_{1j} \\
H_{1i}H_{2j} + H_{2i}H_{1j} \\
H_{2i}H_{2j} \\
H_{3i}H_{1j} + H_{1i}H_{3j} \\
H_{3i}H_{2j} + H_{2i}H_{3j} \\
H_{3i}H_{3j}
\end{bmatrix}^T
$$

After that, we wrote two other functions, `get_intrinsic` and `get_extrinsic`, which compute respectively the $K$ and the pair $[R | t]$.  
The first one computes the Singular Value Decomposition (SVD) of the constraints matrix $V$ (in which, given $n$ planes, $2n \times 6$ equations are stacked) and then extracts the smallest singular vector, which is the solution to the problem. After that, it performs the Cholesky decomposition, finding the intrinsic matrix $K$.  
On the other hand, the second function computes ??(e' giusto dire cosi?)column-wise the rotation matrix $R$ and $t$, starting from the fact that $P = [R | t] = K [r1 \ r2 \ r3 | t]$.

??(DA SCRIVERE MEGLIO)Later on the realization of the project, we had to add this portion of code to the function 
```python
    if t[2] < 0:
        t = -t
        lam = -lam
```

This had to be done because there exists two possible solutions to the problem when computing extrinsics, but only one has the right physical meaning: being the checkerboard in front of the camera, we expect the value of $t_z$ to be positive (since we defined the camera reference frame this way, with $Z > 0$), but sometimes this was not true. In the superimposition task, we observed that, for some images, the value was negative and the cylinder was entering the frame rather that getting out. This corresponded to the WRF to be considered behind the camera, which is clearly unfeasable. So, we are able to detect the wrong solution by checking this value and correct it by taking the opposite, which means taking the opposite scale factor $\lambda$.

??(SENSO FRASE)Now that we have everything required the full core pipeline of Zhang’s calibration method, transitioning from raw images to the estimation of camera parameters is performed: first we define the physical properties of the checkerboard (grid dimensions and square size) and load all the available calibration images. Then, for every image we are given, a planar homography $H$ is computed to relate the world coordinates of the board to the image plane with the `get_homography` function. ??(DA RISCRIVERE)From each $H$, the function `get_v_vector` extracts the $v_{ij}$ vectors to enforce the orthogonality and unit scale constraints required to solve for the camera's internal geometry. These vectors are stacked into a global matrix $V$, representing a system of linear equations: $Vb = 0$. The intrinsic matrix $K$ is recovered by solving the linear system, which is performed by the `get_intrinsics` function. In the end, using the finalized $K$ and the homography as inputs of `get_extrinsics`, we find the specific rotation ($R$) and translation ($t$) matrices of the camera relative to the calibration board.

??(HA SENSO PRINTARE UN OUTPUT DI K, R e t DI ESEMPIO?) -> @@(Nell'esercizio 6.1 ho interpretato i significati dei parametri di $K$, quindi volendo si potrebbe riportare quello. Riguardo $R$ e $t$ secondo me ha meno senso.)

## Task 2 - Total Reprojection Error

For this task we are required to choose one of the calibration images and compute the total reprojection
error, i.e. the distance between the projections (coordinates) of the measured image points and the projections estimated by the geometric model of the camera (perspective projection matrix $P$).

First of all, we defined the function `get_projection_matrix` to compute the $P$ matrix for an image given the intrinsics and extrinsics parameters. After that, we had to project 3D points onto a 2D image plane using the provided projection matrix. Thus, we defined the function `project`, which collects the projected pairs ($u,v$):??(MAGARI SPIEGARE UN PO MEGLIO)

At this point, the pipeline calls these functions and the quadratic error is computed. 

The results we got for image 0, shown as an example, were the following: 
  -  Error: 23.09
  -  Mean error per corner: 0.26 ??(MI VENGONO RISULTATI DIVERSI A ME, poi troppo basso considerato che nel 7 Lore dopo il refinement abbia risultati peggiori)

<div style="
  width: 100%;
  text-align: center;
  margin: 2em 0 3em 0;
">
  <img src="imgs_for_CV_project/red_dots.png"
       alt="Chessboard calibration pattern"
       style="display: block; margin: 0 auto; width: 800px;">
  <div style="margin-top: 0.8em; font-style: italic;">
    Figure 1: Projected corners after calibration.
  </div>
</div>

The second data is the most interesting: a value of ??(DA VEDERE)0.26 means that, on average, the points that the geometric model predicts are located on the image are about a quarter of a pixel away from their actual position in the image. This is considered a good result overall, meaning that the camera model is geometrically accurate.

## Task 3 - Superimposing a Cylinder

The next task requires to superimpose an object, in this case a cylinder, on 25 checkerboards and to visualize the correctness of the previous computations and results. 
To complete the task, we defined the `superimpose_cylinder` function. This function creates a 3D cylinder and renders it onto a specific image. First, it generates a set of 3D points in homogeneous coordinates based on a provided radius, height, and center position ($x, y$) on the world plane. The cylinder is approximated using a user-defined number of sides and vertical slices. Then, using the camera's projection matrix $P$, these 3D points are mapped onto the 2D image plane. Finally, the function uses OpenCV's `polylines` function to draw the cylinder's structure.

??(COSA INTENDI?)In the execution code of the task we recalled the `get_projection_matrix` for each of the 25 images before superimposing the cylinders. Observing the results, we noticed that when the slope of the plane is evident to the human eye, the cylinder is correctly inclined with the plane. When the surface is slightly sloped, so much so that it is imperceptible to the naked eye, it is not to the model and the cylinder superimposed is yet inclined. Here we report three cases of interest of our observations.

### MISSING IMAGES

<div style="
  width: 50%;
  text-align: center;
  margin: 2em 0 3em 0;
">
  <img src="imgs_for_CV_project/cylinder.png"
       alt="Example of superimposed cylinder"
       style="display: block; margin: 0 auto; width: 800px;">
  <div style="margin-top: 0.8em; font-style: italic;">
    Figure 2: Example of superimposed cylinder.
  </div>
</div>

## Task 4 - Standard Deviation of the Principal Point

The exercise asks to analyze how much the uncertainty of the principal point changes while the number of images used to estimate the camera intrinsic is increased. The principal point is the point $(u_0, v_0)$ on the image where the camera’s optical axis intersects the image plane. It is one of the intrinsic parameters and for this reason it is contained in the matrix $K$:

$$
K = \begin{bmatrix} 
\alpha_u & \alpha_u \cot\omega & u_0 \\ 
0 & \alpha_v / \sin\omega & v_0 \\ 
0 & 0 & 1 
\end{bmatrix}
$$

To perform the estimation, a statistical approach is used: several batches of images of size $n\_images \in \{a, \dots, b\}$ are randomly sampled and the standard deviation of $(u_0, v_0)$ is computed for each batch size. We think that this approach is more fair with respect to the combinatorial one, in which all the possible combinations of batches of dimension $n\_images$ are considered to compute the standard deviation. In fact, fixing the number of samples to $n\_samples$ permits to the first approach to create the same number of batches for each size, making the comparison more trustable. This key point is not present in the combinatorial one, as there are more combination of $n$ images than $n+1$. Moreover, the computation is more lightweight, making the code faster to execute. 

In what follows, we can see the results obtained by executing the previous explained approach using $n\_samples = 100$. Since the minimum number of images required to compute the camera intrinsic with the Zhang's method is $3$ and $20$ images are enough to show the standard deviation trend, $n\_images \in \{3, \dots, 20\}$ is selected:

![Standard Deviation vs N Images](imgs_for_CV_project/std_vs_nimages.png)

The uncertainty decreases as the number of images increases: this is an expected behaviour. Using more than $7$ images does not appear to significantly improve the accuracy.

## Task 5 - Comparing the Estimated $R,t$ Pairs

In this task it is required to compare the obtained extrinsic parameters $R$ and $t$ with the provided ground truth. The following methods are used to compute the errors:
- rotation matrix $R$ (**Rotation Error**): given two rotation matrices $R_A$ and $R_B$, the error is defined as:
  $$|\theta| = \left|arccos\left(\frac{tr(R_A R_B) - 1}{2}\right)\right|$$
- translation vector $t$ (**Translation Error**): the error is the Euclidean norm of the difference between the two vectors. 

The ground truth is provided for only five images and its $t$ vectors are estimated in meters rather than millimeters. To account for the scale mismatch, the ground truth is multiplied by $1000$.

Here are the obtained results:

![Ground truth comparison](imgs_for_CV_project/ground_truth_comparison.png)

In both cases, the error seems constant for each image. The Translation Error is around $10$ millimeters and it is probably due to the noise present in the estimation process. The Rotation Error, instead, needs a careful analysis. In fact, it is around $\pi = 3.14$ which represent a rotation of $180°$. This phenomena usually happens when the reference system (world or image) of the two cameras are defined differently, for instance with the axes $x$ and $y$ inverted. Because of that, a further investigation is needed.

First of all, let's see how a cylinder is projected using the ground truth parameters. If the problem is due to the definition of the reference system, this test should be enough to make it visible. Here an example with the image `rgb_0.png` is shown. The respective $R$ and $t$ from the ground truth parameters are used, while the $K$ estimated in Task 1 is selected. A cylinder centered at $(0,0)$ is then projected.

![Cylinder with ground truth parameters](imgs_for_CV_project/cylinder_ground_truth_params.png)

It is evident that:
1. the center $(0,0)$ is not precisely located. This may be caused by the two different estimation processes used to derive $K$, $R$ and $t$. This behaviour is assumed to be normal;
2. the cylinder is projected reversed with respect to our way of projecting, e.g. it is growing away from the camera. This seems to confirm our hypothesis.

Let's try to demonstrate the last point estimating our $R$s and $t$s using $x$ and $y$ inverted. To do that, the function `get_homography` is edited as follow:

```python
# Old function
def get_homography(img_path:str, grid_size:tuple, square_size:int) -> np.ndarray:
    ...
    # finding the (x,y) coordinates wrt the checkerboard
    x_mm = u_index * square_size
    y_mm = v_index * square_size
    ...

################### ↓ ###################

# New function
def get_homography(img_path:str, grid_size:tuple, square_size:int) -> np.ndarray:
    ...
    # finding the (x,y) coordinates wrt the checkerboard
    ## inverting x and y
    x_mm = v_index * square_size
    y_mm = u_index * square_size
    ...
```

Basically, the coordinates of the checkerboard's corners are defined with $x$ and $y$ inverted. This change led to the following result:

![Correct Ground truth comparison](imgs_for_CV_project/correct_ground_truth_comparison.png)

Now the Rotation Error is around $0.02rad = 1°$: this definitely confirm our hypothesis. As for the case of the Translation Error, we assess this last difference to the noise present in the estimation process.


*Clearly, keeping the change to the `get_homography` function means defining a world reference system in which the projected objects would grow away from the camera. We think that this definition is less intuitive, so we decide to restore `get_homography` to its initial version.*

## Task 6 - Our Own Calibration 

It is asked to calibrate a new camera and retrace the previous steps: in our case, our camera smartphone is used. Firstly, $30$ pictures of a $(11, 18)$ checkerboard are taken and then a copy of the previous code is created and executed. The images dimension is $4080$x$3072$.

Since the theory and implementation details are described above, here only the results are discussed. Let's break them down point by point:
1. **Zhang's Calibration method**

    The obtained matrix $K$ is:
    
    $$
        K = \begin{bmatrix} 
        \alpha_u = 3258.001 & s = 7.425 & u_0=2039.796 \\ 
        0 & \alpha_vs = 3246.147 & v_0 = 1412.099 \\ 
        0 & 0 & 1 
        \end{bmatrix}
    $$

    Since $\alpha_u \approx \alpha_v$, the sensor pixel shape can be assumed to be a square. The angle between the axis $u$ and $v$, represented by $s$, is small and can be neglected. The pricipal point $(u_0, v_0)$ is vertically shifted with respect to $(\frac{4080}{2}=2040, \frac{3072}{2} = 1536)$, the expected one in an ideal camera. Even if the presence of misalignment between sensor and lenses may cause it, it is also important to notice that in modern smartphones the image captured by the sensor is not the one shown to the user. In fact, post-processing is generally applied, including also image cropping. This may also explain the notable difference in the vertical coordinate.

2. **Total Reprojection Error**
   
    The total reprojection error obtained is $1185.65$, with a mean error per corner equal to $6.97$. Even if these values are extremely higher with respect to the one previously obtained in the project (respectively $41.28$ and $0.47$), it is important to notice that the different pixel density present in the two images can influence the perception of the error. In fact, the same pixel error is more evident in the image with lower pixel density.

    To perform a fair comparison, the following normalized error is computed:

    $$
    normalized\_error = \frac{\sqrt{\sum_i \left(\frac{u_i - \hat u_i}{width}\right)^2 + \left(\frac{v_i - \hat v_i}{height}\right)^2}}{n\_corners}
    $$

    where:
    - $(\hat u_i,\hat v_i)$ are the coordinates of the projected corner;
    - $(u_i, v_i)$ are the ground truth coordinates of the corner;
    - $(width, height)$ are the dimensions of the image;
    - $n\_corners$ is the number of projected corners.

    In this way, each error is weighted with the respective dimension of the image, obtaining an adimensional value:
    - Old images: $0.0010$;
    - New images: $0.0009$.

    *(The error is computed by collectively considering all corners across the images, e.g. $8 \times 11 \times 81 = 7128$ corners for the old images and $10 \times 17 \times 30 = 5100$ for the new ones.)*
  
    The error is basically the same. Here is an example of the corners projection:

    ![Corners Projection - Phone image](./imgs_for_CV_project/phone_image_corners_projection.png)

3. **Superimposing a cylinder**

    The projection of the cylinder appears as expected in all the 25 images. An example is shown:

    ![Cylinder Projection - Phone image](./imgs_for_CV_project/phone_image_cylinder_projection.png)

4. **Standard deviation of principal point**

    ![Principal Point Standard Deviation - ](./imgs_for_CV_project/phone_image_principal_point.png)

    As the number of images increases, the error decreases and reaches a plateu. The magnitude is significantly higher than the one previosuly observed in the project. In this sense, the analysis proposed in point 2 is still considered valid.

5. **Comparing the estimated $R,t$ pairs**

    Since no ground truth is available for our images, this point is not performed.

## Task 7 - Minimize Reprojection Error via MLE

In this exercise we are asked to refine our estimations by minimising the reprojection error using the Maximum Likelihood Estimation. In fact, by following the approach described by Zhang, we are no longer simply minimising the algebraic error used in the closed-form calibration method, but we are instead minimising the sum of squared reprojection errors, which is equivalent to maximising the likelihood of the observed data.  
It is important to note that the reprojection error measures the difference between the observed image points, which are extracted from the checkerboard images, and the projected image points obtained by projecting the known 3D checkerboard corners onto the image plane using the estimated camera parameters (and therefore the estimated projection matrix).

By following Zhang procedure, we know that the maximum likelihood estimate can be obtained by minimising the following functional: 

$$\sum_{i = 1}^n\sum_{j = 1}^m ||m_{ij} - \hat m(A, R_i, t_i, M_j)||^2$$

where $\hat m(A, R_i, t_i, M_j)$ is the projection of point $M_j$ in image $i$.
Thus, the optimisation minimises the sum of squared errors over all images and points: $\min_\theta \sum_{i, j}||m_{ij} - \hat m(A, R_i, t_i, M_j)||^2$, where $\theta$ is the full parameter vector: 
$$\theta = \{\alpha_u, \gamma, u_0, \alpha_v, v_0, r_1, \ldots, r_N, t_1, \ldots, t_N \}$$
??(se non sbaglio r è composto sempre solo da 3 elementi, quindi invece di $r_1 \dots r_N$ metterei $r_1 \dots r_3$.)
where:
- $K = \begin{bmatrix} \alpha_u & \gamma & u_0 \\ 0 & \alpha_v & v_0 \\ 0 & 0 & 1 \end{bmatrix}$ 
- $\bold{r}$ is the rotation vector in axis-angle (Rodrigues) form
- $\bold{t}$ is the translation vector

In order to minimise our objective, we used, as suggested, the Lebenberg-Marquardt algorithm, which is conveniently implemented in the `scipy.optimize` package. Rotations are converted between matrix and axis-angle representations using Rodrigues' formula, which was implemented from scratch in the `get_rot_axis_from_R` and `get_R_from_axis` functions. So, all we did was applying the least-squares method while minimising the residuals in order to obtain the refined parameters.

After convergence, the reprojection error was evaluated, as usual, on the image indexed $1$:
- **Total Reprojection Error:** 26.31
- **Mean Error per Corner:** 0.30

Comparing these results with the ones obtained in the Exercise 3, we can see a clear improvement: we reduced the total error from 41.28 to 26.31 and the mean error per corner from 0.41 to 0.30. We can therefore conclude that this process worked well, and it refined all the parameters of the camera, both the extrinsic and the intrinsic ones.
??(DA AGGIUNGERE L'IMMAGINE? NON MI SEMBRA MOLTO SIGNIFICATIVA, non credo si vedrebbero molte differenze) -> @@(Sono d'accordo che non sia significativa, penso che gli errori bastino)

## Task 8 - Radial Distortion Compensation

In this task, we had to take into consideration the radial distortion, which is the phenomenon where straight lines appear curved in an image (especially at the periphery of the image), caused by light bending more at the lens edges than the center, making pixels shift radially inward or outward. By explicitly modeling radial lens distortion, we can compensate for it, thus making the model more accurate.

As seen in the Professor's notes, we based our procedure on the two parameter, $k_1$ and $k_2$, radial distortion model: 

$$\begin{cases} \hat{u} = (u - u_0)(1 + k_1r_d^2 + k_2r_d^4) + u_0 \\ \hat{v} = (v - v_0)(1 + k_1r_d^2 + k_2r_d^4) + v_0 \end{cases}$$

where $u, v$ are the ideal projections (in absence of radial distortion), $\hat u, \hat v$ are the actual projections and 

$$r_d^2 = \left(\frac{u - u_0}{\alpha_u}\right)^2 + \left(\frac{v - v_0}{\alpha_v}\right)^2$$

The procedure starts by basically following the initial one: we do not consider distortion and so we estimate our parameters via homographies. This provides an initial estimate of the camera parameters under the ideal pinhole assumption.  
Given the estimated intrinsic matrix $K$ and projection matrices $P_i$, the distortion coefficients $k_1$ and $k_2$ are estimated by solving a linear least-squares problem: for each image, we append the equation system, thus obtaining an overdetermined system: 

$$A \begin{bmatrix} k_1 \\ k_2 \end{bmatrix} = b$$

After estimating all the initial parameters, $K^0, R_i^0, t_i^0, P_i^0 \text{ and } k_1^0, k_2^0$, we had to refine them. To do so, we again applied the Levenberg-Marquardt algorithm to perform a nonlinear reprojection error minimisation. The optimised parameter vector included the intrinsic parameters ($\alpha_u, \alpha_v, u_0, v_0$), the radial distortion coefficients ($k_1, k_2$), the rotation and the translation vectors for each image. Again, minimising the sum of squared residuals corresponds to a Maximum Likelihood Estimation of all camera parameters, which, if we re-used Zhang's notation, would be:

$$\sum_{i = 1}^n \sum_{j = 1}^m ||m_{ij} - \hat m(A, k_1, k_2, R_i, t_i, M_j)||^2$$

It is important to note that in this case the intrinsic parameter $\gamma$ was set to 0, so it was not optimised during the procedure.

??(AGGIUNGERE RISULTATI? QUALI? semplicemente un print dei parametri refined?) -> @@(Secondo me i print risulterebbero un po' 'buttati li'. Siccome la descrizione che fai è prettamente teorica, se ci sono delle parti di codice degne di nota potresti inserirle. In generale concluderei dicendo "lasciamo l'analisi dei risultati ottenuti all'esercizio positivo")

## Task 9 - Total Reprojection Error w/ & w/o Radial Distortion Compensation

The purpose of the last exercise is to quantitavely evaluate the impact of radial distortion compensation. To do so, we compared the total and mean reprojection errors of the standard pinhole camera model, with no distortion, and of the radial distortion-aware model.  
The first model was obviously based on the parameters estimated using Zhang's method, where radial distortion was not considered, whereas the second model was based on the parameters obtained and refined in Exercise 8.

For each observed corner $(u_{obs}, v_{obs})$ and corresponding projected point $(u_{proj}, v_{proj})$, the reprojection error is computed as the Euclidean distance in pixel space: 

$$err = \sqrt{(u_{obs} - u_{proj})^2 + (v_{obs} - v_{proj})^2}$$

The total reprojection error is obtained by summing the error over all points and images, while the mean reprohection error is normalised by the total number of corners.

**Results:**
- Model without radial distortion:
  - Total Error: 5945.02
  - Mean Error: 0.834
- Model with radial distortion:
  - Total Error: 960.08
  - Mean Error: 0.135

The mean reprojection error is reduced by more than a factor of 6, from approximately 0.83 px to 0.14 px. The total reprojection error, on the other hand, decreases by over 80%, indicating a substantial improvement.

To conclude, we can see how effective radial distortion compensation in camera calibration is: while Zhang's initial estimates are valid, ignoring lens distortion leads to significant residual errors. By taking it into consideration, and refining all variables through reprojection error minimisation, we achieved far better and more accurate results, and consequently a more realistic camera model.

# References
- Zhang, Zhengyou. A Flexible New Technique for Camera Calibration a Flexible New Technique for Camera Calibration. Vol. 10, 1999, www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf.
- Wikipedia Contributors. “Axis–Angle Representation.” Wikipedia, Wikimedia Foundation, 8 May 2020, https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

??(avete in mente altre references?)