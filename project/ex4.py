import matplotlib.pyplot as plt
# =========================================================================
V = []          
all_H = []      
all_K = []      
all_N_views = []

for i, img_path in enumerate(images_path):
    
    try:
        _, H = u.get_homography(img_path, grid_size, square_size)
        all_H.append(H)
    except Exception as e:
        print(f"Skipping image {i+1} ('{img_path}') - Homography error: {e}")
        continue 
    
    v_12 = u.get_v_vector(H, 1, 2)
    v_11 = u.get_v_vector(H, 1, 1)
    v_22 = u.get_v_vector(H, 2, 2)
    
    V.append(v_12)
    V.append(v_11 - v_22)
    
    num_constraints = len(V)
    num_imgs = len(V) // 2

    if num_constraints >= 6:
        V_matrix = np.array(V) 
        
        try:
            K_i = u.get_intrinsic(V_matrix)
            all_K.append(K_i)
            all_N_views.append(num_imgs) 
            print(f"K computed with {num_imgs} images (constraints: {num_constraints})")
        
        except (np.linalg.LinAlgError, ValueError) as e:

            print(f"Error with {num_imgs} images. Error: {e}")
            
            if all_K:
                all_K.append(all_K[-1]) 
            else:
                all_K.append(np.full((3, 3), np.nan)) 
            
            all_N_views.append(num_imgs) 
    else:
        print(f"Unsufficient constraints ({num_constraints}). K not computed for image {i+1}.")

if all_K:
    print(f"K final:\n{all_K[-1]}")


all_u0 = np.array(all_K)[:,0,2]
print(f"List of all u0 (all_u): {all_u0}")
all_v0 = np.array(all_K)[:,1,2]
print(f"List of all v0 (all_v): {all_v0}")

std_u0_cumulativa = [np.std(all_u0[:i+1]) for i in range(len(all_u0))]
std_v0_cumulativa = [np.std(all_v0[:i+1]) for i in range(len(all_v0))]


plt.figure(figsize=(10, 6))

plt.plot(all_N_views, std_u0_cumulativa, 'o-', color='skyblue', label='Dev. Std. di $u_0$')
plt.plot(all_N_views, std_v0_cumulativa, 's-', color='coral', label='Dev. Std. di $v_0$')

plt.title('STDV vs. Number of Images')
plt.xlabel('Number of images (N)')
plt.ylabel('Standard deviation $\sigma$ (Pixel)')
plt.grid(True, linestyle=':')
plt.legend()

plt.tight_layout()
plt.show()

# =========================================================================
"""
Zero STDEV (N=3 to N=6): The zero values for N up to 6 (or the first non-zero point)
are due to data redundancy caused by numerical instability.
The calibration requires a minimum of N=3 views (6 constraints). 
When the estimation fails due to noisy data (i.e., the "Matrix is not positive definite" error) for N=4, 5 and 6, 
the code employs a fall-back strategy, repeating the first valid estimate (K_3) in the all_K list.
Since the set of estimates K_3, K_4, K_5, K_6 contains only one unique value K_3, the standard deviation of this set is mathematically zero.
"""
# =========================================================================