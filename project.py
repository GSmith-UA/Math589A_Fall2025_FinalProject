import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================

def power_method(A, x0, maxit, tol):
    """Approximate the dominant eigenvalue and eigenvector of a real symmetric matrix A.

    Parameters
    ----------
    A : (n, n) ndarray
        Real symmetric matrix.
    x0 : (n,) ndarray
        Initial guess for eigenvector (nonzero).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence in relative change of eigenvalue.

    Returns
    -------
    lam : float
        Approximate dominant eigenvalue.
    v : (n,) ndarray
        Approximate unit eigenvector (||v||_2 = 1).
    iters : int
        Number of iterations performed.
    """
    # TODO: implement the power method
    x_current = x0
    x_new = x0
    for k in range(0,maxit):
        x_new = A@x_current
        # mu = np.linalg.norm(x_new) # Standard Power Method eigenvalue
        mu = np.dot(x_new,x_current)/np.dot(x_current,x_current) # Rayleigh Quotient
        x_new = x_new/mu # A will always be symmetric here so mu should not be zero... TODO: consider a tolerance check here
        if np.linalg.norm(x_new - x_current) < tol:
            break
    
    mu = np.dot(x_new,x_current)/np.dot(x_current,x_current) 
    return mu,x_new,k+1

    #raise NotImplementedError("power_method not implemented")


# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================

def svd_compress(image, k):
    """Compute a rank-k approximation of a grayscale image using SVD.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    k : int
        Target rank (1 <= k <= min(m, n)).

    Returns
    -------
    image_k : (m, n) ndarray
        Rank-k approximation of the image.
    rel_error : float
        Relative Frobenius error ||image - image_k||_F / ||image||_F.
    compression_ratio : float
        (Number of stored parameters in image_k) / (m * n).
    """
    m = np.size(image)[0]
    n = np.size(image)[1]
    covMatrix = image.T @ image # This sucks but there is a way around it... we might have to modify power method arguments... 
    # TODO: See if we can get away with not computing A.T@A and running the power method on it...
    fNorm_img = np.sqrt(np.linalg.trace(covMatrix)) # Convenient since I have the covariance matrix already

    svList = []
    vecList = [] # TODO: I need to calculate the left/right singular vectors... find a formula and implement
    vecList_Left = []
    for i in range(0,k):
        maxIterations = 1000
        tol = 1e-6
        eig,eigVec,itnumber = power_method(covMatrix,covMatrix[:][0],maxIterations,tol) #TODO Determine max iterations and error tolerance
        assert(itnumber < maxIterations) # I want to stop everything if I have questionable convergence

        svList.append(np.sqrt(eig))  
        orthoVec = orthogonalize_vector(eigVec, vecList)
        vecList.append(orthoVec)
        vecList_Left.append((1/svList[i])*(image@orthoVec))        

        # Now we deflate
        covMatrix = covMatrix - eig*(orthoVec @ orthoVec.T) # TODO: Think about if this effects speed? overwriting the memory over and over again??
    image_k = svList[0]*(vecList_Left[0])@vecList[0].T
    for i in range(1,k):
        image_k += svList[i]*(vecList_Left[i])@vecList[i].T
    
    # Now that we have the imgApprox... compute its F norm... tedious... find something fast here?
    fNorm_diff = np.sqrt( np.linalg.trace((image - image_k).T@(image - image_k)))
    rel_error = fNorm_diff/fNorm_img

    compression_ratio = k*(m+n+1)/(m*n) 
    
    return image_k,rel_error,compression_ratio
    #raise NotImplementedError("svd_compress not implemented")


# =========================================================
# 3. SVD-based feature extraction
# =========================================================

def svd_features(image, p):
    """Extract SVD-based features from a grayscale image.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    p : int
        Number of leading singular values to use (p <= min(m, n)).

    Returns
    -------
    feat : (p + 2,) ndarray
        Feature vector consisting of:
        [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]
    """
    print("Running svd_features...")
    m,n = np.shape(image)
    covMatrix = image.T@image
    E_total = np.linalg.trace(covMatrix) # This is equiv to frobenius norm squared

    singularValues = []
    rightVectors = []

    maxIterations = 1000
    powerTol = 1e-9

    maxRank = min(m,n)
    k = 0
    while (k < maxRank): # Should compute the all possible singular values...
        if (np.linalg.norm(covMatrix)) < 1e-9:
            break
        eig,eigVec,itnumber = power_method(covMatrix,covMatrix[:][0],maxIterations,powerTol)
        singularValues.append(np.sqrt(eig))
        orthoVec = orthogonalize_vector(eigVec,rightVectors)
        rightVectors.append(orthoVec) 

        covMatrix = covMatrix - eig*(orthoVec @ orthoVec.T)
        k += 1

    missingValues = maxRank - len(singularValues)
    if missingValues > 0: # We must be rank deficient... pad with zeros...
        singularValues = singularValues + [0]*missingValues
    
    sigmaSum = np.sum(singularValues)
    normalizeSigmas = np.array(singularValues)/sigmaSum

    r_9 = maxRank
    r_95 = maxRank
    r_9set = False
    r_95set = False
    runningEnergyTotal = 0
    for i in range(0,maxRank):
        runningEnergyTotal += singularValues[i]**2
        energyRatio = runningEnergyTotal/E_total
        if (energyRatio > 0.9) and not(r_9set):
            r_9 = i+1
            r_9set = True
        if (r_9set) and (energyRatio > 0.95) and not(r_95set):
            r_95 = i+1
            r_95set = True

    feat = normalizeSigmas[:p]
    feat = np.concatenate((feat,np.array([r_9])))
    feat = np.concatenate((feat,np.array([r_95])))

    return feat

    # raise NotImplementedError("svd_features not implemented")


# =========================================================
# 4. Two-class LDA: training
# =========================================================

def lda_train(X, y):
    """Train a two-class LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix (rows = samples, columns = features).
    y : (N,) ndarray
        Labels, each 0 or 1.

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction vector (not necessarily unit length).
    threshold : float
        Threshold in 1D projected space for classifying 0 vs 1.
    """
    # Let's seperate the data
    X0 = X[y == 0]
    X1 = X[y == 1]

    # I want the mean across all the samples...
    mu0 = np.mean(X0,axis=0)
    mu1 = np.mean(X1,axis=0)
    deltaMu = mu1 - mu0

    # Center the matrices...
    C0 = X0 - mu0
    C1 = X1 - mu1

    # Compute the scatter...
    S0 = C0.T@C0
    S1 = C1.T@C1
    Sw = S0+S1

    # I might need to use a custom solver here... 
    # Maybe even a pseudo-inverse... jesus christ...
    # For now just use linalg... TODO: Fix this... see Tychnoff Regularization??
    # Again some linear algebra calls.... might be bad...
    det = np.linalg.det(Sw)
    singular = (np.abs(det) < 1e-9)
    if singular:
        w = np.linalg.pinv(Sw) @ deltaMu
    else:
        w = np.linalg.solve(Sw,deltaMu)

    # And now we can project...
    p0 = np.dot(w,mu0)
    p1 = np.dot(w,mu1)
    threshold = (p0 + p1)/2

    return w,threshold
    # raise NotImplementedError("lda_train not implemented")


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================

def lda_predict(X, w, threshold):
    """Predict class labels using a trained LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    w : (d,) ndarray
        Discriminant direction (from lda_train).
    threshold : float
        Threshold (from lda_train).

    Returns
    -------
    y_pred : (N,) ndarray
        Predicted labels (0 or 1).
    """
    N = np.shape(X)[0]
    # We can compute all the projections using matrix mult
    Z = X@w # Z should be Nx1
    # Loop through and check... pre-filled with zeros so just fill in the needed ones
    y_pred = (Z >= threshold).astype(int) # Slick idea if it can work...
    
    return y_pred
    # raise NotImplementedError("lda_predict not implemented")

# =========================================================
# 6. orthog checking...
# =========================================================
def orthogonalize_vector(new_vector, basis_list):
    """
    Orthogonalizes a new vector against an existing list of orthonormal basis vectors
    (Modified Gram-Schmidt process, which is often more stable).

    Parameters
    ----------
    new_vector : np.ndarray
        The vector to be corrected (e.g., the eigenvector found by the Power Method).
    basis_list : list of np.ndarrays
        The existing list of orthonormal vectors (e.g., vecList).

    Returns
    -------
    corrected_vector : np.ndarray
        The orthonormal vector that is orthogonal to all vectors in basis_list.
    """
    # print("Running ortho vec....")
    # Start with a copy to avoid modifying the original input (eigVec)
    v = new_vector.copy() 
    
    # Perform the Gram-Schmidt subtraction (Projection Subtraction)
    for u in basis_list:
        # u is assumed to be an orthonormal vector from the previous steps.
        # Projection: proj_u(v) = (v . u) * u
        
        # Calculate the scalar projection length (inner product)
        scalar_projection = np.dot(v, u)
        
        # Subtract the projection
        v = v - (scalar_projection * u)
        
    # Final step: Normalize the resulting orthogonal vector
    norm_v = np.linalg.norm(v)
    
    # Handle the case where the vector is near-zero (linearly dependent)
    if norm_v < 1e-12: 
        # Return a zero vector, which simplifies the rank-k reconstruction
        return np.zeros_like(new_vector)
    else:
        # Return the final orthonormal vector
        return v / norm_v
    
# =========================================================
# Simple self-test on the example data
# =========================================================

def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # #################################REMOVE
    # N_TEST_SAMPLES = 100 # Define the number of images you want to use
    
    # # Subset Training Data (using the first N_TEST_SAMPLES images)
    # X_train = X_train[:N_TEST_SAMPLES]
    # y_train = y_train[:N_TEST_SAMPLES]
    
    # # Subset Testing Data (using the first N_TEST_SAMPLES images)
    # X_test = X_test[:N_TEST_SAMPLES]
    # y_test = y_test[:N_TEST_SAMPLES]
    # #################################REMOVE
    # print(f"--- Running Test with Subset of {N_TEST_SAMPLES} samples per set ---")


    # Sanity check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(5, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    # Build feature matrices
    def build_features(X):
        feats = []
        for img in X:
            feats.append(svd_features(img, p))
        return np.vstack(feats)

    try:
        Xf_train = build_features(X_train)
        Xf_test = build_features(X_test)
    except NotImplementedError:
        print("Implement 'svd_features' first to run this example.")
        return

    print("Feature dimension:", Xf_train.shape[1])

    try:
        w, threshold = lda_train(Xf_train, y_train)
    except NotImplementedError:
        print("Implement 'lda_train' first to run this example.")
        return

    try:
        y_pred = lda_predict(Xf_test, w, threshold)
    except NotImplementedError:
        print("Implement 'lda_predict' first to run this example.")
        return

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # This allows students to run a quick local smoke test.
    _example_run()
