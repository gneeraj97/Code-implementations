import pandas as pd
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt

original_stdout = sys.stdout
output_file_name = "output.txt"

with open(output_file_name, "w") as output_file:
    # Redirect stdout to the output file
    sys.stdout = output_file

    # Question a
    print("\n\n Question a ")
    parser = argparse.ArgumentParser(description='Data Mining Assignment')
    parser.add_argument('--filepath', type=str, default="./magic_dataset/magic04.data")
    args = parser.parse_args()
    
    df = pd.read_csv(args.filepath, header = None)
    
    df = np.array(df)
    features = np.array((df[:,:-1]), dtype=np.float64)
    centered = (features - np.mean(features, axis = 0))/np.std(features, axis = 0)
    
    # Transposing the data as we take each data point as a d dimensional vector
    print("This is the centered data = \n ", centered)
    
    # Question b
    print("\n\n Question b")
    def cal_var(data):
        data = data.T
        rows, columns = data.shape
        sum = np.empty((rows, rows))
        for i in range(columns):
            sum += np.dot(data[:, [i]],data[:, [i]].T)
        cal_cov = sum/columns
        return cal_cov
    
    # covariance from our method
    cal_cov = cal_var(centered)
    print("\nCovariance matrix from implemented method = \n", cal_cov)
    
    # covariance from the numpy function
    bias_t =np.cov(centered.T, bias = True)
    print("\n\n\n\nCovariance matrix from the numpy.cov function, (Bias has been put to True i.e dividing by N) = \n", bias_t)
    
    print("\n\nCovariance matrix from the implemented method and numpy.cov are the same matrix")
    
    # Question c
    print("\n\nQuestion c ")
    def norm(v1,v2):
        if len(v1) != len(v2):
            print("Vector not of same length")
            exit()
        sum = 0
        for i in range(len(v1)):
            sum += (v1[i] - v2[i])**2
        return np.sqrt(sum)
    
    def eigen(cal_cov):
        dim,dim = cal_cov.shape
        eigen1 = np.ones((dim,1 ))*2
        curr = eigen1 
        prev = np.ones((dim,1 ))*4
        while (norm(curr, prev)>0.000001):
            prev = curr
            # print("Update sequence")
            curr = np.dot(cal_cov, curr )
            val = np.max(curr)/np.max(prev)
            curr = curr/np.max(curr)
        
        eigen1 = curr/np.sqrt(np.sum([x**2 for x in curr]))
        return eigen1, val
    
    eig, val= eigen(cal_cov)
    
    print("\n\nThe eigen vector obtained from the implemented function is = \n", eig)
    print("\n\nThe eigen value obtained from the implemented function is = ", val)
    
    eigenval_np,eignevec_np  = np.linalg.eig(cal_cov)
    # print(np.linalg.eig(cal_cov.T))
    print("\n\nThe eigen vectors obtained from the linalg.eig function is = \n", (eignevec_np[:,np.argsort(eigenval_np)[::-1]])[:,[0]])
    print("\n\nThe eigen values obtained from the linalg.eig function is = ", (eigenval_np[np.argsort(eigenval_np)[::-1]])[0])
    
    print("\n\nThe eigen vector obtained from the linalg.eig function and the implemented function have the same value but just the opposite direction. Since the subspace has run from -vector to +vector, this will not make a difference")
    
    # Question d
    print("\n\nQuestion d")
    eigenval_np,eignevec_np  = np.linalg.eig(cal_cov)
    dominant_eigen_value = np.argsort(eigenval_np)[::-1]
    eigenval_np_sorted = eigenval_np[dominant_eigen_value]
    dominant_eigen_vectors = eignevec_np[:,dominant_eigen_value[:2]]
    
    projected_data = np.dot(centered,dominant_eigen_vectors )
    variance = np.var(projected_data[:,0]) + np.var(projected_data[:,1])
    
    print("\n\nThe variance of the projected data is = ",variance)
    
    # Question e
    print("\n\nQuestion e ")
    print("\n\nThis is the U matrix = \n\n",eignevec_np[:, dominant_eigen_value])
    print("\n\nThis is the Diagonal matrix = \n\n",np.diag(eigenval_np[dominant_eigen_value]))
    print("\n\nThis is the U^T matrix = \n\n",eignevec_np[:,dominant_eigen_value].T)
    
    # Question f
    print("\n\nQuestion f")
    def mse(data1, data2):
        me = np.trace(cal_var(data1)) - np.trace(cal_var(data2))
        return me
    print("\n\nMSE from the implemented subroutine = ",mse(centered, projected_data))
    print("\n\nMSE from summing the remaining eigen values = ",np.sum(eigenval_np_sorted) - np.sum(eigenval_np_sorted[:2]))
    
    # Question g
    print("\n\nQuestion g\n\n")
    labels = df[:,10].reshape(-1,1)
    combined = np.concatenate((projected_data, labels), axis = 1)
    plt.scatter(combined[combined[:,2]=="h", [0]],combined[combined[:,2]=="h", [1]], c="b", label = "Class h")
    plt.scatter(combined[combined[:,2]=="g", [0]],combined[combined[:,2]=="g", [1]], c="y", label = "Class g")
    plt.legend()
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.title("Plot of Magic dataset along the two principal component")
    plt.show()
    
    # Question h
    print("\n\nQuestion h")
    def pca(data_main, alpha = 0.95):
        data = (data_main - np.mean(data_main, axis = 0))/np.std(data_main, axis = 0)
        cov = cal_var(data)
        eigenval_np,eignevec_np  = np.linalg.eig(cov)
        dominant_eigen_value_index = np.argsort(eigenval_np)[::-1]
        eigenval_np_sorted = eigenval_np[dominant_eigen_value_index]
        eignevec_np_sorted = eignevec_np[:,dominant_eigen_value_index]
        total_var = np.sum(eigenval_np_sorted)
        
        for i in range(len(eigenval_np_sorted)):
            capt = np.sum(eigenval_np_sorted[:i])/total_var
            if capt> alpha:
                components = i
                break
        print("\n\nComponents required to capture the 95% variance is = ", components)
        projected_data = np.dot(data, eignevec_np_sorted[:,:components])
        # print(projected_data)
        
        ms = mse(data.T, projected_data.T)
        # print(total_var-ms)
        # print(projected_data[:10,:])
        return projected_data
    
    final_projected_data = pca(features)
    print("\n\nThe first 10 data points of the projected data basis new component vectors are = \n",final_projected_data[:10,:])

    
    # Replace this with the actual code or script you want to execute

# Restore the original stdout
sys.stdout = original_stdout




