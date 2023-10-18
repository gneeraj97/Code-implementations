import numpy as np
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Data Mining')

parser.add_argument('--filepath', type=str, default="/Users/neerajgupta/Desktop/Computational Data Science/Sem1/Data mining/Homework/HW2/iris.data")
parser.add_argument('--k', type=int, default=3)

args = parser.parse_args()

file_path = args.filepath
with open(file_path, 'r') as file:
    data = file.read()
    # You can then process or split the data into a list of values
    values = data.split()


iris = np.array([x.split(",") for x in values])

groups = args.k

# print(args.k)
# assigned_df = assign_initial(iris, groups)





def assign_initial(data, k_num):
    l = len(data)
    assign = np.zeros((l,1))
    for i in range(k_num):
        start = int(i*(l/k_num))
        end = int((i+1)*(l/k_num))
        assign[start:end]= i+1
    con = np.concatenate((data, assign), axis = 1)
    return con

def multivariate_normal(data,mean, var):
    det_cov = np.linalg.det(var)
    inv_cov = np.linalg.inv(var)
    n = len(mean)
    # print(n)
    
    exponent = -0.5 * np.dot(np.dot((data - mean).T, inv_cov), (data - mean))
    prefactor = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov))
    
    likelihood = prefactor * np.exp(exponent)
    return likelihood

def multivariate_normal_denom(data,mean, var,pc):
    # det_cov = np.linalg.det(var)
    # inv_cov = np.linalg.inv(var)
    n = len(mean)
    # print("expect",var)
    init = 0
    for i in range(len(mean)):
        det_cov = np.linalg.det(var[i])
        inv_cov = np.linalg.inv(var[i])
        exponent = -0.5 * np.dot(np.dot((data - mean[i]).T, inv_cov), (data - mean[i]))
        prefactor = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov))
        likelihood = prefactor * np.exp(exponent)
        init += likelihood*pc[i]
    return init

def EM(data, k_num,numeric_column):
    
    # initial clustering
    data = assign_initial(data, k_num)
    # print(data)
    
    #initialize mean
    unique_groups = np.unique(data[:,5])
    mean = [0]*len(unique_groups)
    # print(mean)
    for i in range(len(unique_groups)):
        mean[i] = np.mean(data[data[:,5]==unique_groups[i],:-2].astype("float"), axis = 0)
    mean = np.array(mean)
    # print(mean)

    #initialize variance
    var = np.zeros((len(unique_groups),numeric_column,numeric_column))
    # print(mean)
    
    
    for i in range(len(unique_groups)):
        var[i,:] = np.cov(data[data[:,5]==unique_groups[i],:-2].astype("float").T)
    var = np.array(var)


    # initialize prior
    pc = [1/k_num]*k_num

    
    # initialize weights
    wij = np.zeros((len(data), k_num))    


    dist = 99
    e = 0.000001
    cnt = 0
    
    while dist > e:
        # print("yes")
        cnt +=1
        # expectation step
        
        for i in range(k_num):
            for j in range(len(data)):
                
                likelihood = multivariate_normal(data[j,:-2].astype("float"),mean[i], var[i])
                
                posterior = likelihood*pc[i]/multivariate_normal_denom(data[j,:-2].astype("float"),mean, var, pc)
                wij[j,i] = posterior
                
    
        # maximization step
        dist = 0
        mu_i_new = [0]*len(unique_groups)
        var_new = np.zeros((len(unique_groups),numeric_column,numeric_column))
        pc_new = [0]*len(unique_groups)
        for i in range(k_num):
            mu_i_new[i] = np.sum(wij[j,i]*data[j,:-2].astype("float") for j in range(len(data)))/np.sum(wij[j,i]for j in range(len(data)))
            var_new[i] = np.sum(wij[j,i]*np.outer(data[j,:-2].astype("float")-mu_i_new[i],(data[j,:-2].astype("float")-mu_i_new[i]).T) for j in range(len(data)))/np.sum(wij[j,i]for j in range(len(data)))
            pc_new[i] = np.sum(wij[j,i]for j in range(len(data)))/len(data)
            dist += np.abs(np.linalg.norm(mean[i] - mu_i_new[i]))
        mean = np.array(mu_i_new)
        var = var_new
        pc = pc_new
        # print(mean)
        # print(dist)
        # print(var)
        
    return mean, cnt, wij, var

def purity(y_true, y_pred):
    k_num = len(np.unique(y_pred))
    
    intersection = 0
    for i in range(k_num):
        cluster = np.where(y_pred == i+1)[0]
        max_ct = max([np.sum(y_true[cluster] == j+1) for j in range(k_num)])
        intersection += max_ct

    purity = intersection/len(y_true)

    return purity
        
final_mean, iterations, weights, covariance = EM(iris,groups,4)
labels = iris[:, 4]

label_to_int = {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}
true_labels = np.array([label_to_int[label] for label in labels])

Em_labels = np.argmax(weights, axis = 1).reshape(-1,1) +1

sorting = np.argsort(np.linalg.norm(final_mean, axis = 1))

print("Mean:")
for i in sorting:
    rounded_mean = [round(val, 3) for val in final_mean[i]]
    print(rounded_mean)

print("\nCovariance:")
for i in sorting:
    rounded_covariance = np.round(covariance[i], 3)
    print(rounded_covariance, "\n")

print("\nIterations:", iterations)

print("\nCluster Membership:")
for i in sorting:
    cluster_members = np.where(Em_labels == i + 1)[0]
    print(", ".join(str(idx) for idx in cluster_members))

print("\nSizes:", end=" ")
for i in sorting:
    cluster_size = np.sum(Em_labels == i + 1)
    print(cluster_size, end=" ")

Purity = purity(true_labels, Em_labels)
print("\n\nPurity:", round(Purity, 3))
