
# Author: Deepsana Shahi
# Feb 2022

# Apply Monte Carlo simulation of 2D Ising model with magnetic field B=0 and J =1.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd



# Initialize the system by creating a NxN lattice using random spin configuration
def init_lattice_state(N):
    lattice_state = np.random.choice([1,-1], size=(N,N))
    return lattice_state

# Create a function to calculate energy
def calc_energy(N,lattice):
    
    E = 0;
    
    # periodic boundary conditions: nearest neighbors
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            spin = lattice[i,j]
            if i+1 == N:
                spinBelow = lattice[0,j]
            else:
                spinBelow = lattice[i+1,j]
            if i-1 == -1:
                spinAbove = lattice[N-1,j]
            else:
                spinAbove = lattice[i-1,j]
            if j+1 == N:
                spinRight = lattice[i,0]
            else:
                spinRight = lattice[i,j+1]
            if j-1 == -1:
                spinLeft = lattice[i,N-1]
            else:
                spinLeft = lattice[i,j-1]
            E += -1/2.*spin*(spinBelow+spinAbove+spinRight+spinLeft)  # divided by 2 in order to avoid double counting the pair of spins
    return E             

def delta_energy(spin_array, J, spin_below, spin_above, spin_right, spin_left):
    
    # The difference in energy for flipping a spin
    
    deltaE = 2*J*spin_array*(spin_below + spin_above + spin_right + spin_left)
    return deltaE

def calc_magnetization(lattice):

    #calculate magnetization for a configuration
    
    mag = np.abs(np.sum(lattice))

    '''
    mag = 0
    for i in range(N):
        for j in range(N):
            mag += lattice[i][j]/weights[i][j]
    '''
    return mag

# Applying Metropolis MC algorithm
def MC_metropolis(N, lattice, beta, num_config):
    
    '''
    Parameters
    N: number of lattice so we have N by N lattice 
    lattice: spin configuration matrix
    beta: beta = 1/kT where k is the boltzmann constant and T is the temperature
    num_config = number of spin configurations
    
    '''
    
    # random spin flip
    for i in range(num_config):
        
       
        a = np.random.randint(0,N)
        b = np.random.randint(0,N)
        #print("Print a, b",a,b)
        spin_array = lattice[a, b]
        
        if a+1 == N:
            new_spinBelow = lattice[0,b]
        else:
            new_spinBelow = lattice[a+1,b]
        if a-1 == -1:
            new_spinAbove = lattice[N-1,b]
        else:
            new_spinAbove = lattice[a-1,b]
        if b+1 == N:
            new_spinRight = lattice[a,0]
        else:
            new_spinRight = lattice[a,b+1]
        if b-1 == -1:
            new_spinLeft = lattice[a,N-1]
        else:
            new_spinLeft = lattice[a,b-1]
                

        dE = delta_energy(spin_array, 1, new_spinBelow, new_spinAbove, new_spinRight, new_spinLeft)
            
        # choose a random number r
        r = np.random.random()
            
        if dE <= 0:
            # flip the spin
            lattice[a,b] = -lattice[a,b]
        elif r < np.exp(-beta*dE):
            lattice[a,b] = -lattice[a,b]
        else:
            pass
                
    return lattice





def main():

    N = 20
    lattice = init_lattice_state(N)
    num_config = 1000
    #energy = calc_energy(5, lattice)
    #print(MC_metropolis(5, lattice, 1/2., 1)) # testing the code to see if the  lattice configuration works for 5 by 5 lattice

    
    # show initial lattice configuration of spins which hasn't reached equilibrium

    
    plt.title('Initial lattice configuration')
    plt.imshow(lattice, cmap = 'RdBu')
    plt.show()
    
    
    
    
    temp_steps = 17
    delta_temp = 0.1
    mc_steps = 100 # total number of configurations 
    
    temp = np.arange(1.5, 3.2, delta_temp)

    E = np.zeros((temp_steps, mc_steps))
    M = np.zeros((temp_steps, mc_steps))
    Config = np.zeros((temp_steps, mc_steps, N, N))
    Config2d = np.zeros((temp_steps))
    array = np.zeros(N**2)
    Xmatrix = np.zeros((mc_steps*temp_steps, N**2))
    for t in range(temp_steps):
        lattice = init_lattice_state(N)
    
    
    
        beta = 1./temp[t]
        thermalize_steps = 4000
        lattice = MC_metropolis(N, lattice, beta, thermalize_steps)
       
        for i in range(mc_steps):
            MC_metropolis(N, lattice, beta, num_config)
            '''
            this MC_metropolis function starts the spin configuration after running thermalize_steps and runs the configuration num_config times and picks the last configuration from it and it repeats mc_steps times.
            '''
            energy = calc_energy(N,lattice)
            magnetization = calc_magnetization(lattice)

            #sum of energy and magnetization after each steps
            E[t][i] = energy
            M[t][i] = magnetization
            lattice = MC_metropolis(N, lattice, beta, num_config)

            Config[t][i] = lattice
           
            for x in range(N):
                for y in range(N):
                    Nsites = N**2
                    k = N*x+y
                    array[k] = Config[t][i][x][y]
            row = t*mc_steps+i

            # Creating Xmatrix to apply PCA
            Xmatrix[row] = array 
    print(Xmatrix[:-2])
    print('Xmatrix length: ',Xmatrix.shape)
           


    plt.figure(figsize=[8, 8])
    plt.plot(E[3], label = "low Temp")  # at t= 1.8
    plt.plot(E[8],label = "critical Temp") #at t = 2.3
    plt.plot(E[-1],label = "high Temp") # at t - 3.1
    plt.legend(loc="upper right")
    plt.xlabel("mc_steps")
    plt.ylabel("Energy")
    plt.show()
    
    
    
    
    
    # sum of all monte carlo steps of energy and magnetization for each temp step
    
    Eave = [np.average(E_temp) for E_temp in E]
    Mave = [np.average(M_temp) for M_temp in M]
    Eave = np.float32(Eave)*(1.0/(N**2))
    Mave = np.float32(Mave)*(1.0/(N**2))

    
    plt.subplot(2,2,1)
    plt.plot(temp, Eave)
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    

    plt.subplot(2,2,2)
    plt.plot(E[-1])
    plt.xlabel("mc_steps")
    plt.ylabel("Energy")
    
    
    plt.subplot(2,2,3)
    plt.plot(temp, Mave)
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")

    plt.subplot(2,2,4)
    plt.plot(M[-1])
    plt.xlabel("mc_steps")
    plt.ylabel("Magnetization")
    plt.show()


    ###########################################################################################
    

    # Using unsupervised machine learning technique: Principal Component Analysis (PCA)


    # fill temp_value with temps
    temp_values = np.hstack([np.repeat(t, mc_steps) for t in temp])

    
    #apply PCA
    pca = PCA(n_components = 2)
    principalComps = pca.fit_transform(Xmatrix)  # Fit the model with Xmatrix and apply the dimensionality reduction on Xmatrix.


    # create a data frame that will have principl component values
    principal_df = pd.DataFrame(data = principalComps, columns = ['PC1','PC2'])
    print('principal components: ',principal_df)


    # Getting the weights of each principal component.
    principal_df_weights = pd.DataFrame(pca.components_)
    print('weights:',principal_df_weights)
    
    
    # Find the explained_variance_ratio. This provides the information about the variance each principal component holds after projecting the data. Sotells you how much information is observed.
    print('Variation for principal component: {}'.format(pca.explained_variance_ratio_))

    
    
    # plot indicating spin configs
    plt.figure(figsize=[12,6])
    plt.scatter(principalComps[:,0], principalComps[:,1], c=temp_values, cmap='coolwarm', alpha=0.5)
    plt.colorbar()
    plt.ylabel('Principal Component 2')
    plt.xlabel('Principal Component 1')
    plt.show()
    

    
    # Plot first  principal component.
    plt.figure(figsize=[8, 8])
    n_comp = 0
    '''
    components_ :ndarray of shape (n_components, n_features) Principal axes in feature space, representing the directions of maximum variance in the data.
    Equivalently, the right singular vectors of the centered input data, parallel to its eigenvectors. The components are sorted by explained_variance_.
    '''
    pc1 = pca.components_[n_comp]/(N**2)
    pc1 = [np.mean(np.abs(np.sum(Xmatrix[temp_values== t, :] * pc1, axis=1))) for t in temp]
    plt.scatter(temp, pc1)
    
    plt.title('PCA 1 component')
    plt.ylabel(r'$<|p_1|>$')
    plt.xlabel('Temperature')
    plt.show()
    

    
    
    

if __name__=="__main__":
    main()
