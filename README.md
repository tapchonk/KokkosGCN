/*
 * ************************************************************************
 *
 *                        Kokkos v. 4.0
 *       Copyright (2022) National Technology  Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
 * See https://kokkos.org/LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
 

 # Kokkos Optimized GCN
 
 This algorithm implements the Graph Convolutional Networks (GCN) algorithm using Kokkos, a programming model for performance portability. 
 It is optimized for both Nvidia GPUs and CPUs using OpenMP. There is also the Tensor-Optimised version.
 
 To run the algorithm, follow these steps:
 
 ## 1. Install CUDA 12.3 (install this with the cuDNN libraries), GCC 12.2.0 and OpenMP4.0/5.0 on your system. (ensure that the necessary modules are loaded in your environment)

 ## 1.b Change the directory for the include and -I and -L folders in the make file to be redirected to where you installed CUDA devkit 12. (Tensor-optimised only)

 ## 1.c Download and unzip the products dataset into the folder containing the version you want to run from (e.g. inside "kokkosGCN"): https://ogb.stanford.edu/docs/nodeprop/
 
 ## 2. Clone the Kokkos GitHub repository by executing the following command in your terminal:
    
    git clone https://github.com/kokkos/kokkos.git kokkos-master
    
 
 ## 3. Ensure that "kokkos-master" is in your home directory.
    
    cd ~/kokkos-master
    
 
 ## 4. Compile the code using the following command for Nvidia GPUs:
    
    make -j KOKKOS_DEVICES=Cuda,OpenMP
    
    or the following command for CPUs:
    
    make -j KOKKOS_DEVICES=OpenMP
    
 
 ## 5. Run the application locally using the following command for Nvidia GPUs:
    
    ./gnnAlgorithm.cuda -Train 2000000 -Test 449029 -Hidden 47 -E 1000 -C 3 -LR 0.2f -AT 80.0f
    
    or the following command for CPUs:
    
    ./gnnAlgorithm.host -Train 2000000 -Test 449029 -Hidden 47 -E 1000 -C 3 -LR 0.2f -AT 80.0f
    
 
 ## 6. To run the application on a batch compute system, prepend `./remoterun.sh` to the above commands.
    Ensure that the script has been updated for the batch compute and for that partition.
    Also ensure that you specify if you want the Cuda-optimised version or the OpenMP-optimised version in the remote run script.
 
 For more information and updates, please visit the Kokkos GitHub repository: [kokkos/kokkos](https://github.com/kokkos/kokkos)

