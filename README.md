# Project 2: Tiled Matrix Multiplication

##### CS-4370-90 - Par. Prog. Many-Core GPUs

##### Professor Liu

##### Nathan Dunn

##### 10-24-19

### Compiling and Running

A source file can be found: 

* ndunn_project2.cu

To compile the program in the Fry environment, run the following command:

`singularity exec --nv /home/containers/cuda92.sif nvcc ndunn_project2.cu -o dunn_mult`

To run the program:

`./dunn_mult`