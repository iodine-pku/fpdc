#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<mpi.h>

#define GRID_SIZE 128
#define M 128
#define N 128
#define MAX_ITER 20000
#define max(a,b) ((a)>(b)?(a):(b))

int main(int argc, char *argv[]) {


//here begins parallelization
MPI_Init( &argc, &argv );
int rank, size;

MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );



// grid
    
    
    float dx, dy;
    dx = (2.0 / (M - 1));
    dy = (2.0 / (N - 1));
    int max_iter = MAX_ITER;
    if(rank==0)
    {
        printf("run parallel Jacobi with grid size: m = %d n = %d max_iter = %d ...\n", M, N, max_iter);
        printf("strategy:collective\n");

        printf("total %d processes\n",size);
    }

// param
    float relax = 1.0;
    float alpha =  0.05;
   // clock_t time0, time1;
    double duration;

// init     
    float phi[M][N]={0.0};  //use two 2d arrays to store, phi[M][N] and phi_old[M][N]
    float f[M][N]={0.0};
    
    int i, j;
    float xx, yy;
    
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        xx = -1.0 + dx * i;
        yy = -1.0 + dy * j;
        phi[i][j] = 0.0;
        f[i][j] = ((((alpha * (1.0 - (xx * xx))) * (1.0 - (yy * yy))) +
                    (2.0 * (1.0 - (xx * xx)))) +
                   (2.0 * (1.0 - (yy * yy))));
      }
    }
    
    
    int k=1;
    float  ax, ay, axy, b, resid;
    
    float phi_old[N][M] =  {0.0};   

    axy = -1.0/2.0;
    ax = -(dy*dy)/(dx*dx);
    ay = -(dx*dx)/(dy*dy);
    b =  2.0 - 2.0 * (ax) - 2.0 * (ay) + alpha * (dx*dx+dy*dy) ;
  
    double time0;
    float res;
    time0 = MPI_Wtime();
    int com;
    com = M/size;     
    double com_time=0.0;
    while (k <= max_iter) {
    //res =0.0;
    double t3,t4;
    t3 = MPI_Wtime();
    MPI_Allgather(&phi[com*rank][0],M*com,MPI_FLOAT,&phi_old[0][0],M*com,MPI_FLOAT,MPI_COMM_WORLD);
    t4 = MPI_Wtime();   //profiling
    com_time = com_time + t4-t3;
    
        for (i = rank*com ; i < (rank+1)*com ; i = i + 1) {
            if(i==0||i==M-1)
            {
                continue;
            }
          for (j = 0; j < N; j++) {   
            if(j==0||j==N-1)
            {
                continue;
            }       
            resid = (axy * (phi_old[i-1][j-1] + phi_old[i-1][j+1] + phi_old[i+1][j-1] + phi_old[i+1][j+1]) + 
                     ax * (phi_old[i - 1][j] + phi_old[i + 1][j]) +
                     ay * (phi_old[i][j - 1] + phi_old[i][j + 1]) + 
                     b * phi_old[i][j]) / (dx*dx+dy*dy) - f[i][j];
            resid *=  (dx*dx+dy*dy) / b; 
            phi[i][j] = phi_old[i][j] - relax * resid; //update 
            //res = res + resid*resid;
          }
            
        }
           // res = sqrt(res) / (M * N);

            //we're not interested in these errors
        if (k%1000==0  && rank==0)
            printf("%dth iteration completed\n", k);
        k = k+1;
    }



    double time1;
    time1 = MPI_Wtime();

    duration = (double)(time1- time0) ;
    double rate;
    rate = com_time/duration*100;
if(rank==0)
{
    printf("Wall time(second): %f\n", duration);
    printf("Communication time(second): %f, takes %f%% of all walltime\n", com_time,rate);
}

if(rank==0)
{  
    int i, j;
    float xx, yy, tmp_err, error = 0.0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            xx = -1.0 + dx * i;
            yy = -1.0 + dy * j;
            tmp_err = phi_old[i][j] - ((1.0 - (xx * xx)) * (1.0 - (yy * yy)));
            error = max(error, fabs(tmp_err));
        }
    }
    printf("Max Error: %.9f\n", error);   
}
    MPI_Finalize();
    return 0;
}