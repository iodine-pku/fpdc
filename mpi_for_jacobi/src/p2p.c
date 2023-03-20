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

double wait_time=0.0;
double sync_time=0.0;
double t5, t6;

// grid
    
    
    float dx, dy;
    dx = (2.0 / (M - 1));
    dy = (2.0 / (N - 1));
    int max_iter = MAX_ITER;
    if(rank==0)
    {
        printf("run parallel Jacobi with gird size: m = %d n = %d max_iter = %d ...\n", M, N, max_iter);
        printf("strategy:p2p\n");
        printf("total %d processes\n",size);
    }

// param
    float relax = 1.0;
    float alpha =  0.05;
   // clock_t time0, time1;
    double duration;

//  init     
    float phi[M][N]={0.0};  
    float f[M][N]={0.0};
    
    int i, j;
    float xx, yy;
    
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        xx = -1.0 + dx * i;
        yy = -1.0 + dy * j;
  //      phi[i][j] = 0.0;
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
 // iteration

    double time0;
    float res;
    time0 = MPI_Wtime();
    int com;
    com = M/size;
    float left_border[M]={0};
    float right_border[M]={0};
    MPI_Request request_1,request_2,request_3,request_4;
    MPI_Status st1,st2,st3,st4;
    int flag = 0;

//begins the BIG LOOP
//double com_time=0.0;
//double tt1,tt2;

if(size==1)  //size 1 needs special dealment
{
    while (k <= max_iter) {

    if(flag==0)
    {
        for (i = rank*com ; i < (rank+1)*com ; i = i + 1) 
            {
                if(i==0||i==M-1)
                {
                    continue;
                }
                for (j = 0; j < N; j++) 
                    {   
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
    }
    if(flag==1)
    {
         for (i = rank*com ; i < (rank+1)*com ; i = i + 1) 
            {
                if(i==0||i==M-1)
                {
                    continue;
                }
                for (j = 0; j < N; j++) 
                    {   
                        if(j==0||j==N-1)
                        {
                            continue;
                        }     
                        
                            resid = (axy * (phi[i-1][j-1] + phi[i-1][j+1] + phi[i+1][j-1] + phi[i+1][j+1]) + 
                                    ax * (phi[i - 1][j] + phi[i + 1][j]) +
                                    ay * (phi[i][j - 1] + phi[i][j + 1]) + 
                                    b * phi[i][j]) / (dx*dx+dy*dy) - f[i][j];
                            resid *=  (dx*dx+dy*dy) / b; 
                            phi_old[i][j] = phi[i][j] - relax * resid; //update 
                            //res = res + resid*resid;
                        
                    }
                        
                    }
    }
    if (k%1000==0  && rank==0)
    printf("%dth iteration completed\n", k);
    k = k+1;
    flag = (flag+1)%2;
    }
}

if(size!=1)
{
double t3,t4;
while (k <= max_iter) {

    
    if(flag==0)  //two M*N arrays are used to store data to save space, flag is used to determine update which of the two arrays
    {
            //communication part for all processes
            if(rank!=0 && rank!=size-1)
            {
            
            MPI_Irecv(left_border,M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_1);
            MPI_Irecv(right_border,M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_2);
            MPI_Isend(&phi_old[rank*com],M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_3);
            MPI_Isend(&phi_old[rank*com+com-1],M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_4);


            }

            if(rank==0)
            {
            
            MPI_Irecv(right_border,M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_2);
            MPI_Isend(&phi_old[rank*com+com-1],M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_4);


            }

             if(rank==size-1)
            {
                            MPI_Irecv(left_border,M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_1);
                            MPI_Isend(&phi_old[rank*com],M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_3);



            }



        
            //computation for middle parts
            for (i = rank*com+1 ; i < (rank+1)*com-1 ; i = i + 1) 
            {
                if(i==0||i==M-1)
                {
                    continue;
                }
                for (j = 0; j < N; j++) 
                    {   
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
        //wait until communication is done,for all processes
        t3 = MPI_Wtime();
        if(rank!=0&&rank!=size-1)
        {   
            MPI_Wait(&request_1,&st1);
            MPI_Wait(&request_2,&st2);
            MPI_Wait(&request_3,&st3);
            MPI_Wait(&request_4,&st4);
            
        }
        if(rank==0)
        {   
            MPI_Wait(&request_2,&st2);
            MPI_Wait(&request_4,&st4);
            
        }
        if(rank==size-1)
        {   
            MPI_Wait(&request_1,&st1);
            MPI_Wait(&request_3,&st3);
            
        }
        t4 = MPI_Wtime();
        wait_time =wait_time+t4-t3;
        // upper border for process rank
            i = rank*com;
        if(rank!=0)
        {
            for (j = 0; j < N; j++) 
                    {   
                        if(j==0||j==N-1)
                        {
                            continue;
                        }     
                        
                            resid = (axy * ( phi_old[i+1][j-1] + phi_old[i+1][j+1]) + 
                                    ax * (  phi_old[i + 1][j]) +
                                    ay * (phi_old[i][j - 1] + phi_old[i][j + 1]) + 
                                    b * phi_old[i][j]+ axy*(left_border[j-1]+left_border[j+1])+ax * left_border[j]) / (dx*dx+dy*dy) - f[i][j];
                            resid *=  (dx*dx+dy*dy) / b; 
                            phi[i][j] = phi_old[i][j] - relax * resid; //update 
                            //res = res + resid*resid;
                        
                    }
        }
        // lower border for process rank

            i = rank*com+com-1;
        if(rank!=size-1)
        {
            for (j = 0; j < N; j++) 
                    {   
                        if(j==0||j==N-1)
                        {
                            continue;
                        }     
                        
                            resid = (axy * ( phi_old[i-1][j-1] + phi_old[i-1][j+1]) + 
                                    ax * (  phi_old[i - 1][j]) +
                                    ay * (phi_old[i][j - 1] + phi_old[i][j + 1]) + 
                                    b * phi_old[i][j]+ axy*(right_border[j-1]+right_border[j+1])+ax * right_border[j]) / (dx*dx+dy*dy) - f[i][j];
                            resid *=  (dx*dx+dy*dy) / b; 
                            phi[i][j] = phi_old[i][j] - relax * resid; //update 
                            //res = res + resid*resid;
                        
                    }
        }
    }  
   if(flag==1)
    {
            //communication part for all processes
            if(rank!=0 && rank!=size-1)
            {
            
            MPI_Irecv(left_border,M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_1);
            MPI_Irecv(right_border,M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_2);
            MPI_Isend(&phi[rank*com],M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_3);
            MPI_Isend(&phi[rank*com+com-1],M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_4);


            }

            if(rank==0)
            {
            
            MPI_Irecv(right_border,M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_2);
            MPI_Isend(&phi[rank*com+com-1],M,MPI_FLOAT,rank+1,111,MPI_COMM_WORLD,&request_4);


            }

             if(rank==size-1)
            {
                MPI_Irecv(left_border,M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_1);
                MPI_Isend(&phi[rank*com],M,MPI_FLOAT,rank-1,111,MPI_COMM_WORLD,&request_3);



            }




            //computation for middle parts
            for (i = rank*com+1 ; i < (rank+1)*com-1 ; i = i + 1) 
            {
                if(i==0||i==M-1)
                {
                    continue;
                }
                for (j = 0; j < N; j++) 
                    {   
                        if(j==0||j==N-1)
                        {
                            continue;
                        }     
                        
                            resid = (axy * (phi[i-1][j-1] + phi[i-1][j+1] + phi[i+1][j-1] + phi[i+1][j+1]) + 
                                    ax * (phi[i - 1][j] + phi[i + 1][j]) +
                                    ay * (phi[i][j - 1] + phi[i][j + 1]) + 
                                    b * phi[i][j]) / (dx*dx+dy*dy) - f[i][j];
                            resid *=  (dx*dx+dy*dy) / b; 
                            phi_old[i][j] = phi[i][j] - relax * resid; //update 
                            //res = res + resid*resid;
                        
                    }
                        
                    }
        //wait until communication is done,for all processes
        t3 = MPI_Wtime();

        if(rank!=0&&rank!=size-1)
        {   
            MPI_Wait(&request_1,&st1);
            MPI_Wait(&request_2,&st2);
            MPI_Wait(&request_3,&st3);
            MPI_Wait(&request_4,&st4);
            
        }
        if(rank==0)
        {    //leftmost process doesn't send information 
            MPI_Wait(&request_2,&st2);
            MPI_Wait(&request_4,&st4);
            
        }
        if(rank==size-1)
        {
            MPI_Wait(&request_1,&st1);
            MPI_Wait(&request_3,&st3);
        }
        t4 = MPI_Wtime();
            wait_time =wait_time+t4-t3;
        // upper border for process rank
            i = rank*com;
        if(rank!=0)
        {
            for (j = 0; j < N; j++) 
                    {   
                        if(j==0||j==N-1)
                        {
                            continue;
                        }     
                        
                            resid = (axy * ( phi[i+1][j-1] + phi[i+1][j+1]) + 
                                    ax * (  phi[i + 1][j]) +
                                    ay * (phi[i][j - 1] + phi[i][j + 1]) + 
                                    b * phi[i][j]+ axy*(left_border[j-1]+left_border[j+1])+ax * left_border[j]) / (dx*dx+dy*dy) - f[i][j];
                            resid *=  (dx*dx+dy*dy) / b; 
                            phi_old[i][j] = phi[i][j] - relax * resid; //update 
                            //res = res + resid*resid;
                        
                    }
        }
        // lower border for process rank

            i = rank*com+com-1;
        if(rank!=size-1)
        {
            for (j = 0; j < N; j++) 
                    {   
                        if(j==0||j==N-1)
                        {
                            continue;
                        }     
                        
                            resid = (axy * ( phi[i-1][j-1] + phi[i-1][j+1]) + 
                                    ax * (  phi[i - 1][j]) +
                                    ay * (phi[i][j - 1] + phi[i][j + 1]) + 
                                    b * phi[i][j]+ axy*(right_border[j-1]+right_border[j+1])+ax * right_border[j]) / (dx*dx+dy*dy) - f[i][j];
                            resid *=  (dx*dx+dy*dy) / b; 
                            phi_old[i][j] = phi[i][j] - relax * resid; //update 
                            //res = res + resid*resid;
                        
                    }
        }
    }  
/*
t5 = MPI_Wtime();
MPI_Barrier(MPI_COMM_WORLD);
t6 = MPI_Wtime();
sync_time =sync_time + t6- t5;
*/
if (k%1000==0  && rank==0)
    printf("%dth iteration completed\n", k);
k = k+1;
flag = (flag+1)%2;
        
}
}


    double time1;
    time1 = MPI_Wtime();

 //   if(rank==0)
 //   {
 //       printf("exited loop stage");
 //   }

    duration = (double)(time1- time0) ;
if(rank==0)
{   double ratio;
    double rate;
    ratio = (wait_time+sync_time)/duration*100;
    rate = (wait_time)/duration*100;

    printf("Wall time(second): %f\n", duration);
    printf("Total wait time(second): %f\n", wait_time);
    //printf("Total sync time(second): %f\n", sync_time);
    printf("wait takes up %f%% of Wall time\n", rate);



}


    float  tmp_err,tp_err, error = 0.0;
    xx = 0.0;
    yy = 0.0;

    float terror;
    for (i = rank*com;i<(rank+1)*com;i++) {
        for (j = 0; j < N; j++) {
            xx = -1.0 + dx * i;
            yy = -1.0 + dy * j;
            tmp_err = phi_old[i][j] - ((1.0 - (xx * xx)) * (1.0 - (yy * yy)));
            tp_err = phi[i][j]- ((1.0 - (xx * xx)) * (1.0 - (yy * yy)));
            error = max(error, fabs(tmp_err));
             error = max(error, fabs(tp_err)); //phi[M][N] and phi_old[M][N] differ minorly(1 iteration), can all be used as final result

        }
    }
    MPI_Reduce(&error,&terror,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0)
    {
    printf("Max Error: %.9f\n", terror);
    }
    MPI_Finalize();
    return 0;
}





