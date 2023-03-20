import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import timeit
from pycuda.compiler import SourceModule

mod2 = SourceModule("""
__global__
void SpMV_kernel_64(double *x, double *y) 
{
    int R1 = blockIdx.x * blockDim.x + threadIdx.x;
    int R2 = blockIdx.y * blockDim.y + threadIdx.y;
    int R3 = blockIdx.z * blockDim.z + threadIdx.z;
    int NX = 128;


    double sum = 0.0;
    for (int bz = -1; bz < 2; bz++) {
        for (int by = -1; by < 2; by++) {
            for (int bx = -1; bx < 2; bx++) {
                if(R1 + bz >= 0 && R1+bz < NX && R2+by >= 0 && R2+by < NX && R3+bx >= 0 && R3+bx < NX)
                sum += x[(R1+bz)*NX*NX+(R2+by)*NX+(R3+bx)];
            }
        }
    }
    y[(R1) * (NX) * (NX) + (R2) * (NX) + R3] = -sum + 27.0 * x[(R1) * (NX) * (NX) + (R2) * (NX) + R3];
}

""")
spmv_gpu_64 = mod2.get_function("SpMV_kernel_64")

mod9 = SourceModule("""
__global__
void apxby_kernel_1d(double *x,  double *y, double *des, double *alpha, double *beta)
{
    int R1 = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = R1;
    des[ind] = alpha[0]*x[ind] + beta[0]*y[ind];
            
}
""")

apxby_gpu_1d = mod9.get_function("apxby_kernel_1d")


mod7 = SourceModule("""
__global__
void copy_kernel(double *src, double *dst)
{
    int R1 = blockIdx.x * blockDim.x + threadIdx.x;
    dst[R1] = src[R1];


}
""")
cpy_gpu = mod7.get_function("copy_kernel")


mod8 = SourceModule("""
__global__ void Dot_Prod(double *x, double *y, double *g_odata) {
 __shared__ double sdata[512];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
sdata[tid] = x[i]*y[i];
__syncthreads();
// do reduction in shared mem
for(unsigned int s=1; s < blockDim.x; s *= 2) {
if (tid % (2*s) == 0) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}
// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


""")
mod10 = SourceModule("""
__global__ void reduce0(double *g_in, double *g_odata) {
 __shared__ double sdata[512];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
sdata[tid] = g_in[i];
__syncthreads();
// do reduction in shared mem
for(unsigned int s=1; s < blockDim.x; s *= 2) {
if (tid % (2*s) == 0) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}
// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
""")
partial_dot = mod8.get_function("Dot_Prod")
partial_red = mod10.get_function("reduce0")

def cal_dpdt(a_gpu,b_gpu,gpu_buffer1,gpu_buffer2,cpu_buffer1,cpu_buffer2):
    '''
    ans = (a,b)/(b,b)
    '''
    partial_dot(a_gpu,b_gpu,gpu_buffer1,block=(512,1,1),grid=(4096,1,1))
    partial_red(gpu_buffer1,gpu_buffer2,block=(512,1,1),grid=(8,1,1))
    
    cuda.memcpy_dtoh(cpu_buffer1,gpu_buffer2)
    cpu_buffer2[0]= np.sum(cpu_buffer1)
    
    partial_dot(b_gpu,b_gpu,gpu_buffer1,block=(512,1,1),grid=(4096,1,1))
    partial_red(gpu_buffer1,gpu_buffer2,block=(512,1,1),grid=(8,1,1))
    cuda.memcpy_dtoh(cpu_buffer1,gpu_buffer2)
    cpu_buffer2[1]= np.sum(cpu_buffer1)
    
    return cpu_buffer2[0]/cpu_buffer2[1]

mod11 = SourceModule("""
__global__
void plus_alpha_vector(double *x,  double *y, double *alpha)
{
    int R1 = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = R1;
    x[ind] = x[ind] + alpha[0]*y[ind];
            
}
""")

mod12 = SourceModule("""
__global__
void subtract_alpha_vector(double *x,  double *y, double *alpha)
{
    int R1 = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = R1;
    x[ind] = x[ind] - alpha[0]*y[ind];
            
}
""")

sub_alpha_gpu = mod12.get_function("subtract_alpha_vector")

plus_alpha_gpu = mod11.get_function("plus_alpha_vector")


def gpu_minres_v2(x0,b,maxit):
    '''
    gpu acceleration of MINRES algorithm
    x0 ,b: numpy.float64 arrays of initial guess and coefficient vector
    maxit: maximum number of iterations
    '''   
    
    #allocating memory
    ans=np.zeros_like(x0).astype(np.float64)
    x=np.zeros_like(x0).astype(np.float64)
    r=np.zeros_like(x0).astype(np.float64)


    cst_one=np.array([1.0]).astype(np.float64)
    cst_neone=np.array([-1.0]).astype(np.float64)
    
    alpha=np.array([1.0]).astype(np.float64)
    beta1=np.array([1.0]).astype(np.float64)
    beta2=np.array([1.0]).astype(np.float64)



    
    x0_gpu = cuda.mem_alloc(x0.nbytes)
    x_gpu = cuda.mem_alloc(x0.nbytes)
    b_gpu = cuda.mem_alloc(x0.nbytes)
    r_gpu = cuda.mem_alloc(x0.nbytes)
    p0_gpu = cuda.mem_alloc(x0.nbytes)
    p1_gpu = cuda.mem_alloc(x0.nbytes)
    p2_gpu = cuda.mem_alloc(x0.nbytes)

    
    s0_gpu = cuda.mem_alloc(x0.nbytes)
    s1_gpu = cuda.mem_alloc(x0.nbytes)
    s2_gpu = cuda.mem_alloc(x0.nbytes)

    
    one_gpu = cuda.mem_alloc(cst_one.nbytes)
    neone_gpu = cuda.mem_alloc(cst_one.nbytes)
    
    alpha_gpu = cuda.mem_alloc(alpha.nbytes)
    beta1_gpu = cuda.mem_alloc(beta1.nbytes)
    beta2_gpu = cuda.mem_alloc(beta2.nbytes)

    
    gpu_buffer1 = cuda.mem_alloc(int(x0.nbytes/512))
    gpu_buffer2 = cuda.mem_alloc(int(x0.nbytes/(512*512)))
    
    cpu_buffer_1=np.zeros(8).astype(np.float64)
    cpu_buffer_2f=np.zeros(2).astype(np.float64)
    cpu_buffer_3=np.zeros(1).astype(np.float64)


    
    temp = cuda.mem_alloc(x0.nbytes)
    
    #print("allocating completed")
    
    #copy data
    cuda.memcpy_htod(x0_gpu, x0)
    cuda.memcpy_htod(b_gpu, b)

    cuda.memcpy_htod(one_gpu, cst_one)
    cuda.memcpy_htod(neone_gpu, cst_neone)
    
    #print("memory copy completed, begin kernel computing")

    #x = np.array(x0)
    cpy_gpu(x0_gpu,x_gpu,block=(512,1,1),grid=(16*16*16,1,1))
    
    #r = b - A @ x0
    spmv_gpu_64(x0_gpu,temp,block=(8,8,8), grid=(16,16,16))
    
    apxby_gpu_1d(
        b_gpu,temp,r_gpu,one_gpu,neone_gpu,
        block=(512,1,1), grid=(16*16*16,1,1))
    
    #p0 = np.array(r)
    cpy_gpu(r_gpu,p0_gpu,block=(512,1,1),grid=(16*16*16,1,1))
    
    #s0 = A @ p0
    spmv_gpu_64(p0_gpu,s0_gpu,block=(8,8,8), grid=(16,16,16))
    
    #p1 = np.array(p0)
    cpy_gpu(p0_gpu,p1_gpu,block=(512,1,1),grid=(16*16*16,1,1))
    
    #s1 = np.array(s0)
    cpy_gpu(s0_gpu,s1_gpu,block=(512,1,1),grid=(16*16*16,1,1))
    
    #print("pre-loop completed, begin iters...")
    
    for iter in range(1,maxit):
    
        #p2 = np.ndarray.copy(p1)
        cpy_gpu(p1_gpu,p2_gpu,block=(512,1,1),grid=(16*16*16,1,1))

        #p1 = np.ndarray.copy(p0)
        cpy_gpu(p0_gpu,p1_gpu,block=(512,1,1),grid=(16*16*16,1,1))

        #s2 = np.ndarray.copy(s1)
        cpy_gpu(s1_gpu,s2_gpu,block=(512,1,1),grid=(16*16*16,1,1))

        #s1 = np.ndarray.copy(s0)
        cpy_gpu(s0_gpu,s1_gpu,block=(512,1,1),grid=(16*16*16,1,1))


        #alpha = np.dot(r,s1)/np.dot(s1,s1)
        cpu_buffer_3[0] = cal_dpdt(r_gpu,s1_gpu,gpu_buffer1,gpu_buffer2,cpu_buffer_1,cpu_buffer_2f)
        cuda.memcpy_htod(alpha_gpu, cpu_buffer_3)

        #x = x + alpha * p1
        plus_alpha_gpu(x_gpu,p1_gpu,alpha_gpu,block=(512,1,1),grid=(4096,1,1))

        #r = r - alpha * s1
        sub_alpha_gpu(r_gpu,s1_gpu,alpha_gpu,block=(512,1,1),grid=(4096,1,1))


        #p0 = np.ndarray.copy(s1)
        cpy_gpu(s1_gpu,p0_gpu,block=(512,1,1),grid=(16*16*16,1,1))


        #s0 = A @ s1
        spmv_gpu_64(s1_gpu,s0_gpu,block=(8,8,8), grid=(16,16,16))

        #beta1 = np.dot(s0,s1)/np.dot(s1,s1)
        cpu_buffer_3[0] = cal_dpdt(s0_gpu,s1_gpu,gpu_buffer1,gpu_buffer2,cpu_buffer_1,cpu_buffer_2f)
        cuda.memcpy_htod(beta1_gpu, cpu_buffer_3)

        #p0 = p0 - beta1* p1
        sub_alpha_gpu(p0_gpu,p1_gpu,beta1_gpu,block=(512,1,1),grid=(4096,1,1))


        #s0 = s0 - beta1* s1
        sub_alpha_gpu(s0_gpu,s1_gpu,beta1_gpu,block=(512,1,1),grid=(4096,1,1))

        if iter>1:

                #beta2 = np.dot(s0,s2)/np.dot(s2,s2)
                cpu_buffer_3[0] = cal_dpdt(s0_gpu,s2_gpu,gpu_buffer1,gpu_buffer2,cpu_buffer_1,cpu_buffer_2f)
                cuda.memcpy_htod(beta2_gpu, cpu_buffer_3)

                #p0 = p0 - beta2* p2
                sub_alpha_gpu(p0_gpu,p2_gpu,beta2_gpu,block=(512,1,1),grid=(4096,1,1))

                #s0 = s0 - beta2* s2
                sub_alpha_gpu(s0_gpu,s2_gpu,beta2_gpu,block=(512,1,1),grid=(4096,1,1))

    
    
    #print("calculation completed, writing back...")
    
    cuda.memcpy_dtoh(x,x_gpu)
    cuda.memcpy_dtoh(r,r_gpu)

    #print("finished")
    return x,r


x_exact = np.ones(128*128*128).astype(np.float64)
x0 = np.zeros(128*128*128).astype(np.float64)
b = np.random.random(128*128*128).astype(np.float64)

spmv_gpu_64(cuda.In(x_exact),cuda.InOut(b),block=(8,8,8),grid=(16,16,16))

st = time.perf_counter()
[x,r]= gpu_minres_v2(x0,b,50)
sp = (time.perf_counter()-st)

relative_error = np.sqrt(np.sum(np.square(x-x_exact)))/np.sqrt(np.sum(np.square(x_exact)))
residue = np.sqrt(np.sum(np.square(r)))

print("relative error={}, residue={}".format(relative_error,residue))
print("elapsed time={}".format(sp))



