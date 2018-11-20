#include <iostream>

__global__ void fillMonoThread(int* vector, std::size_t size){
    for(std::size_t i=0; i<size; ++i){
        vector[i] = 1;
    }
}

void testMonoThread(){
    std::size_t size = 10;
    int* vector_h = new int[size];
    int* vector_d;

    // allocate memory on GPU
    cudaMalloc(&vector_d, size * sizeof(int));

    // run cuda code on GPU
    fillMonoThread<<<1,1>>>(vector_d, size);
    // once it's done transfert GPU data to CPU
    // thus we could use it here, on CPU host
    cudaMemcpy(vector_h, vector_d, size*sizeof(int), cudaMemcpyDeviceToHost);
    for(std::size_t i=0; i<size; ++i){
        std::cout << vector_h[i] << std::endl;
    }
    cudaFree(vector_d);
}


__global__ void fillMutliThreads(int* vector, std::size_t size){
    vector[threadIdx.x] = threadIdx.x;
}


void testMultiThreads(){
    std::size_t size = 1024;
    int* vector_h = new int[size];
    int* vector_d;

    // allocate memory on GPU
    cudaMalloc(&vector_d, size * sizeof(int));

    // run cuda code on GPU
    // on size threads
    fillMutliThreads<<<1,size>>>(vector_d, size);
    // once it's done transfert GPU data to CPU
    // thus we could use it here, on CPU host
    cudaMemcpy(vector_h, vector_d, size*sizeof(int), cudaMemcpyDeviceToHost);
    for(std::size_t i=0; i<size; ++i){
        std::cout << vector_h[i] << ' ' ;
    }
    cudaFree(vector_d);
}


__global__ void fillMutliBlocs(int* vector, std::size_t size){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    vector[threadId] = threadId;
}

void testMultiBlocs(){
    std::size_t size = 1024;
    int* vector_h = new int[size];
    int* vector_d;

    // allocate memory on GPU
    cudaMalloc(&vector_d, size * sizeof(int));

    // run cuda code on GPU
    // on size threads
    fillMutliBlocs<<<2,size/2>>>(vector_d, size);
    // once it's done transfert GPU data to CPU
    // thus we could use it here, on CPU host
    cudaMemcpy(vector_h, vector_d, size*sizeof(int), cudaMemcpyDeviceToHost);
    for(std::size_t i=0; i<size; ++i){
        std::cout << vector_h[i] << ' ' ;
    }
    cudaFree(vector_d);
}


__global__ void addVector(int* v1, int* v2, std::size_t size){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId < size){
        v1[threadId] += v2[threadId];
    }
}

void testAdd(){
    std::size_t const size = 100;
    int* vector0_h = nullptr;
    int* vector1_h = nullptr;

    int* vector0_d = nullptr;
    int* vector1_d = nullptr;

    cudaError_t cudaError;


    // http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g9f93d9600f4504e0d637ceb43c91ebad.html
    // allocate memory host
    cudaMallocHost(&vector0_h, size * sizeof(int));
    cudaMallocHost(&vector1_h, size * sizeof(int));

    // fill vectors
    for(int i=0; i<size; ++i){
        vector0_h[i] = vector1_h[i] = i;
    }

    // allocate memory device
    cudaError = cudaMalloc(&vector0_d, size * sizeof(int));
    if(cudaError != cudaSuccess){
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        throw std::exception();
    }

    cudaError = cudaMalloc(&vector1_d, size * sizeof(int));
    if(cudaError != cudaSuccess){
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        throw std::exception();
    }

    cudaMemcpy(vector0_d, vector0_h, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vector1_d, vector1_h, size * sizeof(int), cudaMemcpyHostToDevice);

    // call cuda Kernel
    // generic block / grid parametrization
    dim3 block(32);

    // little tweak to get a correct grid in case of size/block < 0
    dim3 grid((size-1)/block.x + 1);

    addVector<<<grid, block>>>(vector0_d, vector1_d, size);

    // put back GPU result to CPU
    cudaMemcpy(vector0_h, vector0_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    // check add result vectors
    for(int i=0; i<size; ++i){
        std::cout << vector0_h[i] << ' ' ;
    }

    // free memory on GPU and CPU
    cudaFree(vector0_d);
    cudaFree(vector1_d);
    cudaFreeHost(vector0_h);
    cudaFreeHost(vector1_h);
}


__global__ void addMatrix(int* m1, int* m2, std::size_t width, std::size_t height){
    // index of thread in X dim
    int threadIdX = blockIdx.x * blockDim.x + threadIdx.x;
    // index of thread in Y dim
    int threadIdY = blockIdx.y * blockDim.y + threadIdx.y;

    if(threadIdX < width && threadIdY < height){
        // make a 2D index into 1D
        int matrixIndex = width * threadIdY + threadIdX;
        m1[matrixIndex] += m2[matrixIndex];
    }

}
void testAddMatrice(){
    std::size_t const width = 100;
    std::size_t const height = 100;
    std::size_t const size = width * height;

    int* matrix0_h = nullptr;
    int* matrix1_h = nullptr;

    int* matrix0_d = nullptr;
    int* matrix1_d = nullptr;

    cudaError_t cudaError;


    // http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g9f93d9600f4504e0d637ceb43c91ebad.html
    // allocate memory host
    cudaMallocHost(&matrix0_h, size * sizeof(int));
    cudaMallocHost(&matrix1_h, size * sizeof(int));

    // fill matrixs
    for(int i=0; i<size; ++i){
        matrix0_h[i] = matrix1_h[i] = i;
    }

    // allocate memory device
    cudaError = cudaMalloc(&matrix0_d, size * sizeof(int));
    if(cudaError != cudaSuccess){
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        throw std::exception();
    }

    cudaError = cudaMalloc(&matrix1_d, size * sizeof(int));
    if(cudaError != cudaSuccess){
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        throw std::exception();
    }

    cudaMemcpy(matrix0_d, matrix0_h, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix1_d, matrix1_h, size * sizeof(int), cudaMemcpyHostToDevice);

    // call cuda Kernel
    // generic block / grid parametrization
    dim3 block(32, 32);

    // little tweak to get a correct grid in case of size/block < 0
    dim3 grid((width-1)/block.x + 1, (height-1)/block.y + 1);

    addMatrix<<<grid, block>>>(matrix0_d, matrix1_d, width, height);

    // put back GPU result to CPU
    cudaMemcpy(matrix0_h, matrix0_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    // check add result matrixs
    for(int i=0; i<size; ++i){
        std::cout << matrix0_h[i] << ' ' ;
    }

    // free memory on GPU and CPU
    cudaFree(matrix0_d);
    cudaFree(matrix1_d);
    cudaFreeHost(matrix0_h);
    cudaFreeHost(matrix1_h);
}


__global__ void addVectorShared(int* v1, int* v2, std::size_t size){
    extern __shared__ int shared[];

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = v2[threadId];

    if(threadId < size){
        v1[threadId] += shared[threadIdx.x];
    }
}

void testAddShared(){
    std::size_t const size = 100;
    int* vector0_h = nullptr;
    int* vector1_h = nullptr;

    int* vector0_d = nullptr;
    int* vector1_d = nullptr;

    cudaError_t cudaError;


    // http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g9f93d9600f4504e0d637ceb43c91ebad.html
    // allocate memory host
    cudaMallocHost(&vector0_h, size * sizeof(int));
    cudaMallocHost(&vector1_h, size * sizeof(int));

    // fill vectors
    for(int i=0; i<size; ++i){
        vector0_h[i] = vector1_h[i] = i;
    }

    // allocate memory device
    cudaError = cudaMalloc(&vector0_d, size * sizeof(int));
    if(cudaError != cudaSuccess){
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        throw std::exception();
    }

    cudaError = cudaMalloc(&vector1_d, size * sizeof(int));
    if(cudaError != cudaSuccess){
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        throw std::exception();
    }

    cudaMemcpy(vector0_d, vector0_h, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vector1_d, vector1_h, size * sizeof(int), cudaMemcpyHostToDevice);

    // call cuda Kernel
    // generic block / grid parametrization
    dim3 block(32);

    // little tweak to get a correct grid in case of size/block < 0
    dim3 grid((size-1)/block.x + 1);

    addVectorShared<<<grid, block, block.x * sizeof(int)>>>(vector0_d, vector1_d, size);

    // put back GPU result to CPU
    cudaMemcpy(vector0_h, vector0_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    // check add result vectors
    for(int i=0; i<size; ++i){
        std::cout << vector0_h[i] << ' ' ;
    }

    // free memory on GPU and CPU
    cudaFree(vector0_d);
    cudaFree(vector1_d);
    cudaFreeHost(vector0_h);
    cudaFreeHost(vector1_h);
}


int main(){
//    testMonoThread();
//    testMultiThreads();
//    testMultiBlocs();
//    testAdd();
    testAddShared();
    return 0;
}
