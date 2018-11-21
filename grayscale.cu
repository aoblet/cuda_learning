#include <iostream>
#include <vector>
#include <IL/il.h>


using img = std::vector<unsigned char>;


__global__ void grayscaleKernel(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* out, int width, int height){
	// 2D dimension grid
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidX < width && tidY < height){
		int matrixIndex = width*tidY + tidX;
		out[matrixIndex] = (307 * r[matrixIndex] + 604 * g[matrixIndex] + 113 * b[matrixIndex]) >> 10;

	}
}

int main( int argc, char * argv[] ) {

  std::string filename;
  
  if(argc <= 1){
	std::cerr << "Error: Missing path to image file." << std::endl;
	std::exit(1);
  }
  
  filename = argv[ 1 ];

  unsigned int images[2];
  ilInit();
  ilGenImages(2, images);
  ilBindImage(images[0]);
  ilLoadImage(filename.c_str());

  auto const width = ilGetInteger(IL_IMAGE_WIDTH);
  auto const height = ilGetInteger(IL_IMAGE_HEIGHT);
  auto const size = width*height;
  auto const nbBytesImage = size * sizeof(unsigned char);

  // Image is stored as: rgbrgbrgb....
  unsigned char * data = ilGetData();

  unsigned char* redMatrix_h = nullptr;
  unsigned char* greenMatrix_h = nullptr;
  unsigned char* blueMatrix_h = nullptr;
  unsigned char* out_h = nullptr;
  
  unsigned char* redMatrix_d = nullptr;
  unsigned char* greenMatrix_d = nullptr;
  unsigned char* blueMatrix_d = nullptr;
  unsigned char* out_d = nullptr;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  int const nbStreams = 10;
  cudaStream_t streams[nbStreams];

  for(int i=0; i<nbStreams; ++i){
      cudaStreamCreate(&streams[i]);
  }

  // Divide the image vertically by the number of streams
  int const chunkVectorHeight = height/nbStreams;
  int const chunkVectorHeightBytes = width * chunkVectorHeight * sizeof(unsigned char);

  // Allocate host matrices (R, G, B, and out)
  cudaMallocHost(&redMatrix_h, nbBytesImage);
  cudaMallocHost(&greenMatrix_h, nbBytesImage);
  cudaMallocHost(&blueMatrix_h, nbBytesImage);
  cudaMallocHost(&out_h, nbBytesImage);

  // Fill host matrices
  for(int i=0; i<size; ++i){
	  redMatrix_h[i] = data[i*3];
	  greenMatrix_h[i] = data[(i+1)*3];
	  blueMatrix_h[i] = data[(i+2)*3];
  }

  // Allocate device matrices (R, G, B, and out)
  cudaMalloc(&redMatrix_d, nbBytesImage);
  cudaMalloc(&greenMatrix_d, nbBytesImage);
  cudaMalloc(&blueMatrix_d, nbBytesImage);
  cudaMalloc(&out_d, nbBytesImage);

  // Launch kernek within streams
  for(int i=0; i<nbStreams; ++i){
	  // We split the image vertically
	  // the chunk is the width x chunk vertical
	  // so we need to offset the matrix adress by currentStream
	  // Then we can copy by chunk
	  int offset = width * chunkVectorHeight * i;
      cudaMemcpyAsync(redMatrix_d+offset, redMatrix_h+offset, chunkVectorHeightBytes, cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(greenMatrix_d+offset, greenMatrix_h+offset, chunkVectorHeightBytes, cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(blueMatrix_d+offset, blueMatrix_h+offset, chunkVectorHeightBytes, cudaMemcpyHostToDevice, streams[i]);
  }

  // block/grid dimension setup
  dim3 block(32, 32);
  // grid size is automatically adjusted according the block size and the image size
  dim3 grid((width-1)/block.x + 1, (height-1)/block.y + 1);

  // Launch kernels
  for(int i=0; i<nbStreams; ++i){
	  cudaEventRecord(start, streams[i]);
  	  grayscaleKernel<<<grid, block, 0, streams[i] >>>(redMatrix_d, greenMatrix_d, blueMatrix_d, out_d, width, height);
  }

  // Put back GPU result to CPU by chunk (stream)
  for(int i=0; i<nbStreams; ++i){
	  int offset = width*chunkVectorHeight*i;
      cudaMemcpyAsync(out_h+offset, out_d+offset, chunkVectorHeightBytes, cudaMemcpyDeviceToHost, streams[i]);
  }

  // Synchronize streams since memcpy is async
  for(int i=0; i<nbStreams; ++i){
	  cudaStreamSynchronize(streams[i]);
	  cudaEventRecord(stop, streams[i]);
  }

  // Show elapsed time
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << ms << "ms";

  // Write the image on disk
  ilBindImage( images[ 1 ] );
  ilTexImage( width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_h);
  ilEnable(IL_FILE_OVERWRITE);
  ilSaveImage("out.jpg");
  ilDeleteImages( 2, images ); 

  // Finally free the host and devices ressources
  cudaFreeHost(redMatrix_h);
  cudaFreeHost(greenMatrix_h);
  cudaFreeHost(blueMatrix_h);
  cudaFreeHost(out_h);

  cudaFree(redMatrix_d);
  cudaFree(greenMatrix_d);
  cudaFree(blueMatrix_d);
  cudaFree(out_d);
  return 0;
}
