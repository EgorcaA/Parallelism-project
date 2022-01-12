// module load nvhpc/21.9
//nvc++ gpu.cu && ./a.out -N 500 -ll 15000

#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <iostream>
#include <math.h> 
#include <cstring>
#include <omp.h>
#include <chrono>

#define THREADS_PER_BLOCK 512

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif


std::vector<std::pair<std::string, std::vector<double>>> read_csv(std::string filename){
    std::vector<std::pair<std::string, std::vector<double>>> result;
    std::ifstream myFile(filename);
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    std::string line, colname;
    double val;
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);
        // Create a stringstream from line
        std::stringstream ss(line);
        // Extract each column name
        while(std::getline(ss, colname, ',')){
            // Initialize and add <colname, int vector> pairs to result
            result.push_back({colname, std::vector<double> {}});
        }
    }
    // Read data, line by line
    while(std::getline(myFile, line))
    {
        std::stringstream ss(line);
        int colIdx = 0;
        while(ss >> val){
            result.at(colIdx).second.push_back(val);
            if(ss.peek() == ',') ss.ignore();
            colIdx++;
        }
    }
    myFile.close();
    return result;
}

void print_map(double *rho, int N) {
        
        std::ofstream fout;
        fout.open("out.csv", std::fstream::out); //std::ios::app 
        
        if (!fout.is_open()) std::cout << "ошибка открытия out";
        else{
            for (int i = 0; i < N; ++i) fout<< "name,";
            fout<<std::endl;
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    std::cout.precision(17);
                    fout << std::fixed << rho[i*N + j] << ',';
                }
                fout<<std::endl;
            }
        }
        fout.close();
    }


__global__  void gpu(double *Dx_grid, double *Dz_grid, double *Drho, double *Dx_data, double *Dy_data, double *Dz_data, 
				double *Ddensity_data, int N, int len_data, double alpha, double ll){
    
    int ident = blockIdx.x * blockDim.x + threadIdx.x;
    int i = ident/N;
    int j = ident%N;
    Drho[ident] = 0.;
    for (int k = 0; k < len_data; ++k)
    {
        Drho[ident] += Ddensity_data[k] * exp( -alpha*sqrt(pow(Dx_grid[i] - Dx_data[k], 2.) +
                                                          pow(0          - Dy_data[k], 2.) + 
                                                          pow(Dz_grid[j] - Dz_data[k], 2.) )  );
    }
        
}



int main(int argc, char const *argv[]) {
    int N = 50;
    double ll = 1.;

    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"-N")) {
            N=atoi(argv[++i]);
            
        }
        else if (!strcmp(argv[i],"-ll")) {
            ll=(double)atof(argv[++i]);
        }
    }

    std::cout<< "N   " << N << std::endl;
    std::cout<< "ll   " << ll << std::endl;


    std::vector<std::pair<std::string, std::vector<double>>> data = read_csv("xyz270.csv");

    int len_data = data[0].second.size();
    std::cout<< "Num of str " << len_data << std::endl;

    
    double *x_data = new double [len_data];
    for (int i = 0; i < len_data; ++i) x_data[i] = data[0].second[i];
    double *y_data = new double [len_data];
    for (int i = 0; i < len_data; ++i) y_data[i] = data[1].second[i];
    double *z_data = new double [len_data];
    for (int i = 0; i < len_data; ++i) z_data[i] = data[2].second[i];
    double *density_data = new double [len_data];
    for (int i = 0; i < len_data; ++i) density_data[i] = data[3].second[i];

    double R = 3526229.;
    double alpha = R/10000000000000000.*ll;

    double *x_grid = new double [N];
    for (int i = 0; i < N; ++i) x_grid[i] = -R + (double)i/(N - 1)*2*R;

    double *z_grid = new double [N];
    for (int i = 0; i < N; ++i) z_grid[i] = -R + (double)i/(N - 1)*2*R;

    double *rho =  new double [N*N];
    for (int i = 0; i < N*N; ++i) rho[i] = 0.;


	double *Dz_grid, *Dx_grid, *Drho, *Dx_data, *Dy_data, *Dz_data, *Ddensity_data;
	
    cudaMalloc((void**)&Dx_grid, sizeof(double)*N);	
	cudaMalloc((void**)&Dz_grid, sizeof(double)*N);
	cudaMalloc((void**)&Drho, sizeof(double)*N*N);	
	cudaMalloc((void**)&Dx_data, sizeof(double)*len_data);
	cudaMalloc((void**)&Dy_data, sizeof(double)*len_data);
	cudaMalloc((void**)&Dz_data, sizeof(double)*len_data);	
	cudaMalloc((void**)&Ddensity_data, sizeof(double)*len_data);

    CUDA_CHECK_ERROR(cudaMemcpy(Dx_grid, x_grid, sizeof(double)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(Dz_grid, z_grid, sizeof(double)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(Drho, rho, sizeof(double)*N*N, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(Dx_data, x_data, sizeof(double)*len_data, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(Dy_data, y_data, sizeof(double)*len_data, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(Dz_data, z_data, sizeof(double)*len_data, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(Ddensity_data, density_data, sizeof(double)*len_data, cudaMemcpyHostToDevice));

    float timerValueGPU;
    cudaEvent_t start, stop;
    cudaEventCreate ( &start);
    cudaEventCreate ( &stop);
    cudaEventRecord(start, 0);
    gpu <<< N*N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(Dx_grid, Dz_grid, Drho, Dx_data, Dy_data, Dz_data, Ddensity_data, N, len_data, alpha, ll);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);


    CUDA_CHECK_ERROR(cudaMemcpy(rho, Drho, sizeof(double)*N*N, cudaMemcpyDeviceToHost));
    
    printf ("\n GPU w/o shared memory 1 worker calculation time %f msec\n", timerValueGPU);
    std::cout << "alpha  " <<ll << " N   " << N << "   time    " << timerValueGPU << std::endl;

    std::ofstream fout;
    fout.open("logs.txt", std::fstream::app); //std::ios::app 
    if (!fout.is_open()) std::cout << "ошибка открытия logs";
    else fout<< "gpu "  << "  alpha  " <<ll << " N   " << N << "   time    "<< timerValueGPU << std::endl;
    fout.close();

    print_map(rho, N);

    delete[] z_grid, x_grid, rho, x_data, y_data, z_data, density_data;
    cudaFree( Dz_grid);
    cudaFree(Dx_grid);
    cudaFree(Drho);
    cudaFree(Dx_data);
    cudaFree(Dy_data);
    cudaFree(Dz_data);
    cudaFree(Ddensity_data);
    return 0;
}


