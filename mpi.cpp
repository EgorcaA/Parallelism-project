//module load mpi/openmpi3
//module load gcc/8.3.0

// mpic++ -O0 mpi.cpp 
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
#include "mpi.h"

std::vector<std::pair<std::string, std::vector<double>>> read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<double>>> result;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    double val;

    // Read the column names
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
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val){
            
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }
    }

    // Close file
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

void mpi(double *rho, double *rrho, std::vector<std::pair<std::string, std::vector<double>>> data, double *x_grid,
             double *z_grid, int N, int len_data, double alpha, int ll, int n_threads, int rank, int size){
    
    // double val = 0.;

    double *tmp =  new double [len_data];
    for (int i = 0; i < len_data; ++i) tmp[i] = 0.;
    double *tmp1 =  new double [len_data];
    for (int i = 0; i < len_data; ++i) tmp1[i] = 0.;

    
    // int n_threads = 3;
    // omp_set_num_threads(n_threads);
    // #pragma omp parallel for schedule(dynamic)
    
    for (int j = rank*N/size; j < (rank+1)*N/size; ++j)
        for (int i = 0; i < N; ++i)
        {
            rho[(j - rank*N/size)*N + i] = 0.;
            #pragma omp simd reduction(+:rho[(j - rank*N/size)*N + i])
            for (int k = 0; k < len_data; ++k)
            {
                rho[(j - rank*N/size)*N + i] += data[3].second[k] * exp( -alpha*sqrt(pow(x_grid[i] - data[0].second[k], 2.) +
                                                                  pow(0         - data[1].second[k], 2.) + 
                                                                  pow(z_grid[j] - data[2].second[k], 2.) )  );




            }// все ниже не дало видимых результатов
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = x_grid[i] - data[0].second[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = tmp[l]*tmp[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp1[l] = z_grid[j] - data[2].second[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp1[l] = tmp1[l]*tmp1[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = tmp[l] + tmp1[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp1[l] = data[1].second[l]*data[1].second[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = tmp[l] + tmp1[l];

            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = sqrt(tmp[l]);
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = -alpha*tmp[l];
            // #pragma omp simd
            // for (int l = 0; l < len_data; ++l)
            //     tmp[l] = exp(tmp[l]);

            // #pragma omp simd reduction(+:rho[(j - rank*N/size)*N + i])
            // for (int l = 0; l < len_data; ++l)
            //     rho[(j - rank*N/size)*N + i] += data[3].second[l]*tmp[l];

        }

    

    MPI_Allgather(rho, int(N*N/size), MPI_DOUBLE, rrho, int(N*N/size), MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0)  print_map(rrho, N);
}


int main(int argc, char*argv[]) {
    MPI_Init(&argc,&argv);
    int rank; //id process
    int size; //N processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int n_threads = 3;
    int N = 200;
    double ll =  15000.;

    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"-N")) {
            N=atoi(argv[++i]);
            
        }
        else if (!strcmp(argv[i],"-ll")) {
            ll=(double)atof(argv[++i]);
        }
        else if (!strcmp(argv[i],"-p")) {
            n_threads=atoi(argv[++i]);
        }
    }
    if (rank == 0){
        std::cout<< "N   " << N << std::endl;
        std::cout<< "ll   " << ll << std::endl;
        std::cout<< "size   " << size << std::endl;
    }

    std::vector<std::pair<std::string, std::vector<double>>> data = read_csv("xyz270.csv");

    int len_data = data[0].second.size();
    // std::cout<< "Num of str " << len_data << std::endl;
    
    double R = 3526229.;
    double alpha = R/10000000000000000.*ll;

    double *x_grid = new double [N];
    for (int i = 0; i < N; ++i) x_grid[i] = -R + (double)i/(N - 1)*2*R;

    double *z_grid = new double [N];
    for (int i = 0; i < N; ++i) z_grid[i] = -R + (double)i/(N - 1)*2*R;


    double *rho =  new double [N*N/size];
    for (int i = 0; i < N*N/size; ++i) rho[i] = 0.;

    double *rrho =  new double [N*N];
    for (int i = 0; i < N*N; ++i) rrho[i] = 0.;    

    
    // openmp(rho, data, x_grid, z_grid, N, len_data, alpha, ll, n_threads);

    auto start = std::chrono::high_resolution_clock::now();    
    mpi(rho, rrho, data, x_grid, z_grid, N, len_data, alpha, ll, n_threads, rank, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> durwrite = end - start;
    if (rank == 0){
        std::cout << "alpha  " <<ll << " N   " << N << "   time    " << durwrite.count() << "      size  " << size << std::endl;
        std::ofstream fout;
        fout.open("logs.txt", std::fstream::app); //std::ios::app 
        if (!fout.is_open()) std::cout << "ошибка открытия logs";
        else fout<< "mpi cpu "  << "  alpha  " <<ll << " N   " << N << "   time    "<< durwrite.count() << "  size  " << size <<std::endl;
        fout.close();

    }


    delete[] z_grid, x_grid, rho, rrho;
    MPI_Finalize();
    return 0;
}


