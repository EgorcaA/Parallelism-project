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

void print_map(double **rho, int N) {
        
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
                    fout << std::fixed << rho[i][j] << ',';
                }
                fout<<std::endl;
            }
        }
        fout.close();
    }


void openmp(double **rho, std::vector<std::pair<std::string, std::vector<double>>> data, double *x_grid,
             double *z_grid, int N, int len_data, double alpha, int ll, int n_threads){
    
    // double val = 0.;
    auto start = std::chrono::high_resolution_clock::now();
    // int n_threads = 3;
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            rho[i][j] = 0.;

            for (int k = 0; k < len_data; ++k)
            {
                rho[i][j] += data[3].second[k] * exp( -alpha*sqrt(pow(x_grid[i] - data[0].second[k], 2.) +
                                                                  pow(0         - data[1].second[k], 2.) + 
                                                                  pow(z_grid[j] - data[2].second[k], 2.) )  );
            }
        }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> durwrite = end - start;


    std::cout << "alpha  " <<ll << " N   " << N << "   time    " << durwrite.count() << std::endl;

    std::ofstream fout;
    fout.open("logs.txt", std::fstream::app); //std::ios::app 
    if (!fout.is_open()) std::cout << "ошибка открытия logs";
    else fout<< "openmp num threads  "<< n_threads  << "  alpha  " <<ll << " N   " << N << "   time    "<< durwrite.count() << std::endl;
    fout.close();

    print_map(rho, N);
}



int main(int argc, char const *argv[]) {
    int n_threads = 3;
    int N = 50;
    double ll = 1.;

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

    std::cout<< "N   " << N << std::endl;
    std::cout<< "ll   " << ll << std::endl;


    std::vector<std::pair<std::string, std::vector<double>>> data = read_csv("xyz270.csv");

    int len_data = data[0].second.size();
    std::cout<< "Num of str " << len_data << std::endl;

    
    double R = 3526229.;
    double alpha = R/10000000000000000.*ll;

    double *x_grid = new double [N];
    for (int i = 0; i < N; ++i) x_grid[i] = -R + (double)i/(N - 1)*2*R;

    double *z_grid = new double [N];
    for (int i = 0; i < N; ++i) z_grid[i] = -R + (double)i/(N - 1)*2*R;

    double **rho =  new double* [N];
    for (int i = 0; i < N; ++i) rho[i] = new double [N];
    
    // double val = 0.;


    openmp(rho, data, x_grid, z_grid, N, len_data, alpha, ll, n_threads);
    // acc(rho, data, x_grid, z_grid, N, len_data, alpha, ll, n_threads);

    // std::cout<<rho[0][0];
    
    delete[] z_grid, x_grid;
    for (int i = 0; i < N; ++i) delete[] rho[i];
    delete[] rho;
    return 0;
}


