/* 
  Foundations of Parallel and Distributed Computing , Fall 2022.
  Instructor: Prof. Chao Yang @ Peking University.
  This is a serial implement of SpMV based on CSR format.
  Version 0.1 (date: 11/15/2022)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void read_matrix(int **row_ptr, int **col_ind, double **values, const char *filename, int *num_rows, int *num_cols, int *num_vals) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    // Get number of rows, columns, and non-zero values
    fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);
    
    int *row_ptr_t = (int *) malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *) malloc(*num_vals * sizeof(int));
    double *values_t = (double *) malloc(*num_vals * sizeof(double));
    
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *) malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++) {
        row_occurances[i] = 0;
    }
    
    int row, column;
    double value;
    while (fscanf(file, "%d %d %lf\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        
        row_occurances[row]++;
    }
    
    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++) {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);
    
    // Set the file position to the beginning of the file
    rewind(file);
    
    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++) {
        col_ind_t[i] = -1;
    }
    
    fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);
    int i = 0;
    while (fscanf(file, "%d %d %lf\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }
    fclose(file);
    
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
}

// Serial SpMV using CSR format
void spmv_csr(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, const double *x, double *y) {
    for (int i = 0; i < num_rows; i++) {
        double sum = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_ind[j]];
        }
        
        y[i] = sum;
    }
}





int main(int argc, const char * argv[]) {
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    double *values, elapsed_time;;
    
    int num_repeat = 500;
    int print_mode = 1;
    const char *filename = "./input.mtx";
    
    // Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);

    
    double *x = (double *) malloc(num_rows * sizeof(double)); // solution vector
    double *y = (double *) malloc(num_rows * sizeof(double)); // result vector
    for (int i = 0; i < num_rows; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
    }
    
    // Time the iterations
    clock_t start = clock();
    for (int iter = 0; iter < num_repeat; iter++) {
        spmv_csr(row_ptr, col_ind, values, num_rows, x, y);
        
        // Copy the result to x_{i} at the end of each iteration, and use it in iteration x_{i+1}
        // Scale the value of x_{i} to avoid overflow
        for (int i = 0; i < num_rows; i++) {
            x[i] = y[i]/1e2 + 1.0;
            y[i] = 0.0;
        }
    }
    clock_t stop = clock();
    elapsed_time = (((double) (stop - start)) / CLOCKS_PER_SEC) * 1000; // in milliseconds

    /*
      Parallel code using OpenMP (include the updated part for vecotr x)
    */

    /*
      Check for correctness: Compare item by item with x vector, output maximum absolute error
      (note the x vector is the result vector now)
    */
    
    // Print elapsed time and error
    printf("Serial Running time:  %.4f ms\n", elapsed_time);
    // printf("OpenMP Implementation Running time = \n", );
    // printf("Maximum absolute error:  %.4f ms\n", );
    
    free(row_ptr);
    free(col_ind);
    free(values);
    
    return 0;
}