#include <cstdio>
#include <cudpp_hash.h>
#include <cuda_util.h>
#include <cuda_runtime_api.h>

// input: nsample (1), true_mat (b, m, n)
// output: idx (b, m, nsample)
__global__ void assign_idx_gpu(int b, int n, int m, int nsample, const bool *true_mat, int *idx){
    int batch_index = blockIdx.x;
    true_mat += n*m*batch_index;
    idx += nsample*m*batch_index;
    
    int index = threadIdx.x;
    int stride = blockDim.x;
    
    //j: the jth centroids within the batch
    for (int j = index; j < m; j += stride){
        int cnt = 0;
        for(int i = 0; i < n; i++){
//            printf("batch_index:%d index:%d j:%d i:%d\n", batch_index, index, j, i);
            if(true_mat[j*n + i]){
                idx[j*nsample + cnt] = i;
                cnt++;
            }
        }
        for(; cnt < nsample; cnt++){
            idx[j*nsample + cnt] = idx[j*nsample];
        }
    }
}

//__global__ void assign_idx_gpu(int b, int m, int nsample, const int *true_idx, const int *end_idx, int *idx){
//    int batch_index = blockIdx.x;
//    end_idx += m*batch_index;
//    idx += m*nsample*batch_index;
//    
//    int index = threadIdx.x;
//    int stride = blockDim.x;
//    
//    //j: the jth centroids within the batch
//    for (int j = index; j < m; j += stride){
//        int start_idx_;
//        if (j == 0 && batch_index == 0){
//            start_idx_ = 0;
//        }else{
//            start_idx_ = end_idx[j-1];
//        }
//        int end_idx_ = end_idx[j];
//        
//        int i = 0;
//        for (; i < (end_idx_ - start_idx_); i++){
//            idx[nsample * j + i] = true_idx[start_idx_ + i];
//        }
//        for (; i < nsample; i++){
//            idx[nsample * j + i] = true_idx[end_idx_ - 1];
//        }
//    }
//}
// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
            
    	    float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
//            if (fabsf(x2-x1)<radius && (y2-y1)<radius && (z2-z1)<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}
//compose_insert_items<<<b,256>>>(b, n, grid_size, all_xyz, limits, sizes, d_keys, d_vals);
__global__ void compose_insert_items(int b, int n, float grid_size, const float *all_xyz, const float *limits, const int *sizes, unsigned int *d_keys, unsigned int *d_vals){
    int index = threadIdx.x;
    
    if(index < n){
        int batch_index = blockIdx.x;
        all_xyz += batch_index * n * 3;
        unsigned int *tmp_d_keys = d_keys + batch_index * n;
        unsigned int *tmp_d_vals = d_vals + batch_index * n;
        
        int stride = blockDim.x;
        
        for(int point_idx = index; point_idx < n; point_idx += stride){
            unsigned int x_idx = __float2uint_rd((all_xyz[point_idx*3] - limits[0]) / grid_size) + 1;
            unsigned int y_idx = __float2uint_rd((all_xyz[point_idx*3+1] - limits[2]) / grid_size) + 1;
            unsigned int z_idx = __float2uint_rd((all_xyz[point_idx*3+2] - limits[4]) / grid_size) + 1;
            
            tmp_d_keys[point_idx] = z_idx + sizes[2] * (y_idx + sizes[1] * (x_idx + batch_index * sizes[0]));
            tmp_d_vals[point_idx] = point_idx;
//            printf("b:%d, point_idx:%d, x:%f, y:%f, z:%f, x_idx:%u, y_idx:%u, z_idx:%u, key: %u, val: %u\n", batch_index, point_idx, all_xyz[point_idx*3], all_xyz[point_idx*3+1], all_xyz[point_idx*3+2], x_idx, y_idx, z_idx, tmp_d_keys[point_idx], tmp_d_vals[point_idx]);
        }
    }
}
//compose_queries<<<b,256>>>(b, m, grid_size, centroids_xyz, limits, sizes, d_queries);
__global__ void compose_queries(int b, int m, float grid_size, const float *centroids_xyz, const float *limits, const int *sizes, unsigned int *d_queries){

    int index = threadIdx.x;
    
    if(index < m){
        int stride = blockDim.x;
        
        int batch_index = blockIdx.x;
        centroids_xyz += batch_index * m * 3;
        unsigned int *tmp_d_queries = d_queries + batch_index * m * 27;
        
        unsigned int x_idx = __float2uint_rd((centroids_xyz[index*3] - limits[0]) / grid_size);
        unsigned int y_idx = __float2uint_rd((centroids_xyz[index*3+1] - limits[2]) / grid_size);
        unsigned int z_idx = __float2uint_rd((centroids_xyz[index*3+2] - limits[4]) / grid_size);
        
        int cnt = 0;
        for(int x_offset = 0; x_offset < 3; x_offset++){
            for(int y_offset = 0; y_offset < 3; y_offset++){
                for(int z_offset = 0; z_offset < 3; z_offset++){
                    tmp_d_queries[index*27+cnt] = z_idx + z_offset + sizes[2] * (y_idx + y_offset + sizes[1] * (x_idx + x_offset + batch_index * sizes[0]));  
//                    if(x_offset == 1 && y_offset == 1 && z_offset == 1){
//                        printf("b:%d, centroids_idx:%d, x:%f, y:%f, z:%f, x_grid:%u, y_grid:%u, z_grid:%u, query_self:%u\n", batch_index, index, centroids_xyz[index*3], centroids_xyz[index*3+1], centroids_xyz[index*3+2], x_offset+x_idx, y_offset+y_idx, z_offset+z_idx, tmp_d_queries[index+cnt]);
//                    }
                    cnt++;
                }
            }
        }

    }
}
//    hash_square_idx_gpu<<<b,256>>>(b, n, m, nsample, d_vals_multivalue, d_sorted_idx, d_all_values, idx, pts_cnt);
// , const unsigned int *sorted_idx
__global__ void hash_square_idx_gpu(int b, int n, int m, int nsample, const uint2 *d_vals_multivalue, const unsigned int * d_all_values, int *idx, int *pts_cnt){
    int index = threadIdx.x;
    if(index < m){
        int stride = blockDim.x;
        int batch_index = blockIdx.x;
        unsigned int sorted_idx[27] = {13, 4,10,12,14,16,22, 1,3,5,7,9,11,15,17,19,21,23,25,  0,2,6,8,18,20,24,26};
        
//        if(index == 0 && batch_index == 0){
//            printf("d_vals_multivalue:\n");
//            for(int kk = 0; kk < 2 * 3 * 27; kk++){
//                printf("%u %u\n", d_vals_multivalue[kk].x, d_vals_multivalue[kk].y);
//            }
//        }
        
        idx += batch_index * m * nsample;
        pts_cnt += batch_index * m;
        int query_idx_base = batch_index*m*27+index*27;
        
        int cnt = 0;
        for(int i = 0; i < 27; i++){
            int query_idx = query_idx_base + sorted_idx[i];
//            int query_idx = batch_index*m*27+index*27+i;
//            printf("query_idx: %d ", query_idx);
            unsigned int num_values = d_vals_multivalue[query_idx].y;
//            printf("num_values: %u ", num_values);
//            printf("batch: %d, m: %d, i:%d, query_idx: %d, num_values: %u\n", batch_index, index, i, query_idx, num_values);
            if(num_values > 0){
                for(unsigned int j = 0; j < num_values && cnt < nsample; j++){
                    idx[index*nsample + cnt] = d_all_values[d_vals_multivalue[query_idx].x + j];
                    cnt++;
//                    printf("%d ", idx[index*nsample + cnt]);
                }
            }
        }
        pts_cnt[index] = cnt;
        for(;cnt < nsample;cnt++){
            idx[index*nsample + cnt] = idx[index*nsample];
//            printf("%d ", idx[index*nsample + cnt]);
        }
//        printf("\n");
    }
}

void querySquarePointLauncher(int b, int n, int m, float grid_size, int nsample, const float *all_xyz, const float *centroids_xyz, const float *limits, const int *sizes, int *idx, int *pts_cnt, unsigned int *d_keys, unsigned int *d_vals, unsigned int *d_queries, uint2 *d_vals_multivalue) {
//    printf("Start\n");
    // Allocate the GPU memory.
//    unsigned int *d_keys = NULL, *d_vals = NULL, *d_queries = NULL;
//    uint2 *d_vals_multivalue = NULL;
    
    unsigned int kInputSize = b * n;
//    printf("b %d, n %d, kInputSize: %u\n", b, n, kInputSize);
    
    compose_insert_items<<<b,256>>>(b, n, grid_size, all_xyz, limits, sizes, d_keys, d_vals);
    cudaDeviceSynchronize();
    
    CUDPPHandle theCudpp;
    CUDPPResult result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS){
        fprintf(stderr, "Error initializing CUDPP Library.\n");
        exit(-1);
    }

    CUDPPHashTableConfig config;
    config.type = CUDPP_MULTIVALUE_HASH_TABLE;
    config.kInputSize = kInputSize;
    config.space_usage = 2.0f;
    CUDPPHandle hash_table_handle;
    result = cudppHashTable(theCudpp, &hash_table_handle, &config);
    if (result != CUDPP_SUCCESS){
        fprintf(stderr, "Error in cudppHashTable call in"
                "testHashTable (make sure your device is at"
                "least compute version 2.0\n");
    }
    
    result = cudppHashInsert(hash_table_handle, d_keys,
                                d_vals, kInputSize);
    cudaThreadSynchronize();
//    printf("insert values\n");
    if (result != CUDPP_SUCCESS){
        fprintf(stderr, "Error in cudppHashInsert call in"
                "testHashTable\n");
    }
    
    unsigned int values_size;
    if (cudppMultivalueHashGetValuesSize(hash_table_handle,
                                    &values_size) !=
                                    CUDPP_SUCCESS){
        fprintf(stderr, "Error: "
                "cudppMultivalueHashGetValuesSize()\n");
    }
    
    unsigned int * d_all_values = NULL;
    if (cudppMultivalueHashGetAllValues(hash_table_handle,
                                        &d_all_values) !=
                                        CUDPP_SUCCESS){
        fprintf(stderr, "Error: "
                "cudppMultivalueHashGetAllValues()\n");
    }
    
    
    
    compose_queries<<<b,256>>>(b, m, grid_size, centroids_xyz, limits, sizes, d_queries);
    cudaDeviceSynchronize();
    
    result = cudppHashRetrieve(hash_table_handle,
                                d_queries,
                                d_vals_multivalue,
                                b * m * 27);
    cudaThreadSynchronize();
//    printf("retrieved values\n");
    if (result != CUDPP_SUCCESS){
        fprintf(stderr, "Error in cudppHashRetrieve call\n");
    }

    hash_square_idx_gpu<<<b,256>>>(b, n, m, nsample, d_vals_multivalue, d_all_values, idx, pts_cnt);
    cudaDeviceSynchronize();
//    printf("obtain idx\n");
    
    /// -------------------------------------------- Free the table.    
    result = cudppDestroyHashTable(theCudpp, hash_table_handle);
    if (result != CUDPP_SUCCESS){
        fprintf(stderr, "Error in cudppDestroyHashTable call in"
                "testHashTable\n");
    }

//    CUDA_SAFE_CALL(cudaFree(d_keys));
//    CUDA_SAFE_CALL(cudaFree(d_vals));
//    CUDA_SAFE_CALL(cudaFree(d_queries));
//    CUDA_SAFE_CALL(cudaFree(d_vals_multivalue));

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS){
        printf("Error shutting down CUDPP Library.\n");
    }
//    printf("Ends\n");
}
void assignIdxLauncher(int b, int n, int m, int nsample, const bool *true_mat, int *idx){
    assign_idx_gpu<<<b, 256>>>(b,n,m,nsample,true_mat,idx);
}
void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
//    printf("before queryBallPointLauncher\n");
    query_ball_point_gpu<<<b,256>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
//    printf("after queryBallPointLauncher\n");
    //cudaDeviceSynchronize();
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
//    printf("before groupPointLauncher\n");
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
//    printf("after groupPointLauncher\n");
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,256>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}


