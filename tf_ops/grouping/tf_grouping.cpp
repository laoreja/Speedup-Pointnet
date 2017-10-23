#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include <cudpp_hash.h>
#include <cuda_util.h>

using namespace tensorflow;

REGISTER_OP("AssignIdx")
    .Attr("nsample: int")
    .Input("true_mat: bool")
    .Output("idx: int32")
//    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; 
        c->WithRank(c->input(0), 3, &dims2);// batch_size * npoint * ndataset
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        return Status::OK();
    });
//REGISTER_OP("AssignIdx")
//    .Attr("nsample: int")
//    .Input("true_idx: int32")
//    .Input("end_idx: int32")
//    .Output("idx: int32")
////    .Output("pts_cnt: int32")
//    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//        ::tensorflow::shape_inference::ShapeHandle dims2; 
//        c->WithRank(c->input(1), 2, &dims2);// batch_size * npoint
//        int nsample;
//        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
//        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
//        c->set_output(0, output1);
//        return Status::OK();
//    });
REGISTER_OP("QuerySquarePoint")
    .Attr("grid_size: float")
    .Attr("nsample: int")
    .Input("xyz: float32")
    .Input("centroids_idx: float32")
    .Input("limits: float32")
    .Input("sizes: int32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("QueryBallPoint")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("SelectionSort")
    .Attr("k: int")
    .Input("dist: float32")
    .Output("outi: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("GroupPoint")
    .Input("points: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims2, 2), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("GroupPointGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


//void assignIdxLauncher(int b, int m, int nsample, const int *true_idx, const int *end_idx, int *idx);
//class AssignIdxGpuOp : public OpKernel {
//    public:
//        explicit AssignIdxGpuOp(OpKernelConstruction* context) : OpKernel(context) {
//            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
//            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("AssignIdx expects positive nsample"));
//        }
//
//        void Compute(OpKernelContext* context) override {
//            const Tensor& true_idx_tensor = context->input(0);
//            OP_REQUIRES(context, true_idx_tensor.dims()==1, errors::InvalidArgument("AssignIdx expects 1-D tensor for true_idx."));
//            
//            const Tensor& end_idx_tensor = context->input(1);
//            OP_REQUIRES(context, end_idx_tensor.dims()==2, errors::InvalidArgument("AssignIdx expects (batch_size, npoint) end_idx shape."));
//            int b = end_idx_tensor.shape().dim_size(0); // batch_size
//            int m = end_idx_tensor.shape().dim_size(1); // npoint
//
//            Tensor *idx_tensor = nullptr;
//            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
//
//            auto true_idx_flat = true_idx_tensor.flat<int>();
//            const int *true_idx = &(true_idx_flat(0));
//
//            auto end_idx_flat = end_idx_tensor.flat<int>();
//            const int *end_idx = &(end_idx_flat(0));
//            
//            auto idx_flat = idx_tensor->flat<int>();
//            int *idx = &(idx_flat(0));
//        assignIdxLauncher(b, m, nsample_, true_idx, end_idx, idx);
//        }
//    private:
//        int nsample_;
//};
//REGISTER_KERNEL_BUILDER(Name("AssignIdx").Device(DEVICE_GPU), AssignIdxGpuOp);

void assignIdxLauncher(int b, int n, int m, int nsample, const bool *true_mat, int *idx);
class AssignIdxGpuOp : public OpKernel {
    public:
        explicit AssignIdxGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("AssignIdx expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& true_mat_tensor = context->input(0);
            OP_REQUIRES(context, true_mat_tensor.dims()==3, errors::InvalidArgument("AssignIdx expects (batch_size, npoint, ndataset) for true_mat."));
            
            int b = true_mat_tensor.shape().dim_size(0); // batch_size
            int m = true_mat_tensor.shape().dim_size(1); // npoint
            int n = true_mat_tensor.shape().dim_size(2); // ndataset

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));

            auto true_mat_flat = true_mat_tensor.flat<bool>();
            const bool *true_mat = &(true_mat_flat(0));
            
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
        assignIdxLauncher(b, n, m, nsample_, true_mat, idx);
        }
    private:
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("AssignIdx").Device(DEVICE_GPU), AssignIdxGpuOp);


void querySquarePointLauncher(int b, int n, int m, float grid_size, int nsample, const float *all_xyz, const float *centroids_xyz, const float *limits, const int *sizes, int *idx, int *pts_cnt, unsigned int *d_keys, unsigned int *d_vals, unsigned int *d_queries, uint2 *d_vals_multivalue);
class QuerySquarePointGpuOp : public OpKernel {
    public:
        explicit QuerySquarePointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("grid_size", &grid_size_));
            OP_REQUIRES(context, grid_size_ > 0, errors::InvalidArgument("QuerySquarePoint expects positive grid size"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QuerySquarePoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& all_xyz_tensor = context->input(0);
            OP_REQUIRES(context, all_xyz_tensor.dims()==3 && all_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QuerySquarePoint expects (batch_size, ndataset, 3) all_xyz_tensor shape."));
            int b = all_xyz_tensor.shape().dim_size(0);
            int n = all_xyz_tensor.shape().dim_size(1);

            const Tensor& centroids_xyz_tensor = context->input(1);
            OP_REQUIRES(context, centroids_xyz_tensor.dims()==3 && centroids_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QuerySquarePoint expects (batch_size, npoint, 3) centroids_xyz shape."));
            int m = centroids_xyz_tensor.shape().dim_size(1);
            
            const Tensor& limits_tensor = context->input(2);
            OP_REQUIRES(context, limits_tensor.dims()==1 && limits_tensor.shape().dim_size(0)==6, errors::InvalidArgument("QuerySquarePoint expects (6) limits shape."))
            
            const Tensor& sizes_tensor = context->input(3);
            OP_REQUIRES(context, sizes_tensor.dims()==1 && sizes_tensor.shape().dim_size(0) == 3, errors::InvalidArgument("QuerySquarePoint expects (3) sizes shape."))

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));
            
            Tensor keys_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({b*n}), &keys_tensor));
            
            Tensor vals_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({b*n}), &vals_tensor));
            
            Tensor vals_multivalue_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({b*m*27*2}), &vals_multivalue_tensor));
            
            Tensor queries_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({b*m*27}), &queries_tensor));
            

            auto all_xyz_flat = all_xyz_tensor.flat<float>();
            const float *all_xyz = &(all_xyz_flat(0));
            
            auto centroids_xyz_flat = centroids_xyz_tensor.flat<float>();
            const float *centroids_xyz = &(centroids_xyz_flat(0));
            
            auto limits_flat = limits_tensor.flat<float>();
            const float *limits = &(limits_flat(0));
            
            auto sizes_flat = sizes_tensor.flat<int>();
            const int *sizes = &(sizes_flat(0));
            
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            
            auto keys_flat = keys_tensor.flat<int>();
            unsigned int *keys = (unsigned int *)&(keys_flat(0));
            
            auto vals_flat = vals_tensor.flat<int>();
            unsigned int *vals = (unsigned int *)&(vals_flat(0));
            
            auto queries_flat = queries_tensor.flat<int>();
            unsigned int *queries = (unsigned int *)&(queries_flat(0));
            
            auto vals_multivalue_flat = vals_multivalue_tensor.flat<int>();
            uint2 *vals_multivalue = reinterpret_cast<uint2*> (&(vals_multivalue_flat(0)));
            
//            printf("Before launcher in cpp\n");
            
            querySquarePointLauncher(b, n, m, grid_size_, nsample_, all_xyz, centroids_xyz, limits, sizes, idx, pts_cnt, keys, vals, queries, vals_multivalue);         
        }
    private:
        float grid_size_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QuerySquarePoint").Device(DEVICE_GPU), QuerySquarePointGpuOp);


void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
class QueryBallPointGpuOp : public OpKernel {
    public:
        explicit QueryBallPointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU), QueryBallPointGpuOp);

void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out);
class SelectionSortGpuOp : public OpKernel {
    public:
        explicit SelectionSortGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("SelectionSort expects positive k"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& dist_tensor = context->input(0);
            OP_REQUIRES(context, dist_tensor.dims()==3, errors::InvalidArgument("SelectionSort expects (b,m,n) dist shape."));
            int b = dist_tensor.shape().dim_size(0);
            int m = dist_tensor.shape().dim_size(1);
            int n = dist_tensor.shape().dim_size(2);

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,n}, &outi_tensor));
            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,n}, &out_tensor));

            auto dist_flat = dist_tensor.flat<float>();
            const float *dist = &(dist_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            selectionSortLauncher(b,n,m,k_,dist,outi,out);
        }
    private:
        int k_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionSort").Device(DEVICE_GPU), SelectionSortGpuOp);


void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            groupPointLauncher(b,n,c,m,nsample,points,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU),GroupPointGpuOp);

void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points);
class GroupPointGradGpuOp: public OpKernel{
    public:
        explicit GroupPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPointGrad expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            const Tensor& grad_out_tensor=context->input(2);
            OP_REQUIRES(context,grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==m && grad_out_tensor.shape().dim_size(2)==nsample && grad_out_tensor.shape().dim_size(3)==c, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample, channel) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,c}, &grad_points_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            cudaMemset(grad_points, 0, sizeof(float)*b*n*c);
            groupPointGradLauncher(b,n,c,m,nsample,grad_out,idx,grad_points);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_GPU),GroupPointGradGpuOp);


