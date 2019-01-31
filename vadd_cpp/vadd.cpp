#define DEBUG 1

#define CL_HPP_ENABLE_EXCEPTIONS

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#define HTYPE int
#define MAX_SOURCE_SIZE 100000
#define SIZE 100

const char *getErrorString(cl_int error);
int main()
{
    // define data
    HTYPE a_h[SIZE], b_h[SIZE], c_h[SIZE];
    for (int i = 0; i < SIZE; i++) {
        a_h[i] = i;
        b_h[i] = i;
        c_h[i] = 0;
    }

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform plat = cl::Platform();
        if (platforms.size() > 0)
            plat = platforms[0]; // pick the first platform that's found
        else 
        {
            printf("no OpenCL platform found\n");
            exit(0);
        }

        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        printf("Found %d devices\n", devices.size());

        cl::Context context = cl::Context(devices[0]);
        printf("Context Info:\n");
        std::vector<cl::Device> dev = context.getInfo<CL_CONTEXT_DEVICES>();
        char devName[1000];
        std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Buffer a_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE*sizeof(HTYPE), a_h);
        cl::Buffer b_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE*sizeof(HTYPE), b_h);
        cl::Buffer c_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SIZE*sizeof(HTYPE), c_h);
        cl::CommandQueue q = cl::CommandQueue(context, devices[0], 0);
        //read file
        FILE* vadd_source_file = fopen("vadd.cl", "rb");
        if (vadd_source_file == NULL) printf("Failed to lead kernel source\n");
        const char* vadd_source = (char*) malloc(MAX_SOURCE_SIZE);
        size_t source_size = fread((void*)vadd_source, 1, MAX_SOURCE_SIZE, vadd_source_file);
        cl::Program program = cl::Program(context, vadd_source, CL_TRUE);
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        // create kernel
        cl::Kernel vadd_cl = cl::Kernel(program, "vadd");
        std::cout << vadd_cl.getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
        std::cout << vadd_cl.getInfo<CL_KERNEL_NUM_ARGS>() << std::endl;

        vadd_cl.setArg(0, a_d);
        vadd_cl.setArg(1, b_d);
        vadd_cl.setArg(2, c_d);


        auto q_context =  q.getInfo<CL_QUEUE_CONTEXT>();
        auto q_dev = q_context.getInfo<CL_CONTEXT_DEVICES>();
        auto q_dev_name = q_dev[0].getInfo<CL_DEVICE_NAME>();
        std::cout << q_dev_name << std::endl;


        q.enqueueWriteBuffer(a_d, CL_TRUE, 0, sizeof(HTYPE)*SIZE, a_h);
        q.enqueueWriteBuffer(b_d, CL_TRUE, 0, sizeof(HTYPE)*SIZE, b_h);
        cl::NDRange global = {SIZE};
        q.enqueueNDRangeKernel(vadd_cl, cl::NullRange, global, cl::NullRange);
        q.enqueueReadBuffer(c_d, CL_TRUE, 0, SIZE*sizeof(HTYPE), c_h);
        q.finish();

        for (int i = 0; i < SIZE; i++) printf("c_h[%i] = %i\n", i, c_h[i]);
        
        
    } catch (cl::Error &e) {
        std::cout << getErrorString(e.err()) << std::endl;
    }

    return 0;
}


const char *getErrorString(cl_int error) {
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}


