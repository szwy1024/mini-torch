// MPS 矩阵乘法的 C ABI；这里不包含张量求导或后端选择逻辑。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

static MPSMatrix* make_matrix(id<MTLDevice> device, size_t rows, size_t columns)
{
    NSUInteger stride = [MPSMatrixDescriptor rowBytesFromColumns:columns
                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                            columns:columns
                                           rowBytes:stride
                                           dataType:MPSDataTypeFloat32];
    id<MTLBuffer> buffer = [device newBufferWithLength:rows * stride
                                             options:MTLResourceStorageModeShared];
    if (!buffer)
        return nil;
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
}

// MPS 的行跨度可能包含填充，不能把紧密排列的 Python 数组直接整块复制。
static void upload(MPSMatrix* matrix, const float* source, size_t rows, size_t columns)
{
    auto* destination = static_cast<char*>(matrix.data.contents);
    for (size_t row = 0; row < rows; ++row)
        std::memcpy(destination + row * matrix.rowBytes,
                    source + row * columns, columns * sizeof(float));
}

static void download(MPSMatrix* matrix, float* destination, size_t rows, size_t columns)
{
    auto* source = static_cast<const char*>(matrix.data.contents);
    for (size_t row = 0; row < rows; ++row)
        std::memcpy(destination + row * columns,
                    source + row * matrix.rowBytes, columns * sizeof(float));
}

struct Workspace
{
    MPSMatrix* left;
    MPSMatrix* right;
    MPSMatrix* result;
    MPSMatrixMultiplication* kernel;
};

struct MetalRuntime
{
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::map<std::tuple<size_t, size_t, size_t>, Workspace> workspaces;
    std::mutex mutex;

    MetalRuntime(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : device(device), queue(queue)
    {
    }
};

extern "C" void* mini_metal_create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !MPSSupportsMTLDevice(device))
            return nullptr;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            return nullptr;
        return std::make_unique<MetalRuntime>(device, queue).release();
    }
}

extern "C" void mini_metal_destroy(void* context)
{
    std::unique_ptr<MetalRuntime> owner(static_cast<MetalRuntime*>(context));
}

extern "C" const char* mini_metal_device_name(void* context)
{
    return static_cast<MetalRuntime*>(context)->device.name.UTF8String;
}

extern "C" int mini_metal_matmul(void* context, const float* left, const float* right,
                                 float* output, size_t rows, size_t inner, size_t columns)
{
    @autoreleasepool
    {
        auto& runtime = *static_cast<MetalRuntime*>(context);
        // 同一形状复用缓冲区；每次重新上传输入，不能把旧权重当作缓存。
        std::lock_guard<std::mutex> guard(runtime.mutex);
        auto key = std::make_tuple(rows, inner, columns);
        auto found = runtime.workspaces.find(key);
        if (found == runtime.workspaces.end())
        {
            Workspace workspace{
                make_matrix(runtime.device, rows, inner),
                make_matrix(runtime.device, inner, columns),
                make_matrix(runtime.device, rows, columns),
                [[MPSMatrixMultiplication alloc] initWithDevice:runtime.device
                                                    resultRows:rows
                                                 resultColumns:columns
                                               interiorColumns:inner]};
            if (!workspace.left || !workspace.right || !workspace.result || !workspace.kernel)
                return 1; // 无法分配矩阵或创建 kernel。
            found = runtime.workspaces.emplace(key, workspace).first;
        }
        auto& workspace = found->second;
        upload(workspace.left, left, rows, inner);
        upload(workspace.right, right, inner, columns);
        id<MTLCommandBuffer> command = [runtime.queue commandBuffer];
        if (!command)
            return 2;
        [workspace.kernel encodeToCommandBuffer:command
                                     leftMatrix:workspace.left
                                    rightMatrix:workspace.right
                                   resultMatrix:workspace.result];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return 3; // GPU 执行未成功，不读回结果。
        download(workspace.result, output, rows, columns);
        return 0;
    }
}
