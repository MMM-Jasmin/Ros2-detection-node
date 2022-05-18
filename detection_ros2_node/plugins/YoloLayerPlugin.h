#ifndef YOLO_LAYER_PLUGIN_H
#define YOLO_LAYER_PLUGIN_H

#include <NvInfer.h>
#include <string>
#include <vector>

#define MAX_ANCHORS 6

#ifndef NOEXCEPT
#if NV_TENSORRT_MAJOR >= 8
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif
#endif

namespace Yolo
{
struct alignas(float) Detection
{
	float bbox[4]; // x, y, w, h
	float det_confidence;
	float class_id;
	float class_confidence;
};
} // namespace Yolo

namespace nvinfer1
{
class YoloLayerPlugin : public IPluginV2IOExt
{
public:
	YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y);
	YoloLayerPlugin(const void* data, size_t length);

	~YoloLayerPlugin() override = default;

	int getNbOutputs() const NOEXCEPT override { return 1; }

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override;

	int initialize() NOEXCEPT override { return 0; }

	void terminate() NOEXCEPT override;

	size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override { return 0; }

#if NV_TENSORRT_MAJOR >= 8
	int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override;
#else
	int enqueue(int batchSize, const void* const * inputs, void** outputs, void* workspace, cudaStream_t stream) NOEXCEPT override;
#endif

	size_t getSerializationSize() const NOEXCEPT override;

	void serialize(void* buffer) const NOEXCEPT override;

	bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const NOEXCEPT override
	{
		return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
	}

	const char* getPluginType() const NOEXCEPT override { return "YoloLayer_TRT"; }

	const char* getPluginVersion() const NOEXCEPT override { return "1"; }

	void destroy() NOEXCEPT override { delete this; }

	IPluginV2IOExt* clone() const NOEXCEPT override;

	void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override { mPluginNamespace = pluginNamespace; }

	const char* getPluginNamespace() const NOEXCEPT override { return mPluginNamespace; }

	DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const NOEXCEPT override { return DataType::kFLOAT; }

	bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const NOEXCEPT override { return false; }

	bool canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT override { return false; }

	void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) NOEXCEPT override {}

	void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) NOEXCEPT override {}

	void detachFromContext() NOEXCEPT override {}

private:
	void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

	int mThreadCount = 64;
	int mYoloWidth, mYoloHeight, mNumAnchors;
	float mAnchorsHost[MAX_ANCHORS * 2];
	float* mAnchors; // allocated on GPU
	int mNumClasses;
	int mInputWidth, mInputHeight;
	float mScaleXY;

	const char* mPluginNamespace;
};

class YoloPluginCreator : public IPluginCreator
{
public:
	YoloPluginCreator();

	~YoloPluginCreator() override = default;

	const char* getPluginName() const NOEXCEPT override;

	const char* getPluginVersion() const NOEXCEPT override;

	const PluginFieldCollection* getFieldNames() NOEXCEPT override;

	IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override;

	IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override;

	void setPluginNamespace(const char* libNamespace) NOEXCEPT override
	{
		mNamespace = libNamespace;
	}

	const char* getPluginNamespace() const NOEXCEPT override
	{
		return mNamespace.c_str();
	}

private:
	static PluginFieldCollection mFC;
	static std::vector<nvinfer1::PluginField> mPluginAttributes;
	std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

}; // namespace nvinfer1

#endif // YOLO_LAYER_PLUGIN_H
