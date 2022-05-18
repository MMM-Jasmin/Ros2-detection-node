/*
 * yolo_layer.cu
 *
 * This code was originally written by wang-xinyu under MIT license.
 * I took it from:
 *
 *     https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4
 *
 * and made necessary modifications.
 *
 * - JK Jung
 */

#include "YoloLayerPlugin.h"

#include <cassert>
#include <iostream>
#include <math_constants.h>

namespace
{
// Write values into buffer
template<typename T>
void write(char*& buffer, const T& val)
{
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

// Read values from buffer
template<typename T>
void read(const char*& buffer, T& val)
{
	val = *reinterpret_cast<const T*>(buffer);
	buffer += sizeof(T);
}

#define CHECK_YOLO(status)                                    \
	do                                                        \
	{                                                         \
		auto ret = status;                                    \
		if (ret != 0)                                         \
		{                                                     \
			std::cerr << "Cuda failure in file '" << __FILE__ \
					  << "' line " << __LINE__                \
					  << ": " << ret << std::endl;            \
			abort();                                          \
		}                                                     \
	} while (0)
} // namespace

namespace nvinfer1
{
YoloLayerPlugin::YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y)
{
	mYoloWidth  = yolo_width;
	mYoloHeight = yolo_height;
	mNumAnchors = num_anchors;
	memcpy(mAnchorsHost, anchors, num_anchors * 2 * sizeof(float));
	mNumClasses  = num_classes;
	mInputWidth  = input_width;
	mInputHeight = input_height;
	mScaleXY     = scale_x_y;

	CHECK_YOLO(cudaMalloc(&mAnchors, MAX_ANCHORS * 2 * sizeof(float)));
	CHECK_YOLO(cudaMemcpy(mAnchors, mAnchorsHost, mNumAnchors * 2 * sizeof(float), cudaMemcpyHostToDevice));
}

YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
{
	const char* d = reinterpret_cast<const char*>(data);
#ifndef NDEBUG
	const char* a = d;
#endif
	read(d, mThreadCount);
	read(d, mYoloWidth);
	read(d, mYoloHeight);
	read(d, mNumAnchors);
	memcpy(mAnchorsHost, d, MAX_ANCHORS * 2 * sizeof(float));
	d += MAX_ANCHORS * 2 * sizeof(float);
	read(d, mNumClasses);
	read(d, mInputWidth);
	read(d, mInputHeight);
	read(d, mScaleXY);

	CHECK_YOLO(cudaMalloc(&mAnchors, MAX_ANCHORS * 2 * sizeof(float)));
	CHECK_YOLO(cudaMemcpy(mAnchors, mAnchorsHost, mNumAnchors * 2 * sizeof(float), cudaMemcpyHostToDevice));

	assert(d == a + length);
}

void YoloLayerPlugin::serialize(void* buffer) const NOEXCEPT
{
	char* d = static_cast<char*>(buffer);
#ifndef NDEBUG
	char* a = d;
#endif
	write(d, mThreadCount);
	write(d, mYoloWidth);
	write(d, mYoloHeight);
	write(d, mNumAnchors);
	memcpy(d, mAnchorsHost, MAX_ANCHORS * 2 * sizeof(float));
	d += MAX_ANCHORS * 2 * sizeof(float);
	write(d, mNumClasses);
	write(d, mInputWidth);
	write(d, mInputHeight);
	write(d, mScaleXY);

	assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const NOEXCEPT
{
	return sizeof(mThreadCount) +
		   sizeof(mYoloWidth) + sizeof(mYoloHeight) +
		   sizeof(mNumAnchors) + MAX_ANCHORS * 2 * sizeof(float) +
		   sizeof(mNumClasses) +
		   sizeof(mInputWidth) + sizeof(mInputHeight) +
		   sizeof(mScaleXY);
}

void YoloLayerPlugin::terminate() NOEXCEPT
{
	CHECK_YOLO(cudaFree(mAnchors));
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT
{
	assert(index == 0);
	assert(nbInputDims == 1);
	assert(inputs[0].d[0] == (mNumClasses + 5) * mNumAnchors);
	assert(inputs[0].d[1] == mYoloHeight);
	assert(inputs[0].d[2] == mYoloWidth);
	// output detection results to the channel dimension
	int totalsize = mYoloWidth * mYoloHeight * mNumAnchors * sizeof(Yolo::Detection) / sizeof(float);
	return Dims3(totalsize, 1, 1);
}

// Clone the plugin
IPluginV2IOExt* YoloLayerPlugin::clone() const NOEXCEPT
{
	YoloLayerPlugin* p = new YoloLayerPlugin(mYoloWidth, mYoloHeight, mNumAnchors, (float*)mAnchorsHost, mNumClasses, mInputWidth, mInputHeight, mScaleXY);
	p->setPluginNamespace(mPluginNamespace);
	return p;
}

inline __device__ float sigmoidGPU(float x)
{
	return 1.0f / (1.0f + __expf(-x));
}

inline __device__ float scale_sigmoidGPU(float x, float scale)
{
	return scale * sigmoidGPU(x) - (scale - 1.0f) * 0.5f;
}

// CalDetection(): This kernel processes 1 yolo layer calculation.  It
// distributes calculations so that 1 GPU thread would be responsible
// for each grid/anchor combination.
// NOTE: The output (x, y, w, h) are between 0.0 and 1.0
//       (relative to orginal image width and height).
__global__ void CalDetection(const float* input, float* output, int yolo_width, int yolo_height, int num_anchors,
							 const float* anchors, int num_classes, int input_w, int input_h, float scale_x_y)
{
	int idx         = threadIdx.x + blockDim.x * blockIdx.x;
	Yolo::Detection* det  = ((Yolo::Detection*)output) + idx;
	int total_grids = yolo_width * yolo_height;
	if (idx >= total_grids * num_anchors) return;

	int anchor_idx         = idx / total_grids;
	idx                    = idx - total_grids * anchor_idx;
	int info_len           = 5 + num_classes;
	const float* cur_input = input + anchor_idx * (info_len * total_grids);

	int class_id;
	float max_cls_logit = -CUDART_INF_F; // minus infinity
	for (int i = 5; i < info_len; ++i)
	{
		float l = cur_input[idx + i * total_grids];
		if (l > max_cls_logit)
		{
			max_cls_logit = l;
			class_id      = i - 5;
		}
	}
	float max_cls_prob = sigmoidGPU(max_cls_logit);
	float box_prob     = sigmoidGPU(cur_input[idx + 4 * total_grids]);

	int row = idx / yolo_width;
	int col = idx % yolo_width;

	det->bbox[0] = (col + scale_sigmoidGPU(cur_input[idx + 0 * total_grids], scale_x_y)) / yolo_width;  // [0, 1]
	det->bbox[1] = (row + scale_sigmoidGPU(cur_input[idx + 1 * total_grids], scale_x_y)) / yolo_height; // [0, 1]
	det->bbox[2] = __expf(cur_input[idx + 2 * total_grids]) * anchors[2 * anchor_idx] / input_w;        // [0, 1]
	det->bbox[3] = __expf(cur_input[idx + 3 * total_grids]) * anchors[2 * anchor_idx + 1] / input_h;    // [0, 1]

	det->bbox[0] -= det->bbox[2] / 2; // shift from center to top-left
	det->bbox[1] -= det->bbox[3] / 2;

	det->det_confidence   = box_prob;
	det->class_id         = class_id;
	det->class_confidence = max_cls_prob;
}

void YoloLayerPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize)
{
	int num_elements = batchSize * mNumAnchors * mYoloWidth * mYoloHeight;

	CalDetection<<<(num_elements + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>(inputs[0], output, mYoloWidth, mYoloHeight, mNumAnchors, (const float*)mAnchors, mNumClasses, mInputWidth, mInputHeight, mScaleXY);
}

#if NV_TENSORRT_MAJOR >= 8
int32_t YoloLayerPlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT
#else   // NV_TENSORRT_MAJOR < 8
int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
#endif  // NV_TENSORRT_MAJOR
{
	forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
	return 0;
}

YoloPluginCreator::YoloPluginCreator()
{
	mPluginAttributes.clear();

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields   = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const NOEXCEPT
{
	return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const NOEXCEPT
{
	return "1";
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames() NOEXCEPT
{
	return &mFC;
}

IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT
{
	assert(!strcmp(name, getPluginName()));
	const PluginField* fields = fc->fields;
	int yolo_width, yolo_height, num_anchors = 0;
	float anchors[MAX_ANCHORS * 2];
	int num_classes;
	int input_width, input_height;
	float scale_x_y = 1.0;

	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "yoloWidth"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			yolo_width = *(static_cast<const int*>(fields[i].data));
		}
		else if (!strcmp(attrName, "yoloHeight"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			yolo_height = *(static_cast<const int*>(fields[i].data));
		}
		else if (!strcmp(attrName, "numAnchors"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			num_anchors = *(static_cast<const int*>(fields[i].data));
		}
		else if (!strcmp(attrName, "numClasses"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			num_classes = *(static_cast<const int*>(fields[i].data));
		}
		else if (!strcmp(attrName, "inputWidth"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			input_width = *(static_cast<const int*>(fields[i].data));
		}
		else if (!strcmp(attrName, "inputHeight"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			input_height = *(static_cast<const int*>(fields[i].data));
		}
		else if (!strcmp(attrName, "anchors"))
		{
			assert(num_anchors > 0 && num_anchors <= MAX_ANCHORS);
			assert(fields[i].type == PluginFieldType::kFLOAT32);
			memcpy(anchors, static_cast<const float*>(fields[i].data), num_anchors * 2 * sizeof(float));
		}
		else if (!strcmp(attrName, "scaleXY"))
		{
			assert(fields[i].type == PluginFieldType::kFLOAT32);
			scale_x_y = *(static_cast<const float*>(fields[i].data));
		}
	}
	assert(yolo_width > 0 && yolo_height > 0);
	assert(anchors[0] > 0.0f && anchors[1] > 0.0f);
	assert(num_classes > 0);
	assert(input_width > 0 && input_height > 0);
	assert(scale_x_y >= 1.0);

	YoloLayerPlugin* obj = new YoloLayerPlugin(yolo_width, yolo_height, num_anchors, anchors, num_classes, input_width, input_height, scale_x_y);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT
{
	YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;
} // namespace nvinfer1
