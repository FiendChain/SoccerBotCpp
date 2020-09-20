#include "Model.h"

#include <cstdio>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include <vector>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

Model::Model(const char *filepath) {
  // Load model
  mModel = tflite::FlatBufferModel::BuildFromFile(filepath);
  if (mModel == NULL) {
    fprintf(stderr, "Failed to load model: %s\n", filepath);
    exit(1);
  }

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*mModel, resolver);
  builder(&mInterpreter);
  TFLITE_MINIMAL_CHECK(mInterpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(mInterpreter->AllocateTensors() == kTfLiteOk);
  
  // Check if IO exists
  const auto& inputs = mInterpreter->inputs();
  const auto& outputs = mInterpreter->outputs();

  TFLITE_MINIMAL_CHECK(inputs.size() >= 1);
  TFLITE_MINIMAL_CHECK(outputs.size() >= 1);

  const auto& input_tensor = mInterpreter->input_tensor(0);
  const auto& input_dims = input_tensor->dims;
  mInputHeight = input_dims->data[1];
  mInputWidth = input_dims->data[2];
  mInputChannels = input_dims->data[3];
  mInputSize = mInputWidth * mInputHeight * mInputChannels;
  
  mResizeBuffer = std::vector<uint8_t>(mInputSize);

  mInputBuffer = mInterpreter->typed_input_tensor<float>(0);
  mOutputBuffer = mInterpreter->typed_output_tensor<float>(0);
}

Model::Output Model::predict(uint8_t *image_buffer, int image_width, int image_height, int image_channels) {
  stbir_resize_uint8(
    image_buffer, image_width, image_height , 0,
    mResizeBuffer.data(), mInputWidth, mInputHeight, 0, 
    mInputChannels);

  for (int i = 0; i < mInputSize; i++) {
    mInputBuffer[i] = (float)mResizeBuffer[i] / 255.0;
  }

  TFLITE_MINIMAL_CHECK(mInterpreter->Invoke() == kTfLiteOk);

  Model::Output output;
  output.confidence = mOutputBuffer[2];
  output.x = mOutputBuffer[0];
  output.y = mOutputBuffer[1];
  return output;
}

void Model::show_info() {
  printf("Model minimum runtime: '%s'\n", mModel->GetMinimumRuntime().c_str());
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(mInterpreter.get());

  const auto& inputs = mInterpreter->inputs();
  const auto& outputs = mInterpreter->outputs();
  printf("total inputs: %d\n", inputs.size());
  printf("total outputs: %d\n", outputs.size());

  for (int i = 0; i < inputs.size(); i++) {
    const auto& tensor = mInterpreter->input_tensor(i);
    const auto& dims = tensor->dims;
    printf("input[%d] = %s ( ", i, TfLiteTypeGetName(tensor->type));
    for (int j = 0; j < dims->size; j++) {
      printf("%d ", dims->data[j]);
    }
    printf(") node=%d\n", inputs[i]);
  }

  for (int i = 0; i < outputs.size(); i++) {
    const auto& tensor = mInterpreter->output_tensor(i);
    const auto& dims = tensor->dims;
    printf("output[%d] = %s ( ", i, TfLiteTypeGetName(tensor->type));
    for (int j = 0; j < dims->size; j++) {
      printf("%d ", dims->data[j]);
    }
    printf(") node=%d\n", outputs[i]);
  }
}


