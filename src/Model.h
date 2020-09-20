#ifndef _MODEL_H_
#define _MODEL_H_

#include <cstdint>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

class Model 
{
public:
    typedef struct Output {
        float x;
        float y;
        float confidence;
    } Output;
public:
    Model(const char *filepath);
    Output predict(uint8_t *image, int image_width, int image_height, int image_channels);
    void show_info();
private:
    int mInputWidth, mInputHeight, mInputChannels, mInputSize;
    std::vector<uint8_t> mResizeBuffer;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
    float *mInputBuffer;
    float *mOutputBuffer;
};

#endif