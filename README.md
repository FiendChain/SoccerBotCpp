## Introduction
Test app using tensorflow-lite. App for loading in tflite model from soccerbot project (https://github.com/FiendChain/soccerbot).

## Compiling
***compiler***: msys2 mingw64 gcc/g++ version 9.2.0

***libraries***:
- tensorflow-lite static library 
    - use fork with mingw64 fixes (https://github.com/FiendChain/tensorflow.git)
    - replaced ```<sys/mman.h>``` with (https://github.com/witwall/mman-win32.git)
    - follow guide for raspberry pi (https://www.tensorflow.org/lite/guide/build_rpi) to download dependencies
    - fixed defines to exclude missing dependency ```<bitswap.h>```
        - file: tensorflow/lite/tools/make/downloads/farmhash/src/farmhash.cc
        - ```c
            #else
            // line 170
            // #undef bswap_32
            // #undef bswap_64
            // #include <byteswap.h>

            // our new define (from MSVC define block)
            #undef bswap_32
            #undef bswap_64
            #define bswap_32(x) _byteswap_ulong(x)
            #define bswap_64(x) _byteswap_uint64(x)
            // end edit
            #endif
            ```
    - run build_lib.sh instead of build_rpi_lib.sh (refer to linked guide)
    - built static library is located in ```tensorflow/lite/tools/make/gen/windows_x86_64/lib/```
    - move built libtensorflow-lite.a into ```vendor/tensorflow/lib/```
    - create include headers for tensorflow-lite
        - ```rsync -avm --include='*.h' -f 'hide,! */' $(TENSORREPO)/tensorflow/lite/ $(PROJECTREPO)/vendor/tensorflow/include/tensorflow/lite/```
    - copy ```flatbuffers``` library to project 
      - ```cp -R $(TENSORREPO)/tensorflow/lite/tools/make/downloads/flatbuffers/include/flatbuffers $(PROJECTREPO)/vendor/tensorflow/include/flatbuffers```
- stb image library (https://github.com/nothings/stb)
  - install to vendor/stb

## Specs
- Binary size ~5.0MB
- Considerably smaller than a python script running a TFLite model