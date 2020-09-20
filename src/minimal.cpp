#include <cstdio>
#include "Model.h"

#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

#include <stdlib.h>
#include <stdio.h>

namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

bool sort_paths(fs::path& a, fs::path& b) {
  return (a.string().compare(b.string()) < 0);
}

int main(int argc, char* argv[]) {
  const char *DEFAULT_MODEL_PATH = "C:/Users/acidi/Coding/Projects/SoccerBot/assets/models/cnn_113_80_quantized.tflite";
  const char *DEFAULT_OUTPUT_PATH = "C:/Users/acidi/Downloads/tensorflow-lite-r2.3/output.log";

  if (argc <= 1) {
    fprintf(stderr, "%s <image_dir> [tflite model] [output log]\n", argv[0]);
    return 1;
  }
  const char *filename = (argc >= 3) ? argv[2] : DEFAULT_MODEL_PATH;
  const char *output_filepath = (argc >= 4) ? argv[3] : DEFAULT_OUTPUT_PATH;
  

  fs::path root_dir(argv[1]);

  if (!fs::is_directory(root_dir)) {
    fprintf(stderr, "%s is not a valid directory\n", argv[1]);
    return 1;
  }

  printf("Loading model: %s\n", filename);
  Model model(filename);

  printf("log file: %s\n", output_filepath);
  std::ofstream log_file(output_filepath);
  log_file << "name x y confidence\n";
  // FILE *fp = fopen(output_filepath, "w+");
  // fprintf(fp, "name x y confidence\n");

  std::vector<fs::path> paths;

  for (auto&p: fs::directory_iterator(root_dir.string())) {
    if (!p.is_regular_file()) {
      continue;
    }

    paths.push_back(p.path());
  }

  std::sort(paths.begin(), paths.end(), sort_paths);

  const int N = paths.size();

  for (int i = 0; i < N; i++) {
    const auto& p = paths[i];

    const std::string &image_filepath = p.string(); 
    const std::string &image_filename = p.filename().string();

    int x, y, src_channels;
    int channels = 3;
    uint8_t *image_buffer = stbi_load(image_filepath.c_str(), &x, &y, &src_channels, channels);
    if (image_buffer == NULL) {
      printf("unable to open image: %s\n", image_filepath.c_str());
      continue;
    }

    if (src_channels < channels) {
      printf("image %s has insufficient channels %d (required %d)\n", 
        image_filepath.c_str(), src_channels, channels);
      continue;
    }

    // printf("Loaded image: %s (%dx%dx%d)\n", image_filepath.c_str(), x, y, channels);
    const auto& output = model.predict(image_buffer, x, y, channels);
    stbi_image_free(image_buffer);
    // printf("%s x=%.3f y=%.3f confidence=%.3f\n", 
    //   image_filepath.c_str(), output.x, output.y, output.confidence);

    log_file << image_filename << ' ' << output.x << ' ' << output.y << ' ' << output.confidence << '\n';
    // fprintf(fp, "%s %.5f %.5f %.5f\n", image_filename.c_str(), output.x, output.y, output.confidence);

    if (i % 100 == 0) {
      printf("%d/%d\r", i, N);
    }
  }
  printf("\n");

  log_file.close();
  // fclose(fp);


  return 0;
}
