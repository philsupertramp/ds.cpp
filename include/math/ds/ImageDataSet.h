#pragma once

#include "DataSet.h"

//! only include Magick++ if needed
#if MATH_IMAGE_PROCESSING
  #include <Magick++.h>
  #include <Magick++/Image.h>
#endif

#include <filesystem>

/**
 * Image data set representation
 */
class ImageDataSet : public DataSet
{
public:
  /**
   * default constructor
   * @param inputCount
   * @param outputCount
   */
  ImageDataSet(size_t inputCount, size_t outputCount)
    : DataSet(inputCount, outputCount) { }

  /**
   * prepares a new set of data based on passed filePath
   * @param filePath path name of directory to prepare
   */
  virtual void PrepareDirectory(const char* filePath) {
    /** identifies "classes" as files in nested directories */

    std::cout << "Moving files..." << std::flush;
    std::filesystem::remove_all(trainDirectory);
    std::filesystem::copy(
    filePath,
    trainDirectory,
    std::filesystem::copy_options::update_existing | std::filesystem::copy_options::recursive);

    totalCount        = 0;
    size_t classCount = 0;
    classNames.clear();
    for(const auto& entry : std::filesystem::directory_iterator(trainDirectory)) {
      if(entry.is_directory()) {
        classNames.push_back(entry.path().filename());
        auto dirIter = std::filesystem::directory_iterator(entry);
        totalCount += std::count_if(begin(dirIter), end(dirIter), [](auto& elem) { return elem.is_regular_file(); });
        classCount += 1;
      }
    }
    trainingCount     = (int)(totalCount * 0.8);
    validationCount   = (int)(totalCount - trainingCount);
    Training.Input    = MatrixDS<double>(0, trainingCount, InputCount);
    Training.Output   = MatrixDS<double>(0, trainingCount, OutputCount);
    Validation.Input  = MatrixDS<double>(0, validationCount, InputCount);
    Validation.Output = MatrixDS<double>(0, validationCount, OutputCount);

    size_t index = 0;
    for(const auto& entry : std::filesystem::directory_iterator(trainDirectory)) {
      if(entry.is_directory()) {
        MatrixDS<bool> elem(false, 1, classCount);
        elem(0, index) = true;
        classes.push_back(elem);
        index += 1;
      }
    }
    std::cout << "done!\nDirectory \"" << trainDirectory << "\" ready." << std::endl;
    std::cout << "Found " << totalCount << " files belonging to " << classCount << " classes.\nUsing "
              << validationCount << " files for validation\n"
              << std::flush;
  }

  /**
   * loads data set into memory
   *
   *
   * we split the dataset into
   * - 80% training
   * - 20% validation
   * -  0% test
   *
   * Example: 100 images => 80 training, 20 validation, 0 test
   */
  void Cache() {
#if MATH_IMAGE_PROCESSING
    std::cout << "Resizing files and saving into memory..." << std::flush;
    size_t count = 0, i = 0, classCount = 0;
    size_t trainingIter = 0, validationIter = 0;
    for(const auto& classDirectory : std::filesystem::directory_iterator(trainDirectory)) {
      bool hadEntry = false;
      if(!classDirectory.is_directory() || classCount >= classes.size()) continue;
      for(const auto& entry : std::filesystem::directory_iterator(classDirectory)) {
        if(!entry.is_regular_file()) continue;

        hadEntry = true;
        Magick::Image image;
        image.verbose(verbose);
        image.magick("JPG");
        image.read(std::string(entry.path()));
        // PZ: we don't scale but resample, to enforce dimension width * height and ignore loss in content
        image.resample(Magick::Point(imageWidth, imageHeight));
        Set* target;
        bool is_validation =
        (trainingIter >= trainingCount || ((count % (int)totalCount * (validationShare)) == 0 && count != 0 && validationIter < validationCount));
        target = is_validation ? &Validation : &Training;
        Magick::PixelData pixelBlob(image, "RGBA", Magick::FloatPixel);

        auto* in = (float*)pixelBlob.data();

        for(i = 0; i < InputCount; i++) { (*target).Input[is_validation ? validationIter : trainingIter][i] = in[i]; }
        for(i = 0; i < OutputCount; i++) {
          (*target).Output[is_validation ? validationIter : trainingIter][i] = classes[classCount][0][i];
        }
        if(is_validation) {
          validationIter += 1;
        } else {
          trainingIter += 1;
        }
        count += 1;
      }
      classCount += hadEntry;
    }
#endif
  }

public:
  //! desired image height
  size_t imageHeight = 180;
  //! desired image width
  size_t imageWidth = 180;
  //! percentage of validation data
  double validationShare = 0.2;
  //! target directory for training
  const char* trainDirectory = "../../resources/image_classification/training/";
  //! representation of all classes
  std::vector<MatrixDS<bool>> classes;
  //! representation of all class names
  std::vector<std::string> classNames;
  //! total number of data records
  size_t totalCount = 0;
  //! number of training data records
  size_t trainingCount = 0;
  //! number of validation data records
  size_t validationCount = 0;
};
