#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include "checkSizes.h"


/**
 * @brief Error checks the sizes given by the user.
 * 
 * @param Train The number of training examples.
 * @param Test The number of testing examples.
 * @param Hidden The number of hidden neurons.
 * @param Epochs The number of epochs to run.
 * @param LearningRate The learning rate.
 * @param AccuracyThreshold The accuracy threshold.
 * 
 * @return exit(-1) if one or more sizes are invalid.
 */
void checkSizes( long long &Train,
                 long long &Test,
                 long long &Hidden,
                 long long &Epochs,
                 long long &Convolutions,
                 float &LearningRate,
                 float &AccuracyThreshold   ) {
  if (Train==-1) {
    fprintf(stderr, "  No size given, using default size Train = 800\n");
    Train = 800;
  }
  if (Test==-1) {
    fprintf(stderr, "  No size given, using default size Test = 200\n");
    Test = 200;
  }
  if (Hidden==-1) {
    fprintf(stderr, "  No size given, using default size Hidden = 47\n");
    Hidden = 47;
  }
  if (Epochs==-1) {
    fprintf(stderr, "  No size given, using default size Epochs = 10\n");
    Epochs = 10;
  }
  if (Convolutions==-1) {
    fprintf(stderr, "  No size given, using default size Convolutions = 1\n");
    Convolutions = 1;
  }
  if (LearningRate==-0.0f) {
    fprintf(stderr, "  No learning rate given, using default learning rate = 0.01\n");
    LearningRate = 0.1f;
  }
  if (AccuracyThreshold==-0.0f) {
    fprintf(stderr, "  No accuracy threshold given, using default accuracy threshold = 100.1 to run all epochs\n");
    AccuracyThreshold = 100.1f;
  }
}