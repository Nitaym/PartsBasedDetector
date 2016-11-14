/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    main.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 27, 2012
 */

#include <stdio.h>
#include <omp.h>
#include <boost/filesystem.hpp>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
#ifdef WITH_MATLABIO
	#include "MatlabIOModel.hpp"
#endif
#include "Visualize.hpp"
#include "types.hpp"
#include "nms.hpp"
#include "Rect3.hpp"
#include "DistanceTransform.hpp"
using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    // check arguments
    if (argc != 3 && argc != 4) {
        printf("Usage: PartsBasedDetector model_file image_file [depth_file]\n");
        exit(-1);
    }

    // determine the type of model to read
    string ext = boost::filesystem::path(argv[1]).extension().string();

    // load the image from file
    Mat_<float> depth;
    Mat im = imread(argv[2]);
    if (im.empty()) {
        printf("Image not found or invalid image format\n");
        exit(-4);
    }
    if (argc == 4) {
        depth = imread(argv[3], IMREAD_ANYDEPTH);
        // convert the depth image from mm to m
        depth = depth / 1000.0f;
    }

    vector<Candidate> final_candidates;
    string model_name;

    static const int NUM_TESTS_PROCESSORS = 4;
    static const int NUM_TESTS_PER_THREAD = 30;

    PartsBasedDetector<float> pbds[NUM_TESTS_PROCESSORS];
    for (int n = 0; n < NUM_TESTS_PROCESSORS; n++) {
        boost::scoped_ptr<Model> model;
        if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0) {
            model.reset(new FileStorageModel);
        }
#ifdef WITH_MATLABIO
            else if (ext.compare(".mat") == 0) {
            model.reset(new MatlabIOModel);
        }
#endif
        else {
            printf("Unsupported model format: %s\n", ext.c_str());
            exit(-2);
        }
        bool ok = model->deserialize(argv[1]);
        if (!ok) {
            printf("Error deserializing file\n");
            exit(-3);
        }
        pbds[n].distributeModel(*model);
        if (n==0)
            model_name = model->name();
    }


    omp_set_nested(1);
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_TESTS_PROCESSORS);

    //clock_t tic = ::clock();
    std::time_t tic = std::time(NULL);
    #pragma omp parallel for
    for (int n = 0; n < NUM_TESTS_PROCESSORS; n++) {
        vector<Candidate> candidates;

        for (int k = 0; k < NUM_TESTS_PER_THREAD; k++) {
            candidates.clear();
            // detect potential candidates in the image
            pbds[n].detect(im, depth, candidates);
        }

        printf("Number of candidates: %ld\n", candidates.size());

        if (n == 0)
            final_candidates = candidates;
    }
    std::time_t toc = std::time(NULL);


    printf("Total took %.3lf sec. Average detection took %.3lf sec\n",
           std::difftime(toc, tic), std::difftime(toc, tic)/(NUM_TESTS_PROCESSORS*NUM_TESTS_PER_THREAD));

	// display the best candidates
	Visualize visualize(model_name);
	SearchSpacePruning<float> ssp;
        Mat canvas;
	if (final_candidates.size() > 0) {
	    Candidate::sort(final_candidates);
	    //Candidate::nonMaximaSuppression(im, candidates, 0.2);
	    visualize.candidates(im, final_candidates, canvas, true);
        //visualize.image(canvas);
        cvtColor(canvas, canvas, cv::COLOR_RGB2BGR);
        imwrite("detected.jpg", canvas);
	    // waitKey();
	}
	return 0;
}
